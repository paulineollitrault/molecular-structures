"""
Transition1x-style NEB sampling pipeline for a single reaction.

Pipeline for one (R, TS, P) triplet:
  1. IDPP-interpolate a two-segment path R -> TS -> P
  2. NEB (no climbing) to a loose Fmax convergence
  3. CI-NEB (climbing) to a tighter Fmax convergence
  4. While optimizing, accumulate the per-iteration Fmax. Whenever the
     cumulative sum since the last save crosses `save_threshold`, the full
     band (positions, energies, forces) is dumped to an HDF5 group.

HDF5 layout (mirrors Transition1x):
    /atomic_numbers           (n_atoms,)               int
    /iter_0000/
        positions             (n_images, n_atoms, 3)   float, Angstrom
        energies              (n_images,)              float, eV
        forces                (n_images, n_atoms, 3)   float, eV/Angstrom
        attrs: stage, neb_iter, fmax, cumulative_fmax
    /iter_0001/
        ...
    attrs: charge, spin, xc, basis, n_images, k_spring, comment
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import h5py
import numpy as np
from ase import Atoms
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS


# ---------------------------------------------------------------------------
# Initial-path construction
# ---------------------------------------------------------------------------

def relax_endpoint(
    atoms: Atoms,
    calc,
    fmax: float = 0.01,
    max_steps: int = 200,
    label: str = "endpoint",
    logger=print,
) -> Atoms:
    """Local minimum of `atoms` under `calc`. Returns the optimized copy."""
    a = atoms.copy()
    a.calc = calc
    opt = BFGS(a, logfile=None)
    t0 = time.time()
    opt.run(fmax=fmax, steps=max_steps)
    logger(f"[neb] relaxed {label}: fmax->{(a.get_forces()**2).sum(axis=1).max()**0.5:.4f} "
           f"E={a.get_potential_energy():.4f} eV ({time.time()-t0:.1f}s, "
           f"steps={opt.nsteps}, converged={opt.converged()})")
    return a


def build_idpp_path(
    reactant: Atoms,
    ts: Atoms,
    product: Atoms,
    n_images: int = 12,
) -> List[Atoms]:
    """Build a two-segment IDPP-interpolated path R -> TS -> P.

    `n_images` is the total number of images including R, TS, and P; the TS
    is shared between the two halves. Requires n_images >= 5.
    """
    if n_images < 5:
        raise ValueError("n_images must be >= 5 to fit R, TS, P plus interior")
    n_left = n_images // 2 + 1          # images in R..TS segment, inclusive
    n_right = n_images - n_left + 1     # images in TS..P segment, inclusive

    def _interp_segment(a: Atoms, b: Atoms, n: int) -> List[Atoms]:
        images = [a.copy()] + [a.copy() for _ in range(n - 2)] + [b.copy()]
        NEB(images, allow_shared_calculator=True).interpolate(method="idpp")
        return images

    left = _interp_segment(reactant, ts, n_left)
    right = _interp_segment(ts, product, n_right)
    path = left + right[1:]  # drop duplicate TS at the seam
    assert len(path) == n_images
    return path


# ---------------------------------------------------------------------------
# Saving rule
# ---------------------------------------------------------------------------

@dataclass
class SaveState:
    """Tracks cumulative Fmax and writes saved iterations to HDF5."""

    h5_path: str
    save_threshold: float = 0.1
    cumulative: float = 0.0
    save_idx: int = 0
    iter_idx: int = 0
    history: List[dict] = field(default_factory=list)

    def open(self, atoms_ref: Atoms, meta: dict):
        os.makedirs(os.path.dirname(self.h5_path), exist_ok=True)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset(
                "atomic_numbers",
                data=np.asarray(atoms_ref.get_atomic_numbers(), dtype=np.int32),
            )
            for k, v in meta.items():
                f.attrs[k] = v

    def _maybe_save(self, neb: NEB, fmax: float, stage: str, force_save: bool = False):
        self.cumulative += fmax
        save_now = force_save or self.cumulative >= self.save_threshold

        if save_now:
            n_imgs = len(neb.images)
            n_atoms = len(neb.images[0])
            positions = np.empty((n_imgs, n_atoms, 3), dtype=float)
            forces = np.empty((n_imgs, n_atoms, 3), dtype=float)
            energies = np.empty(n_imgs, dtype=float)

            # Reuse already-computed values from the most recent NEB.get_forces():
            # neb.real_forces is (n_imgs, n_atoms, 3) with endpoints zeroed; neb.energies (n_imgs,).
            # We refill endpoints from the calculators (cheap — they're cached).
            for i, img in enumerate(neb.images):
                positions[i] = img.get_positions()
            energies[:] = neb.energies
            forces[:] = neb.real_forces
            # Replace endpoint forces (which NEB leaves as zero) with the true PES forces
            for i in (0, len(neb.images) - 1):
                try:
                    forces[i] = neb.images[i].get_forces()
                except Exception:
                    forces[i] = 0.0

            with h5py.File(self.h5_path, "a") as f:
                grp = f.create_group(f"iter_{self.save_idx:04d}")
                grp.create_dataset("positions", data=positions)
                grp.create_dataset("energies", data=energies)
                grp.create_dataset("forces", data=forces)
                grp.attrs["stage"] = stage
                grp.attrs["neb_iter"] = self.iter_idx
                grp.attrs["fmax"] = fmax
                grp.attrs["cumulative_fmax"] = self.cumulative

            self.history.append(
                dict(
                    save_idx=self.save_idx,
                    iter=self.iter_idx,
                    stage=stage,
                    fmax=fmax,
                    cumulative_fmax=self.cumulative,
                )
            )
            self.save_idx += 1
            self.cumulative = 0.0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class NEBConfig:
    n_images: int = 12
    k_spring: float = 0.1
    method: str = "improvedtangent"
    neb_fmax: float = 0.15           # 0.1-0.2 eV/A range per strategy doc Sec. 4
    max_steps_neb: int = 500
    save_threshold: float = 0.1
    save_first_iter: bool = True
    # BFGS step control (Schreiner et al. used alpha=70, maxstep=0.03)
    bfgs_alpha: float = 70.0
    bfgs_maxstep: float = 0.03
    # Endpoint relaxation (strategy doc step 1)
    relax_endpoints: bool = True
    endpoint_fmax: float = 0.05      # eV/A; 0.01 is paper, 0.05 is fine for sampling
    endpoint_max_steps: int = 200


def attach_calculators(
    images: List[Atoms],
    calc_factory: Optional[Callable[[], object]] = None,
    shared_calc: Optional[object] = None,
):
    """Attach calculators to every image. Either provide a factory that
    returns a fresh calculator per image, or a single shared calculator
    instance (used when the calculator is stateless / heavy to load)."""
    if (calc_factory is None) == (shared_calc is None):
        raise ValueError("exactly one of calc_factory or shared_calc required")
    for img in images:
        img.calc = shared_calc if shared_calc is not None else calc_factory()


def run_neb_pipeline(
    reactant: Atoms,
    ts: Atoms,
    product: Atoms,
    h5_path: str,
    calc_factory: Optional[Callable[[], object]] = None,
    shared_calc: Optional[object] = None,
    cfg: NEBConfig = NEBConfig(),
    meta: Optional[dict] = None,
    logger=print,
):
    """Run the plain-NEB pipeline (no CI-NEB) and save iteration snapshots.

    Provide exactly one of `calc_factory` (one fresh calculator per image) or
    `shared_calc` (single calculator instance used by every image).
    """
    if (calc_factory is None) == (shared_calc is None):
        raise ValueError("exactly one of calc_factory or shared_calc required")

    if cfg.relax_endpoints:
        # For endpoint relaxation we attach a calculator instance per endpoint.
        # When `shared_calc` is provided we reuse the same instance; otherwise
        # build a fresh one from the factory.
        relax_calc_r = shared_calc if shared_calc is not None else calc_factory()
        relax_calc_p = shared_calc if shared_calc is not None else calc_factory()
        reactant = relax_endpoint(reactant, relax_calc_r,
                                  fmax=cfg.endpoint_fmax,
                                  max_steps=cfg.endpoint_max_steps,
                                  label="reactant", logger=logger)
        product = relax_endpoint(product, relax_calc_p,
                                 fmax=cfg.endpoint_fmax,
                                 max_steps=cfg.endpoint_max_steps,
                                 label="product", logger=logger)
        # TS is NOT relaxed (relaxing would slide it off the saddle to a minimum).

    logger(f"[neb] building IDPP path with target n_images={cfg.n_images}")
    images = build_idpp_path(reactant, ts, product, n_images=cfg.n_images)
    logger(f"[neb] actual path length: {len(images)} images")
    attach_calculators(images, calc_factory=calc_factory, shared_calc=shared_calc)
    use_shared = shared_calc is not None

    meta_full = dict(
        n_images=len(images),
        k_spring=cfg.k_spring,
        method=cfg.method,
        neb_fmax=cfg.neb_fmax,
        save_threshold=cfg.save_threshold,
    )
    if meta:
        meta_full.update(meta)
    state = SaveState(h5_path=h5_path, save_threshold=cfg.save_threshold)
    state.open(images[0], meta=meta_full)

    logger(f"[neb] NEB (no climb) to Fmax < {cfg.neb_fmax} eV/A")
    neb = NEB(images, k=cfg.k_spring, climb=False, method=cfg.method,
              allow_shared_calculator=use_shared)
    opt = BFGS(neb, logfile="-", alpha=cfg.bfgs_alpha, maxstep=cfg.bfgs_maxstep)
    state.iter_idx = 0

    def cb_neb():
        fmax = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
        state._maybe_save(
            neb,
            fmax=fmax,
            stage="neb",
            force_save=(state.iter_idx == 0 and cfg.save_first_iter),
        )
        state.iter_idx += 1

    opt.attach(cb_neb, interval=1)
    t0 = time.time()
    opt.run(fmax=cfg.neb_fmax, steps=cfg.max_steps_neb)
    logger(f"[neb] done in {time.time() - t0:.1f}s, "
           f"converged={opt.converged()}, total saves={state.save_idx}")

    # Always save the final band
    if state.history and state.history[-1]["iter"] != state.iter_idx - 1:
        fmax = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
        state._maybe_save(neb, fmax=fmax, stage="neb_final", force_save=True)

    # Record history table in the H5 file
    with h5py.File(h5_path, "a") as f:
        if state.history:
            f.create_dataset(
                "save_history",
                data=np.asarray(
                    [(h["save_idx"], h["iter"], h["fmax"], h["cumulative_fmax"])
                     for h in state.history],
                    dtype=[("save_idx", "i4"), ("iter", "i4"),
                           ("fmax", "f8"), ("cumulative_fmax", "f8")],
                ),
            )
            f.create_dataset(
                "save_stages",
                data=np.asarray([h["stage"].encode("utf-8") for h in state.history]),
            )

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_xyz(path: str) -> Atoms:
    return read(path)
