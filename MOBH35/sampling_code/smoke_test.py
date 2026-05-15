"""
Fast smoke test for the NEB pipeline. Uses a trivial EMT-like dummy
calculator (constant energy, zero forces) so the IDPP, NEB-callback,
cumulative-Fmax saving, and HDF5 output paths exercise without running DFT.

Run with:
    conda activate tmc
    python smoke_test.py
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from neb_pipeline import NEBConfig, load_xyz, run_neb_pipeline


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RXN_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "rxn_6"))


class HarmonicDummy(Calculator):
    """E = 0.5 * sum (r - r0)^2 toward a fixed reference; tiny forces.

    Cheap and well-defined so BFGS converges in a handful of steps.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, ref_positions, k=0.01, **kwargs):
        super().__init__(**kwargs)
        self._ref = np.asarray(ref_positions, dtype=float)
        self._k = k

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        diff = pos - self._ref
        energy = 0.5 * self._k * float((diff ** 2).sum())
        forces = -self._k * diff
        self.results["energy"] = energy
        self.results["forces"] = forces


def main():
    reactant = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_start.xyz"))
    ts = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_ts.xyz"))
    product = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_end.xyz"))

    # Each image gets its own dummy calc, anchored to the *image's own* starting
    # geometry, so the band feels weak restoring forces toward its IDPP shape.
    # We need a factory that returns distinct calculators per image — so capture
    # the appropriate reference at call-time.
    ref_stack = [reactant.get_positions(), ts.get_positions(), product.get_positions()]
    counter = {"i": 0}

    def calc_factory():
        ref = ref_stack[counter["i"] % len(ref_stack)]
        counter["i"] += 1
        return HarmonicDummy(ref_positions=ref, k=0.02)

    with tempfile.TemporaryDirectory() as tmp:
        out_h5 = os.path.join(tmp, "smoke.h5")
        cfg = NEBConfig(
            n_images=7,
            k_spring=0.1,
            neb_fmax=1e-3,        # tight enough to force many iterations
            max_steps_neb=4,
            save_threshold=0.05,
        )
        state = run_neb_pipeline(
            reactant=reactant, ts=ts, product=product,
            h5_path=out_h5,
            calc_factory=calc_factory,
            cfg=cfg,
            meta=dict(reaction="smoke"),
        )

        with h5py.File(out_h5, "r") as f:
            keys = list(f.keys())
            iter_keys = sorted(k for k in keys if k.startswith("iter_"))
            print(f"[smoke] top-level keys: {keys}")
            print(f"[smoke] saved iteration groups: {len(iter_keys)}")
            for k in iter_keys[:3]:
                g = f[k]
                print(f"  {k}: positions={g['positions'].shape}, "
                      f"energies={g['energies'].shape}, "
                      f"fmax={g.attrs['fmax']:.4f}, stage={g.attrs['stage']}")
            assert len(iter_keys) >= 1
            assert "atomic_numbers" in f
            assert f["atomic_numbers"].shape[0] == len(reactant)

    print(f"[smoke] OK — pipeline produced {state.save_idx} snapshots")


if __name__ == "__main__":
    main()
