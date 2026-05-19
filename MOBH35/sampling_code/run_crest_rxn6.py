"""
CREST conformer sampling for MOBH35 reaction 6, with UMA-OMol25 re-scoring.

Workflow:
  1. Run CREST + GFN2-xTB on each of (reactant, product, TS), separately.
     Each produces an ensemble of conformers (crest_conformers.xyz).
  2. Load UMA-OMol25 once. For every conformer, run a UMA single-point to
     get UMA energies and forces.
  3. Write everything to a single HDF5 file mirroring the Transition1x-like
     schema used by run_rxn6_neb.py, but with a 'crest_<region>' stage tag.

Outputs:
  MOBH35/rxn_6/samples/crest/{r,p,ts}/    (CREST working directories)
  MOBH35/rxn_6/samples/crest/conformers.h5

Note on charge / spin
---------------------
CREST is invoked with the same charge/spin used for the NEB runs (--charge,
--spin). For UMA-OMol the spin field is *multiplicity* (2S+1); for CREST it
is `-uhf` = 2S.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

# Avoid OMP double-init on macOS (UMA -> torch)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
from ase.io import read, write

from neb_pipeline import load_xyz


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))           # MOBH35/
RXN_DIR = os.path.join(ROOT, "rxn_6")
OUT_DIR = os.path.join(RXN_DIR, "samples", "crest")


REGIONS = [
    ("r", "MOBH35_06_start.xyz"),
    ("p", "MOBH35_06_end.xyz"),
    ("ts", "MOBH35_06_ts.xyz"),
]


TOML_TEMPLATE = """input = "input.xyz"
runtype = "imtd-gc"
threads = {threads}

[[calculation.level]]
method = "tblite"
tblite_level = "{gfn_level}"
chrg = {charge}
uhf = {spin_2s}
etemp = {etemp}
maxscc = {maxscc}
"""


def run_crest(input_xyz: str, work_dir: str, charge: int, spin_2s: int,
              threads: int, gfn_level: str = "gfn2", etemp: float = 1500.0,
              maxscc: int = 1000) -> str:
    """Run CREST in `work_dir` via a TOML config. Returns crest_conformers.xyz."""
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(input_xyz, os.path.join(work_dir, "input.xyz"))

    toml_path = os.path.join(work_dir, "crest.toml")
    with open(toml_path, "w") as f:
        f.write(TOML_TEMPLATE.format(
            threads=threads, gfn_level=gfn_level,
            charge=charge, spin_2s=spin_2s,
            etemp=etemp, maxscc=maxscc,
        ))

    cmd = ["crest", "--input", "crest.toml"]
    log_path = os.path.join(work_dir, "crest.log")
    print(f"[crest] running in {work_dir} (TOML: etemp={etemp}, maxscc={maxscc})")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=logf,
                              stderr=subprocess.STDOUT)
    print(f"[crest] finished in {time.time() - t0:.0f}s (exit {proc.returncode})")
    if proc.returncode != 0:
        print(f"[crest] WARNING non-zero exit. Tail of {log_path}:")
        with open(log_path) as f:
            tail = f.readlines()[-20:]
            print("".join(tail))

    # Prefer the optimized conformer ensemble; if CREST's multilevel
    # optimization failed (common on tight-field organometallics), fall
    # back to the raw metadynamics trajectory. For ML-dataset purposes
    # the unoptimized MD snapshots are *more* useful (broader sampling).
    for candidate in ("crest_conformers.xyz", "crest_rotamers.xyz",
                      "crest_rotamers_0.xyz", "crest_dynamics.trj"):
        p = os.path.join(work_dir, candidate)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return os.path.join(work_dir, "crest_conformers.xyz")  # nonexistent


def evaluate_with_uma(conformer_files: dict[str, str], uma_model: str,
                      charge: int, spin_mult: int, h5_out: str,
                      meta_extra: dict | None = None):
    """Load UMA once, score every conformer in every file."""
    from uma_calculator import load_uma, make_uma_calculator, stamp_charge_spin

    print(f"[uma] loading {uma_model} on cpu (first call ~3.5 min)...")
    t0 = time.time()
    predict_unit = load_uma(uma_model, device="cpu")
    calc = make_uma_calculator(predict_unit, task_name="omol")
    print(f"[uma] loaded in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(h5_out), exist_ok=True)
    with h5py.File(h5_out, "w") as f:
        if meta_extra:
            for k, v in meta_extra.items():
                f.attrs[k] = v
        f.attrs["driver"] = "GFN2-xTB (CREST) + UMA-OMol25 re-eval"
        f.attrs["uma_model"] = uma_model
        f.attrs["charge"] = charge
        f.attrs["spin_multiplicity"] = spin_mult

        # atomic_numbers is the same across all conformers (CREST preserves order)
        any_path = next(iter(conformer_files.values()))
        if not any_path or not os.path.exists(any_path):
            raise RuntimeError(f"No conformer file at {any_path}")
        # CREST writes crest_dynamics.trj as a multi-frame XYZ. ASE doesn't
        # auto-detect .trj as xyz, so force format='xyz' for any trj file.
        def _read_one(p, idx):
            fmt = "xyz" if p.endswith(".trj") else None
            return read(p, index=idx, format=fmt)
        def _read_all(p):
            fmt = "xyz" if p.endswith(".trj") else None
            return read(p, index=":", format=fmt)
        sample = _read_one(any_path, 0)
        f.create_dataset("atomic_numbers",
                         data=np.asarray(sample.get_atomic_numbers(), dtype=np.int32))

        save_idx = 0
        for region, path in conformer_files.items():
            if not os.path.exists(path):
                print(f"[uma] skipping {region} — no file at {path}")
                continue
            ensemble = _read_all(path)
            print(f"[uma] {region}: scoring {len(ensemble)} conformers...")
            t1 = time.time()
            for i, atoms in enumerate(ensemble):
                stamp_charge_spin(atoms, charge=charge,
                                  spin_multiplicity=spin_mult)
                atoms.calc = calc
                e = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                grp = f.create_group(f"conf_{save_idx:05d}")
                grp.create_dataset("positions",
                                   data=np.asarray(atoms.get_positions(),
                                                   dtype=float))
                grp.create_dataset("forces", data=forces)
                grp.attrs["energy"] = e
                grp.attrs["region"] = region
                grp.attrs["region_idx"] = i
                grp.attrs["stage"] = f"crest_{region}"
                save_idx += 1
            print(f"[uma] {region} done in {time.time() - t1:.1f}s "
                  f"(total saves so far: {save_idx})")

        f.attrs["n_conformers"] = save_idx
    print(f"[uma] wrote {save_idx} conformer records to {h5_out}")
    return save_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0,
                   help="2S (unpaired electrons). Singlet=0 -> -uhf 0, -mult 1.")
    p.add_argument("--threads", type=int, default=max(os.cpu_count() // 2, 4))
    p.add_argument("--uma-model", default="uma-s-1p2",
                   choices=["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"])
    p.add_argument("--skip-crest", action="store_true",
                   help="Reuse existing CREST output dirs (just rerun UMA re-eval)")
    p.add_argument("--only", choices=["r", "p", "ts"], default=None,
                   help="Run CREST only on this region (still re-scores all available).")
    p.add_argument("--out", default=os.path.join(OUT_DIR, "conformers.h5"))
    args = p.parse_args()

    conformer_paths: dict[str, str] = {}
    if not args.skip_crest:
        for region, xyz_name in REGIONS:
            if args.only and region != args.only:
                continue
            work_dir = os.path.join(OUT_DIR, region)
            confs = run_crest(
                input_xyz=os.path.join(RXN_DIR, xyz_name),
                work_dir=work_dir,
                charge=args.charge,
                spin_2s=args.spin,
                threads=args.threads,
            )
            conformer_paths[region] = confs
    # Always look on disk for ensembles to score (handles --skip-crest too)
    for region, _ in REGIONS:
        if region in conformer_paths:
            continue
        for candidate in ("crest_conformers.xyz", "crest_rotamers.xyz",
                          "crest_rotamers_0.xyz", "crest_dynamics.trj"):
            p = os.path.join(OUT_DIR, region, candidate)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                conformer_paths[region] = p
                break

    if not conformer_paths:
        print("No conformer files found. Did CREST finish?")
        return 1

    evaluate_with_uma(
        conformer_files=conformer_paths,
        uma_model=args.uma_model,
        charge=args.charge,
        spin_mult=args.spin + 1,
        h5_out=args.out,
        meta_extra=dict(reaction="MOBH35_06"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
