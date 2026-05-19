"""
Export CREST conformer HDF5 (conformers.h5) to XYZ trajectories.

Layout differs from the NEB HDF5: each top-level group is `conf_NNNNN` with
positions (n_atoms, 3), forces (n_atoms, 3), and attrs {energy, region, stage}.

Usage:
    python crest_h5_to_xyz.py <h5_path>                # one per region
    python crest_h5_to_xyz.py <h5_path> --combined     # single 2672-frame file
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", help="conformers.h5 produced by run_crest_rxn6.py")
    p.add_argument("--combined", action="store_true",
                   help="Write one combined XYZ instead of per-region files.")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.h5)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        Z = f["atomic_numbers"][...]
        syms = [chemical_symbols[z] for z in Z]
        keys = sorted(k for k in f if k.startswith("conf_"))

        by_region: dict[str, list[Atoms]] = {}
        all_frames: list[Atoms] = []
        for k in keys:
            g = f[k]
            pos = g["positions"][...]
            e = float(g.attrs["energy"])
            region = g.attrs["region"]
            if isinstance(region, bytes):
                region = region.decode()
            a = Atoms(symbols=syms, positions=pos)
            a.info["comment"] = f"{k} region={region} E={e:.6f}eV"
            by_region.setdefault(region, []).append(a)
            all_frames.append(a)

    if args.combined:
        out = os.path.join(out_dir, "crest_all.xyz")
        write(out, all_frames)
        print(f"wrote {len(all_frames)} frames to {out}")
    else:
        for region, frames in by_region.items():
            out = os.path.join(out_dir, f"crest_{region}.xyz")
            write(out, frames)
            print(f"{region}: wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
