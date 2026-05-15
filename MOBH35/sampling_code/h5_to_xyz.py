"""
Read the NEB HDF5 produced by run_rxn6_neb.py and export the structures.

Usage:
    python h5_to_xyz.py <h5_path> --out all.xyz           # all 1716 frames in one file
    python h5_to_xyz.py <h5_path> --out band.xyz --mode band-per-save
                                                            # 143 multi-frame files, one per save
    python h5_to_xyz.py <h5_path> --view                  # open ase gui on the trajectory
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import chemical_symbols


def load_frames(h5_path, mode="all"):
    """Yield ASE Atoms objects.

    mode='all'           : every (save, image) pair, ordered save then image
    mode='per-image-traj': all save_idx for each image_idx (a movie of one image over the run)
    mode='last-band'     : 12 images from the final save (the converged band)
    """
    with h5py.File(h5_path, "r") as f:
        Z = f["atomic_numbers"][...]
        symbols = [chemical_symbols[z] for z in Z]
        iter_keys = sorted(k for k in f.keys() if k.startswith("iter_"))

        for k in iter_keys:
            g = f[k]
            positions = g["positions"][...]
            energies = g["energies"][...]
            n_images = positions.shape[0]
            neb_iter = int(g.attrs["neb_iter"])
            fmax = float(g.attrs["fmax"])

            if mode == "last-band" and k != iter_keys[-1]:
                continue

            for i in range(n_images):
                a = Atoms(symbols=symbols, positions=positions[i])
                a.info["comment"] = (
                    f"{k} image={i} neb_iter={neb_iter} "
                    f"E={energies[i]:.6f}eV fmax={fmax:.3f}"
                )
                yield a, k, i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5", help="HDF5 file produced by the NEB pipeline")
    p.add_argument("--out", default=None,
                   help="Output .xyz path. Default: same dir as h5, name 'all_frames.xyz'.")
    p.add_argument("--mode", default="all",
                   choices=["all", "last-band", "band-per-save"],
                   help="all: 12*N frames; last-band: 12 frames; band-per-save: one file per save.")
    p.add_argument("--view", action="store_true",
                   help="Launch `ase gui` on the resulting xyz.")
    args = p.parse_args()

    out = args.out or os.path.join(os.path.dirname(args.h5), "all_frames.xyz")

    if args.mode == "band-per-save":
        out_dir = out if out.endswith("/") else out
        if not out_dir.endswith("/"):
            out_dir = out + "_perband"
        os.makedirs(out_dir, exist_ok=True)
        with h5py.File(args.h5, "r") as f:
            Z = f["atomic_numbers"][...]
            symbols = [chemical_symbols[z] for z in Z]
            iter_keys = sorted(k for k in f.keys() if k.startswith("iter_"))
            for k in iter_keys:
                g = f[k]
                positions = g["positions"][...]
                energies = g["energies"][...]
                frames = []
                for i in range(positions.shape[0]):
                    a = Atoms(symbols=symbols, positions=positions[i])
                    a.info["comment"] = f"{k} image={i} E={energies[i]:.6f}"
                    frames.append(a)
                write(os.path.join(out_dir, f"{k}.xyz"), frames)
        print(f"wrote {len(iter_keys)} files into {out_dir}/")
        if args.view:
            print("(--view ignored for band-per-save; open any file with `ase gui`.)")
        return

    frames = [a for a, _, _ in load_frames(args.h5, mode=args.mode)]
    write(out, frames)
    print(f"wrote {len(frames)} frames to {out}")

    if args.view:
        os.execvp("ase", ["ase", "gui", out])


if __name__ == "__main__":
    main()
