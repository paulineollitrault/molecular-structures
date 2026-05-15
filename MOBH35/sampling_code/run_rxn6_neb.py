"""
Run the NEB sampling pipeline on MOBH35 reaction 6 (Fe complex, 37 atoms),
driven by MACE-MP-0 as the cheap force field.

Usage:
    conda activate tmc
    python run_rxn6_neb.py             # full run with defaults
    python run_rxn6_neb.py --smoke     # short run for sanity check

Outputs to MOBH35/rxn_6/samples/neb_iterations.h5
"""

from __future__ import annotations

import argparse
import os
import sys

from neb_pipeline import NEBConfig, load_xyz, run_neb_pipeline


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))  # MOBH35/
RXN_DIR = os.path.join(ROOT, "rxn_6")
OUT_DIR = os.path.join(RXN_DIR, "samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a quick test: few images, few steps")
    parser.add_argument("--driver", default="uma",
                        choices=["xtb", "mace", "uma"],
                        help="Cheap driver for energies/forces during NEB.")
    parser.add_argument("--n-images", type=int, default=12)
    parser.add_argument("--mace-model", default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--uma-model", default="uma-s-1p2",
                        choices=["uma-s-1p2", "uma-s-1p1", "uma-m-1p1"])
    parser.add_argument("--dtype", default="float64",
                        choices=["float32", "float64"])
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0,
                        help="2S (unpaired electrons). Singlet=0.")
    parser.add_argument("--save-threshold", type=float, default=0.1,
                        help="Cumulative-Fmax threshold for saving a snapshot, eV/A")
    parser.add_argument("--no-relax-endpoints", action="store_true",
                        help="Skip the initial endpoint relaxation step.")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "neb_iterations.h5"))
    args = parser.parse_args()

    reactant = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_start.xyz"))
    ts = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_ts.xyz"))
    product = load_xyz(os.path.join(RXN_DIR, "MOBH35_06_end.xyz"))

    print(f"[run] reactant: {reactant.get_chemical_formula()}, "
          f"ts: {ts.get_chemical_formula()}, product: {product.get_chemical_formula()}")
    print(f"[run] charge={args.charge}, spin(2S)={args.spin}")

    relax = not args.no_relax_endpoints
    if args.driver == "mace":
        from mace_calculator import load_mace
        print(f"[run] driver: MACE-MP-0 ({args.mace_model}, {args.dtype})")
        calc = load_mace(model=args.mace_model, default_dtype=args.dtype)
        run_kwargs = dict(shared_calc=calc)
        driver_meta = dict(driver="MACE-MP-0",
                           mace_model=args.mace_model, dtype=args.dtype)
    elif args.driver == "xtb":
        from xtb_calculator import make_xtb
        print(f"[run] driver: GFN2-xTB (tblite)")
        run_kwargs = dict(
            calc_factory=lambda: make_xtb(charge=args.charge, spin=args.spin),
        )
        driver_meta = dict(driver="GFN2-xTB")
    elif args.driver == "uma":
        from uma_calculator import load_uma, make_uma_calculator, stamp_charge_spin
        print(f"[run] driver: UMA-OMol25 ({args.uma_model}, cpu)")
        # spin (2S) -> multiplicity (2S+1) for OMol convention
        mult = args.spin + 1
        for a in (reactant, ts, product):
            stamp_charge_spin(a, charge=args.charge, spin_multiplicity=mult)
        predict_unit = load_uma(args.uma_model, device="cpu")
        calc = make_uma_calculator(predict_unit, task_name="omol")
        run_kwargs = dict(shared_calc=calc)
        driver_meta = dict(driver="UMA-OMol25", uma_model=args.uma_model)
    else:
        raise ValueError(args.driver)

    if args.smoke:
        cfg = NEBConfig(
            n_images=max(5, args.n_images // 2),
            k_spring=0.1,
            neb_fmax=0.5,
            max_steps_neb=20,
            save_threshold=args.save_threshold,
            relax_endpoints=relax,
        )
    else:
        cfg = NEBConfig(n_images=args.n_images,
                        save_threshold=args.save_threshold,
                        relax_endpoints=relax)

    meta = dict(
        reaction="MOBH35_06",
        charge=args.charge,
        spin=args.spin,
        formula=str(reactant.get_chemical_formula()),
        **driver_meta,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    state = run_neb_pipeline(
        reactant=reactant,
        ts=ts,
        product=product,
        h5_path=args.out,
        cfg=cfg,
        meta=meta,
        **run_kwargs,
    )

    print(f"[run] wrote {state.save_idx} iteration snapshots to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
