"""
UMA-OMol25 (Meta/FAIRChem) driver for the NEB sampling pipeline.

UMA is a universal MLIP trained on multiple datasets. The OMol25 task head
covers organic and organometallic molecules including transition metals.

For UMA on the OMol task, charge and spin MULTIPLICITY are read from
atoms.info['charge'] and atoms.info['spin'] at every calculation. We attach
these per-image via the factory.
"""

from __future__ import annotations

import os

# Avoid OpenMP double-init from PyTorch + libomp on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional


def load_uma(
    model_name: str = "uma-s-1p2",
    device: Optional[str] = None,
):
    """Build a single MLIPPredictUnit. Shared across all NEB images.

    `model_name`: 'uma-s-1p2' (small, latest), 'uma-s-1p1', 'uma-m-1p1'.
    `device`: 'cpu' or 'cuda'. (MPS is not directly supported by fairchem's
              get_predict_unit type signature; fall back to 'cpu' on Apple.)
    """
    from fairchem.core import pretrained_mlip

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return pretrained_mlip.get_predict_unit(model_name, device=device)


def make_uma_calculator(
    predict_unit,
    task_name: str = "omol",
):
    """Build a FAIRChemCalculator from a shared predict_unit."""
    from fairchem.core import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit, task_name=task_name)


def stamp_charge_spin(atoms, charge: int = 0, spin_multiplicity: int = 1):
    """UMA-OMol needs charge and spin (multiplicity) per Atoms object.

    spin_multiplicity = 2S+1: 1=singlet, 2=doublet, 3=triplet, ...
    """
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(spin_multiplicity)
    return atoms
