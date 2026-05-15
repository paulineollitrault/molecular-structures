"""
MACE-MP-0 driver for the NEB sampling pipeline.

A single MACE model is shared across all NEB images via
ase.mep.NEB(..., allow_shared_calculator=True). Each call to the model is
stateless w.r.t. the previous image, so sharing is safe and saves memory.

This module exposes:
    load_mace(model='medium', default_dtype='float64', dispersion=False)
        -> a single MACECalculator instance to attach to every image.
"""

from __future__ import annotations

from typing import Optional


def load_mace(
    model: str = "medium",
    default_dtype: str = "float64",
    dispersion: bool = False,
    device: Optional[str] = None,
):
    """Build a single MACE-MP-0 calculator.

    `model` may be 'small', 'medium', or 'large' (or a path to a .model file).
    `default_dtype` should be 'float64' for geometry optimization / NEB,
    'float32' for faster MD.
    """
    from mace.calculators import mace_mp

    kwargs = dict(model=model, default_dtype=default_dtype, dispersion=dispersion)
    if device is not None:
        kwargs["device"] = device
    return mace_mp(**kwargs)
