"""
GFN2-xTB driver via tblite for the NEB sampling pipeline.

GFN2-xTB (Bannwarth, Ehlert, Grimme, JCTC 2019) is a semiempirical tight-binding
method that is well-validated on organometallic complexes including transition
metals. Energies in eV, forces in eV/Angstrom (ASE convention).
"""

from __future__ import annotations


def make_xtb(
    method: str = "GFN2-xTB",
    charge: int = 0,
    spin: int = 0,
    electronic_temperature: float = 1500.0,
    max_iterations: int = 1000,
    mixer_damping: float = 0.2,
    **kwargs,
):
    """Build a TBLite calculator.

    method: 'GFN2-xTB' (default), 'GFN1-xTB', or 'IPEA1-xTB'.
    spin: number of unpaired electrons (2S). Singlet = 0.

    Defaults are tuned to avoid SCF failures on transition-metal complexes:
        - electronic_temperature=1500 K (Fermi smearing softens d-orbital
          near-degeneracies; default 300 K often diverges on Fe complexes)
        - max_iterations=1000 (default 250 too tight for stubborn cases)
        - mixer_damping=0.2 (stronger damping than default 0.4 = lower mix rate)
    """
    from tblite.ase import TBLite

    return TBLite(
        method=method,
        charge=charge,
        multiplicity=spin + 1,
        electronic_temperature=electronic_temperature,
        max_iterations=max_iterations,
        mixer_damping=mixer_damping,
        verbosity=0,
        **kwargs,
    )
