"""
Minimal ASE Calculator wrapping PySCF RKS with density fitting.

Defaults are tuned for cheap-but-reasonable organometallic geometry:
    xc = 'bp86', basis = 'def2-svp', density-fitted J (RI-J).

Energies in eV, forces in eV/Angstrom (ASE convention).
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree


class PySCFCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        xc: str = "bp86",
        basis: str = "def2-svp",
        charge: int = 0,
        spin: int = 0,
        density_fit: bool = True,
        auxbasis: str | None = "def2-universal-jfit",
        max_cycle: int = 200,
        conv_tol: float = 1e-8,
        grid_level: int = 3,
        verbose: int = 0,
        chkfile: str | None = None,
        **kwargs,
    ):
        """
        spin = 2S (number of unpaired electrons), per PySCF convention.
        For a singlet (multiplicity 1): spin = 0.
        """
        super().__init__(**kwargs)
        self.xc = xc
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.density_fit = density_fit
        self.auxbasis = auxbasis
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.grid_level = grid_level
        self.verbose = verbose
        self.chkfile = chkfile
        self._last_dm = None

    def _build_mol(self, atoms):
        from pyscf import gto

        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()  # Angstrom
        atom_spec = [(s, tuple(p)) for s, p in zip(symbols, positions)]

        mol = gto.M(
            atom=atom_spec,
            unit="Angstrom",
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=self.verbose,
        )
        return mol

    def _make_mf(self, mol):
        from pyscf import dft

        if self.spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = self.xc
        mf.grids.level = self.grid_level
        mf.max_cycle = self.max_cycle
        mf.conv_tol = self.conv_tol
        if self.chkfile is not None:
            mf.chkfile = self.chkfile
        if self.density_fit:
            mf = mf.density_fit(auxbasis=self.auxbasis)
        return mf

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        mol = self._build_mol(self.atoms)
        mf = self._make_mf(mol)

        # warm-start SCF from previous density matrix if shape matches
        if self._last_dm is not None and self._last_dm.shape[-1] == mol.nao:
            e = mf.kernel(dm0=self._last_dm)
        else:
            e = mf.kernel()
        self._last_dm = mf.make_rdm1()

        # PySCF gradient: Hartree/Bohr, shape (n_atoms, 3)
        grad = mf.nuc_grad_method().kernel()
        forces_ev_a = -grad * (Hartree / Bohr)

        self.results["energy"] = float(e) * Hartree  # Hartree -> eV
        self.results["forces"] = np.asarray(forces_ev_a, dtype=float)


def make_calculator(charge=0, spin=0, xc="bp86", basis="def2-svp", **kwargs):
    """Convenience factory — each NEB image needs its own calculator instance."""
    return PySCFCalculator(xc=xc, basis=basis, charge=charge, spin=spin, **kwargs)
