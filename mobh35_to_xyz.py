"""
Split MOBH35/ct1c01126_si_003.txt into individual .xyz files.

Each structure block in the source file looks like:
    <name>
    <natoms>
    <blank>
    <element> <x> <y> <z>
    ...
    <blank lines>

Output: XYZ/MOBH35_<name>.xyz
"""

import os
import re

SRC = "MOBH35/ct1c01126_si_003.txt"
OUT_DIR = "XYZ"

HEADER_RE = re.compile(r"^\d+_[A-Za-z_]+\s*$")
ATOM_RE = re.compile(r"^[A-Z][a-z]?\s+-?\d")

os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC) as f:
    lines = [ln.rstrip() for ln in f]

i = 0
count = 0
while i < len(lines):
    if HEADER_RE.match(lines[i]):
        name = lines[i].strip()
        natoms = int(lines[i + 1].strip())
        # collect the next `natoms` atom lines, skipping blanks
        atoms = []
        j = i + 2
        while len(atoms) < natoms and j < len(lines):
            if ATOM_RE.match(lines[j].strip()):
                atoms.append(lines[j].strip())
            j += 1
        assert len(atoms) == natoms, f"{name}: got {len(atoms)} atoms, expected {natoms}"

        out_path = os.path.join(OUT_DIR, f"MOBH35_{name}.xyz")
        with open(out_path, "w") as out:
            out.write(f"{natoms}\n{name}\n")
            for a in atoms:
                out.write(a + "\n")
        count += 1
        i = j
    else:
        i += 1

print(f"Wrote {count} xyz files to {OUT_DIR}/")
