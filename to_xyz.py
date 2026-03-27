"""
Example:
python to_xyz.py MB08-165structures 001 turbomole
"""


import sys
from ase.io import read, write

dataset = sys.argv[1]
molecule = sys.argv[2]
input_format = sys.argv[3]

mol = read(f'{dataset}/{molecule}', format=input_format)
write(f'XYZ/{dataset}_{molecule}.xyz', mol)

