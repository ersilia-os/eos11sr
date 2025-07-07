# imports
import os
import csv
import sys
import numpy as np
from rdkit import Chem

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
from utils_molecules import convert_fp_to_embV2, calculate_morgan_fingerprints

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
molecules = [Chem.MolFromSmiles(mol) for mol in smiles_list]
morgan_fps = calculate_morgan_fingerprints(molecules, radius=2, nbits=16384)
emfps = convert_fp_to_embV2(morgan_fps, size=16)
outputs = emfps

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    header = ["dim_{0}".format(str(i).zfill(4)) for i in range(1024)]
    writer.writerow(header)  # header
    for o in outputs:
        writer.writerow(o)
