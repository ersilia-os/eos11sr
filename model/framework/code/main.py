# imports
import os
import csv
import sys
import numpy as np
from rdkit import Chem
from ersilia_pack_utils.core import write_out, read_smiles
# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
from utils_molecules import convert_fp_to_embV2, calculate_morgan_fingerprints

# read SMILES from .csv file, assuming one column with header
cols, smiles_list = read_smiles(input_file)

print(smiles_list)
# run model
molecules = [Chem.MolFromSmiles(mol) for mol in smiles_list]
morgan_fps = calculate_morgan_fingerprints(molecules, radius=2, nbits=16384)
emfps = convert_fp_to_embV2(morgan_fps, size=64)
outputs = emfps

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len
print(outputs)

header = ["dim_{0}".format(str(i).zfill(3)) for i in range(256)]

write_out(outputs,header,output_file,dtype='float32')
# write output in a .csv file
# with open(output_file, "w") as f:
#     writer = csv.writer(f)
#     header = ["dim_{0}".format(str(i).zfill(3)) for i in range(256)]
#     writer.writerow(header)  # header
#     for o in outputs:
#         writer.writerow(o)
