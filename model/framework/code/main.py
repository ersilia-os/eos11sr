# imports
import os
import csv
import sys
import numpy as np
from rdkit import Chem
from collections import Counter
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

# run model
molecules = [Chem.MolFromSmiles(mol) for mol in smiles_list]
morgan_fps = calculate_morgan_fingerprints(molecules, radius=2, nbits=16384).astype(int)
emfps = convert_fp_to_embV2(morgan_fps, size=512)
outputs = emfps

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# for i, j in zip(outputs, morgan_fps):
# 	print(Counter(j)[0], len(j), Counter(i)[0], len(i))

# write output
header = ["dim_{0}".format(str(i).zfill(2)) for i in range(32)]
write_out(outputs,header,output_file,dtype='float32')
