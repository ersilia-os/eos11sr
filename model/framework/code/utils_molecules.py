import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations
import psutil
import time
import pickle

from random import shuffle

import rdkit 
from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.*')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------
# Helpers: SMILES <-> Mol
# ---------------------------

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        import traceback
        print(f"[mol_from_smiles] Error converting SMILES -> Mol. SMILES='{smiles}'. Error: {e}")
        traceback.print_exc()
        return None

def canonical_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception as e:
        import traceback
        print(f"[canonical_smiles] Error canonicalizing SMILES='{smiles}'. Error: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# Descriptors
# ---------------------------

def custom_descriptors(mol, descriptor_list, missingVal):
    """
    Return None if mol is None; otherwise compute only the requested descriptors.
    Verbose prints on errors.
    """
    if mol is None:
        print("[custom_descriptors] Skipped: mol is None")
        return None
    descriptor_dictionary = {}
    for name, function in Descriptors._descList:
        if name in descriptor_list:
            try:
                val = function(mol)
            except Exception as e:
                import traceback
                print(f"[custom_descriptors] Error with descriptor '{name}': {e}")
                traceback.print_exc()
                val = missingVal
            descriptor_dictionary[name] = val
    return descriptor_dictionary

def get_custom_descriptors(molecular, case, descriptor_list):
    '''
    Calculate selected descriptors for a list of molecules and organize results in a DataFrame.
    Rows corresponding to invalid molecules will be None -> converted to all-NaN rows in the DataFrame.
    Includes detailed prints to trace Nones and errors.
    '''
    if case == 'smiles':
        mols = []
        for i, smile in enumerate(molecular):
            m = Chem.MolFromSmiles(smile)
            if m is None:
                print(f"[get_custom_descriptors] Index {i}: SMILES '{smile}' -> Mol is None (row will be NaNs).")
            mols.append(m)
        allDescrs = [
            (custom_descriptors(m, descriptor_list, 0) if m is not None else None)
            for m in tqdm(mols, desc='Mols')
        ]
    elif case == 'mol_object':
        for i, m in enumerate(molecular):
            if m is None:
                print(f"[get_custom_descriptors] Index {i}: Mol is None (row will be NaNs).")
        allDescrs = [
            (custom_descriptors(m, descriptor_list, 0) if m is not None else None)
            for m in tqdm(molecular, desc='Mols')
        ]
    else:
        raise ValueError("case must be 'smiles' or 'mol_object'")
    df = pd.DataFrame(allDescrs)
    return df

def get_all_descriptors(molecular, case):
    '''
    Calculate all RDKit descriptors for a list of molecules and organize results in a DataFrame.
    Rows corresponding to invalid molecules will be None -> converted to all-NaN rows in the DataFrame.
    Includes detailed prints to trace Nones and errors.
    '''
    if case == 'smiles':
        mols = []
        for i, smile in enumerate(molecular):
            m = Chem.MolFromSmiles(smile)
            if m is None:
                print(f"[get_all_descriptors] Index {i}: SMILES '{smile}' -> Mol is None (row will be NaNs).")
            mols.append(m)
        allDescrs = [
            (getMolDescriptors(m, 0) if m is not None else None)
            for m in tqdm(mols, desc='Mols')
        ]
    elif case == 'mol_object':
        for i, m in enumerate(molecular):
            if m is None:
                print(f"[get_all_descriptors] Index {i}: Mol is None (row will be NaNs).")
        allDescrs = [
            (getMolDescriptors(m, 0) if m is not None else None)
            for m in tqdm(molecular, desc='Mols')
        ]
    else:
        raise ValueError("case must be 'smiles' or 'mol_object'")
    df = pd.DataFrame(allDescrs)
    return df

def getMolDescriptors(mol, missingVal=0):
    '''
    Calculate the full list of descriptors for a molecule.
    Returns None immediately if mol is None. Verbose prints on errors.
    '''
    if mol is None:
        print("[getMolDescriptors] Skipped: mol is None")
        return None
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except Exception as e:
            import traceback
            print(f"[getMolDescriptors] Error with descriptor '{nm}': {e}")
            traceback.print_exc()
            val = missingVal
        res[nm] = val
    return res


# ---------------------------
# Fingerprints
# ---------------------------

def calculate_morgan_fingerprints(molecules, radius=2, nbits=2048):
    """
    Calculate Morgan fingerprints for a list of RDKit Mol objects.
    If a mol is None or errors, append None and print details.
    Returns an object-dtype array so Nones can coexist.
    """
    fingerprints_list = []
    for i, mol in enumerate(tqdm(molecules, desc='MFP')):
        if mol is None:
            print(f"[calculate_morgan_fingerprints] Skipped: mol index {i} is None")
            fingerprints_list.append(None)
            continue
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=nbits, useChirality=True, useFeatures=True
            )
            fingerprint_array = np.array(fingerprint)
            fingerprints_list.append(fingerprint_array)
        except Exception as e:
            import traceback
            print(f"[calculate_morgan_fingerprints] Error at mol index {i}: {e}")
            traceback.print_exc()
            fingerprints_list.append(None)
    return np.array(fingerprints_list, dtype=object)

def single_mfp(mol, radius, nbits):
    """
    Returns a string representation of the bit vector with '&' separators, or None if mol is None.
    Verbose prints on errors.
    """
    if mol is None:
        print("[single_mfp] Skipped: mol is None")
        return None
    try:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nbits, useChirality=True, useFeatures=True
        )
        return "&".join(str(item) for item in fingerprint)
    except Exception as e:
        import traceback
        print(f"[single_mfp] Error: {e}")
        traceback.print_exc()
        return None


# ---------------------------
# FP Embedding / Utilities
# ---------------------------

def convert_fp_to_embV2(vector, size):
    """
    Converts a fingerprint vector into an embedded vector using a specified size.
    If vector contains None values, filter them out before calling this function.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer. Try size = 16, or size = 32")

    # Helpful diagnostic
    if isinstance(vector, np.ndarray) and vector.dtype == object:
        print("[convert_fp_to_embV2] Warning: input array has dtype=object. "
              "Ensure you filtered out None entries before stacking.")

    if len(vector.shape) == 1:
        rows = 1 
        cols = vector.shape[0]
    else:
        rows, cols = vector.shape

    narrays = cols // size

    mask = 2**np.arange(size, dtype=np.float64)
    mask = mask / np.sum(mask)
    bigMask = np.tile(mask, (narrays, 1))

    tensorMask = np.tile(bigMask, (rows, 1))
    tensorMask = tensorMask.reshape((rows * narrays, size))
    vector_reshape = vector.reshape((rows * narrays, size))
    
    mfp_masked = tensorMask * vector_reshape
    mfp_maskedDotted = np.sum(mfp_masked, axis=1)
    
    if len(vector.shape) == 1:
        return mfp_maskedDotted.reshape((rows, narrays))
    else:
        return mfp_maskedDotted.reshape((rows, narrays)).squeeze()


# ---------------------------
# DataFrame Utilities
# ---------------------------

def normalize_dataframe(df):
    """
    Normalize the values in each column of a Pandas DataFrame to the range [0, 1].
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def columns_without_nan_inf(dataframe):
    """
    Returns a list of columns that do not contain NaN or infinity values.
    """
    nan_columns = dataframe.columns[dataframe.isna().any()].tolist()
    inf_columns = dataframe.columns[dataframe.isin([float('inf'), -float('inf')]).any()].tolist()
    nan_or_inf_columns = list(set(nan_columns + inf_columns))
    columns_wo = [col for col in dataframe.columns if col not in nan_or_inf_columns]
    return columns_wo

def common_elements(list1, list2):
    """
    Returns a list containing only the elements common to both input lists.
    """
    return list(set(list1) & set(list2))

def remove_nan_inf(df):
    """
    Remove columns containing NaN or Inf values from the dataframe.
    """
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    return df_cleaned

def remove_non_numeric(df):
    """
    Remove columns containing non-numeric values from the dataframe.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    return df_numeric

def remove_correlated_features(df, threshold=0.9):
    """
    Remove correlated features from the dataframe.
    """
    corr_matrix = df.corr().abs()
    # Mask retained for clarity, though not used downstream:
    _mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    features_to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)]
    df_filtered = df.drop(columns=features_to_drop)
    return df_filtered

def get_minimal_columns(dfFeatures, threshold=0.95):
    """
    Returns the minimal list of columns that describe all features by removing highly correlated columns.
    """
    corr_matrix = dfFeatures.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    remaining_columns = [column for column in dfFeatures.columns if column not in to_drop]
    return remaining_columns


# ---------------------------
# Bit collision diagnostics
# ---------------------------

def identify_bit_collisions(mol, radius, nbits):
    """
    Identify bit collisions in a Morgan fingerprint.
    Returns (None, None) if mol is None. Verbose prints on errors.
    """
    if mol is None:
        print("[identify_bit_collisions] Skipped: mol is None")
        return None, None
    try:
        info = {}
        _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info, useChirality=True, useFeatures=True)
        bit_counts = np.zeros(nbits, dtype=int)
        for bit, paths in info.items():
            bit_counts[bit] = len(paths)
        collisions = np.where(bit_counts > 1)[0]
        return collisions, bit_counts
    except Exception as e:
        import traceback
        print(f"[identify_bit_collisions] Error: {e}")
        traceback.print_exc()
        return None, None
