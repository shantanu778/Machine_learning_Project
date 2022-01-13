import numpy as np
import os
from os import path
from importlib import import_module
#import_module("ffeature_extraction")
from feature_extraction import FeatureExtractor

# ========== IMPORT DATA ======================================================

# training set
X_train = np.load('dataset/X_train.npy')
X_train = np.array(X_train, dtype=np.int32)
X_train = X_train.reshape(1000,16,15)

# testing set
X_test = np.load('dataset/X_test.npy')
X_test = np.array(X_test, dtype=np.int32)
X_test = X_test.reshape(1000,16,15)

# ========== TRANSFORM & EXPORT ===============================================

# create subdirectory if necessary
file_path = 'dataset/extracted_features'
if not path.isdir(file_path):
    os.mkdir(file_path)

for extr_method in FeatureExtractor.VALID_METHODS:

    cur_path = path.join(file_path, extr_method)

    ft_structure = FeatureExtractor(method=extr_method)
    X_train_transf = ft_structure.transform(X_train, cell_size=2)
    X_test_transf = ft_structure.transform(X_test, cell_size=2)

    # create subdirectory if necessary
    if not path.isdir(cur_path):
        os.mkdir(cur_path)

    np.save(path.join(cur_path, 'X_train.npy'), X_train_transf, allow_pickle=True)
    np.save(path.join(cur_path,'X_test.npy'), X_test_transf, allow_pickle=True)