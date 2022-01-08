import numpy as np
from feature_extractor import FeatureExtractor

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

for extr_method in FeatureExtractor.VALID_METHODS:
    ft_structure = FeatureExtractor(method=extr_method)
    X_train_transf = ft_structure.fit_transform(X_train, cell_size=2)
    X_test_transf = ft_structure.fit_transform(X_test, cell_size=2)

    np.save(f'dataset/extracted_features/{extr_method}/X_train.npy', X_train_transf, allow_pickle=True)
    np.save(f'dataset/extracted_features/{extr_method}/X_test.npy', X_test_transf, allow_pickle=True)