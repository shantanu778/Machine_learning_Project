# Machine Learning Project 2021-2022
# Digit classification
# KNN with features

# Libraries and directory
import os
os.chdir('C:\\Users\\Veera Ruuskanen\\Machine_learning_Project\\KNN') 
import joblib
import numpy as np
import pandas as pd
from os import path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from importlib import import_module
from feature_extraction import FeatureExtractor

# Import data 
# training set
X_train = np.load('data/X_train.npy')
X_train = np.array(X_train, dtype=np.int32)
X_train = X_train.reshape(1000,16,15)

# testing set
X_test = np.load('data/X_test.npy')
X_test = np.array(X_test, dtype=np.int32)
X_test = X_test.reshape(1000,16,15)

# create subdirectory if necessary
file_path = 'data/extracted_features'
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


# ================ PIPELINE ================
# Initialise objects relevant preprocessing classes.
scaler = StandardScaler()
pca = PCA(svd_solver='auto')
classifier = KNeighborsClassifier(n_neighbors=3)

# Combine into a pipeline.
# PCA and scaler are populated in the param_grid.
pipe = Pipeline(steps=[('scaler', 'passthrough'), ('pca', 'passthrough'), ('classifier', classifier)])

# ========== GRID SEARCH ======================================================

# y_train is the same for all models
y_train = np.load('dataset/y_train.npy')
extr_methods = ['structure', 'hog', 'gradient', 'hotspots']

for extr_method in extr_methods:

    # x_train differs according to the feature extraction method used.
    file_path = path.join('data/extracted_features', extr_method)
    X_train = np.load(path.join(file_path, 'X_train.npy'))

    # Specify parameter space to search.
    param_grid = [
        {'scaler': [None],
        'pca': [None],  
        'classifier__n_neighbors': np.arange(1,30,2)},

        {'scaler': [scaler],
        'pca': [None], 
        'classifier__n_neighbors': np.arange(1,30,2)},

        {'scaler': [None],
        'pca': [pca],
        'pca__n_components': [np.linspace(2, X_train.shape[1], X_train.shape[1]-1, dtype=int)], 
        'classifier__n_neighbors': np.arange(1,30,2)},

        {'scaler': [scaler],
        'pca': [pca],
        'pca__n_components': [np.linspace(2, X_train.shape[1], X_train.shape[1]-1, dtype=int)], 
        'classifier__n_neighbors': np.arange(1,30,2)},
        ]

    # Initialise the GridSearchCV object and run the grid search. 
    search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=8, cv=5, verbose=4)
    search.fit(X_train, y_train)

    # Print test accuracy and parameters. Note: this refers to the highest
    # mean accuracy on the test datasets during k-fold cross-validation.
    print("=====================================================")
    print(f'\n{extr_method.capitalize}')
    print('Best parameter (CV score=%0.3f):' % search.best_score_)
    print(search.best_params_)

    # Print training accuracy of optimal model, trained on the entire train set. 
    print(search.best_estimator_.score(X_train, y_train))

    # Save compressed version of the optimal model.
    joblib.dump(search.best_estimator_, f'KNN_optimal_{extr_method}.joblib')

    # Store output of parameter sweep as pd.DataFrame and export.
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(f'KNN_optimal_{extr_method}.csv', sep=',', header=True)

'''Use code below to test accuracy of all models constructed with the
various feature extraction methods.'''

y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

extr_methods = ['structure', 'hog', 'gradient', 'hotspots']

for extr_method in extr_methods:

    # Load datasets.
    file_path = path.join('data/extracted_features', extr_method)
    X_train = np.load(path.join(file_path, 'X_train.npy'))
    X_test = np.load(path.join(file_path, 'X_test.npy'))

    # Load classifier
    classifier = joblib.load(f'KNN_optimal_{extr_method}.joblib')
    print("==================================================================")
    print(f"\nTraining score for {extr_method}: ", classifier.score(X_train, y_train))
    print(f"Testing score for {extr_method}: ", classifier.score(X_test, y_test))
    print(classifier.get_params())
    print("\n")