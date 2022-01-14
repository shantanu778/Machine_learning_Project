import numpy as np
import pandas as pd
import os
from os import path
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.pipeline import Pipeline


# ========== ANALYSIS PIPELINE ================================================

'''Preprocessing (standardization), followed by principal component analysis to 
reduce dimensionality, and a linear classifier (logistic regression). We 
optimize number of principal components to add as features (max. 100 to adhere 
to the rule of thumb that N > 10n), type of penalty, solver, and the 
regularization strength. C refers to the inverse of the regularization strength, 
with smaller values representing stronger regularization. 
'''

# Initialise objects relevant preprocessing classes.
scaler = StandardScaler()
pca = PCA(svd_solver='auto')
classifier = LogisticRegression(max_iter=500, multi_class='multinomial')

# Combine into a pipeline.
# PCA and scaler are populated in the param_grid.
pipe = Pipeline(steps=[('scaler', 'passthrough'), ('pca', 'passthrough'), ('classifier', classifier)])


# ========== GRID SEARCH ======================================================

# y_train is the same for all models
y_train = np.load('dataset/y_train.npy')
extr_methods = ['structure', 'hog', 'gradient', 'hotspots']

for extr_method in extr_methods:

    # x_train differs according to the feature extraction method used.
    file_path = path.join('dataset/extracted_features', extr_method)
    X_train = np.load(path.join(file_path, 'X_train.npy'))

    # Specify parameter space to search.
    param_grid = [
        {'scaler': [None],
        'pca': [None], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__penalty': ['none', 'l1', 'l2'], 
        'classifier__solver': ['saga', 'sag']},

        {'scaler': [scaler],
        'pca': [None], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__penalty': ['none', 'l1', 'l2'], 
        'classifier__solver': ['saga', 'sag']},

        {'scaler': [None],
        'pca': [pca],
        'pca__n_components': [np.linspace(2, X_train.shape[1], X_train.shape[1]-1, dtype=int)], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__penalty': ['none', 'l1', 'l2'], 
        'classifier__solver': ['saga', 'sag']},

        {'scaler': [scaler],
        'pca': [pca],
        'pca__n_components': [np.linspace(2, X_train.shape[1], X_train.shape[1]-1, dtype=int)], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__penalty': ['none', 'l1', 'l2'], 
        'classifier__solver': ['saga', 'sag']}
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
    joblib.dump(search.best_estimator_, f'lr_optimal_{extr_method}.joblib')

    # Store output of parameter sweep as pd.DataFrame and export.
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(f'lr_optimal_{extr_method}.csv', sep=',', header=True)
