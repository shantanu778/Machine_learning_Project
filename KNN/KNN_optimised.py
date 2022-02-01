# Machine Learning Project 2021-2022
# Digit classification
# KNN with scaler and PCA

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

# Import data (train is 50% of all digit data -> 1000)
X_train = np.load('data\\X_train.npy')
y_train = np.load('data\\y_train.npy')
X_test = np.load('data\\X_test.npy')
y_test = np.load('data\\y_test.npy')

# Initialise objects relevant preprocessing classes
scaler = StandardScaler()
pca = PCA(svd_solver='auto')
classifier = KNeighborsClassifier(n_neighbors=3)

# Combine into a pipeline
pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('classifier', classifier)])

# Specify parameter space to search
param_grid = {
    'pca__n_components': np.linspace(2, 100, 100-1, dtype=int), 
    'classifier__n_neighbors': np.arange(1,30,2)
}

# Initialise the GridSearchCV object and run the grid search. 
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=8, cv=5, verbose=4)
search.fit(X_train, y_train)

# Print test accuracy and parameters of optimal model. Note: this refers to 
# the highest mean accuracy on the test datasets during k-fold cross-
# validation. 
print("Best parameter (CV score=%0.3f):" % search.best_score_) # mean acc
print(search.best_params_)

# Print training accuracy of optimal model, trained on the entire train set. 
print(search.best_estimator_.score(X_train, y_train))

# Export model and results
# Save compressed version of the optimal model.
joblib.dump(search.best_estimator_, 'KNN_optimal.joblib')

# Store output of parameter sweep as pd.DataFrame and export.
results = pd.DataFrame(search.cv_results_)
results.to_csv('KNN_optimal.csv', sep=',', header=True)

# =================== TEST =========================

classifier = joblib.load('KNN_optimal.joblib')
results = pd.read_csv('KNN_optimal.csv', sep=',')
print(results)

# Verify test error on train set. 
print(classifier.score(X_train, y_train))

# Estimate risk. 
print(classifier.score(X_test, y_test)) #96.7%