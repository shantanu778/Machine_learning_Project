import numpy as np
import pandas as pd
import os
from os import path
import joblib

'''Use commented code below to test accuracy of a single model.'''

# classifier = joblib.load('lr_optimal.joblib')
# results = pd.read_csv('lr_optimal.csv', sep=',')
# #print(results)

# # Load train data to verify test error on train set. 
# X_train = np.load('dataset/X_train.npy')
# y_train = np.load('dataset/y_train.npy')

# # Load test data to estimate risk. 
# X_test = np.load('dataset/X_test.npy')
# y_test = np.load('dataset/y_test.npy')

# print("==================================================================")
# print(f"\nTraining score LogisticRegression: ", classifier.score(X_train, y_train))
# print(f"Testing scoreLogisticRegression: ", classifier.score(X_test, y_test))
# print(classifier.get_params())

'''Use code below to test accuracy of all models constructed with the
various feature extraction methods.'''

y_train = np.load('dataset/y_train.npy')
y_test = np.load('dataset/y_test.npy')

extr_methods = ['structure', 'hog', 'gradient', 'hotspots']

for extr_method in extr_methods:

    # Load datasets.
    file_path = path.join('dataset/extracted_features', extr_method)
    X_train = np.load(path.join(file_path, 'X_train.npy'))
    X_test = np.load(path.join(file_path, 'X_test.npy'))

    # Load classifier
    classifier = joblib.load(f'lr_optimal_{extr_method}.joblib')
    print("==================================================================")
    print(f"\nTraining score for {extr_method}: ", classifier.score(X_train, y_train))
    print(f"Testing score for {extr_method}: ", classifier.score(X_test, y_test))
    print(classifier.get_params())
    print("\n")
