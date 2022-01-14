# Data splitting for digit classification

import numpy as np
import os

# Splitting data into test and train sets

def split_digits(X, y):
    X_train = np.array(X[0:100])
    y_train = np.array(y[0:100])

    end = 300

    for i in range(9):
        start = end - 100
        X_train = np.append(X_train, X[start:end], axis=0)
        y_train = np.append(y_train, y[start:end], axis=0)
        end += 200 

    X_test = np.array(X[100:200])
    y_test = np.array(y[100:200])
    
    end = 400

    for i in range(9):
        start = end - 100
        X_test = np.append(X_test, X[start:end], axis=0)
        y_test = np.append(y_test, y[start:end], axis=0)
        end += 200 

    return X_train, y_train, X_test, y_test


# Set directory
os.chdir('C:\\Users\\Veera Ruuskanen\\Documents\\Uni\\Master\\2nd year\\Machine Learning\\Project\\data_scripts\\data')
path = 'mfeat-pix.txt'
with open(path,'r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)] # create labels

X = np.array(X, dtype=np.int32)
y = np.array(y)

N = X.shape[0]
n = X.shape[1]

X_train, y_train, X_test, y_test = split_digits(X,y)

# Export data

np.save('X_train.npy', X_train, allow_pickle=True)
np.save('y_train.npy', y_train, allow_pickle=True)
np.save('X_test.npy', X_test, allow_pickle=True)
np.save('y_test.npy', y_test, allow_pickle=True)