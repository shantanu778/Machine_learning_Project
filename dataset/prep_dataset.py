import numpy as np
import pandas as pd

def split_digits(X, y):
    '''
    Implementing the customary split of the Digits dataset into 1000 training 
    and 1000 testing examples. Rows 0-100 are training examples, 101-200 are 
    testing examples, 201-300 are training examples, and so on.
    ---------------
    Input arguments: 
    X:  matrix of dimension 2000 x 240 contains intensity values of 240 pixels 
        for 2000 greyscale images of digits, values range from [0,6]
    y:  vector of length 2000 contains labels {0,1,2,3,4,5,6,7,8,9}
    '''

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


# ========== IMPORT DATA ======================================================

path = 'dataset.txt'
with open(path,'r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)] # create labels


X = np.array(X, dtype=np.int32)
y = np.array(y)

N = X.shape[0]
n = X.shape[1]

X_train, y_train, X_test, y_test = split_digits(X,y)


# ========== EXPORT DATA ======================================================

np.save('X_train.npy', X_train, allow_pickle=True)
np.save('y_train.npy', y_train, allow_pickle=True)
np.save('X_test.npy', X_test, allow_pickle=True)
np.save('y_test.npy', y_test, allow_pickle=True)
