import numpy as np
import pandas as pd
import joblib

classifier = joblib.load('lr_optimal.joblib')
results = pd.read_csv('lr_optimal.csv', sep=',')
#print(results)

# Load train data to verify test error on train set. 
X_train = np.load('/Users/noaschwensfeier/python_ml/Machine_learning_Project/dataset/X_train.npy')
y_train = np.load('/Users/noaschwensfeier/python_ml/Machine_learning_Project/dataset/y_train.npy')
classifier.score(X_train, y_train)

# Load test data to estimate risk. 
X_test = np.load('/Users/noaschwensfeier/python_ml/Machine_learning_Project/dataset/X_test.npy')
y_test = np.load('/Users/noaschwensfeier/python_ml/Machine_learning_Project/dataset/y_test.npy')
classifier.score(X_test, y_test)