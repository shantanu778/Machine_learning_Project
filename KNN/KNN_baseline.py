# Machine Learning Project 2021-2022
# Digit classification
# Baseline KNN, no optimisation

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

# ====================== BASELINE KNN =========================

# Split train data into train (80%) and validation (20%) sets (so as to not use test data until optimal model is found)
trainData,valData,trainLabel,valLabel = train_test_split(X_train,y_train,test_size=0.2,random_state=84)
print("training data points: {}".format(len(trainLabel))) # 800
print("validation data points: {}".format(len(valLabel))) # 200

# Build models for different values of K and check which gives best accuracy
kVals = np.arange(1,30,2)
for k in kVals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData,trainLabel)
    
    # Evaluate the model and update the accuracies list
    score = model.score(valData, valLabel)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))

# k = 3 gives best accuracy of 97%

# Train model with k = 3, now using all of the training data again
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

# Try prediction using the model above
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))

# Results in 97% accuracy