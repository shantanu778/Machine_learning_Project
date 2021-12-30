from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np


from feature_extraction import structure, hog, lbp, gradient, hotspots 




with open('../dataset/dataset.txt','r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)]
X = np.array(X, dtype=np.int32)
X = X.reshape(2000,16,15)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.50,random_state=43)

#Structure
features_train = structure(X_train)
features_test = structure(X_test)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
print("Structure: {}".format(clf.score(features_test, y_test,)))





#HOG 
features_train = hog(X_train)
features_test = hog(X_test)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
print("HOG: {}".format(clf.score(features_test, y_test,)))


#Gradient
features_train = gradient(X_train,100)
features_test = gradient(X_test,100)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
clf.score(features_test, y_test,)
print("Gradient: {}".format(clf.score(features_test, y_test,)))




#Hotspots
features_train = hotspots(X_train)
features_test = hotspots(X_test)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
clf.score(features_test, y_test,)
print("Hotspots: {}".format(clf.score(features_test, y_test,)))




#LBP
features_train = lbp(X_train)
features_test = lbp(X_test)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
clf.score(features_test, y_test,)
print("LBP: {}".format(clf.score(features_test, y_test,)))




