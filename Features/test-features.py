from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from feature_extraction import structure, hog, lbp, gradient, hotspots 
from sklearn.pipeline import Pipeline



with open('../dataset/dataset.txt','r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)]
X = np.array(X, dtype=np.int32)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.50,random_state=43)


#PCA
scaler = StandardScaler()
pca = PCA(n_components=100,svd_solver='auto')
classifier = svm.SVC(kernel='rbf')
pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('classifier', classifier)])
pipe.fit(X_train,y_train)
print("PCA: {}".format(pipe.score(X_test, y_test,)))



X_train = X_train.reshape(1000,16,15)
X_test = X_test.reshape(1000,16,15)

#Structure
features_train = structure(X_train,cell_size=2)
features_test = structure(X_test,cell_size=2)
clf = svm.SVC(kernel='rbf').fit(features_train, y_train)
print("Structure: {}".format(clf.score(features_test, y_test,)))





#HOG 
features_train = hog(X_train)
features_test = hog(X_test)
pipe.fit(features_train,y_train)
print("HOG: {}".format(pipe.score(features_test, y_test,)))


#Gradient
features_train = gradient(X_train)
features_test = gradient(X_test)
pipe.fit(features_train,y_train)
print("Gradient: {}".format(pipe.score(features_test, y_test,)))




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




