import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

with open('dataset.txt','r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)]

X = np.array(X, dtype=np.int32)
print(X.shape)

X.reshape(2000,16,15)


def one_hot(Y):
    one = np.zeros((len(Y),max(Y)+1))
    one[np.arange(len(Y)),Y] = 1
    return one
    
def reverse(x):
    return (6-x)*(255/6)



with open('dataset.txt','r') as data:
    X = [line.strip().split("  ") for line in data]
    y = [i for i in range(10) for j in range(200)]
    
X = np.array(X, dtype=np.int32)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.50,random_state=43)

#one hot encoding
y_train = one_hot(y_train)
y_test = one_hot(y_test)

#normalize
X_train = X_train/6
X_test = X_test/6


X_mean = np.mean(X_train,axis=0)
X_mean.shape


#centering
X_train = X_train-X_mean
X_test = X_test-X_mean


#covariance matrix
C = np.cov(np.transpose(X_train))
C.shape

#SVD
U, S, V = np.linalg.svd(C, full_matrices=True)


# m prinicipal component of PCA
m = 100
Um = U[:,:100] 
Um.shape


#get features
def get_feature(row):
    return np.transpose(Um).dot(row)
features = np.apply_along_axis(get_feature,1,X_train)
features.shape


# linear regression classifier
F = np.transpose(features)
W_t = ((np.linalg.inv(F.dot(np.transpose(F)))).dot(F)).dot(y_train)
W = np.transpose(W_t) 
W.shape


#MSE 
def get_output(row):
    return W.dot(row)
error = y_train -  np.apply_along_axis(get_output,1,features)
error_norm = np.linalg.norm(error,axis=1)
MSE = np.sum(error_norm)/1000


#misclassification
def get_output(row):
    return np.argmax(W.dot(row))

def teacher_output(row):
    return np.argmax(row)

output = np.apply_along_axis(get_output,1,features)
teacher = np.apply_along_axis(teacher_output,1,y_train)
error = np.where(output==teacher,0,1)
error_rate = np.sum(error)/1000

print("Training Error: ", error_rate)

#test 
#get features
def get_feature(row):
    return np.transpose(Um).dot(row)
features_test = np.apply_along_axis(get_feature,1,X_test)


#misclassification
def get_output(row):
    return np.argmax(W.dot(row))

def teacher_output(row):
    return np.argmax(row)

output = np.apply_along_axis(get_output,1,features_test)
teacher = np.apply_along_axis(teacher_output,1,y_test)

## It is a Line for Test
error = np.where(output==teacher,0,1)
error_rate = np.sum(error)/1000
print("Testing Error: ", error_rate)
