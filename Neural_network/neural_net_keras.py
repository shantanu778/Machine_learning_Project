import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# training set
X_train = np.load('../dataset/X_train.npy')
X_train = np.array(X_train, dtype=np.int32)
y_train = np.load('../dataset/y_train.npy')

# testing set
X_test = np.load('../dataset/X_test.npy')
X_test = np.array(X_test, dtype=np.int32)
y_test = np.load('../dataset/y_test.npy')

def one_hot(Y):
    one_hot_Y= np.zeros((len(Y), max(Y) + 1))
    one_hot_Y[np.arange(len(Y)), Y] = 1
    return one_hot_Y

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.50,random_state=43)
def preprocess_data(x, y):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 240,1)
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = one_hot(y)
    y = y.reshape(y.shape[0], 10)
    return x, y


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)



model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (240,1)), 
    tf.keras.layers.Dense(32, activation = 'tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation = 'tanh'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'sgd', loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 100)


model.evaluate(X_test, y_test)



