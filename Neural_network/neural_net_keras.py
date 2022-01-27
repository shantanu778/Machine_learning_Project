from gc import callbacks
from time import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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
#X_test, y_test = preprocess_data(X_test, y_test)


# Keras model
def create_model(nodes=8,lr_rate = 0.001, activation = None, dropout=0.0):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (240,1)), 
        tf.keras.layers.Dense(nodes, activation = activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(nodes, activation = activation),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = lr_rate), loss='categorical_crossentropy', metrics = ['accuracy'])
    return model



start=time()
# Parameters
nodes = [32, 64, 16]
activation = ['tanh', 'relu', 'sigmoid']
lr_rate = [0.005, 0.01, 0.001]
dropout = [0.2, 0.3, 0.4]
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=200)

param_grid = dict(nodes=nodes, lr_rate=lr_rate, activation=activation, dropout=dropout)

grid = GridSearchCV(estimator=model, param_grid=param_grid, refit = True, cv=5, verbose=2)

# Early Stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=10,
    mode='min', restore_best_weights=True
)

grid_result = grid.fit(X_train, y_train ,verbose=0, callbacks=[es])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("total time:",time()-start)


'''
grid_predictions = grid_result.predict(X_test.reshape(X_test.shape[0], 240,1)) 
print(classification_report(y_test, grid_predictions)) 
'''

model = create_model(grid_result.best_params_['nodes'],grid_result.best_params_['lr_rate'],grid_result.best_params_['activation'], grid_result.best_params_['dropout'])
model.fit(X_train, y_train, epochs = 200, batch_size=None)
y_pred= model.predict(X_test.reshape(X_test.shape[0], 240,1)) 
print(classification_report(y_test, np.argmax(y_pred, axis=1))) 


