from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from feature_extraction import FeatureExtractor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


# training set
X_train = np.load('../dataset/X_train.npy')
X_train = np.array(X_train, dtype=np.int32)
y_train = np.load('../dataset/y_train.npy')

# testing set
X_test = np.load('../dataset/X_test.npy')
X_test = np.array(X_test, dtype=np.int32)
y_test = np.load('../dataset/y_test.npy')

scaler = StandardScaler()
pca = PCA(svd_solver='auto')
classifier = svm.SVC()
pipe = Pipeline(steps=[('scaler', 'passthrough'), ('pca', 'passthrough'), ('classifier', classifier)])

extr_methods = ['structure', 'hog', 'gradient', 'hotspots', 'lbp']

for extr_method in extr_methods:


    ft_structure = FeatureExtractor(method=extr_method)
    X_train_transf = ft_structure.transform(X_train, cell_size=2)
    X_test_transf = ft_structure.transform(X_test, cell_size=2)

    # Specify parameter space to search.
    param_grid = [
        {'scaler': [None],
        'pca': [None], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__kernel': ['linear', 'rbf', 'poly']
        },

        {'scaler': [scaler],
        'pca': [None], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__kernel': ['linear', 'rbf', 'poly']},

        {'scaler': [None],
        'pca': [pca],
        'pca__n_components': [i for i in range(2, X_train_transf.shape[1]+1)], 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__kernel': ['linear', 'rbf', 'poly']
        },

        {'scaler': [scaler],
        'pca': [pca],
        'pca__n_components': np.linspace(2, X_train_transf.shape[1], X_train_transf.shape[1]-1, dtype=int), 
        'classifier__C': np.logspace(-4, 2, num=10),
        'classifier__kernel': ['linear', 'rbf', 'poly']
        }
        ]

    # Initialise the GridSearchCV object and run the grid search. 
    search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=8, cv=5)
    search.fit(X_train_transf, y_train)

    # Print test accuracy and parameters. Note: this refers to the highest
    # mean accuracy on the test datasets during k-fold cross-validation.
    print("=====================================================")
    print(f'\n{extr_method}')
    print('Best parameter (CV score=%0.3f):' % search.best_score_)
    print(search.best_params_)

    # Print training accuracy of optimal model, trained on the entire train set. 
    print(search.best_estimator_.score(X_test_transf, y_test))