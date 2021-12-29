import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.pipeline import Pipeline

# ========== IMPORT DATA ======================================================

X_train = np.load('../dataset/X_train.npy')
y_train = np.load('../dataset/y_train.npy')

# ========== ANALYSIS PIPELINE ================================================

'''Preprocessing (standardization), followed by principal component analysis to 
reduce dimensionality, and a linear classifier (logistic regression). We 
optimize number of principal components to add as features (max. 100 to adhere 
to the rule of thumb that N > 10n), type of penalty, solver, and the 
regularization strength. C refers to the inverse of the regularization strength, 
with smaller values representing stronger regularization. 
'''

# Initialise objects relevant preprocessing classes.
scaler = StandardScaler()
pca = PCA(svd_solver='auto')
classifier = LogisticRegression(max_iter=500, multi_class='multinomial')

# Combine into a pipeline.
pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('classifier', classifier)])

# Specify parameter space to search.
param_grid = {
    'pca__n_components': np.linspace(2, 100, 100-1, dtype=int), 
    'classifier__C': np.logspace(-4, 2, num=10),
    'classifier__penalty': ['none', 'l1', 'l2'], 
    'classifier__solver': ['saga', 'sag']
}


# ========== GRID SEARCH ======================================================

# Initialise the GridSearchCV object and run the grid search. 
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=8, cv=5, verbose=4)
search.fit(X_train, y_train)

# Print test accuracy and parameters of optimal model. Note: this refers to 
# the highest mean accuracy on the test datasets during k-fold cross-
# validation. 
print("Best parameter (CV score=%0.3f):" % search.best_score_) # mean acc
print(search.best_params_)

# Print training accuracy of optimal model, trained on the entire train set. 
print(search.best_estimator_.score(X_train, y_train))

# ========== EXPORT MODEL =====================================================

# Save compressed version of the optimal model.
joblib.dump(search.best_estimator_, 'lr_optimal.joblib')

# Store output of parameter sweep as pd.DataFrame and export.
results = pd.DataFrame(search.cv_results_)
results.to_csv('lr_optimal.csv', sep=',', header=True)
