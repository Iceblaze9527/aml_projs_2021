import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# functions

# 1.outlier detection
def outlier_detection(X):
    cov = EllipticEnvelope(random_state=0).fit(X)
    cov.predict(X)
    return cov.correct_covariance(X)

# 2.feature selection 
def feature_selection(X):
    pca = PCA()
    return pca.fit_transform(X)

# 3.imputation of missing values
def missing_values(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    imp.transform(X)
    return imp.transform(X)

# read in data
def read_csv(file_name):
    data = pd.read_csv(file_name).values[:,1:]
    return data


# main

# read csv files
X_train = read_csv('X_train.csv')
X_test = read_csv('X_test.csv')
y = read_csv('y_train.csv')

# fill in missing values
X_test = missing_values(X_test)
X_train = missing_values(X_train)

# remove outliers
X_test = outlier_detection(X_test)
X_train = outlier_detection(X_train)

# reduce dimension
X_test = feature_selection(X_test)


# linera/lasso/ridge regression

#score
# from sklearn.metrics import r2_score
# score = r2_score(y, y_pred)
# print(score)


