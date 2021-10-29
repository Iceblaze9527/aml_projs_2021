import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score


#parameters
folds = 10 #define the k-fold
#for SVR
gamma = 'scale'
tol = 1e-10
C = 500
epsilon = 0.001


# functions

# 1.outlier detection
def outlier_detection(X):
    cov = EllipticEnvelope(random_state=0).fit(X)
    cov.predict(X)
    return cov.correct_covariance(X)

# 2.feature selection 
def feature_selection(X_test, X_train):
    pca = PCA()
    pca.fit(X_test)
    return pca.transform(X_test), pca.transform(X_train)

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
# X_test = outlier_detection(X_test)
# X_train = outlier_detection(X_train)
#sth went wrong here

# reduce dimension
X_test, X_train = feature_selection(X_test, X_train)
#print(X_train, X_test, y)

# kernel ridge regression
kf = KFold(n_splits=folds)
prediction = np.zeros(X_test.shape[0])
#--------------------------------------------
for train_index, test_index in kf.split(X_train):
    X_training, X_testing = X_train[train_index], X_train[test_index]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
    
    regr = SVR(kernel = 'rbf', gamma=gamma,tol=tol,C=C, epsilon=epsilon)
    regr.fit(X_training, y_train)
    y_pred = regr.predict(X_testing)
    # print(y_pred)
    
    #i actually do my outlier detection here
    score = r2_score(y_test, y_pred)
    if score > 0.06:
        prediction += regr.predict(X_test)
        #print(prediction)
    else:
        folds -= 1
    print(score)
y_pred = prediction/folds
#print(y_pred)

# output
filename = 'submatt.csv'
with open(filename, 'w') as output_file:
    output_file.write('id,y\n')
    for i in range(776):
        output_file.write(str(i))
        output_file.write(',')
        output_file.write(str(y_pred[i]))
        output_file.write('\n')