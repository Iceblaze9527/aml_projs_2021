import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import RobustScaler

#parameters
folds = 20 #define the k-fold
random_state = 500 #shuffle of the kfold
k = 200 #how many features in feature reduction are kept

#for SVR
gamma = 'auto'
tol = 1e-8
C = 150
epsilon = 0.01


# functions

# 1.outlier detection:
def outlier_detection(X):
    transformer = RobustScaler().fit(X)
    return transformer.transform(X)

# 2.feature selection:
def feature_selection(X_test, X_train, y_train):
    X_new = SelectKBest(k=k).fit(X_train, y_train)
    return X_new.transform(X_test), X_new.transform(X_train)

# 3.imputation of missing values:
def missing_values(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X)
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
X_test, X_train = feature_selection(X_test, X_train, y.ravel())


# SVR with kernel, because it is not linear
kf = KFold(n_splits=folds, shuffle = True, random_state = random_state)
prediction = np.zeros(X_test.shape[0])
score_ = 0
for train_index, test_index in kf.split(X_train):
    X_training, X_testing = X_train[train_index], X_train[test_index]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
    
    regr = SVR(kernel = 'rbf', gamma=gamma,tol=tol,C=C, epsilon=epsilon)
    regr.fit(X_training, y_train)
    y_pred = regr.predict(X_testing)
    
    #dismiss bad scores
    score = r2_score(y_test, y_pred)
    if score > 0.4:
        prediction += regr.predict(X_test)
        score_ += score
    else:
        folds -= 1
    print(score)
y_pred = prediction/folds
score = score_/folds
print("-------------------------------------------")
print(score)

# output
filename = 'submatt.csv'
with open(filename, 'w') as output_file:
    output_file.write('id,y\n')
    for i in range(776):
        output_file.write(str(i))
        output_file.write(',')
        output_file.write(str(y_pred[i]))
        output_file.write('\n')