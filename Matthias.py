import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler


#parameters
folds = 20 #define the k-fold

#for SVR
gamma = 'scale'
tol = 1e-3
C = 100
epsilon = 0.1


# functions

# 1.outlier detection: does not work, maybe not necessary because I filter bellow in the regression part
def outlier_detection(X_test, X_train):
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_test), scaler.transform(X_train)

# 2.feature selection: might work now
def feature_selection(X_test, X_train, y_train):
    f_test, _ = f_regression(X_train, y_train,center=False)
    f_test /= np.max(f_test)
    f_test_idx = f_test.argsort()[-C:][::-1]
    
    mi = mutual_info_regression(X_train, y_train,n_neighbors=5)
    mi /= np.max(mi)
    
    mi_idx = mi.argsort()[-C:][::-1]
    idx_filtered = np.intersect1d(f_test_idx,mi_idx)
    
    return X_test[:,idx_filtered], X_train[:,idx_filtered]

# 3.imputation of missing values: should work
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
# print(X_train)
# print('------------------')

# fill in missing values
X_test = missing_values(X_test)
X_train = missing_values(X_train)
# print(X_train)
# print('------------------')

# remove outliers
X_test, X_train = outlier_detection(X_test, X_train)
# print(X_train)

# reduce dimension
X_test, X_train = feature_selection(X_test, X_train, y.ravel())
print(X_train)
print('------------------')
#print(X_train, X_test, y)

# SVR with kernel, because it is not linear
kf = KFold(n_splits=folds, shuffle = True, random_state = 256)
prediction = np.zeros(X_test.shape[0])
for train_index, test_index in kf.split(X_train):
    X_training, X_testing = X_train[train_index], X_train[test_index]
    y_train, y_test = y[train_index].ravel(), y[test_index].ravel()
    
    regr = SVR(kernel = 'rbf', gamma=gamma,tol=tol,C=C, epsilon=epsilon)
    regr.fit(X_training, y_train)
    y_pred = regr.predict(X_testing)
    # print(y_pred)
    
    #dismiss bad scores
    score = r2_score(y_test, y_pred)
    if score > 0.4:
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