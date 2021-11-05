import os
import joblib

import pandas as pd
import numpy as np

# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer#, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

root_path = './'
X_train_path = 'X_train.csv'
X_test_path = 'X_test.csv'
y_train_path = 'y_train.csv'
y_test_path = 'y_test_yutong_v8.csv'

random_state = 30

# Feature Selection
num_features = np.arange(175,251)
n_estimators = np.arange(100,190, step=15)

# Regresion & Model Selection
best_limit = 10
num_KFold = 10
svr_param_grid = {
    'svr__C': np.arange(80,101), 
    'svr__gamma': np.arange(1e-3, 1e-2, step=1e-3)}


def data_raw(root_path, data_path):
    return pd.read_csv(os.path.join(root_path, data_path)).values[:,1:]

def median_imp(X_raw):
    imp_med = SimpleImputer(missing_values=np.nan, strategy='median')
    return imp_med.fit_transform(X_raw)

def knn_imp(X_raw):##to-do, still a comp intensive methods and suffers from curse of dimensionality and outliers
    pass

# def mice_imp(X_raw):#use after fea sel! this method is comp expensive, and is an unstable implementation based on docs
#     imp_mice = IterativeImputer(missing_values=np.nan, initial_strategy='median')
#     return imp_mice.fit_transform(X_raw)


def feat_sel(X_raw, y, score_func, num_features):
    return SelectKBest(score_func = score_func, k=num_features).fit(X_raw, y.ravel())

# def train_val_split(X_train, y_train, val_size):
#     return train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

def iforest(X_raw, y, n_estimators):
    iso = IsolationForest(n_estimators = n_estimators, random_state=random_state).fit_predict(X_raw)
    return X_raw[np.where(iso==1)], y[np.where(iso==1)]

if __name__ == '__main__':
    X_train_raw = data_raw(root_path, X_train_path)
    X_test_raw = data_raw(root_path, X_test_path)
    y_train_raw = data_raw(root_path, y_train_path)

    X_train_raw = median_imp(X_train_raw)
    X_test_raw = median_imp(X_test_raw)

    best_estimators = [(0,None)] * best_limit

    for num_fea in num_features:
        selector = feat_sel(X_train_raw, y_train_raw, f_regression, num_fea)
        X_train_reduced = selector.transform(X_train_raw)
        X_test = selector.transform(X_test_raw)

        for num_est in n_estimators:
            X_train, y_train = iforest(X_train_reduced, y_train_raw, num_est)

            pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
            search = GridSearchCV(pipe, svr_param_grid, scoring='r2', n_jobs=-1, cv=num_KFold)

            reg = search.fit(X_train, y_train.ravel())

            best_estimators.append((reg.best_score_, reg.best_estimator_))
            best_estimators.sort(key=lambda x:x[0], reverse=True)
            best_estimators.pop()

    for n in range(best_limit):
        if best_estimators[n][1] is not None:
            print("No.%d Estimator: "%(n+1), best_estimators[n][1])
            print("No.%d CV Score: "%(n+1), best_estimators[n][0])
            joblib.dump(best_estimators[n][1], os.path.join(root_path, 'model_score_%.3f.pkl'%(best_estimators[n][0])))

# X_train_raw, X_val, y_train, y_val = train_val_split(X_train_raw, y_train, val_size)
# y_val_pred = reg.predict(X_val)
# print("Val R2 Score: ", r2_score(y_val, y_val_pred))

# Output
# df_result = pd.DataFrame(data = y_test_pred, columns=['y'])
# df_result.to_csv(path_or_buf=os.path.join(root_path,y_test_path), index_label='id')