import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Lasso
import sklearn.linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import sklearn.svm
import sklearn.utils
import sklearn.ensemble

from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K


folds = 10

def initialImputation(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X)
    return imp.transform(X)


#indicatorMat is a ndarray of same shape as X; the vectors at the outlier-positions are also removed from indicatorMat. It is intended to remember the positions of the values that were missing
def outlier_detection(X, y, indicatorMat = None): #use isolation forest
    isoForest = IsolationForest(n_estimators=200, max_samples="auto", bootstrap=False, contamination=0.05)
    outlierResults = isoForest.fit_predict(X)
    print(f"number of outliers: {(outlierResults != 1).sum()} outliers in {X.shape[0]} samples")
    #print(f"anomaly scores: {isoForest.score_samples(X)}")
    #plt.hist(isoForest.decision_function(X))
    #plt.show()
    if indicatorMat is None:
        return X[outlierResults == 1], y[outlierResults == 1]
    else:
        return X[outlierResults == 1], y[outlierResults == 1], indicatorMat[outlierResults == 1]


def feature_selection(X_test, X_train, y_train, missingValsTrain=None, missingValsTest=None): #missingValsTrain and missingValsTest can be used to remember where there were missing features in the input data (i.e. the same columns are selected on these ndarrays)
    l1_lambda = 0.2
    scaler = StandardScaler()
    scaler.fit(X_train)
    #print(f"means: {scaler.mean_}")
    normalizedX_train = scaler.transform(X_train)
    lasso = Lasso(alpha=l1_lambda).fit(normalizedX_train, y_train)
    selectionModel = SelectFromModel(lasso, prefit=True)
    lassoFilteredIdxs = selectionModel.get_support(indices=True)
    #selectedX_train = selectionModel.transform(X_train)
    selectedFeaturesMatt = feature_selection_matt(X_test, X_train, y_train)
    #print(f"shape of original X_train: {X_train.shape}, shape of feature-selected X_train: {selectedX_train.shape}")
    #selectedX_test = selectionModel.transform(X_test)
    print(f"lasso feature selection with l1_lambda = {l1_lambda} selected {len(lassoFilteredIdxs)} features")
    selectedIdxs = np.intersect1d(lassoFilteredIdxs, selectedFeaturesMatt)
    #selectedIdxs = np.union1d(lassoFilteredIdxs, selectedFeaturesMatt)
    print(f"selected {len(selectedIdxs)} out of {X_train.shape[1]} features")
    selectedX_train = X_train[:, selectedIdxs]
    selectedX_test = X_test[:, selectedIdxs]
    if missingValsTrain is None or missingValsTest is None:
        return selectedX_test, selectedX_train
    elif missingValsTrain is not None and missingValsTest is not None:
        return selectedX_test, selectedX_train, missingValsTrain[:,selectedIdxs], missingValsTest[:,selectedIdxs]
    else:
        raise NotImplementedError("missingValsTrain and missingValsTest must either be both given or both None")

def feature_selection_matt(X_test, X_train, y_train): #feature selection from Matt; using to for ensemble with feature selection from lasso
    C = 400
    f_test, _ = f_regression(X_train, y_train,center=False)
    f_test /= np.max(f_test)
    f_test_idx = f_test.argsort()[-C:][::-1]
    
    mi = mutual_info_regression(X_train, y_train,n_neighbors=5)
    mi /= np.max(mi)
    
    mi_idx = mi.argsort()[-C:][::-1]
    idx_filtered = np.intersect1d(f_test_idx,mi_idx)
    print(f"selected {len(idx_filtered)} features using f_regression and mutual_info_regression")
    
    #return X_test[:,idx_filtered], X_train[:,idx_filtered]
    return idx_filtered #modified to return indices instead of filtered data


def missing_values(X): #imputing missing values
    #imp = IterativeImputer(n_nearest_features=200, initial_strategy='mean')
    imp = sklearn.impute.KNNImputer(n_neighbors=30, missing_values=np.nan)
    return imp.fit_transform(X)

def rescale(X_test, X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    normalizedX_train = scaler.transform(X_train)
    normalizedX_test = scaler.transform(X_test)
    return normalizedX_test, normalizedX_train

# read in data
def read_csv(file_name):
    data = pd.read_csv(file_name).values[:,1:]
    return data


def createModel():
    model = keras.Sequential()
    l1_lambda = 0.01
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, kernel_regularizer=keras.regularizers.l1(l1_lambda), bias_regularizer=keras.regularizers.l1(l1_lambda)))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=[det_coeff])
    return model

def det_coeff(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



class BaseRegressorSetup: #Base class for regressors; constructor initializes internal train and test sets with some general preprocessing (which might not be suitable for all regressors used

    def __init__(self, X_train, Y_train, X_test): #X_train and X_test are assumed to be unmodified (i.e. directly read from the csv)
        self.paramDescription = ""
        
        missingTrainEntries = X_train == np.nan
        missingTestEntries = X_test == np.nan
        
        X_train = initialImputation(X_train)
        X_test = initialImputation(X_test)
        
        # remove outliers
        X_train, Y_train, missingTrainEntries = outlier_detection(X_train, Y_train, missingTrainEntries)
        
        X_test, X_train = rescale(X_test, X_train)
        
        X_train[missingTrainEntries] = np.nan
        X_test[missingTestEntries] = np.nan
        
        # fill in missing values
        X_test = missing_values(X_test)
        X_train = missing_values(X_train)
        
        
        Y_train = Y_train.ravel()
        
        X_test, X_train = feature_selection(X_test, X_train, Y_train)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def getRegressor(self):
        raise NotImplementedError("getRegressor needs to be implemented in a subclass")

    def getFittedRegressor(self, shuffle=False):
        if shuffle:
            X_train, Y_train = sklearn.utils.shuffle(self.X_train, self.Y_train)
        else:
            X_train = self.X_train
            Y_train = self.Y_train

        return self.getRegressor().fit(X_train, Y_train)

    def crossValidate(self, nFolds, score='r2', shuffle=False):
        if shuffle:
            X_train, Y_train = sklearn.utils.shuffle(self.X_train, self.Y_train)
        else:
            X_train = self.X_train
            Y_train = self.Y_train
        return sklearn.model_selection.cross_validate(self.getRegressor(), X_train, y=Y_train, scoring=score, cv=nFolds, return_train_score=True)

class ElasticNetRegressorSetup(BaseRegressorSetup):

    def getRegressor(self):
        regr = sklearn.linear_model.ElasticNetCV(cv=10).fit(self.X_train, y=self.Y_train)
        self.paramDescription = "_alpha_" + str(regr.alpha_) + "_l1_ratio_" + str(regr.l1_ratio_)
        regr = sklearn.linear_model.ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_)
        return regr

class GradientBoostingRegressorSetup(BaseRegressorSetup):

    def __init__(self, X_train, Y_train, X_test): #X_train and X_test are assumed to be unmodified (i.e. directly read from the csv)
        missingTrainEntries = X_train == np.nan
        missingTestEntries = X_test == np.nan
        
        X_train = initialImputation(X_train)
        X_test = initialImputation(X_test)
        
        # remove outliers
        X_train, Y_train, missingTrainEntries = outlier_detection(X_train, Y_train, missingTrainEntries)
        
        
        Y_train = Y_train.ravel()
        
        X_test, X_train, missingTrainEntries, missingTestEntries = feature_selection(X_test, X_train, Y_train, missingTrainEntries, missingTestEntries)
        X_train[missingTrainEntries] = np.nan #HistGradientBoostingRegressor can handle missing values
        X_test[missingTestEntries] = np.nan

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        #self.maxLeafNodes = 20
        self.setParams()
    
    def setParams(self, maxLeafNodes=30, earlyStoppingTol=1e-2):
        self.maxLeafNodes = maxLeafNodes
        self.earlyStoppingTol = earlyStoppingTol

    def getRegressor(self):
        self.paramDescription = "maxLeafNodes_" + str(self.maxLeafNodes) + "_earlyStoppingTol_" + str(self.earlyStoppingTol)
        regr = sklearn.ensemble.HistGradientBoostingRegressor(loss='squared_error', max_leaf_nodes=self.maxLeafNodes, early_stopping=True, tol=self.earlyStoppingTol, scoring='r2')
        return regr


# main

# read csv files
X_train = read_csv('X_train.csv')
X_test = read_csv('X_test.csv')
y = read_csv('y_train.csv')

#missingTrainEntries = X_train == np.nan
#missingTestEntries = X_test == np.nan
#
#X_train = initialImputation(X_train)
#X_test = initialImputation(X_test)
#
## remove outliers
#X_train, y, missingTrainEntries = outlier_detection(X_train, y, missingTrainEntries) #TODO: maybe add second iteration of imputing missing values after outlier detection
#
#X_test, X_train = rescale(X_test, X_train)
#
#X_train[missingTrainEntries] = np.nan
#X_test[missingTestEntries] = np.nan
#
## fill in missing values
#X_test = missing_values(X_test)
#X_train = missing_values(X_train)
#
#
#y = y.ravel()
#
#X_test, X_train = feature_selection(X_test, X_train, y.ravel())
#
#
##regr = Lasso(alpha=0.1)
#regr = sklearn.linear_model.ElasticNetCV(cv=folds).fit(X_train, y)
#regr = sklearn.linear_model.ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_)


#gamma = 'scale'
#tol = 1e-3
#C = 100
#epsilon = 0.1
#regr = SVR(kernel = 'rbf', gamma=gamma,tol=tol,C=C, epsilon=epsilon)
#X_train, y = sklearn.utils.shuffle(X_train, y)
#cvScores = cross_val_score(regr, X_train, y.ravel(), cv=folds)

#folds = 8
#batchSize = 64
#nEpochs = 200
#
#histories = []
#kf = KFold(n_splits=8, shuffle=True)
#for trainIdx, testIdx in kf.split(X_train):
#    model = createModel()
#    history = model.fit(X_train[trainIdx], y[trainIdx], batch_size=batchSize, epochs=nEpochs, validation_data=(X_train[testIdx], y[testIdx]))
#    print(history.history.items())
#    histories.append(history)
#
#subplotCols = 4
#fig, axs = plt.subplots((len(histories) + 1)//subplotCols, subplotCols, sharey=True, sharex=True)
#for i, history in enumerate(histories):
#    axs[i//subplotCols, i % subplotCols].plot(history.history['loss'], color='blue')
#    axs[i//subplotCols, i % subplotCols].plot(history.history['val_loss'], color='orange')
#plt.show()
#
#fig, axs = plt.subplots((len(histories) + 1)//subplotCols, subplotCols, sharey=True, sharex=True)
#for i, history in enumerate(histories):
#    axs[i//subplotCols, i % subplotCols].plot(history.history['val_det_coeff'], color='orange')
#plt.show()

regSetup = GradientBoostingRegressorSetup(X_train, y, X_test)

crossValResults = regSetup.crossValidate(nFolds=10, shuffle=True)

plt.title("cross validation scores (coefficients of determination)")
plt.xlabel("fold number")
plt.ylabel("score")
plt.plot(crossValResults['train_score'], color='blue', label='train_score')
plt.plot(crossValResults['test_score'], color='orange', label='test_score')
plt.show()


testOut = 'subfab_' + type(regSetup).__name__ + "_" + regSetup.paramDescription + '.csv'
regr = regSetup.getFittedRegressor(shuffle=True)

if isinstance(regSetup, GradientBoostingRegressorSetup):
    print(f"did {regr.n_iter_} training iterations")
    print(f"train scores:\n{regr.train_score_}")
    print(f"val scores:\n{regr.validation_score_}")
    plt.plot(regr.train_score_, color='blue', label='train_score')
    plt.plot(regr.validation_score_, color='orange', label='test_score')
    plt.show()

yPred = regr.predict(regSetup.X_test)
assert(len(yPred) == X_test.shape[0])
outDf = pd.DataFrame({'y': yPred})
outDf.to_csv(testOut, index_label="id")
