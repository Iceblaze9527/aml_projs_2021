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

from abc import ABC, abstractmethod #for abstract classes

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("sklearnex is installed. Patched sklearn to use the optimized algorithms.")
except ImportError or ModuleNotFoundError:
    print("sklearnex (intel extension for accelerating sklearn) is not installed, not using it.")


folds = 10
useTestResultsFromCrossValidationEnsemble = False

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

def isolationForestOutlierDetection(X, contamination=0.05, n_estimators=200, max_samples="auto", bootstrap=False):
    """
    Returns a mask where the inliers are 1 and the outliers are 0.
    """
    isoForest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap, contamination=contamination)
    outlierResults = isoForest.fit_predict(X)
    print(f"number of outliers: {(outlierResults != 1).sum()} outliers in {X.shape[0]} samples")
    return outlierResults == 1

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
    #selectedIdxs = np.intersect1d(lassoFilteredIdxs, selectedFeaturesMatt)
    #selectedIdxs = selectedFeaturesMatt
    selectedIdxs = lassoFilteredIdxs
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

def lassoFeatureSelection(X_train, y_train, l1_lambda):
    """
    Returns a list of the indices of the selected features.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    #print(f"means: {scaler.mean_}")
    normalizedX_train = scaler.transform(X_train)
    lasso = Lasso(alpha=l1_lambda).fit(normalizedX_train, y_train)
    selectionModel = SelectFromModel(lasso, prefit=True)
    lassoFilteredIdxs = selectionModel.get_support(indices=True)
    return lassoFilteredIdxs

def fRegressionFeatureSelection(X_train, y_train, numFeatures):
    """
    Returns a list of the indices of the selected features.
    """
    f_test, _ = f_regression(X_train, y_train,center=False)
    f_test /= np.max(f_test)
    f_test_idx = f_test.argsort()[-numFeatures:][::-1]
    return f_test_idx

def mutualInfoRegressionFeatureSelection(X_train, y_train, numFeatures, nNeighbors=5):
    mi = mutual_info_regression(X_train, y_train,n_neighbors=nNeighbors)
    mi /= np.max(mi)
    mi_idx = mi.argsort()[-numFeatures:][::-1]
    return mi_idx

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

def intersect(*args):
    if len(args) == 1:
        return args[0]
    I = np.intersect1d(args[0], args[1])
    for i in range(2, len(args)):
        I = np.intersect1d(args[i], I)
    return I

def missing_values(X): #imputing missing values
    #imp = IterativeImputer(n_nearest_features=200, initial_strategy='mean')
    imp = sklearn.impute.KNNImputer(n_neighbors=30, missing_values=np.nan)
    return imp.fit_transform(X)

def impute_KNN(X, n_neighbors=30):
    imp = sklearn.impute.KNNImputer(n_neighbors=n_neighbors, missing_values=np.nan)
    return imp.fit_transform(X)

def impute_MICE(X, n_nearest_features=200, initial_strategy='mean'):
    imp = IterativeImputer(n_nearest_features=n_nearest_features, initial_strategy=initial_strategy)
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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, kernel_regularizer=keras.regularizers.l1(l1_lambda), bias_regularizer=keras.regularizers.l1(l1_lambda)))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=[det_coeff])
    return model

def det_coeff(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class BaseRegressorWrapper(sklearn.base.BaseEstimator):
    """
    Class to allow RegressorSetup to be used with models that are not from sklearn. Implement a subclass as a wrapper around a model from another ML library.
    """
    @abstractmethod
    def fit(self, X, y):
        return self
    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    @abstractmethod
    def getActualRegressor(self):
        pass

    def displayAdditionalInformation(self): #displays additional information about model training, e.g. train and test losses or scores
        pass

class KerasWrapper(BaseRegressorWrapper):
    def __init__(self, kerasModel: tf.keras.Model, nEpochs, batch_size=None, verbose="auto", callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False
):
        self.model = kerasModel
        self.batch_size=batch_size
        self.nEpochs=nEpochs
        self.verbose=verbose
        self.callbacks=callbacks
        self.validation_split=validation_split
        self.validation_data=validation_data
        self.shuffle=shuffle
        self.class_weight=class_weight
        self.sample_weight=sample_weight
        self.initial_epoch=initial_epoch
        self.steps_per_epoch=steps_per_epoch
        self.validation_steps=validation_steps
        self.validation_batch_size=validation_batch_size
        self.validation_freq=validation_freq
        self.max_queue_size=max_queue_size
        self.workers=workers
        self.use_multiprocessing=use_multiprocessing

        self.history = None

    def fit(self, X, y, validation_split = None, validation_data = None):
        if validation_split is None:
            validation_split = self.validation_split
        if validation_data is None:
            validation_data = self.validation_data
        self.history = self.model.fit(X, y, batch_size=self.batch_size, epochs=self.nEpochs, verbose=self.verbose, callbacks=self.callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=self.shuffle, class_weight=self.class_weight, sample_weight=self.sample_weight, initial_epoch=self.initial_epoch, steps_per_epoch=self.steps_per_epoch, validation_steps=self.validation_steps, validation_batch_size=self.validation_batch_size, validation_freq=self.validation_freq, max_queue_size=self.max_queue_size, workers=self.workers, use_multiprocessing=self.use_multiprocessing)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def displayAdditionalInformation(self):
        if self.history is not None:
            print(f"History: {self.history.history.items()}")
            if 'val_det_coeff' in self.history.history.items():
                plt.subplot(1, 2, 2)
                plt.plot(self.history.history['val_det_coeff'], color='orange', label='val score')
                plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], color='blue', label='train loss')
            plt.plot(self.history.history['val_loss'], color='orange', label='val loss')
            plt.show()
        return self.history

class BaseRegressorSetup: #Base class for regressors; constructor initializes internal train and test sets with some general preprocessing (which might not be suitable for all regressors used

    def __init__(self, X_train, Y_train, X_test): #X_train and X_test are assumed to be unmodified (i.e. directly read from the csv)
        self.paramDescription = ""
        self.prepareDatasets(X_train, Y_train, X_test)


    def prepareDatasets(self, X_train, Y_train, X_test, basicImputationForOutlierDetection=True, normalize=True):
        missingTrainEntries = np.isnan(X_train)
        missingTestEntries = np.isnan(X_test)
        
        if basicImputationForOutlierDetection:
            X_train = initialImputation(X_train)
            X_test = initialImputation(X_test)
        
        # remove outliers
        X_train, Y_train, missingTrainEntries = self.detectOutliers(X_train, Y_train, missingTrainEntries)
        
        if normalize:
            X_test, X_train = rescale(X_test, X_train)
        
        X_train[missingTrainEntries] = np.nan
        X_test[missingTestEntries] = np.nan
        
        # fill in missing values
        X_train, X_test = self.imputeMissing(X_train, X_test)
        
        Y_train = Y_train.ravel()
        
        X_train, X_test = self.selectFeatures(X_train, Y_train, X_test)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def shuffleTrainingData(self):
        self.X_train, self.Y_train = sklearn.utils.shuffle(self.X_train, self.Y_train)

    def getRegressor(self):
        raise NotImplementedError("getRegressor needs to be implemented in a subclass")

    def getFittedRegressor(self, shuffle=False):
        if shuffle:
            X_train, Y_train = sklearn.utils.shuffle(self.X_train, self.Y_train)
        else:
            X_train = self.X_train
            Y_train = self.Y_train

        return self.getRegressor().fit(X_train, Y_train)

    def getParamDescription(self):
        return self.paramDescription

    def crossValidate(self, nFolds, score='r2', shuffle=False):
        if shuffle:
            X_train, Y_train = sklearn.utils.shuffle(self.X_train, self.Y_train)
        else:
            X_train = self.X_train
            Y_train = self.Y_train
        return sklearn.model_selection.cross_validate(self.getRegressor(), X_train, y=Y_train, scoring=score, cv=nFolds, return_train_score=True)

    @classmethod
    def selectFeatures(cls, X_train, Y_train, X_test):
        selectedFeatures = intersect(lassoFeatureSelection(X_train, Y_train, l1_lambda=0.2), fRegressionFeatureSelection(X_train, Y_train, numFeatures=400), mutualInfoRegressionFeatureSelection(X_train, Y_train, numFeatures=400))
        return X_train[:, selectedFeatures], X_test[:, selectedFeatures]

    @classmethod
    def detectOutliers(cls, X_train, Y_train, indicatorMat = None):
        mask = isolationForestOutlierDetection(X_train)
        if indicatorMat is None:
            return X_train[mask], Y_train[mask]
        else:
            return X_train[mask], Y_train[mask], indicatorMat[mask]

    @classmethod
    def imputeMissing(cls, X_train, X_test):
        return impute_KNN(X_train), impute_KNN(X_test)



class ElasticNetRegressorSetup(BaseRegressorSetup):

    def getRegressor(self):
        regr = sklearn.linear_model.ElasticNetCV(cv=10).fit(self.X_train, y=self.Y_train)
        self.paramDescription = "_alpha_" + str(regr.alpha_) + "_l1_ratio_" + str(regr.l1_ratio_)
        regr = sklearn.linear_model.ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_)
        return regr

class GradientBoostingRegressorSetup(BaseRegressorSetup):

    def __init__(self, X_train, Y_train, X_test): #X_train and X_test are assumed to be unmodified (i.e. directly read from the csv)
        missingTrainEntries = np.isnan(X_train)
        missingTestEntries = np.isnan(X_test)
        
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
        self.paramDescription = "maxLeafNodes_" + str(self.maxLeafNodes) + "_earlyStoppingTol_" + str(self.earlyStoppingTol)
    
    def setParams(self, maxLeafNodes=30, earlyStoppingTol=1e-2):
        self.maxLeafNodes = maxLeafNodes
        self.earlyStoppingTol = earlyStoppingTol

    def getRegressor(self):
        regr = sklearn.ensemble.HistGradientBoostingRegressor(loss='squared_error', max_leaf_nodes=self.maxLeafNodes, early_stopping=True, tol=self.earlyStoppingTol, scoring='r2')
        return regr

class RandomForestRegressorSetup(BaseRegressorSetup):

    def __init__(self, X_train, Y_train, X_test):
        super().__init__(X_train, Y_train, X_test)
        self.setParams()
        self.paramDescription = "nTrees_" + str(self.nTrees) + "_maxFeatures_" + str(self.maxFeatures) + "_maxDepth_" + str(self.maxDepth) + "_minSamplesSplit_" + str(self.minSamplesSplit) + "_bootstrap_" + str(self.bootstrap) + "_maxLeafNodes_" + str(self.maxLeafNodes)

    def setParams(self, nTrees=400, maxFeatures=None, maxDepth=None, minSamplesSplit=2, bootstrap=True, maxLeafNodes=None):
        self.nTrees = nTrees
        self.maxFeatures = maxFeatures
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.bootstrap = bootstrap
        self.maxLeafNodes = maxLeafNodes

    def getRegressor(self):
        regr = sklearn.ensemble.RandomForestRegressor(n_estimators=self.nTrees, max_depth=self.maxDepth, min_samples_split=self.minSamplesSplit, bootstrap=self.bootstrap, max_features=self.maxFeatures, max_leaf_nodes=self.maxLeafNodes, n_jobs=-1)
        return regr

DEFAULT_N_EPOCHS = 50
class MlpRegressor(KerasWrapper):

    def __init__(self, inputShape, hiddenLayerSizes=[10], dropout=0.5, nEpochs=DEFAULT_N_EPOCHS):
        self.inputShape = inputShape
        self.hiddenLayerSizes = hiddenLayerSizes
        self.dropout=dropout
        model = self.getNewModel()
        super().__init__(model, nEpochs=nEpochs)
    
    def getNewModel(self):
        #x = keras.Input(shape=self.X_train.shape[1:])
        x = keras.Input(shape=self.inputShape)
        y = x
        for hiddSz in self.hiddenLayerSizes:
            y = layers.Dense(hiddSz, activation='relu')(y)
            y = layers.Dropout(self.dropout)(y)
        y = layers.Dense(1)(y)
        model = keras.Model(inputs=x, outputs=y)
        model.compile(loss="mean_squared_error", optimizer='adam', metrics=[det_coeff])
        #model.summary()
        return model

    def set_params(self, inputShape=None, hiddenLayerSizes=[10], dropout=0.5, nEpochs=None):
        self.hiddenLayerSizes = hiddenLayerSizes
        self.dropout=dropout
        if nEpochs is not None:
            self.nEpochs = nEpochs
        if inputShape is not None:
            self.inputShape = inputShape
        self.model = self.getNewModel()
        return self

    def reset(self):
        self.model = self.getNewModel()
        return self

    def get_params(self, deep=True):
        return {
            "hiddenLayerSizes": self.hiddenLayerSizes,
            "dropout": self.dropout,
            "nEpochs": self.nEpochs,
            "inputShape": self.inputShape
        }


#class MlpRegressorSetup(BaseRegressorSetup, sklearn.base.BaseEstimator, KerasWrapper):
class MlpRegressorSetup(BaseRegressorSetup):
    

    def __init__(self, X_train, Y_train, X_test, hiddenLayerSizes=[10], dropout=0.5, nEpochs=DEFAULT_N_EPOCHS):
        self.prepareDatasets(X_train, Y_train, X_test)
        self.setParams(hiddenLayerSizes=hiddenLayerSizes, dropout=dropout, nEpochs=nEpochs)
        #model = self.getNewModel()
        #KerasWrapper.__init__(self, model, nEpochs=nEpochs)

    def setParams(self, hiddenLayerSizes=[10], dropout=0.5, nEpochs=DEFAULT_N_EPOCHS, **kwargs): #**kwargs is for accepting parameters from MLPRegressor

        #self.hiddenLayerSizes = hiddenLayerSizes
        #self.dropout=dropout
        #self.nEpochs = nEpochs
        self.model = MlpRegressor(inputShape=self.X_train.shape[1:], hiddenLayerSizes=hiddenLayerSizes, dropout=dropout, nEpochs=nEpochs)
        #self.model = self.getNewModel()
    
    def getParamDescription(self):
        def numbersToString(nrs, separator="-"):
            s = ""
            for n in nrs:
                s += str(n) + separator
            return s[:-1]
        return "hiddenLayerSizes_" + numbersToString(self.model.hiddenLayerSizes) + "_dropout_" + str(self.model.dropout)


    def getRegressor(self):
        #first reset model
        return self.model.reset()



    def findBestParams(self, nFolds=10, paramGrid = {"hiddenLayerSizes": [[10], [20, 10]], "dropout": [0.25, 0.3, 0.4, 0.5]}):
        """
        Finds the best parameter combination by exhaustive search; k-fold cross validation is used to determine the performance of each parameter combination.
        After completion, this functions sets the parameters of this RegressorSetup to the best combination and prints this combination.
        """
        gscv = sklearn.model_selection.GridSearchCV(estimator=self.model, param_grid=paramGrid, n_jobs=1, cv=nFolds)
        gscv.fit(self.X_train, self.Y_train)
        self.model.set_params(**gscv.best_params_)
        print("best parameter combination found: " + str(gscv.best_params_))
        return gscv.cv_results_


def testResultsFromCrossValidationEnsembleSingleRegressor(regSetup : BaseRegressorSetup, nFolds, shuffle=True):
    """
    Cross validate the regSetup and use each of the models used for cross validation to predict the test results.
    The test results from the individual models are then reduced to a single result using a weighted average, with the r2_score as weights (folds with negative scores are ignored).
    """
    kf = KFold(n_splits=folds, shuffle=shuffle)
    pred = np.zeros(regSetup.X_test.shape[0])
    predNormalization = 0
    trainScores = []
    valScores = []
    for trainIdxs, testIdxs in kf.split(regSetup.X_train):
        xTrain = regSetup.X_train[trainIdxs]
        yTrain = regSetup.Y_train[trainIdxs]
        xVal = regSetup.X_train[testIdxs]
        yVal = regSetup.Y_train[testIdxs]

        regr = regSetup.getRegressor()
        regr.fit(xTrain, yTrain)
        yValPred = regr.predict(xVal)
        score = r2_score(yVal, yValPred)
        valScores.append(score)
        trainScores.append(r2_score(yTrain, regr.predict(xTrain)))
        
        if score > 0:
            predNormalization += score
            pred += score * regr.predict(regSetup.X_test)
    pred /= predNormalization

    crossValResults = {"train_score": trainScores, "test_score": valScores}
    return pred, crossValResults

def crossValidateEnsemble(regSetups, nFolds: int, returnCrossValidationEnsembleResults: bool, shuffle=True):
    """
    Cross validates an ensemble of regressors. The training and test sets of all regSetups must contain the same samples, in the same order (but do not necessarily have to contain the same imputed values and other preprocessing such as scaling is allowed)
    """
    kf = KFold(n_splits=folds, shuffle=shuffle)
    pred = np.zeros(regSetups[0].X_test.shape[0])
    predNormalization = 0
    trainScores = []
    valScores = []
    for trainIdxs, testIdxs in kf.split(regSetups[0].X_train):
        yTrain = regSetups[0].Y_train[trainIdxs]
        yVal = regSetups[0].Y_train[testIdxs]
        yValPredTotal = np.zeros(yVal.shape)
        yValNormalization = 0
        yTestPredCurSplit = np.zeros(pred.shape)
        yTestNormCurSplit = 0
        yTrainTotal = np.zeros(yTrain.shape)
        for regSetup in regSetups:
            xTrain = regSetup.X_train[trainIdxs]
            xVal = regSetup.X_train[testIdxs]
            xTest = regSetup.X_test

            regr = regSetup.getRegressor()
            regr.fit(xTrain, yTrain)
            yValPred = regr.predict(xVal)
            
            weight = max(r2_score(yVal, yValPred), 0)

            yValNormalization += weight
            yValPredTotal += yValPred * weight

            yTrainTotal += regr.predict(xTrain) * weight

            yTestNormCurSplit += weight
            yTestPredCurSplit += regr.predict(xTest) * weight
        
        yTrainTotal /= yValNormalization
        yValPredTotal /= yValNormalization
        yTestPredCurSplit /= yTestNormCurSplit
        score = r2_score(yVal, yValPredTotal)
        valScores.append(score)
        trainScores.append(r2_score(yTrain, yTrainTotal))

        if score > 0:
            predNormalization += score
            pred += score * yTestNormCurSplit
    pred /= predNormalization

    crossValResults = {"train_score": trainScores, "test_score": valScores}
    if returnCrossValidationEnsembleResults:
        return pred, crossValResults #pred is the prediction of the total ensemble (consisting of nFolds estimators per regSetup) for the test dataset (regSetups[0].X_Test)
    else:
        return crossValResults


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

#ensemble = [GradientBoostingRegressorSetup(X_train, y, X_test), RandomForestRegressorSetup(X_train, y, X_test)]


regSetup = GradientBoostingRegressorSetup(X_train, y, X_test)
#regSetup = RandomForestRegressorSetup(X_train, y, X_test)

#regSetup = MlpRegressorSetup(X_train, y, X_test, hiddenLayerSizes=[20, 10], dropout=0.25, nEpochs=300)
#regSetup.shuffleTrainingData()
#regr = regSetup.getRegressor()
#regr.fit(regSetup.X_train, regSetup.Y_train, validation_split=0.2)
##print(regSetup.findBestParams())
#regr.displayAdditionalInformation()

if "ensemble" in vars() and len(ensemble) > 1:
    if not useTestResultsFromCrossValidationEnsemble:
        print("Note: useTestResultsFromCrossValidationEnsemble is False, but ensemble is defined, so a cross validation ensemble is used anyway")
    yPred, crossValResults = crossValidateEnsemble(ensemble, nFolds=folds, returnCrossValidationEnsembleResults=True, shuffle=True)
    testOut = f"subfab_ensemble_{folds}foldCV"
    for regSetup in ensemble:
        testOut += "-" + type(regSetup).__name__ + "_" + regSetup.getParamDescription()
    testOut += ".csv"
else:
    if useTestResultsFromCrossValidationEnsemble:
        yPred, crossValResults = testResultsFromCrossValidationEnsembleSingleRegressor(regSetup, nFolds=folds)
        testOut = 'subfab_' + str(folds) + "foldCvEnsemble_" + type(regSetup).__name__ + "_" + regSetup.getParamDescription() + '.csv'
    else:
        crossValResults = regSetup.crossValidate(nFolds=folds, shuffle=True)
        testOut = 'subfab_' + type(regSetup).__name__ + "_" + regSetup.getParamDescription() + '.csv'

plt.title("cross validation scores (coefficients of determination)")
plt.xlabel("fold number")
plt.ylabel("score")
plt.plot(crossValResults['train_score'], color='blue', label='train_score')
plt.plot(crossValResults['test_score'], color='orange', label='test_score')
plt.show()


if not useTestResultsFromCrossValidationEnsemble or ("ensemble" not in vars() or len(ensemble) <= 1):
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
