import os

import pandas as pd
import numpy as np

# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer#, KNNImputer #, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, GridSearchCV

root_path = './'
X_train_path = 'X_train.csv'
X_test_path = 'X_test.csv'
y_train_path = 'y_train.csv'
# y_test_path = 'y_test_yutong_v10.csv'

random_state = 30

# Param: Feature Selection
l1_lambda = 0.2
num_features = 225
n_estimators = 100

# Param: Regresion & Model Selection
nFolds = 10
svr_param_grid = {
    'C': np.arange(80,101), 
    'gamma': np.arange(1e-3, 1e-2, step=1e-3)}

## @Fabian's implementation of imputation
"""
Remarks:
(More on imputation later)
"""
def imputation(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X)
    return imp.transform(X)

# def missing_values(X): #imputing missing values
#     #imp = IterativeImputer(n_nearest_features=200, initial_strategy='mean')
#     imp = KNNImputer(n_neighbors=30, missing_values=np.nan)
#     return imp.fit_transform(X)

## @Fabian's implementation of isolation forest with some param modification based on handy experiments
"""
Some modifications:
n_estimators: a param worth tuning (altho not of primary concern). I get best results at range 100-150.
max_samples, bootstrap, contamination: insensitive param, use default instead
random_state: fixed to get consistent results
(More on indicatorMat later)
"""

def outlier_detection(X, y, indicatorMat = None):
    isoForest = IsolationForest(n_estimators=n_estimators, random_state=random_state)# n_estimators is set globally
    outlierResults = isoForest.fit_predict(X)
    print(f"number of outliers: {(outlierResults != 1).sum()} outliers in {X.shape[0]} samples")
    #print(f"anomaly scores: {isoForest.score_samples(X)}")
    #plt.hist(isoForest.decision_function(X))
    #plt.show()
    if indicatorMat is None:
        return X[outlierResults == 1], y[outlierResults == 1]
    else:
        return X[outlierResults == 1], y[outlierResults == 1], indicatorMat[outlierResults == 1]


## @Fabian's implementation of feature selection. Modified some part to facilitate my own understanding of the code and ensemble
"""
Remarks:
I simply used SelectKBest() to filter features based on f_regression method. 
But I carefully tuned num_features, the best result is around 225.
We need to decide what's the best strategy here.
"""

def feature_selection(X_test, X_train, y_train, missingValsTrain=None, missingValsTest=None):
    # selectedFeaturesFabian = feature_selection_fabian(X_train, y_train)
    # selectedFeaturesMatt = feature_selection_matt(X_train, y_train)
    selectedFeaturesYutong = feature_selection_yutong(X_train, y_train)
    
    # selectedIdxs = selectedFeaturesFabian
    #selectedIdxs = selectedFeaturesMatt
    selectedIdxs = selectedFeaturesYutong
    #selectedIdxs = np.intersect1d(selectedFeaturesFabian, selectedFeaturesMatt)## functools.reduce(np.intersect1d, all_lists) for multiple intersection
    #selectedIdxs = np.union1d(selectedFeaturesFabian, selectedFeaturesMatt)
    print(f"selected {len(selectedIdxs)} out of {X_train.shape[1]} features")
    
    selectedX_train = X_train[:, selectedIdxs]
    selectedX_test = X_test[:, selectedIdxs]
    
    if missingValsTrain is None or missingValsTest is None:
        return selectedX_test, selectedX_train
    elif missingValsTrain is not None and missingValsTest is not None:
        return selectedX_test, selectedX_train, missingValsTrain[:,selectedIdxs], missingValsTest[:,selectedIdxs]
    else:
        raise NotImplementedError("missingValsTrain and missingValsTest must either be both given or both None")

def feature_selection_fabian(X_train, y_train): 
    normalizedX_train = StandardScaler().fit_transform(X_train)
    
    lasso = Lasso(alpha=l1_lambda).fit(normalizedX_train, y_train)## l1_lambda is set globally
    selectionModel = SelectFromModel(lasso, prefit=True)
    
    return selectionModel.get_support(indices=True)

## this is actually an older version of @Matt's work thus is subjected to change
def feature_selection_matt(X_train, y_train):
    C = 400
    f_test, _ = f_regression(X_train, y_train, center=False)
    f_test /= np.max(f_test)
    f_test_idx = f_test.argsort()[-C:][::-1]
    
    mi = mutual_info_regression(X_train, y_train, n_neighbors=5)
    mi /= np.max(mi)
    
    mi_idx = mi.argsort()[-C:][::-1]
    idx_filtered = np.intersect1d(f_test_idx,mi_idx)
    print(f"selected {len(idx_filtered)} features using f_regression and mutual_info_regression")
    
    #return X_test[:,idx_filtered], X_train[:,idx_filtered]
    return idx_filtered #modified to return indices instead of filtered data

def feature_selection_yutong(X_train, y_train):
    kbest = SelectKBest(score_func = f_regression, k=num_features).fit(X_train, y_train)## num_features is set globally
    
    return kbest.get_support(indices=True)

# Regressor Class from Fabian
"""
Remark:
1. About preprocessing
- I think you are doing outlier detection first then feature selection.
- but in my experiments on svr, doing feature selection first gives systematically better results (abt 0.04 improvement on r^2)
- I cannot guarantee if this also applies to other methods, but anyway the processing order needs to be justified and the performance evaluated.

- And I don't quite get why you reintroduce the missing values and use other imputation methods, imho the nans should be gotten rid of asap...
- also, both IterativeImputer and KNNImputer suffer from high dimensionality (~200 features are still a high-dim case, and you are even doing this before feature selection...)

2. About code design
- current implementation does not allow setting the params of a single regressor from the outside, so I use **kwargs in the base regressor
- prediction methods are introduced (may not be necessary for your whole design)
"""

class BaseRegressorSetup:
    def __init__(self, X_train, Y_train, X_test, **kwargs): #X_train and X_test are assumed to be unmodified (i.e. directly read from the csv)
        self.paramDescription = ""
        self.kwargs = kwargs

        ## I still leave the missing entry vars here, but since I don't understand the benefit of the refilling, I am not using it.
        missingTrainEntries = X_train == np.nan
        
        X_train = imputation(X_train)
        X_test = imputation(X_test) 
        X_test, X_train = feature_selection(X_test, X_train, Y_train)
        X_train, Y_train, missingTrainEntries = outlier_detection(X_train, Y_train, missingTrainEntries)
        
        """
        Remark: a slight mistake on original rescaling implementation
        standard scaler is used to normalize data, where the data should be subtracted by its own mean and divided by its variance.
        So X_train and X_test should be scaled respectively instead of using the train scaler to transform the test data.
        Even though we often assume the training set and test set have the same distribution, unfortunate splits can happen 
        and in this case we will introduce error to the results
        """
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
       
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
    
    def getRegressor(self):
        raise NotImplementedError("getRegressor needs to be implemented in a subclass")

    def getFittedRegressor(self, is_shuffle=False):
        if is_shuffle:
            X_train, Y_train = shuffle(self.X_train, self.Y_train, random_state=random_state)
        else:
            X_train = self.X_train
            Y_train = self.Y_train

        return self.getRegressor().fit(X_train, Y_train)

    def crossValidate(self, nFolds, score='r2', is_shuffle=False):
        if is_shuffle:
            X_train, Y_train = shuffle(self.X_train, self.Y_train, random_state=random_state)
        else:
            X_train = self.X_train
            Y_train = self.Y_train
        return cross_validate(self.getRegressor(), X_train, y=Y_train, scoring=score, cv=nFolds, return_train_score=True)
    
    ## temp method for prediction
    def getPrediction(self):
        X_test = self.X_test
        return self.getFittedRegressor().predict(X_test)

## my implementation of SVR (using grid search to find an optimal regressor)
class SupportVectorRegressorSetup(BaseRegressorSetup):
    def __init__(self, X_train, Y_train, X_test, svr_param_grid=None, nFolds=None):
        super().__init__(X_train, Y_train, X_test, svr_param_grid=svr_param_grid, nFolds=nFolds)
        self.paramDescription = ""

    def getRegressor(self):
        regr = GridSearchCV(SVR(), self.kwargs['svr_param_grid'], scoring='r2', n_jobs=-1, cv=self.kwargs['nFolds'])# n_jobs=-1: use all cpu cores
        return regr

if __name__ == '__main__':
    pass
    # read_csv = lambda file_name: pd.read_csv(file_name).values[:,1:]
    
    # X_train_raw = read_csv(os.path.join(root_path, X_train_path))
    # X_test_raw = read_csv(os.path.join(root_path, X_test_path))
    # y_train_raw = read_csv(os.path.join(root_path, y_train_path)).ravel()

    # svr = SupportVectorRegressorSetup(X_train_raw, y_train_raw, X_test_raw, svr_param_grid=svr_param_grid, nFolds=nFolds)
    
    # y_test_pred = svr.getPrediction()

    # df_result = pd.DataFrame(data = y_test_pred, columns=['y'])
    # df_result.to_csv(path_or_buf=os.path.join(root_path,y_test_path), index_label='id')