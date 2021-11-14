import sklearn
import numpy as np
import pandas as pd
import sklearn.pipeline

class RegressorEnsemble:
    def __init__(self, *regressors, removeStandardScalers=False, strategy='median'):
        """
        An ensemble of regressors. Strategy can be either 'median', 'mean' or a function object that takes a numpy array of shape (len(regressors), n) and returns a numpy array of shape (n,), where element i in the returned array depends only on elements [:, i] in the input array.
        """
        if removeStandardScalers:
            self.regressors = []
            for regr in regressors:
                if isinstance(regr, sklearn.pipeline.Pipeline):
                    steps = []
                    for label, step in regr.named_steps.items():
                        if not isinstance(step, sklearn.preprocessing.StandardScaler):
                            steps.append((label, step))
                    if len(steps) > 1:
                        self.regressors.append(sklearn.pipeline.Pipeline(steps))
                    else:
                        self.regressors.append(steps[0][1])
                else:
                    self.regressors.append(regr)
        else:
            self.regressors = regressors
        if strategy == 'median':
            self.strategy = lambda x: np.median(x, axis=0)
        elif strategy == 'mean':
            self.strategy = lambda x: np.mean(x, axis=0)
        else:
            self.strategy = strategy

    def predict(self, X):
        individualPreds = np.stack([regr.predict(X) for regr in self.regressors], axis=0)
        return self.strategy(individualPreds)

    def fit(self, X, y):
        for regr in self.regressors:
            regr.fit(X, y)
        return self


