import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ARDRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline

class Model():
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def _estimator_(self, X, y):
        pass
    
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        pass

class LinearRegression_log(Model):
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.model_name = "LinearRegression_log"

        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        lr = TransformedTargetRegressor(
            regressor = LinearRegression(),
            func=np.log1p,
            inverse_func=np.expm1
        )
        lr_rint = TransformedTargetRegressor(
            regressor=lr,
            inverse_func=np.rint
        )
        model = make_pipeline(
            lr_rint
        )        
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred