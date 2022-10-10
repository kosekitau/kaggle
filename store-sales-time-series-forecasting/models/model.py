import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

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
        
class Ridge_log(Model):
    
    def __init__(self, alpha, standardscaler_features, drop_features, n_jobs=-1):
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.model_name = "Ridge_log"
        self.drop_features = drop_features
        self.standardscaler_features = list(set(standardscaler_features) - set(self.drop_features)) # 標準化をかけるカラム

        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        ridge = TransformedTargetRegressor(
            regressor = Ridge(alpha=self.alpha),
            func=np.log1p,
            inverse_func=np.expm1
        )
        ridge_rint = TransformedTargetRegressor(
            regressor=ridge,
            inverse_func=np.rint
        )
        ct = ColumnTransformer([("StandardScaler", StandardScaler(), self.standardscaler_features)], remainder="passthrough")
        model = make_pipeline(
            ct,
            ridge_rint
        )        
        self.model = model
    
    def fit(self, X, y):
        X = X.drop(self.drop_features, axis=1)
        self.model.fit(X, y)
        self.coef_ = self.model["transformedtargetregressor"].regressor_.regressor_.coef_.reshape(-1, 1)
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
        
        