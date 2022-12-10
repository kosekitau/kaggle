import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Ridge_rint:
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.model_name = "Ridge_rint"

        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        ridge_rint = TransformedTargetRegressor(
            regressor=Ridge(alpha=self.alpha),
            inverse_func=np.rint
        )
        model = make_pipeline(
            StandardScaler(),
            ridge_rint
        )        
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
        
class RandomForest_rint:
    
    def __init__(self, n_estimators, max_depth, random_state, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_name = "RandomForest"
        self.feature_importances_ = None
        
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        randomforest_rint = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        model = TransformedTargetRegressor(
            regressor=randomforest_rint,
            inverse_func=np.rint
        )
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.feature_importances_ = self.model.regressor_.feature_importances_
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred  