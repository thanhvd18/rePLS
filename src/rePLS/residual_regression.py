import numpy as np
import copy 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.linalg import pinv as pinv2

class rePLS(BaseEstimator, RegressorMixin):

    def __init__(self, Z, n_components):
        self.Z = Z
        self.n_components = n_components
        # initial model for calculating residuals
        self.reg_Zy = LinearRegression()
        self.reg_ZX = LinearRegression()
        
        # residual regression
        self.residual_model = PLSRegression(n_components=n_components)

    def fit(self, X, y=None):
        self.reg_Zy.fit(self.Z, y)
        self.reg_ZX.fit(self.Z, X)

        y_residuals = y - self.reg_Zy.predict(self.Z)
        X_residuals = X - self.reg_ZX.predict(self.Z)


        # Zero-centering residuals if not already zero-centered
        self._mean_y_residuals = np.mean(y_residuals, axis=0)
        self._mean_X_residuals = np.mean(X_residuals, axis=0)
        
        y_residuals -= np.mean(y_residuals, axis=0)
        X_residuals -= np.mean(X_residuals, axis=0)

        
        # Regression model for residuals
        self.residual_model.fit(X_residuals, y_residuals)

        self.P = self.residual_model.x_loadings_ 
        self.Q = self.residual_model.y_loadings_

        # coefficients are not affected by confounders
        self.PQ = self.residual_model.coef_

    def predict(self, X, y=None, Z=None):
        if Z is None:
            Z = self.Z
        X_residuals = X - self.reg_ZX.predict(Z)

        # Zero-centering residuals for prediction
        X_residuals -= self._mean_X_residuals
        

        preds = self.residual_model.predict(X_residuals) + self._mean_y_residuals + self.reg_Zy.predict(Z)

        return np.array(preds)
    def predict_with_components(self, X,components=None, Z=None):
        """
        Predict using a specified number of components from the residual model.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Input data for prediction.
        - components: list, optional
            List of components to use for prediction. If None, all components are used.
        - Z: array-like, shape (n_samples, n_confounders), optional
            Confounding variables. If None, the original Z provided during initialization is used.

        Returns:
        - preds: array-like, shape (n_samples,)
            Predictions based on the specified list of components.
        """
        if Z is None:
            Z = self.Z
        X_residuals = X - self.reg_ZX.predict(Z)
        X_residuals -= np.mean(X_residuals, axis=0)

        if components is not None:
            x_weights = self.residual_model.x_weights_[:, components]
            x_loadings = self.residual_model.x_loadings_[:, components]
            y_loadings = self.residual_model.y_loadings_[:, components]
            x_rot = x_weights @ np.linalg.pinv(x_loadings.T @ x_weights)
            coef_ = (x_rot @ y_loadings.T).T
            coef_ = (coef_ * self.residual_model._y_std) / self.residual_model._x_std
            preds_residual = X_residuals @ coef_.T + self.residual_model.intercept_
        else:
            preds_residual = self.residual_model.predict(X_residuals)
        return preds_residual + self._mean_y_residuals + self.reg_Zy.predict(Z)

class rePCR(BaseEstimator, RegressorMixin):

    def __init__(self,Z,n_components):
        self.Z = Z
        self.n_components = n_components
        #initial model for calculating residual 
        self.reg_Zy = LinearRegression()
        self.reg_ZX = LinearRegression()
        
        #residual regression
        self.residual_model = make_pipeline(PCA(n_components=n_components), LinearRegression())


            
    def fit(self, X, y=None):
        self.reg_Zy.fit(self.Z, y)
        self.reg_ZX.fit(self.Z, X)        
        
        #Calculate residuals        
        y_residuals = y - self.reg_Zy.predict(self.Z)
        X_residuals = X - self.reg_ZX.predict(self.Z)
        
        self._mean_y_residuals = np.mean(y_residuals, axis=0)
        self._mean_X_residuals = np.mean(X_residuals, axis=0)
        
        y_residuals -= np.mean(y_residuals, axis=0)
        X_residuals -= np.mean(X_residuals, axis=0)
        
        #Regession model for residual         
        self.residual_model.fit(X_residuals, y_residuals)
        


    
    def predict(self, X, y=None,Z=None):        
        if Z is None:
            Z = self.Z
        X_residuals = X - self.reg_ZX.predict(Z)
        preds = self.residual_model.predict(X_residuals) + self.reg_Zy.predict(Z) +self._mean_y_residuals
        
        return np.array(preds) 
class reMLR(BaseEstimator, RegressorMixin):

    def __init__(self,Z):
        self.Z = Z
        
        #initial model for calculating residual 
        self.reg_Zy = LinearRegression()
        self.reg_ZX = LinearRegression()
        
        #residual regression
        self.residual_model = LinearRegression()


            
    def fit(self, X, y=None):
        #fit LR
        self.reg_Zy.fit(self.Z, y)
        self.reg_ZX.fit(self.Z, X)        
        
        #Calculate residuals        
        y_residuals = y - self.reg_Zy.predict(self.Z)
        X_residuals = X - self.reg_ZX.predict(self.Z)
        
        y_residuals -= np.mean(y_residuals, axis=0)
        X_residuals -= np.mean(X_residuals, axis=0)
        
        #Regession model for residual         
        self.residual_model.fit(X_residuals, y_residuals)
        


    
    def predict(self, X, y=None,Z=None):        
        if Z is None:
            Z = self.Z
        X_residuals = X - self.reg_ZX.predict(Z)
        preds = self.residual_model.predict(X_residuals) + self.reg_Zy.predict(Z)
        return np.array(preds) 
    