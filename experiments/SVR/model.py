import numpy as np
import joblib
from sklearn.svm import SVR as svr
import torch
from sklearn.preprocessing import StandardScaler


class SVR:
    """
    Una clase para encapsular un modelo Support Vector Regression (SVR) de scikit-learn.
    """
    def __init__(self, kernel, C, gamma, epsilon):
        self.scaler=StandardScaler()
        self.model = svr(
            kernel = kernel, 
            C = C, 
            gamma = gamma, 
            epsilon = epsilon) 


  
    def predict(self, X: torch.tensor) -> np.ndarray:
        self.scaler.transform(X)
        predictions = self.model.predict(X)
        return predictions


    def save(self, path: str = 'svr_model.joblib'):
        joblib.dump(self.model, path)
        print(f"Model saved as {path}")


    def load(self, path: str = 'svr_model.joblib'):
        self.model = joblib.load(path)
        print(f"Model loaded'{path}'")


    def train(self, xtrain, ytrain):
        xtrain = self.scaler.fit_transform(xtrain)
        print("\n Training SVR...")
        self.model.fit(xtrain, ytrain)
        print(f"Número de vectores de soporte: {len(self.model.support_vectors_)}")

    def test(self, xtest):

        ypred = self.predict(xtest)

        return ypred
