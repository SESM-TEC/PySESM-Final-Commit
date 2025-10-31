import numpy as np
import joblib
from sklearn.svm import SVR as svr
import torch
import time


class SVR:
    """
    Una clase para encapsular un modelo Support Vector Regression (SVR) de scikit-learn.
    """
    def __init__(self, kernel, C, gamma, epsilon):
        self.model = svr(
            kernel = kernel, 
            C = C, 
            gamma = gamma, 
            epsilon = epsilon) 


  
    def predict(self, X: torch.tensor) -> np.ndarray:
        predictions = self.model.predict(X)
        return predictions


    def save(self, path: str = 'svr_model.joblib'):
        joblib.dump(self.model, path)
        print(f"Model saved as {path}")


    def load(self, path: str = 'svr_model.joblib'):
        self.model = joblib.load(path)


    def train(self, xtrain, ytrain):
        print("\n Training SVR...")

        start_time = time.time()
        self.model.fit(xtrain, ytrain)
        end_time = time.time()
        print(f"Número de vectores de soporte: {len(self.model.support_vectors_)}")
        
        return end_time - start_time

    def test(self, xtest):
        print("\n Testing SVR...")
        ypred = self.predict(xtest)

        return ypred
