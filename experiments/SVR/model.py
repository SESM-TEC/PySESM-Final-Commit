import numpy as np
import joblib
from sklearn.svm import SVR as svr
import torch

class SVR:
    """
    Una clase para encapsular un modelo Support Vector Regression (SVR) de scikit-learn.
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1):
        self.model = svr(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    def fit(self, X: torch.tensor, y: torch.tensor):
        print("Training SVR...")
        X = X.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        self.model.fit(X, y)
        print(f"Número de vectores de soporte: {len(self.model.support_vectors_)}")
        
    def predict(self, X: torch.tensor) -> np.ndarray:
        predictions = self.model.predict(X)
        return predictions

    def save(self, path: str = 'svr_model.joblib'):
        joblib.dump(self.model, path)
        print(f"Modelo guardado exitosamente en '{path}'")

    def load(self, path: str = 'svr_model.joblib'):
        self.model = joblib.load(path)
        print(f"Modelo cargado exitosamente desde '{path}'")
