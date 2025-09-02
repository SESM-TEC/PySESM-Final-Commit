import chaospy as cp
import numpy as np
import torch

class PCE():
    def __init__(self, order: int = 3):
        self.order : int = order

    def train(self, X_torch, y_torch):

        X_train, y_train = self.preprocess_torch(X_torch, y_torch)

        lo, hi = np.min(X_train), np.max(X_train)
        dists = cp.Iid(cp.Uniform(lo, hi), X_train.shape[1])
        expansion = cp.orth_ttr(self.order, dists)  # total-degree truncation
        coeffs = cp.fit_regression(expansion, X_train.T, y_train)
        self.pce_model = cp.sum(coeffs * expansion)
        
    def test(self, X_torch):    
        X_test = self.preprocess_torch(X_torch)
        y_pred = self.pce_model(*X_test.T)
        return y_pred

    def preprocess_torch(self, X_torch, y_torch=None):
        """
        Convert PyTorch tensors to NumPy arrays suitable for Chaospy.
        """
        # Convert to numpy and float64
        X_np = X_torch.detach().cpu().numpy().astype(np.float64)
        # Ensure shapes
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        if y_torch is not None:
            y_np = y_torch.detach().cpu().numpy().astype(np.float64)
            if y_np.ndim > 1:
                y_np = y_np.ravel()
            return X_np, y_np

        return X_np
