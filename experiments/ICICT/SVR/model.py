"""Support Vector Regression wrapper used in the ICICT experiments.

This module provides a thin wrapper around scikit-learn's `SVR` to keep a
consistent interface with the rest of the experiment code (methods `train`
and `test`).
"""

import time

import torch
from sklearn.svm import SVR as svr

# Allow parameter name `C` (capital) to match common SVR notation/configs
# pylint: disable=invalid-name


class SVR:
    """Encapsula un `sklearn.svm.SVR` con métodos `train` y `test`.

    Args:
        kernel (str): Kernel type for the SVR.
        C (float): Regularization parameter (kept uppercase to match configs).
        gamma (str|float): Kernel coefficient.
        epsilon (float): Epsilon parameter in the epsilon-SVR model.
    """

    def __init__(self, kernel, C, gamma, epsilon):
        self.model = svr(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    def train(self, xtrain, ytrain):
        """Fit the SVR model and return elapsed training time (seconds)."""
        print("\n Training SVR...")
        start_time = time.time()
        self.model.fit(xtrain, ytrain)
        end_time = time.time()
        print(f"Número de vectores de soporte: {len(self.model.support_vectors_)}")
        return end_time - start_time

    def test(self, xtest):
        """Predict using the trained model and return a torch tensor."""
        print("\n Testing SVR...")
        ypred = self.model.predict(xtest)
        ypred = torch.from_numpy(ypred)
        return ypred
