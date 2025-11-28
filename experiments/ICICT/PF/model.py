"""Polynomial Features + Lasso pipeline used in the ICICT experiments.

This module provides a thin wrapper around scikit-learn's
`PolynomialFeatures` + `Lasso` pipeline so the experiment code can call
`train` and `test` with consistent return types (training time and torch
predictions respectively).
"""

import time

import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


class PF:
    """Pipeline that fits polynomial features followed by a Lasso regressor.

    Args:
        order (int): Degree of the polynomial features.
        alpha (float): Regularization strength for Lasso.
        include_bias (bool): Whether to include the bias term in polynomial features.
        max_iter (int): Maximum iterations for the Lasso solver.
    """

    def __init__(
        self,
        order: int = 3,
        alpha: float = 0.01,
        include_bias: bool = False,
        max_iter: int = 10000,
    ):

        self.poly_lasso = Pipeline(
            [
                ("poly_features", PolynomialFeatures(degree=order, include_bias=include_bias)),
                ("lasso_reg", Lasso(alpha=alpha, max_iter=max_iter)),
            ]
        )

    def train(self, x_train, y_train):
        """Fit the polynomial+lasso pipeline.

        Returns the elapsed training time in seconds.
        """
        print("\n Training PF...")
        start_time = time.time()
        self.poly_lasso.fit(x_train, y_train)
        end_time = time.time()
        return end_time - start_time

    def test(self, x_test):
        """Predict with the fitted pipeline and return a torch tensor."""
        print("\n Testing PF...")
        y_pred = self.poly_lasso.predict(x_test)
        y_pred = torch.from_numpy(y_pred)
        return y_pred
