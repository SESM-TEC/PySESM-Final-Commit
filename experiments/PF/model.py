from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

class PF():
    def __init__(self, order: int=3, alpha: float=0.01, include_bias: bool = False, max_iter:int=10000):
        self.poly_lasso = Pipeline([
            ("poly_features", PolynomialFeatures(degree=order, include_bias=include_bias)),
            ("lasso_reg", Lasso(alpha=alpha, max_iter=max_iter))
        ])

    def train(self, X_train, y_train):
        self.poly_lasso.fit(X_train, y_train)

    def test(self, X_test):
        y_pred = self.poly_lasso.predict(X_test)
        return y_pred