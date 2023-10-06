import torch
import numpy as np
from tqdm import tqdm


class Wrapper:
    """
    Initialize the wrapper with a given DictLayer model.
    Args:
        model (DictLayer): The DictLayer model we will train.
    """
    def __init__(self, model, optimizer = torch.optim.SGD):
        self.model = model
        self.losses = []
        self.optimizer = optimizer

    def fit(self, X, y, epochs, h, alpha, log_losses=True):
        """
        Let's train the model.

        Args:
            X (Tensor): Input data of shape (n_samples, n_features).
            y (Tensor): Target data of shape (n_samples,).
            epochs (int): Number of training epochs.
            h (Tensor): Sparse vector for selecting basis functions.
            alpha (float): Learning rate for optimization.
            log_losses (bool): Whether to log losses during training (default True).
        """
        optimizer = self.optimizer(self.model.parameters(), lr=alpha)
        criterion = torch.nn.MSELoss()

        for i in tqdm(range(epochs), desc='Training model...'):
            self.model.forward(X)
            y_pred = self.model.dictionary @ h
            loss = criterion(y_pred, y) # + lambda*1/(la suma de las trzas de las inversas matrices de covarianza)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if log_losses:
                self.losses.append(loss.item())

    def predict(self, X, h):
        """
        Args:
            X (Tensor): Input data of shape (n_samples, n_features).
            h (Tensor): Sparse vector for selecting basis functions.

        Returns:
            y_pred (Tensor): Predicted values.
        """
        self.model.forward(X)
        h = h.float()
        self.model.dictionary = self.model.dictionary.float()
        y_pred = self.model.dictionary @ h
        return y_pred
