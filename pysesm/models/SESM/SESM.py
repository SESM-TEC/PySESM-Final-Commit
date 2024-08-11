import logging

import torch
import numpy as np
import time

from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction
from pysesm.models.DictLayer import DictLayer
from pysesm.models.ISTALayer import ISTALayer


class SESM(torch.nn.Module):
    """
    A custom PyTorch module for implementing a surrogate model that uses the SESM architecture.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Attributes:
        n_samples (int): The number of samples taken from the original function.
        n_features (int): The number of input features or dimensions.
        seed (float): Random seed to be used in the model.
        ista_layer (ISTALayer): Module layer that holds and adjusts the sparse vectors.
        dictionary_layer (DictLayer): Module layer that holds and adjusts the dictionary parameters (ρ, μ).
        loss_stats (dict): A dictionary containing the different types of loss history.
            - 'loss_mean' (list): A history of the mean loss values.
            - 'loss_std' (list): A history of the standard loss values.
            - 'loss_max' (list): A history of the max loss values.
            - 'loss_min' (list): A history of the min loss values.
        elapsed_time (float): Time consumed by the SESM fit process.
        model_epochs (int): Number of training epochs for the model.
        ista_epochs (int): Number of training epochs for the ISTA layer.
        ista_alpha (float): Learning rate for the ISTA layer.
        ista_lambd (float): Regularization parameter for the ISTA layer.
        dictionary_alpha (float): Learning rate for the dictionary layer.
        mu_epochs (int): Number of training epochs for the dictionary layer adjusting only the mu parameter.
        rho_epochs (int): Number of training epochs for the dictionary layer adjusting only the rho parameter.
        weight_decay (float): Penalty rate applied to the loss function to prevent overfitting.
    """

    def __init__(self, n_samples: int, psi: ApproximateSurrogateFunction, seed: float, model_epochs: int,
                 ista_epochs: int, ista_alpha: float, ista_lambd: float, mu_epochs: int, rho_epochs: int,
                 dictionary_alpha: float, weight_decay):
        """
        Method that initializes the SESM.

        Args:
            n_samples (int): The number of samples taken from the original function.
            psi (ApproximateSurrogateFunction): The function used for generating the model's dictionary.
            seed (float): Random seed to be used in the model.
            model_epochs (int): Number of training epochs for the model.
            ista_epochs (int): Number of training epochs for the ISTA layer.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            mu_epochs (int): Number of training epochs for the dictionary layer adjusting only the mu parameter.
            rho_epochs (int): Number of training epochs for the dictionary layer adjusting only the rho parameter.
            dictionary_alpha (float): Learning rate for the dictionary layer.
            weight_decay (float): Penalty rate applied to the loss function to prevent overfitting.
        """
        super(SESM, self).__init__()

        self.n_samples = n_samples
        self.n_features = psi.n_features
        self.seed = seed
        self.ista_alpha = ista_alpha
        self.ista_epochs = ista_epochs
        self.model_epochs = model_epochs
        self.ista_lambd = ista_lambd
        self.mu_epochs = mu_epochs
        self.rho_epochs = rho_epochs
        self.dictionary_alpha = dictionary_alpha
        self.weight_decay = weight_decay
        self.elapsed_time = 0
        self.loss_stats = {
            "loss_mean": [],
            "loss_std": [],
            "loss_max": [],
            "loss_min": [],
        }

        # Instantiate model layers
        self.ista_layer = ISTALayer(self.n_features, self.seed)

        self.dictionary_layer = DictLayer(
            n_samples=self.n_samples,
            psi=psi,
            alpha=self.dictionary_alpha
        )

    @property
    def ista_layer_losses(self):
        """Returns the losses for the ISTA layer and acts as an attribute due to the @property decorator."""
        return self.ista_layer.losses

    @property
    def dictionary_layer_losses(self):
        """Returns the losses for the Dictionary layer and acts as an attribute due to the @property decorator."""
        return self.dictionary_layer.losses

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Trains the model by learning a sparse vector and a dictionary that represent the original function, and it
        redefines the weight each time its called.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """

        self.dictionary_layer.initialize_layer(X)
        self.ista_layer.initialize_h_vector()

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.forward(X, y)

            epoch_end_time = time.time()
            self.elapsed_time += (epoch_end_time - epoch_start_time)

            logging.info(f'Fit epoch {epoch + 1} Loss_ISTA: {self.ista_layer_losses[-1]} Loss_Dictionary: {self.dictionary_layer_losses[-1]} \n')

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Trains the model by learning a sparse vector and a dictionary that represent the original function without redefining the weight each time.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """

        if self.dictionary_layer.dictionary is None:
            self.dictionary_layer.initialize_layer(X)

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.forward(X, y)

            epoch_end_time = time.time()
            self.elapsed_time += (epoch_end_time - epoch_start_time)

            logging.info(
                f'Partial fit epoch {epoch + 1} Loss_ISTA: {self.ista_layer_losses[-1]} Loss_Dictionary: {self.dictionary_layer_losses[-1]} \n')

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Computes the forward pass for the model.

        Args:
           X (torch.Tensor): Input data of shape (n_samples, n_features).
           y (torch.Tensor): Target data of shape (n_samples,).
        """

        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.mu_epochs,
            h=self.ista_layer.h,
            mu_flag=True
        )

        self.loss_analysis(self.mu_epochs)

        self.dictionary_layer.partial_fit(
            X=X,
            y=y,
            epochs=self.rho_epochs,
            h=self.ista_layer.h,
            rho_flag=True,
        )

        self.loss_analysis(self.rho_epochs)

        self.ista_layer.fit(
            y=y,
            epochs=self.ista_epochs,
            dictionary=self.dictionary_layer.dictionary,
            alpha=self.ista_alpha,
            lambd=self.ista_lambd,
            weight_decay=self.weight_decay
        )

    def predict(self, X: torch.Tensor, custom_ista_layer: ISTALayer = None) -> torch.Tensor:
        """
        Predicts the value of a function using the learned sparse vector and dictionary.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            custom_ista_layer (ISTALayer): New ISTA Layer to be used in the prediction process

        Returns:
            torch.Tensor: The predicted values for each sample of the objective function.
        """

        if custom_ista_layer is not None:
            self.ista_layer = custom_ista_layer

        with torch.no_grad():
            self.dictionary_layer.dictionary = self.dictionary_layer.forward(X)

        dictionary = self.dictionary_layer.dictionary.double()
        h = self.ista_layer.h.double()

        return dictionary @ h

    def loss_analysis(self, dict_epochs: int) -> None:
        current_loss = self.dictionary_layer.losses[-dict_epochs:]
        self.loss_stats["loss_mean"].append(np.mean(current_loss))
        self.loss_stats["loss_std"].append(np.std(current_loss))
        self.loss_stats["loss_max"].append(np.max(current_loss))
        self.loss_stats["loss_min"].append(np.min(current_loss))
