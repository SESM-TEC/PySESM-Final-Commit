import torch
import numpy as np
import matplotlib.pyplot as plt
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
        weight_decay: TODO: Add description for weight_decay
    """

    def __init__(self, n_samples: int, psi: ApproximateSurrogateFunction, seed: float, model_epochs: int,
                 ista_epochs: int, ista_alpha: float, ista_lambd: float, mu_epochs: int, rho_epochs: int,
                 dictionary_alpha: float, weight_decay):
        """
        Function that initializes the class SESM us

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
            weight_decay: TODO: Add description for weight_decay
            dictionary_alpha (float): Learning rate for the dictionary layer.
        """
        super(SESM, self).__init__()

        self.n_samples = n_samples
        self.n_features = psi.n_features
        self.seed = seed

        self.ista_layer = ISTALayer(psi.n_functions, seed)
        self.dictionary_layer = DictLayer(n_samples, psi)

        # TODO: Replaced with function using the @property decorator
        # self.losses_ISTA = []
        # self.losses_Dictionary = []

        # TODO: Unused attribute?
        self.val_losses = []

        self.loss_stats = {
            "loss_mean": [],
            "loss_std": [],
            "loss_max": [],
            "loss_min": [],
        }

        # TODO: Renamed time -> elapsed_time because it could create naming issues with the library time.
        self.elapsed_time = 0
        self.ista_alpha = ista_alpha
        self.ista_epochs = ista_epochs
        self.model_epochs = model_epochs
        self.ista_lambd = ista_lambd
        self.mu_epochs = mu_epochs
        self.rho_epochs = rho_epochs
        self.dictionary_alpha = dictionary_alpha
        self.weight_decay = weight_decay

    @property
    def losses_ISTA(self):
        """Returns the losses for the ISTA layer and acts as an attribute due to the @property decorator."""
        return self.ista_layer.losses

    @property
    def losses_Dictionary(self):
        """Returns the losses for the Dictionary layer and acts as an attribute due to the @property decorator."""
        return self.dictionary_layer.losses

    # TODO: X_test and y_test should not be used to fit the model as they must only serve to validate effectiveness
    # TODO: Rename X_train, y_train -> X, y as the test tensors should not be passed
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Trains the model by learning a sparse vector and a dictionary that represent the original function.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """

        if self.dictionary_layer.dictionary is None:
            self.dictionary_layer.forward(X, rho_flag=False, mu_flag=False)

        for epoch in range(self.model_epochs):
            epoch_start_time = time.time()

            self.dictionary_layer.fit(
                X=X,
                y=y,
                epochs=self.mu_epochs,
                h=self.ista_layer.h,
                alpha=self.dictionary_alpha,
                rho_flag=False,
                mu_flag=True
            )

            self.dictionary_layer.fit(
                X=X,
                y=y,
                epochs=self.rho_epochs,
                h=self.ista_layer.h,
                alpha=self.dictionary_alpha,
                rho_flag=True,
                mu_flag=False
            )

            # TODO: Should it only use the y?
            self.ista_layer.fit(
                y=y,
                epochs=self.ista_epochs,
                dictionary=self.dictionary_layer.dictionary,
                alpha=self.ista_alpha,
                lambd=self.ista_lambd,
                weight_decay=self.weight_decay
            )

            epoch_end_time = time.time()

            self.elapsed_time += (epoch_end_time - epoch_start_time)

            # TODO: No need to have the values in two places at a time
            # self.losses_ISTA.append(self.ista_layer.losses[-1])
            # self.losses_Dictionary.append(self.dictionary_layer.losses[-1])

            self.loss_analysis(self.mu_epochs)
            self.loss_analysis(self.rho_epochs)

            # TODO: Change printing to logging
            print(
                f'Epoch {epoch + 1} Loss_ISTA: {self.losses_ISTA[-1]} Loss_Dictionary: {self.losses_Dictionary[-1]} \n')

            # TODO: Verify code relevance and delete it
            # with torch.no_grad():
            #   criterion = torch.nn.MSELoss()
            #   y_pred = self.predict(X_test)
            #   val_loss = criterion(y_pred, y_test)
            #   self.val_losses.append(val_loss.item())
            #   self.dictionary_layer.forward(X_train, rho_flag=False, mu_flag=False)

    # TODO: Renamed ista_layer -> custom_ista_layer to better understand why a whole layer is being passed as param
    def predict(self, X: torch.Tensor, custom_ista_layer: ISTALayer = None):
        """
        Predicts the value of a function using the learned sparse vector and dictionary.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            custom_ista_layer (ISTALayer): New ISTA Layer to be used in the prediction process

        Returns:
            torch.Tensor: The predicted values for each sample of the objective function.
        """

        # TODO: Maybe defining a setting method and setting the ISTA layer if needed before calling predict is a better approach
        # when external block ista layer is needed
        if custom_ista_layer is not None:
            self.ista_layer = custom_ista_layer
        with torch.no_grad():
            self.dictionary_layer.forward(X, rho_flag=False, mu_flag=False)

        dictionary = self.dictionary_layer.dictionary.double()
        h = self.ista_layer.h.double()
        return dictionary @ h

    def loss_analysis(self, dict_epochs):
        current_loss = self.dictionary_layer.losses[-dict_epochs:]
        self.loss_stats["loss_mean"].append(np.mean(current_loss))
        self.loss_stats["loss_std"].append(np.std(current_loss))
        self.loss_stats["loss_max"].append(np.max(current_loss))
        self.loss_stats["loss_min"].append(np.min(current_loss))

    # TODO: Maybe plotting function must be abstracted outside of the class to preserve the Single Responsibility principle
    def plot(self, n_samples, samples, savefig=False, filepath=None):
        n_plots = 4
        plot_elevs = [30, 60, 90, 30]
        plot_azims = [30, 60, 90, 120]

        # grids = torch.meshgrid(*samples)

        # xy_grid = torch.stack([grids[n].ravel() for n in range(self.n_features)], dim=1)

        # pdf_values = self.predict(xy_grid).reshape((n_samples,) * self.n_features).detach()

        pdf_values = self.predict(samples).detach()

        if (self.n_features == 2):
            # X = grids[0]
            # Y = grids[1]

            X = samples[:, 0]
            Y = samples[:, 1]
        else:
            # reduced_xy_grid = self.pca(xy_grid, 2)

            reduced_xy_grid = self.pca(samples, 2)

            # X = reduced_xy_grid[:, 0].reshape((n_samples,) * self.n_features)
            # Y = reduced_xy_grid[:, 1].reshape((n_samples,) * self.n_features)
            X = reduced_xy_grid[:, 0]
            Y = reduced_xy_grid[:, 1]

        fig = plt.figure(figsize=(8, 8))

        for i in range(n_plots):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            ax.scatter(X.numpy(), Y.numpy(), pdf_values.numpy(), c=pdf_values.numpy(), cmap='plasma')
            ax.view_init(elev=plot_elevs[i], azim=plot_azims[i])

        plt.tight_layout()

        if savefig:
            plt.savefig(filepath)

        plt.show()

    # TODO: Maybe plotting function must be abstracted outside of the class to preserve the Single Responsibility principle
    def plot_loss(self, ylim=0.1, savefig=False, filepath=None):
        plt.plot(self.losses)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.ylim(0, ylim)

        if savefig:
            plt.savefig(filepath)

        plt.show()

    # TODO: Cant figure out what does this function do
    def pca(self, X: torch.Tensor, n_components: int = 2):
        torch.manual_seed(1024)

        mean = torch.mean(X, dim=0)
        std = torch.std(X, dim=0)

        # Standardize the data
        X_std = (X - mean) / std

        # Step 2 & 3: Compute SVD
        U, S, V = torch.pca_lowrank(X_std)

        # Step 4: Select the number of principal components
        U = U[:, :n_components]

        # Step 5: Project data onto lower-dimensional space
        X_reduced = torch.mm(U, torch.diag(S[:n_components]))

        return X_reduced
