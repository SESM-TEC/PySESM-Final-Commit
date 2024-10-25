import logging

import torch
from sklearn.metrics import mean_squared_error

from pysesm.base_functions.sub_block_partition import predict_on_test_set
from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction
from pysesm.models.Blocks.UniformPartitionManager import UniformPartitionManager
from pysesm.models.SESM.SESM import SESM


class BSESM(SESM):

    def __init__(self,
                 n_samples: int,
                 n_features: int,
                 l_functions: int,
                 eig_range,
                 mu_range,
                 vector_range,
                 model_epochs: int,
                 ista_epochs: int,
                 rho_epochs: int,
                 mu_epochs: int,
                 ista_alpha: float,
                 ista_lambd: float,
                 dictionary_alpha: float,
                 weight_decay: float,
                 surrogate_function: ApproximateSurrogateFunction,
                 dfngroup,
                 iter,
                 T: list[int],
                 seed,
                 logger,
                 debug=True):
        """
        Initialize the SESMS model with a sequential approach.

        Args:
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of input features.
            l_functions (int): Number of latent functions used in the model.
            eig_range (tuple): Range for eigenvalues during dictionary creation.
            mu_range (tuple): Range for mu parameter during dictionary creation.
            vector_range (tuple): Range for vector parameter initialization.
            model_epochs (int): Number of epochs for the overall model training.
            ista_epochs (int): Number of epochs for the ISTA layer training.
            rho_epochs (int): Number of epochs for adjusting the rho parameter in the dictionary layer.
            mu_epochs (int): Number of epochs for adjusting the mu parameter in the dictionary layer.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            dictionary_alpha (float): Learning rate for the dictionary layer.
            weight_decay (float): Weight decay for regularization to prevent overfitting.
            surrogate_function (ApproximateSurrogateFunction): The surrogate function used to create the dictionary.
            dfngroup: Grouping information for the functions (implementation-specific).
            iter (int): Iteration count of the experiment.
            seed (int): Random seed for reproducibility.
            logger: Logger instance to capture runtime information.
            debug (bool): Flag to enable or disable debug mode. Default is True.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.l_functions = l_functions
        self.eig_range = eig_range
        self.mu_range = mu_range
        self.vector_range = vector_range
        self.dfngroup = dfngroup
        self.iter = iter
        self.T = torch.tensor(T)
        self.partition_manager = UniformPartitionManager(logger, self.T, n_functions=l_functions)
        self.logger = logger
        self.debug = debug

        super().__init__(
            n_samples=n_samples,
            psi=surrogate_function,
            seed=seed,
            model_epochs=model_epochs,
            ista_epochs=ista_epochs,
            ista_alpha=ista_alpha,
            ista_lambd=ista_lambd,
            mu_epochs=mu_epochs,
            rho_epochs=rho_epochs,
            dictionary_alpha=dictionary_alpha,
            weight_decay=weight_decay
        )

    def partial_fit(self, X, y, h=None):
        self.partition_manager.add_points(X, y)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        long_h = torch.cat(tuple([block.h for block in active_blocks])).resize(len(active_blocks) * self.l_functions, 1)
        long_h.t()

        max_points_in_block = len(max(active_blocks, key=lambda block: len(block.X)).X)
        empty_X = [0 for _ in range(self.n_features)]

        filled_active_blocks_X = []
        filled_active_blocks_y = []

        for block in active_blocks:
            filled_active_blocks_X.extend(
                [
                    block.normalized_X[index] if len(block.normalized_X) > index else empty_X
                    for index in range(max_points_in_block)
                ])

            filled_active_blocks_y.extend(
                [
                    block.y[index] if len(block.normalized_X) > index else 0
                    for index in range(max_points_in_block)
                ]
            )

        filled_active_blocks_X = torch.tensor(filled_active_blocks_X, dtype=torch.float32)
        filled_active_blocks_y = torch.tensor(filled_active_blocks_y, dtype=torch.float32)

        super().partial_fit(filled_active_blocks_X, filled_active_blocks_y, long_h)

    def forward(self, X: torch.Tensor, y: torch.Tensor, block_size: int = 1) -> None:
        super().forward(X, y, block_size)

    def predict(self, X_test):
        """
        Predict the output using the trained SESM model with sub-blocks.

        Args:
            X_test (torch.Tensor): Input features for prediction.
            list_sub_blocks (list): A list of sub-blocks used for the prediction process.

        Returns:
            torch.Tensor: Predicted values for the test set.
        """
        return predict_on_test_set(X_test, super(), self.T, self.partition_manager.blocks)

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Perform a forward pass for model evaluation with sub-blocks.

        Args:
            X (torch.Tensor): Input features for evaluation.
            y (torch.Tensor): True target values for evaluation.
            list_sub_blocks (list): A list of sub-blocks to be used for evaluation.

        Returns:
            tuple: A tuple containing:
                - Predicted values (torch.Tensor).
                - Training time (float): Time taken for the training process (in minutes).
                - Mean squared error (float): MSE between predicted and true target values.
        """
        y_pred = self.predict(X)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.clone().detach(), y)
        return y_pred, time, mse
