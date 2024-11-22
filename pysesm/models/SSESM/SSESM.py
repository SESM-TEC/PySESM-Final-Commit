from pysesm.functions import SurrogateFunction
from pysesm.models.SESM.SESM import SESM
from pysesm.blocks import UniformPartitionManager


import numpy as np
import torch
from sklearn.metrics import mean_squared_error


class SSESM(SESM):
    """
    A PyTorch module extending the SESM architecture to implement a surrogate model
    using a sequential approach. This class is designed for function approximation
    and surrogate modeling tasks by utilizing sub-block partitioning.
    """

    def __init__(self,
                 n_samples: int,
                 n_features: int,
                 n_functions: int,
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
                 surrogate_function: SurrogateFunction,
                 permutation_times: int,
                 dfngroup,
                 iter,
                 seed,
                 logger,
                 T: list[int],
                 initial_bounds=None,
                 debug=True):
        """
        Initialize the SSESM model with a sequential approach.

        Args:
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of input features.
            n_functions (int): Number of latent functions used in the model.
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
            permutation_times (int): Number of times to permute the dataset for training.
            dfngroup: Grouping information for the functions (implementation-specific).
            iter (int): Iteration count of the experiment.
            seed (int): Random seed for reproducibility.
            logger: Logger instance to capture runtime information.
            T (int): The scaling factor for normalization.
            debug (bool): Flag to enable or disable debug mode. Default is True.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_functions = n_functions
        self.eig_range = eig_range
        self.mu_range = mu_range
        self.vector_range = vector_range
        self.permutation_times = permutation_times
        self.dfngroup = dfngroup
        self.iter = iter
        self.T = torch.tensor(T)
        self.partition_manager = UniformPartitionManager(logger, self.T, n_functions=n_functions,
                                                         initial_bounds=initial_bounds)
        self.logger = logger
        self.debug = debug
        self.calculate_y_pred = lambda dictionary, h: torch.matmul(dictionary, h)

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
            weight_decay=weight_decay,
            calculate_y_pred=self.calculate_y_pred
        )

    def partial_fit(self, X, y):
        self.partition_manager.add_points(X, y)
        self.partition_manager.init_ista_per_block(self.n_functions,
                                                   self.seed,
                                                   self.ista_alpha,
                                                   self.ista_alpha,
                                                   self.weight_decay,
                                                   self.calculate_y_pred)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        for _ in range(self.permutation_times):
            selected_indexes = np.random.permutation(len(active_blocks))
            permuted_list_sub_blocks = [active_blocks[i] for i in selected_indexes]
            for block in permuted_list_sub_blocks:
                self.ista_layer = block.ista_layer
                X_torch = torch.tensor(block.normalized_X, dtype=torch.float32)
                y_torch = torch.tensor(block.target, dtype=torch.float32)
                super().partial_fit(X_torch, y_torch)

    def predict(self, X, y):
        """
        Predict the output using the trained SESM model with sub-blocks.

        Args:
            X_test (torch.Tensor): Input features for prediction.
            -
        Returns:
            torch.Tensor: Predicted values for the test set.
        """
        active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)

        y_pred_per_block = [0 for _ in range(len(y))]
        for block in active_blocks:
            self.ista_layer = block.ista_layer
            X_torch = torch.tensor(block.normalized_X, dtype=torch.float32)
            block_pred = super().predict(X_torch)
            for i, pos in enumerate(block.positions):
                y_pred_per_block[pos] = block_pred[i]

        return torch.tensor(y_pred_per_block, dtype=torch.float32)

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
        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.clone().detach(), y)
        return y_pred, time, mse
