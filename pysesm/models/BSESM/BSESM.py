import numpy as np
import torch

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
                 T: int,
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
        self.T = T
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

    def partial_fit(self, X, y):
        self.partition_manager.add_points(X, y)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        long_h = torch.column_stack(tuple([block.h for block in active_blocks]))
        self.ista_layer.h.data = long_h

        max_points_in_block = max(active_blocks, key=lambda block: len(block.X))

        standardized_points = np.empty((max_points_in_block, self.l_functions * len(active_blocks)))
        for index, block in enumerate(active_blocks):
            evaluated_X = self.psi(block.normalized_X.mT, self.theta_parameter_vector, True, True)
            result = torch.zeros(max_points_in_block)
            result[:len(evaluated_X)] = evaluated_X
            standardized_points[index] = result

        super().partial_fit(torch.tensor(standardized_points).mT, y)
