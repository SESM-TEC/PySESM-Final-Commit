from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction
from pysesm.models.Blocks.UniformPartitionManager import UniformPartitionManager
from pysesm.models.SESM.SESM import SESM
import torch
import numpy as np

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
        self.partition_manager = UniformPartitionManager(self.T)
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
        for block in self.partition_manager.blocks:
            y = torch.tensor(block.target, dtype=torch.float32)
            X = torch.tensor(np.array(block.get_X()), dtype=torch.float32)
            # self.ista_layer = block.ista_layer TODO: Needed?
            super().partial_fit(X, y)