import torch
from sklearn.metrics import mean_squared_error

from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction
from pysesm.models.Blocks.PartitionBlock import PartitionBlock
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
                 initial_bounds=None,
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
        self.partition_manager = UniformPartitionManager(logger, self.T, n_functions=l_functions,
                                                         initial_bounds=initial_bounds)
        self.logger = logger
        self.debug = debug
        self.calculate_y_pred = lambda dictionary, h: torch.bmm(dictionary, h).squeeze(-1).flatten()

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

    def _arrange_batch_h(self, blocks: list[PartitionBlock]):
        return torch.stack([block.h for block in blocks]).unsqueeze(-1)

    def _fill_block_points(self, blocks: list[PartitionBlock]):
        filled_active_blocks_X = []
        filled_active_blocks_y = []
        empty_X = [0 for _ in range(self.n_features)]
        max_points_in_block = len(max(blocks, key=lambda each_block: len(each_block.X)).X)

        for block in blocks:
            filled_active_blocks_X.append(
                torch.cat([
                    block.normalized_X,
                    torch.tensor(
                        [empty_X for i
                         in range(max_points_in_block - block.normalized_X.shape[0])]
                        , dtype=torch.float32)
                ])
            )
            filled_active_blocks_y += block.target + [0 for _ in range(max_points_in_block - len(block.target))]

        filled_active_blocks_X = torch.cat(filled_active_blocks_X)
        filled_active_blocks_y = torch.tensor(filled_active_blocks_y, dtype=torch.float32)

        return filled_active_blocks_X, filled_active_blocks_y, max_points_in_block

    def partial_fit(self, X, y):
        self.partition_manager.add_points(X, y)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        long_h = self._arrange_batch_h(active_blocks)

        filled_active_blocks_X, filled_active_blocks_y, max_points_in_block = self._fill_block_points(active_blocks)

        self.ista_layer.h.data = long_h.data
        super().partial_fit(
            filled_active_blocks_X,
            filled_active_blocks_y,
            max_points_in_block,
            len(active_blocks)
        )

        trained_h = self.ista_layer.h.squeeze(-1)

        for index, block in enumerate(active_blocks):
            block.h = trained_h[index]

    def forward(self, X: torch.Tensor, y: torch.Tensor, max_points_in_block: int = 0,
                active_blocks_count: int = 0) -> None:
        super().forward(X, y, max_points_in_block, active_blocks_count)

    def predict(self, X, y) -> [torch.Tensor, torch.Tensor]:  # type: ignore
        """
        Predict the output using the trained SESM model with sub-blocks.

        Args:
            X_test (torch.Tensor): Input features for prediction.
            list_sub_blocks (list): A list of sub-blocks used for the prediction process.

        Returns:
            torch.Tensor: Predicted values for the test set.
        """

        active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)
        long_h = self._arrange_batch_h(active_blocks)
        self.ista_layer.h.data = long_h.data

        filled_active_blocks_X, filled_active_blocks_y, max_points_in_block = self._fill_block_points(active_blocks)

        y_pred = super().predict(filled_active_blocks_X, max_points_in_block, len(active_blocks))
        print("y_pred.shape", y_pred.shape)
        y_pred_per_block = [0 for _ in range(len(y))]
        i = 0
        for block in active_blocks:
            for pos in block.positions:
                y_pred_per_block[pos] = y_pred[i]
                i += 1
        y_pred_per_block = torch.tensor(y_pred_per_block, dtype=torch.float32)

        for i in range(len(y)):
            print(f"Value at i {i}, Target is {y[i]}, Predicted value is {y_pred_per_block[i]}")

        return y_pred_per_block

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
        print("y_pred", y_pred)
        print("target", y)
        print("y_pred shape", y_pred.shape)
        time = self.elapsed_time / 60
        print(time)
        mse = mean_squared_error(y_pred.clone().detach(), y)
        return y_pred, time, mse
