from pysesm.functions import SurrogateFunction
from pysesm.blocks import PartitionBlock, UniformPartitionManager
from pysesm.enums import SurrogateFunctionEnum, EvaluationFuncEnum
from pysesm.models.SESM.SESM import SESM


import logging
import torch
from typing import Union
from sklearn.metrics import mean_squared_error


class BSESM(SESM):
    """
    A Batch-based Surrogate Model for Sequential Estimation of Sparse Models (BSESM).
    
    This class extends the SESM (Sequential Estimation of Sparse Models) model by incorporating
    a batch processing approach, which organizes data into blocks for more efficient learning and
    prediction. The model uses a surrogate function to generate a dictionary and iteratively adjusts 
    the parameters during training.
    
    """
    def __init__(
        self,
        n_features: int,
        n_functions: int,
        model_epochs: int,
        ista_epochs: int,
        rho_epochs: int,
        mu_epochs: int,
        ista_alpha: float,
        ista_lambd: float,
        dictionary_alpha: float,
        psi: Union[SurrogateFunction, SurrogateFunctionEnum],
        dfngroup,
        seed: int,
        logger: logging.Logger,
        initial_bounds=None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Args:
            n_samples (int): Number of samples in the dataset.
            n_features (int): Number of input features.
            n_functions (int): Number of latent functions used in the model.
            model_epochs (int): Number of epochs for the overall model training.
            ista_epochs (int): Number of epochs for the ISTA (Iterative Shrinkage-Thresholding Algorithm) layer training.
            rho_epochs (int): Number of epochs for adjusting the rho parameter in the dictionary layer.
            mu_epochs (int): Number of epochs for adjusting the mu parameter in the dictionary layer.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            dictionary_alpha (float): Learning rate for the dictionary layer.
            surrogate_function (Union[SurrogateFunction, SurrogateFunctionEnum]): The surrogate function used to create the dictionary.
            dfngroup: Grouping information for the functions (specific to the implementation).
            T (list[int]): A list of the number of elements in each block for partitioning.
            seed (int): Random seed for reproducibility.
            logger (logging.Logger): Logger instance to capture runtime information.
            iter (int, optional): Iteration count of the experiment (default is 1).
            initial_bounds (optional): Initial bounds for the dictionary (default is None).
            debug (bool, optional): Flag to enable or disable debug mode (default is False).
        """
        self.dfngroup = dfngroup
        self.partition_manager = UniformPartitionManager(
            logger=logger,
            T=kwargs.get("T"),
            n_functions=n_functions,
            initial_bounds=initial_bounds,
        )

        super().__init__(
            n_functions=n_functions,
            n_features=n_features,
            psi=psi,
            model_epochs=model_epochs,
            ista_epochs=ista_epochs,
            ista_alpha=ista_alpha,
            ista_lambd=ista_lambd,
            mu_epochs=mu_epochs,
            rho_epochs=rho_epochs,
            dictionary_alpha=dictionary_alpha,
            seed=seed,
            logger=logger,
            debug=debug,
            evaluation_func=EvaluationFuncEnum.BMM_MULT,
            **kwargs,
        )

    def _arrange_batch_h(self, blocks: list[PartitionBlock]):
        """
        Arrange and stack the 'h' vectors from each block in a batch.

        Args:
            blocks (list[PartitionBlock]): List of partition blocks containing sparse vectors.

        Returns:
            torch.Tensor: A tensor with stacked 'h' vectors from all blocks.
        """
        return torch.stack([block.h for block in blocks]).unsqueeze(-1)

    def _fill_block_points(self, blocks: list[PartitionBlock]):
        """
        Fill missing points in the blocks with padding, ensuring consistent size across all blocks.

        Args:
            blocks (list[PartitionBlock]): List of partition blocks to process.

        Returns:
            tuple: A tuple containing:
                - filled_active_blocks_X (torch.Tensor): Features for all blocks, padded where necessary.
                - filled_active_blocks_y (torch.Tensor): Targets for all blocks, padded where necessary.
                - max_points_in_block (int): Maximum number of points in any block.
        """
        filled_active_blocks_X = []
        filled_active_blocks_y = []
        empty_X = [0 for _ in range(self.n_features)]
        max_points_in_block = len(
            max(blocks, key=lambda each_block: len(each_block.X)).X
        )

        for block in blocks:
            filled_active_blocks_X.append(
                torch.cat(
                    [
                        block.normalized_X,
                        torch.tensor(
                            [
                                empty_X
                                for i in range(
                                    max_points_in_block - block.normalized_X.shape[0]
                                )
                            ],
                            dtype=normalized_X.dtype,
                        ),
                    ]
                )
            )
            filled_active_blocks_y += block.target + [
                0 for _ in range(max_points_in_block - len(block.target))
            ]

        filled_active_blocks_X = torch.cat(filled_active_blocks_X)
        filled_active_blocks_y = torch.tensor(
            filled_active_blocks_y, dtype=torch.float32
        )

        return filled_active_blocks_X, filled_active_blocks_y, max_points_in_block


    def evaluation_func(self, dictionary: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Concrete implementation of the evaluation function for SSESM.
        This performs standard 2D matrix multiplication.
        """
        return torch.bmm(dictionary, h).squeeze(-1).flatten()
    
    def partial_fit(self, X, y, *_):
        """
        Perform a partial fit on the model using the provided data, updating the dictionary and sparse representation.

        Args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Target vector.
        """
        self.partition_manager.add_points(X, y)
        active_blocks = self.partition_manager.retrieve_active_blocks()

        long_h = self._arrange_batch_h(active_blocks)

        filled_active_blocks_X, filled_active_blocks_y, max_points_in_block = (
            self._fill_block_points(active_blocks)
        )

        self.ista_layer.h.data = long_h.data
        super().partial_fit(
            filled_active_blocks_X,
            filled_active_blocks_y,
            dictionary_shape=(len(active_blocks), max_points_in_block, self.n_functions)
        )

        trained_h = self.ista_layer.h.squeeze(-1)

        for index, block in enumerate(active_blocks):
            block.h = trained_h[index]
            if self.debug:
                # Logging the complete tensor with its content
                self.logger.debug(
                    f"Block at index {block.block_index}: Sparse vector 'h' learned: {trained_h[index]} (Tensor with shape {trained_h[index].shape})"
                )


    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        dictionary_shape: tuple
    ) -> None:
        super().forward(X, y, dictionary_shape)
        """
        Perform a forward pass for the model.

        Args:
            X (torch.Tensor): Feature matrix for the forward pass.
            y (torch.Tensor): Target vector for the forward pass.
            dictionary_shape (tuple, optional): Specifies the shape of the evaluated dictionary before
                                            computing the loss. If not provided, the default shape is used.
        """
        super().forward(X, y,dictionary_shape)

    def predict(self, X, y) -> [torch.Tensor, torch.Tensor]:  # type: ignore
        """
        Make predictions using the trained BSESM model.

        Args:
            X (torch.Tensor): Input features for prediction.
            y (torch.Tensor): Target values corresponding to the input features.

        Returns:
            torch.Tensor: Predicted values for the input data.
        """

        active_blocks = self.partition_manager.retrieve_test_active_blocks(X, y)
        long_h = self._arrange_batch_h(active_blocks)
        self.ista_layer.h.data = long_h.data

        filled_active_blocks_X, filled_active_blocks_y, max_points_in_block = (
            self._fill_block_points(active_blocks)
        )

        y_pred = super().predict(
            filled_active_blocks_X, dictionary_shape=(len(active_blocks), max_points_in_block, self.n_functions)
        )

        y_pred_per_block = [0 for _ in range(len(y))]
        i = 0

        for block in active_blocks:
            for pos in block.positions:
                y_pred_per_block[pos] = y_pred[i]
                i += 1
        y_pred_per_block = torch.tensor(y_pred_per_block, dtype=torch.float32)

        return y_pred_per_block

    def performance_stats(self, X: torch.Tensor, y: torch.Tensor):
        """
        Evaluate the model performance by making predictions and calculating the mean squared error.

        Args:
            X (torch.Tensor): Feature matrix for evaluation.
            y (torch.Tensor): True target values for evaluation.

        Returns:
            tuple: A tuple containing:
                - Predicted values (torch.Tensor).
                - Training time (float): Time taken for training, in minutes.
                - Mean squared error (float): MSE between predicted and true target values.
        """
        y_pred = self.predict(X, y)
        time = self.elapsed_time / 60
        mse = mean_squared_error(y_pred.clone().detach(), y)
        return y_pred, time, mse
