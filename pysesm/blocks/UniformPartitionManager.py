from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer

import logging
from copy import deepcopy
from typing import Union, Callable
import numpy as np
import torch


DEFAULT_BLOCKS_PER_DIM = 4

def squeeze_factor(y: np.ndarray):
    """
    Calculates a squeezing factor for a given set of values.

    Args:
        y (np.ndarray): An array containing numeric values.

    Returns:
        float: The squeezing factor. If the maximum value in y exceeds 1, returns 1 / max(y).
        Otherwise, returns 1.0.
    """
    e_f = 0.0
    max_y = max(y)
    if max_y > 1:
        e_f = 1 / max_y
    else:
        e_f = 1.0
    return e_f


class UniformPartitionManager(BlockManager):
    """
    A class to manage a uniform partitioning of the input space into blocks.

    The UniformPartitionManager divides the space into uniformly sized blocks, assigns points to these blocks,
    and configures or adjusts local models within each block.

    Args:
        logger (logging.Logger): Logger instance for recording messages and warnings.
        T (torch.Tensor): A tensor defining the number of blocks per dimension.
        n_functions (int): Number of functions or features of interest in each block.
        initial_bounds (np.ndarray, optional): Initial bounds for the partitioning, shaped as (2, n_features).
            - The first row contains the lower bounds.
            - The second row contains the upper bounds.
            If not provided, it is automatically calculated from the input data.
        threshold (float, optional): Threshold for determining block activity (default is 0).
    """

    def __init__(self, logger: logging.Logger, T: torch.Tensor, n_functions, initial_bounds: np.ndarray = None, threshold: float = 0):
        """
        Initializes the UniformPartitionManager with the provided parameters.

        Args:
            logger (logging.Logger): Logger instance for recording messages and warnings.
            T (torch.Tensor): A tensor defining the number of blocks per dimension.
            n_functions (int): Number of functions or features of interest in each block.
            initial_bounds (np.ndarray, optional): Initial bounds for the partitioning.
                If not provided, bounds are automatically derived from the data.
            threshold (float, optional): Threshold for determining block activity.
        """
        super().__init__()

        self.T = T
        self.n_functions = n_functions
        self.initial_bounds = initial_bounds
        self.threshold = threshold
        self.logger = logger
        self.blocks = None
        self.block_size = None
        self.X = None
        self.y = None
        self._vectorized_normalization = np.vectorize(lambda x: x.normalize())

    def _find_block(self, X: torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the block corresponding to a given point.

        Args:
            X (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """
        # TODO: Find efficient way to find block (currently being worked on by Joshua)
        for index in np.ndindex(self.blocks.shape):
            block: PartitionBlock = self.blocks[index]
            if block.is_point_in_block(X):
                return block

        logging.warning("Could not find a block for point {}", X)
        return None

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        """
        Updates the arrangement of blocks based on the input data.

        This method initializes or adjusts the block arrangement and sizes, ensuring coverage of the input space.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        # If no T is given, create a T with a default size
        if self.T is None:
            self.T = torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())])

        # When no points and no blocks have been created
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.vstack(
                    [torch.min(X, dim=0).values,
                     torch.max(X, dim=0).values])  # Calculates the range covered by the X vector
                logging.warning('[UniformPartitionManager] No initial bounds provided, using calculated one {}'.format(
                    self.initial_bounds
                ))

            # Space to be partitioned
            delta = self.initial_bounds[1] - self.initial_bounds[0]
            self.block_size = torch.div(delta, self.T)

            self.blocks = np.empty(self.T.numpy(), dtype=PartitionBlock)

            for index in np.ndindex(self.blocks.shape):
                self.blocks[index] = PartitionBlock(self.initial_bounds[0], index, self.block_size)
        else:
            new_max_x = torch.max(X)
            new_min_x = torch.min(X)

    def _configure_blocks(self, init_h: bool = True):
        """
        Configures each block with its expected squeeze factor and initializes sparse vectors if required.

        Args:
            init_h (bool, optional): Whether to initialize the sparse vector `h` for each block (default is True).
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if len(block.y) != 0:
                block.amplitude = squeeze_factor(block.y)
                if init_h:
                    block.h = torch.nn.Parameter(torch.rand(self.n_functions), requires_grad=True)
                    # block.h.data /= block.h.data.sum()
                    self.logger.debug(
                        f"Created random vector for block at index {index}, created sparse vector h: {block.h}")

                block.target = [value * block.amplitude for value in block.y]

    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Maps input points to their respective sub-blocks.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target data corresponding to the input points.
        """
        for i in range(X.shape[0]):
            selected_block = self._find_block(X[i])
            if selected_block is not None:
                selected_block.new_point(X[i], y[i], i)

    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """
        self._update_block_arrangement(X)

        self._map_points(X, y)
        self._vectorized_normalization(self.blocks)
        self._configure_blocks()

    def init_ista_per_block(self, n_functions: int, seed: int, ista_alpha: float, ista_lambd: float, weight_decay: float,
                            evaluation_func:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Initializes an ISTA layer for each block.

        Args:
            n_functions (int): Number of functions or features for the ISTA layer.
            seed (int): Random seed for initialization.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            weight_decay (float): Weight decay penalty for the ISTA layer.
            evaluation_func (Callable): Function for evaluating the ISTA layer.
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            block.ista_layer = ISTALayer(
                n_functions=n_functions,
                random_seed=seed,
                alpha=ista_alpha,
                lambd=ista_lambd,
                weight_decay=weight_decay,
                evaluation_func=evaluation_func
            )

    def retrieve_active_blocks(self):
        """
        Retrieves all active blocks in the partition.

        Returns:
            List[PartitionBlock]: A list of active blocks.
        """
        return [self.blocks[index] for index in np.ndindex(self.blocks.shape) if self.blocks[index].is_active]

    def retrieve_test_active_blocks(self, X, y):
        """
        Retrieves active blocks for testing purposes.

        Args:
            X (torch.Tensor): Test input data of shape (n_samples, n_features).
            y (torch.Tensor): Test target data of shape (n_samples,).

        Returns:
            List[PartitionBlock]: A list of active blocks corresponding to the test data.
        """
        # Copy blocks without X and y
        test_blocks = deepcopy(self.blocks)

        # Save temporarily current blocks
        temp_current_blocks = self.blocks
        self.blocks = test_blocks

        # Map and normalize points into test blocks
        self._map_points(X, y)
        self._vectorized_normalization(self.blocks)
        self._configure_blocks(init_h=False)

        # Retrieved mapped test blocks and return to usual blocks
        test_active_blocks = self.retrieve_active_blocks()
        self.blocks = temp_current_blocks

        return test_active_blocks
