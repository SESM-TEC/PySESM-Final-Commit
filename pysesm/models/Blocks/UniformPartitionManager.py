import logging
from typing import Union

import numpy as np
import torch

from pysesm.models.Blocks.BlockManager import BlockManager
from pysesm.models.Blocks.PartitionBlock import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer
DEFAULT_BLOCKS_PER_DIM = 4


def squeeze_factor(y: np.ndarray):
    """
    Calculates a squeezing factor for a given set of values.

    Args:
    - Y (iterable): An iterable containing numeric values.

    Returns:
    - float: The squeezing factor. If the maximum value in Y is greater than 1, returns the reciprocal of the maximum value. Otherwise, returns 1.0.
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

    """

    def __init__(self, logger, T: torch.Tensor, n_functions, initial_bounds: np.ndarray = None, threshold: float = 0):
        """

        Args:
            T(int):
            initial_bounds:
            threshold:
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
        # TODO: Find efficient way to find block
        for index in np.ndindex(self.blocks.shape):
            if self.blocks[index].is_point_in_block(X):
                return self.blocks[index]
        return None

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        # If no T is given, create a T with a default size
        if self.T is None:
            self.T = torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())])

        # When no points and no blocks have been created
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.vstack(
                    [torch.min(X, dim=0).values, torch.max(X, dim=0).values])  # Calculates the range covered by the X vector
                logging.warning('[UniformPartitionManager] No initial bounds provided using calculated one {}'.format(
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

    def _configure_blocks(self):
        """
        Configures the blocks with their expected squeeze factor
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if len(block.y) != 0:
                block.amplitude = squeeze_factor(block.y)
                block.h = torch.nn.Parameter(torch.rand(self.n_functions), requires_grad=True)
                block.h.data /= block.h.data.sum()
                block.target = [value * block.amplitude for value in block.y]

    def _data_mapping(self, X: torch.Tensor):
        """
        Maps input data to a normalized range and separates it into integer and fractional parts.

        Args:
        - X (torch.Tensor): A tensor containing the input data.

        Returns:
        - t (numpy.ndarray): An array of integer parts of the normalized data.
        - x_n (torch.Tensor): A tensor of fractional parts of the normalized data.
        """
        delta = torch.max(X) - torch.min(X)
        eps = delta * 1e-6
        norm_x = ((X - torch.min(X)) / (delta + eps)) * self.T
        t = norm_x.numpy().astype(int)
        x_n = norm_x - t
        return t, x_n

    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Locate points in their respective sub-blocks.

        Args:
        - x_n (np.ndarray): The normalized points between 0 and 1.
        - t (np.ndarray): The integer part of the normalized points.
        - y (np.ndarray) : The output values associated with the samples
        """
        for i in range(X.shape[0]):
            selected_block = self._find_block(X[i])
            selected_block.new_point(X[i], y[i])

    def add_points(self, X: torch.Tensor, y: np.ndarray):
        self._update_block_arrangement(X)

        self._map_points(X, y)
        self._vectorized_normalization(self.blocks)
        self._configure_blocks()
    
    def init_ista_per_block(self, n_features: int, seed:int, ista_alpha: float,ista_lambd: float,weight_decay: float, h = None):
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            block.ista_layer = ISTALayer(
                n_functions=n_features,
                random_seed=seed,
                alpha=ista_alpha,
                lambd=ista_lambd,
                weight_decay=weight_decay,
                h=h
                )
    
    def retrieve_active_blocks(self):
        return [self.blocks[index] for index in np.ndindex(self.blocks.shape) if self.blocks[index].is_active]

    def retrieve_test_blocks(self, X, y):
        #Copy blocks
        #Clone blocks
        #Delete x and y from blocks

        self._map_points(X, y)
        self._vectorized_normalization(self.blocks)
        self._configure_blocks()

