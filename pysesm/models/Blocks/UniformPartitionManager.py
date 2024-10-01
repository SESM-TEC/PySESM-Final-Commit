import numpy as np
import torch

from pysesm.models.Blocks.BlockManager import BlockManager
from pysesm.models.Blocks.PartitionBlock import PartitionBlock

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

    def __init__(self, logger, T: torch.Tensor(int), initial_bounds: np.ndarray = None, threshold: float = 0):
        """

        Args:
            T(int):
            initial_bounds:
            threshold:
        """
        super().__init__()

        self.T = T
        self.initial_bounds = initial_bounds
        self.threshold = threshold
        self.logger = logger
        self.blocks = None
        self.X = None
        self.y = None

    def _find_block(self, X: torch.Tensor):
        # TODO: Find efficient way to find block
        for block in self.blocks:
            if block.is_point_in_block(X):
                return block
        return -1

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        def calculate_next_index(block_index: list[int]) -> list[int] | int:
            # If is the last index return -1
            if block_index == (self.T - 1):
                return -1
            # If first dim haven't reach max size increase it
            if block_index[0] < (self.T[0] - 1):
                block_index[0] += 1
                return block_index
            else:
                # If so, update all dims that have reach max size
                block_index[0] = 0
                for block_dim in range(X.dim()[1:]):
                    if block_index[block_dim] == self.T[block_dim]:
                        block_index[block_dim] = 0
                    else:
                        block_index[block_dim] += 1
                        break
                return block_index

        # If no T is given, create a T with a default size
        if self.T is None:
            self.T = torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())])

        # When no points and no blocks have been created
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.column_stack(
                    [torch.min(X), torch.max(X)])  # Calculates the range covered by the X vector

            # Space to be partitioned
            delta = torch.sum(self.initial_bounds * torch.tensor([-1, 1]))
            block_size = delta / self.T
            block_count = torch.prod(self.T)

            self.blocks = np.empty(self.T, dtype=PartitionBlock)

            for index in np.ndindex(self.blocks.shape):
                self.blocks[index] = PartitionBlock(index, block_size)
        else:
            new_max_x = torch.max(X)
            new_min_x = torch.min(X)

    def _configure_blocks(self):
        """
        Configures the blocks with their expected squeeze factor
        """
        for block in self.blocks:
            if len(block.output_values) != 0:
                block.amplitude = squeeze_factor(block.y)
                # block.ista_layer = ISTALayer(l_functions, SEED) TODO: Is this needed? Yes it is
                block.target = [value * block.amplitude for value in block.output_values]

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

    def _normalize_blocks(self):
        map(lambda x: x.normalize(), self.blocks)

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
        self._normalize_blocks()
        self._configure_blocks()

    def retrieve_active_blocks(self):
        return [block for block in self.blocks if block.is_active]
