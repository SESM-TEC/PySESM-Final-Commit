from pysesm.models.Blocks.BlockManager import BlockManager
import numpy as np
import torch

from pysesm.models.Blocks.PartitionBlock import PartitionBlock


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
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def _configure_blocks(self):
        """
        Configures the blocks with their expected squeeze factor
        """
        for block in self.blocks:
            if len(block.output_values) != 0:
                block.amplitude = squeeze_factor(block.target)
                # block.ista_layer = ISTALayer(l_functions, SEED) TODO: Is this needed?
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

    def _distribute_items_in_blocks(self, t: np.ndarray, x_n: torch.Tensor, y: np.ndarray):
        """
        Locate points in their respective sub-blocks.

        Args:
        - x_n (np.ndarray): The normalized points between 0 and 1.
        - t (np.ndarray): The integer part of the normalized points.
        - y (np.ndarray) : The output values associated with the samples
        """
        for i in range(x_n.shape[0]):
            point = x_n[i]
            location = t[i]
            block = self.blocks[location[0] * self.T + location[1]]
            # TODO: Verify if is valid
            block.new_point(point, y[i])

    def add_points(self, X: torch.Tensor, y: np.ndarray):
        t, x_n = self._data_mapping(X)
        self._distribute_items_in_blocks(t, x_n, y)
        self._configure_blocks()

    def create_blocks(self):
        self.blocks = np.empty((self.T ** 2), dtype=object)

        for index in range(self.T ** 2):
            self.blocks[index] = PartitionBlock()
