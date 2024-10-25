import copy

import torch


class PartitionBlock:
    """
    Represents a sub-block in a 2D grid.

    Attributes:
    - vertices (np.ndarray): The vertices of the sub-block.
    - amplitude (int): The amplitude of the sub-block.
    - samples_inside (list): List of samples inside the sub-block.
    - output_values (list): List of output values.
    - index (list): List of index of the original point X

    Methods:
    - add_point(point): Add a point to the sub-block.
    """

    def __init__(self, space_bound: torch.Tensor, block_index: tuple[int, ...], block_size: torch.Tensor, amplitude: int = 1, h=None, ista_layer=None):
        self.block_index = block_index
        self.block_size = block_size
        eps = torch.finfo(torch.float32).eps
        base_edge = space_bound + torch.mul(torch.tensor(block_index), block_size)
        self.block_scope = torch.stack((base_edge - eps, base_edge + block_size + eps))
        self.h = h
        self.amplitude = amplitude
        self.X = []
        self.y = []
        self.normalized_X = None
        self.predicted_output = []
        self.ista_layer = ista_layer

    def new_point(self, point_x, point_y):
        self.X.append(point_x)
        self.y.append(point_y)

    @property
    def is_active(self):
        return len(self.X) > 0

    def is_point_in_block(self, point_x):
        return torch.all(self.block_scope[0] <= point_x) and torch.all(point_x <= self.block_scope[1])

    def normalize(self):
        tensor_X = torch.stack(self.X)
        self.normalized_X = (tensor_X - self.block_scope[0])/self.block_scope[1]

    def clone_test(self):
        cloned_block = copy.copy(self)

