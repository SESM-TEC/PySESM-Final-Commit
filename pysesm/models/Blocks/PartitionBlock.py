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

    def __init__(self, block_index: tuple[int, ...], block_size: torch.Tensor, amplitude=1, h=None):
        self.block_index = block_index
        self.block_size = block_size
        base_edge = block_index * block_size
        self.block_scope = torch.tensor([base_edge, base_edge + block_size])
        self.h = h
        self.amplitude = amplitude
        self.X = []
        self.y = []
        self.normalized_X = []
        self.predicted_output = []

    def new_point(self, point_x, point_y):
        self.X.append(point_x)
        self.y.append(point_y)

    @property
    def is_active(self):
        return len(self.X) > 0

    def is_point_in_block(self, point_x):
        return self.block_scope[0] <= point_x <= self.block_scope[1]

    def normalize(self):
        tensor_X = torch.tensor(self.X)
        eps = self.block_scope * 1e-6
        self.normalized_X = ((tensor_X - torch.min(tensor_X)) / (self.block_scope + eps))
