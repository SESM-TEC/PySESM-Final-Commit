import copy
import torch
class PartitionBlock:
    """
    Represents a sub-block in a 2D grid.
    Attributes:
    - space_bound (torch.Tensor[float]): Bounds of each dimension (e.g. tensor([-2.1,-2.1]))
    - amplitude (float): The squeeze_factor of targets (self.y).
    - block_index (tuple): index of this block in the array of blocks.
    - block_size (torch.Tensor[float]): Space length of each dimension.
    - 
    Methods:
    - add_point(point): Add a point to the sub-block.
    """

    def __init__(
        self,
        space_bound: torch.Tensor,
        block_index: tuple[int, ...],
        block_size: torch.Tensor,
        amplitude: int = 1,
        h=None,
        ista_layer=None,
        device=None
    ):  
        print("space_bound",space_bound)
        print("block_index",block_index)
        print("block_size", block_size)

        self.block_index = block_index
        self.block_size = block_size
        self.device = device
        
        self.space_bound = space_bound.to(self.device)
        self.block_size = block_size.to(self.device)

        eps = torch.finfo(torch.float32).eps
        base_edge = self.space_bound + torch.mul(torch.tensor(block_index, device=self.device), self.block_size)
        self.block_scope = torch.stack((base_edge - eps, base_edge + self.block_size + eps)).to(self.device)
        
        self.h = h
        self.amplitude = amplitude
        self.X = []
        self.normalized_X = None
        self.y = []
        self.positions = []
        self.target = []
        self.predicted_output = []
        self.ista_layer = ista_layer

    def new_point(self, point_x, point_y, pos):
        point_x = point_x.to(self.device)
        point_y = point_y.to(self.device)
        self.X.append(point_x)
        self.y.append(point_y)
        self.positions.append(pos)


    def clear_points(self):
        """Remove all points"""
        self.X=[]
        self.normalized_X=None
        self.y=[]
        self.target = []
        self.position=[]


    @property
    def is_active(self):
        return len(self.X) > 0

    def is_point_in_block(self, point_x):
        
        point_x = point_x.to(self.device)

        return torch.all(self.block_scope[0] <= point_x) and torch.all(
            point_x <= self.block_scope[1]
        )

    def normalize(self):
        tensor_X = torch.stack(self.X).to(self.device)
        min_vals = self.block_scope[0].to(self.device)
        sizes = self.block_size.to(self.device)
        self.normalized_X = (tensor_X - min_vals) / sizes

    def clone_test(self):
        cloned_block = PartitionBlock.__new__(PartitionBlock)
        cloned_block.block_index = self.block_index
        cloned_block.block_size = self.block_size
        cloned_block.block_scope = self.block_scope
        cloned_block.h = self.h
        cloned_block.amplitude = self.amplitude
        cloned_block.X = []
        cloned_block.y = []
        cloned_block.normalized_X = None
        cloned_block.positions = []
        cloned_block.target = []
        cloned_block.predicted_output = []
        cloned_block.ista_layer = self.ista_layer
        cloned_block.device = self.device 
        return cloned_block

    def __deepcopy__(self, memo):
        # Use the custom clone_test method for deep copying
        cloned_block = self.clone_test()
        # Add the cloned object to the memo dictionary to handle circular references
        memo[id(self)] = cloned_block
        return cloned_block
