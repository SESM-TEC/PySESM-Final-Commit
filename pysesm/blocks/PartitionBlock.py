'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

SESM Base Class

Provides the basic functionality of the Sparse-Encoded Surrogate Model.

Authors: The SESM Team 

License: 
'''

import copy
import torch
from typing import Optional # Keep Optional for explicit None types

# Import the specific base class for type hinting
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer

class PartitionBlock:
    """
    Represents a sub-block in an N-dimensional grid.

    A PartitionBlock defines a specific region within the overall N-dimensional
    input space and manages the data points (X, y) that fall within its boundaries.
    It holds references to its own data, an associated sparse coding layer,
    and properties needed for local model evaluation.

    Attributes:
    - space_origin (torch.Tensor): The minimum coordinates (origin) of the entire
                                   N-dimensional partitioned input space (global lower bound).
                                   Shape: (n_features,).
    - block_index (Tuple[int, ...]): A tuple representing the N-dimensional grid index of this block
                                     (e.g., (0, 1, 0) for a 3D grid). Length is `n_features`.
    - block_size (torch.Tensor): The N-dimensional side lengths of this specific block
                                 (e.g., [dim1_size, dim2_size, ...]). Shape: (n_features,).
    - block_scope (torch.Tensor): A [2, n_features] tensor defining the actual
                                  lower and upper bounds (min/max corners) of this block in
                                  the N-dimensional input space. This is the block's bounding box (bbox).
    - amplitude (float): A squeezing/scaling factor applied to target values (y)
                         within this block. Used for normalization.
    - X (list[torch.Tensor]): List of original, unnormalized input data points (features)
                              that have been assigned to this block. Each element is
                              a tensor of shape (n_features,).
    - normalized_X (Optional[torch.Tensor]): Stacked and normalized (0 to 1) version of X,
                                          relative to the block's `block_scope`.
                                          Shape: (n_samples_in_block, n_features).
    - y (list[torch.Tensor]): List of original, unnormalized target values (outputs)
                              corresponding to points in X. Each element is a tensor.
    - target (Optional[torch.Tensor]): Stacked and amplitude-scaled version of y.
                                    This is the target used for sparse coding.
                                    Shape: (n_samples_in_block, output_dim).
    - positions (list[int]): List of original indices of points added to this block,
                             allowing mapping back to the global dataset.
    - predicted_output (list): List of predicted values for points within this block.
    - sparse_coding_layer (Optional[SparseCodingBaseLayer]): The specific sparse coding layer
                                                        instance associated with this block.
                                                        This layer holds the block's 'h' vector.
    - device (torch.device): The PyTorch device (CPU/GPU) where this block's tensors reside.
    """

    def __init__(
        self,
        space_origin: torch.Tensor,
        block_index: Tuple[int, ...], # Use Tuple for clarity
        block_size: torch.Tensor,
        device: torch.device,
    ):
        self.block_index = block_index
        self.block_size = block_size.to(device)
        self.device = device
        
        self.space_origin = space_origin.to(self.device)

        eps = torch.finfo(torch.float32).eps
        # Calculate the base edge of this specific block based on its index
        # (e.g., for block (i, j), its origin is global_origin + i*block_size_x + j*block_size_y)
        base_edge_of_this_block = self.space_origin + torch.mul(
            torch.tensor(block_index, device=self.device), self.block_size
        )
        # Define the block's specific bounding box (min_coords, max_coords)
        self.block_scope = torch.stack((
            base_edge_of_this_block - eps,
            base_edge_of_this_block + self.block_size + eps
        )).to(self.device)
        
        self.amplitude: float = 1.0
        self.X: list[torch.Tensor] = []
        self.normalized_X: Optional[torch.Tensor] = None
        self.y: list[torch.Tensor] = []
        self.target: Optional[torch.Tensor] = None
        self.positions: list[int] = []
        self.predicted_output: list = []
        self.sparse_coding_layer: Optional[SparseCodingBaseLayer] = None
        
    def new_point(self, point_x: torch.Tensor, point_y: torch.Tensor, pos: int):
        """
        Adds a new data point (x, y) to this block along with its original index.
        Points are stored in their original, unnormalized form.

        Args:
            point_x (torch.Tensor): The N-dimensional input features of the point.
                                    Expected shape: (n_features,).
            point_y (torch.Tensor): The target value(s) of the point.
                                    Expected shape: (output_dim,) or scalar.
            pos (int): The original index of the point in the global dataset.
        """
        self.X.append(point_x.to(self.device))
        self.y.append(point_y.to(self.device))
        self.positions.append(pos)


    def clear_points(self):
        """Removes all data points and their derived values from the block."""
        self.X = []
        self.normalized_X = None
        self.y = []
        self.target = None
        self.positions = []


    @property
    def is_active(self) -> bool:
        """
        Determines if the block is active (contains any data points).

        Returns:
            bool: True if the block has points, False otherwise.
        """
        return len(self.X) > 0

    def is_point_in_block(self, point_x: torch.Tensor) -> bool:
        """
        Checks if a given point's N-dimensional coordinates fall within this block's `block_scope`.

        Args:
            point_x (torch.Tensor): The N-dimensional input features of the point to check.
                                    Expected shape: (n_features,).

        Returns:
            bool: True if the point is within the block's N-dimensional bounds, False otherwise.
        """
        point_x = point_x.to(self.device)
        # Check if all dimensions of point_x are >= lower bound AND <= upper bound
        return torch.all(self.block_scope[0] <= point_x) and torch.all(
            point_x <= self.block_scope[1]
        )

    def normalize_points(self):
        """
        Normalizes the X coordinates of all points within the block to a [0, 1] range,
        relative to the block's own `block_scope`.
        Requires X to be stacked as a tensor first.
        The `normalized_X` attribute will have shape (n_samples_in_block, n_features).
        """
        if not self.X:
            self.normalized_X = None
            return

        tensor_X = torch.stack(self.X).to(self.device)
        min_vals = self.block_scope[0].to(self.device)
        sizes = self.block_size.to(self.device) 

        # Protect against division by zero for any dimension that has zero size.
        # This might indicate malformed block sizes or degenerate dimensions.
        sizes[sizes == 0] = 1.0 

        self.normalized_X = (tensor_X - min_vals) / sizes

    def calculate_amplitude_and_target(self):
        """
        Calculates the squeezing amplitude factor for the block's 'y' values
        and derives the 'target' (scaled 'y') tensor used for sparse coding.
        This ensures target values are within a normalized range suitable for training.
        The `target` attribute will have shape (n_samples_in_block, output_dim).
        """
        if not self.y:
            self.amplitude = 1.0
            self.target = None
            return

        # Stack y values to compute max absolute value. Ensure they are tensors.
        stacked_y = torch.stack([val if isinstance(val, torch.Tensor) else torch.tensor(val, device=self.device) for val in self.y])

        # If y values are scalars, unsqueeze to make it (n_samples, 1) before max()
        if stacked_y.dim() == 0:
             stacked_y = stacked_y.unsqueeze(0)
        
        max_y_abs = stacked_y.abs().max()

        if max_y_abs > 1:
            self.amplitude = 1.0 / max_y_abs.item() # Get scalar float value
        else:
            self.amplitude = 1.0

        # Create target tensor by applying amplitude.
        # Ensure target is 2D: (n_samples_in_block, output_dim)
        # If original y was (n_samples,) or (n_samples, scalar), make it (n_samples, 1)
        self.target = torch.stack([value * self.amplitude for value in self.y]).to(self.device)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)
        self.target = self.target.detach() # Detach from any prior graph if y was part of one
