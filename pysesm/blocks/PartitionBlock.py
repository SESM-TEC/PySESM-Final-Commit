"""
Partition Block Class.
 
Represents a single partition block within the input space, managing local
data points, normalization, and associated sparse coding layers.
 
Authors: The SESM Team 
Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import torch


from pysesm.base_types import TensorProxy


# Import the specific base class for type hinting
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer

class PartitionBlock:
    """
    Represents a sub-block in an N-dimensional grid.

    A PartitionBlock defines a specific region within the overall N-dimensional
    input space and manages the data points (X, y) that fall within its boundaries.
    It holds references to its own data, an associated sparse coding layer,
    and properties needed for local model evaluation.

    X and y are implemented as lists of tensors to optimize adding points to them.

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
    - normalized_X (Optional[TensorProxy]): Stacked and normalized (0 to 1) version of X,
                                          relative to the block's `block_scope`.
                                          Shape: (n_samples_in_block, n_features).
    - y (list[torch.Tensor]): List of original, unnormalized target values (outputs)
                              corresponding to points in X. Each element is a tensor.
    - target (Optional[TensorProxy]): Stacked and amplitude-scaled version of y.
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
        block_index: tuple[int, ...], # Use Tuple for clarity
        block_size: torch.Tensor,
        device: torch.device,
    ):
        self.block_index = block_index
        self.block_size = block_size.to(device)
        self.device = device
        
        self.space_origin = space_origin.to(self.device)

        # Calculate the base edge of this specific block based on its index
        # (e.g., for block (i, j), its origin is global_origin + i*block_size_x + j*block_size_y)
        base_edge_of_this_block = self.space_origin + torch.mul(
            torch.tensor(block_index, device=self.device), self.block_size
        )
        # Define the block's specific bounding box (min_coords, max_coords)
        self.block_scope = torch.stack((
            base_edge_of_this_block,
            base_edge_of_this_block + self.block_size
        )).to(self.device)
        
        self.amplitude: float = 1.0
        self.X: list[torch.Tensor] = []
        self.normalized_X: TensorProxy | None = None
        self.y: list[torch.Tensor] = []
        self.target: TensorProxy | None = None
        self.positions: list[int] = []
        self.predicted_output: list = []
        self.sparse_coding_layer: SparseCodingBaseLayer | None = None

        # Cache for device-specific data copies
        self._device_data_cache = {}
        
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

    def append_points(self, points_x: torch.Tensor, points_y: torch.Tensor, positions: list[int]):
        """
        Adds multiple data points (X, y) to this block along with their original indices.
        Points are stored in their original, unnormalized form.

        Args:
            points_x (torch.Tensor): A tensor of N-dimensional input features for the points.
                                     Expected shape: (n_samples_to_add, n_features).
            points_y (torch.Tensor): A tensor of target values for the points.
                                     Expected shape: (n_samples_to_add, output_dim) or (n_samples_to_add,).
            positions (list[int]): A list of original indices of the points in the global dataset.
                                   Length must match n_samples_to_add.
        """
        if points_x.shape[0] != points_y.shape[0] or points_x.shape[0] != len(positions):
            raise ValueError(
                f"Dimension mismatch: points_x ({points_x.shape[0]} samples), "
                f"points_y ({points_y.shape[0]} samples), and positions ({len(positions)} samples) "
                f"must have the same number of samples."
            )

        # Convert to list of tensors if needed (for consistency with `self.X` and `self.y` being lists)
        # However, for efficiency, if these are always processed as stacked tensors,
        # consider changing `self.X` and `self.y` to be single tensors that are concatenated.
        # For now, sticking to the existing list-of-tensors pattern for `self.X` and `self.y`.

        # Move to device and append
        self.X.extend([p_x.to(self.device) for p_x in points_x])
        self.y.extend([p_y.to(self.device) for p_y in points_y])
        self.positions.extend(positions)

        
    def clear_points(self):
        """Removes all data points and their derived values from the block."""
        self.X = []
        self.normalized_X = None
        self.y = []
        self.target = None
        self.positions = []
        self._device_data_cache.clear()


    def is_active(self,threshold=0) -> bool:
        """
        Determines if the block is active (contains any data points).

        Returns:
            bool: True if the block has points, False otherwise.
        """
        return len(self.X) > threshold
    

    def normalize_points(self, preserve_aspect_ratio: bool = True):
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

        if preserve_aspect_ratio:
            divisor = sizes.max()
        else:
            divisor = sizes

        # Protect against division by zero for any dimension that has zero size.
        # This might indicate malformed block sizes or degenerate dimensions.
        divisor[divisor == 0] = 1.0 if isinstance(divisor, torch.Tensor) else (1.0 if divisor == 0 else divisor)

        self.normalized_X = TensorProxy((tensor_X - min_vals) / divisor)


    def _create_target_tensor(self) -> torch.Tensor | None:
        """
        Private helper: Stacks and formats 'y' data into a 2D tensor,
        then scales it by `self.amplitude`. Does NOT calculate amplitude.
        Assumes self.y is populated and self.amplitude is set.
        """
        if not self.y:
            return None

        stacked_y = torch.stack([val.to(self.device) if isinstance(val, torch.Tensor) else torch.tensor(val, device=self.device) for val in self.y])
        
        # Ensure target is at least 2D (N_samples, output_dim)
        if stacked_y.dim() > 2: # Si es (N, 1, 1) o más, aplanar la última dim
            stacked_y = stacked_y.squeeze(-1) # Resultado (N, 1)
        elif stacked_y.dim() < 2: # Si es (N,), añadir una dim
            stacked_y = stacked_y.unsqueeze(-1) # Resultado (N, 1)
        
        # Apply the current amplitude
        target_tensor = stacked_y * self.amplitude
        return target_tensor.detach()

        
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

        # Stack y values. Ensure individual y elements are tensors.
        # If `y` is a list of scalar tensors (e.g., [tensor(1.0), tensor(2.0)]),
        # stacking them results in a 1D tensor (e.g., tensor([1.0, 2.0])).
        # If `y` is a list of 1D tensors (e.g., [tensor([1.0]), tensor([2.0])]),
        # stacking them results in a 2D tensor (e.g., tensor([[1.0],[2.0]])).
        # If `y` is a list of N-dim tensors, stacking them results in (N_samples, ...).
        stacked_y = torch.stack([val if isinstance(val, torch.Tensor) else torch.tensor(val, device=self.device) for val in self.y])
        
        # Handle scalar y inputs for max_y_abs calculation
        if stacked_y.dim() == 0:
            max_y_abs = stacked_y.abs().item() # Get scalar directly
        else:
            max_y_abs = stacked_y.abs().max().item()

        if max_y_abs > 1:
            self.amplitude = 1.0 / max_y_abs
        else:
            self.amplitude = 1.0

        # Now, use the new helper method to create the target tensor
        target_tensor = self._create_target_tensor()
        self.target = TensorProxy(target_tensor) if target_tensor is not None else None
        
    def prepare_target_for_inference(self):
        """
        Prepares the 'target' tensor for inference/testing using this block's 'y' data
        and its already set 'amplitude' (which should be the learned value).
        Does NOT recalculate amplitude.
        """
        if not self.y:
            self.target = None
            return

        # Directly use the helper method; self.amplitude is assumed to be set already.
        target_tensor = self._create_target_tensor()
        self.target = TensorProxy(target_tensor) if target_tensor is not None else None
