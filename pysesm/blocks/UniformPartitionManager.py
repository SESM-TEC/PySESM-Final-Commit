'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Abstract class for all Block Managers

Authors: The SESM Team 

License: 
'''

from dataclasses import dataclass # Ensure dataclass is imported
from collections.abc import Callable
import logging

import numpy as np
import torch

from pysesm.base_types import TensorBatch

from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingConfig
from pysesm.factories.SparseCodingFactory import SparseCodingFactory

# Update BlockManager import:
from .BlockManager import BlockManager, BlockManagerConfig # Import both class and config
from .PartitionBlock import PartitionBlock

# Default number of blocks per dimension if not specified
DEFAULT_BLOCKS_PER_DIM = 4

@dataclass(kw_only=True)
class UniformPartitionConfig(BlockManagerConfig):
    """Configuration for UniformPartitionManager.
    
    This class defines the configuration parameters needed to set up uniform
    partitioning of the input space into blocks with optional overlap between
    adjacent blocks for smooth transitions.
    """
    
    # Number of blocks per dimension - can be int (uniform) or tensor (per-dim)
    T: torch.Tensor | int = DEFAULT_BLOCKS_PER_DIM
    
    # Bounding box coordinates: array of shape (2, n_dims) with [min, max] corners
    # None means bounds will be inferred from data
    initial_bounds: np.ndarray | None = None
    
    # Number of points in a block that must be surpassed to be considered active.
    activity_threshold: int = 0
    
    # Block overlap ratio (0-1) for smooth transitions between blocks
    # None=no overlap, float=uniform overlap, tensor=per-dimension overlap
    overlap_ratio: float | torch.Tensor | None = None
    
    
class UniformPartitionManager(BlockManager):
    """
    A class to manage a uniform partitioning of the input space into
    blocks.

    The UniformPartitionManager divides the space into uniformly sized
    blocks, assigns points to these blocks, and configures or adjusts
    local models within each block.

    """

    CONFIG_CLASS = UniformPartitionConfig 

    
    def __init__(self,
                 config: UniformPartitionConfig,
                 logger: logging.Logger,
                 sparse_coding_layer_hook = None
                 ):
        """
        Initializes the UniformPartitionManager with the provided parameters.

        Args:
            config (UniformPartitionConfig): configuration of the partition manager.
            logger (logging.Logger): Logger instance for recording messages and warnings.            
            sparse_coding_layer_hook: function to be attached to all block's sparse coding layers
        """
        super().__init__(config=config, logger=logger)
        
        self.T = config.T

        if self.T is not None: # T can be None if not provided initially, handled in _update_block_arrangement
            # If T is an int, it will be converted to tensor in _update_block_arrangement
            # If it's already a tensor, validate its values here.
            if isinstance(self.T, torch.Tensor) and (self.T <= 0).any():
                raise ValueError(f"All values in 'T' (blocks per dimension) must be positive integers. Got: {self.T}")
        
        self.initial_bounds = config.initial_bounds
        self.activity_threshold = config.activity_threshold

        self.blocks = None
        self.block_size = None
        
        self.sparse_coding_layer_hook = sparse_coding_layer_hook

        # Helper for normalizing X coordinates in each block.
        self._vectorized_normalization = np.vectorize(lambda x: x.normalize_points())

    def _find_block(self, x: torch.Tensor) -> PartitionBlock | None:
        """Finds the block corresponding to a given point.

        This will return one block only, whose scope covers the point.
        If a point lies exactly at the boundary between two blocks,
        the block with the highest index is chosen.

        Args:
            x (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.

        """

        normalized = torch.floor( (x - self.initial_bounds[0]) / self.block_size).long()

        # Verificación rápida con operaciones vectorizadas
        if torch.any(normalized < 0) or torch.any(normalized >= self.T):
            self.logger.warning("Could not find a block for point %s", x)
            return None
    
        index = tuple(normalized.cpu().numpy())
        return self.blocks[index]
        
    def _is_point_in_block(self, x: torch.Tensor, block: PartitionBlock) -> bool:
        """
        Checks if a given point's N-dimensional coordinates fall within this
        block's `block_scope`.

        Args:
            x (torch.Tensor): The N-dimensional input features of the point to
                              check.
                              Expected shape: (n_features,).
            block (PartitionBlock): The block to check against.            

        Returns:
            bool: True if the point is within the block's N-dimensional bounds,
            False otherwise.
        """
        x = x.to(self.device)
        # Check if all dimensions of point_x are >= lower bound AND <= upper bound
        return torch.all(block.block_scope[0] <= x) and torch.all(x <= block.block_scope[1])
        

    
    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        """
        Updates the arrangement of blocks based on the input data.

        This method initializes or adjusts the block arrangement and sizes, ensuring coverage of the input space.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        

        if self.initial_bounds is not None and isinstance(self.initial_bounds, np.ndarray):
            # If it's still None at this point, it will be inferred later.
            # If it's already a Tensor, it's fine.
            self.initial_bounds = torch.from_numpy(self.initial_bounds).to(self.device)

        # If no T is given initially (i.e., self.T is None), or if it's a single integer:
        if self.T is None or isinstance(self.T, int):
            num_features = X.shape[1] if X.dim() > 1 else 1 # Number of features for the input space
            # Create T as a tensor, ensuring all values are positive
            T_val_to_use = self.T if self.T is not None else DEFAULT_BLOCKS_PER_DIM
            if isinstance(T_val_to_use, int) and T_val_to_use <= 0:
                raise ValueError(f"Default or provided integer 'T' ({T_val_to_use}) must be a positive integer.")
            
            self.T = torch.tensor([T_val_to_use for _ in range(num_features)], device=self.device)

        # Ensure T is a tensor for subsequent operations
        self.T = self.T.to(self.device)

        # Validate that all T values are positive after conversion
        if (self.T <= 0).any():
            raise ValueError(f"All values in 'T' (blocks per dimension) must be positive integers. Got: {self.T}")

        # When no points and no blocks have been created (first call)
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.vstack(
                    [torch.min(X, dim=0).values, torch.max(X, dim=0).values]
                ).to(self.device)   # Calculates the range covered by the X vector
                self.logger.warning(
                    f"[{self.__class__.__name__}] No initial bounds provided, using calculated one {self.initial_bounds}"
                )
            else:
                # If initial_bounds was provided and is now a tensor, ensure it's on device
                self.initial_bounds = self.initial_bounds.to(self.device)

            # Space to be partitioned
            delta = self.initial_bounds[1] - self.initial_bounds[0]

            self.block_size = torch.div(delta, self.T.float()).to(self.device)

            if self.config.overlap_ratio is None:
                self.overlap = torch.zeros_like(self.block_size,device=self.device)
            else:
                self.overlap = self.block_size * torch.tensor(self.config.overlap_ratio,device=self.device)

            self.blocks = np.empty(self.T.cpu().numpy(), dtype=object) 

            for index in np.ndindex(self.blocks.shape):
                self.blocks[index] = PartitionBlock(space_origin = self.initial_bounds[0],
                                                    block_index = index, 
                                                    block_size = self.block_size,
                                                    device=self.device)
        else:
            new_max_x = torch.max(X).to(self.device)
            new_min_x = torch.min(X).to(self.device)

            # TODO: We need to fix this and reset all blocks with the new box if the new limits are large enough
            self.logger.debug(f"[{self.__class__.__name__}] Blocks already exist. Skipping re-arrangement for new points outside bounds.")
            

    def _map_points(self, X: torch.Tensor, y: torch.Tensor, expand_scope: bool = True):
        """
        Maps input points to their respective sub-blocks.

        Args:
            X (torch.Tensor):    Input data of shape (n_samples, n_features). y
            y (torch.Tensor):    Target data corresponding to the input points.
            expand_scope (bool): If True, allows blocks to expand their scope to
                                 include points outside their current bounds.

        """
        X = X.to(self.device)
        y = y.to(self.device)

        # Ensure y is always 2D (n_samples, output_dim) for consistent slicing later
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        assigned_points_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=self.device)

        if expand_scope:
            overlap = self.overlap
        else:
            overlap = torch.zeros_like(self.overlap,device=self.device)
        
        # Iterate over all blocks in the manager's grid
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]

            lower_bound = block.block_scope[0]-overlap
            upper_bound = block.block_scope[1]+overlap
            
            # Combine both checks to get points within the extended block
            points_in_extended_block_mask = ( (X >= lower_bound) * (X <= upper_bound) ).all(dim=1)
            
            # Get the actual points (X_selected) and their corresponding targets (y_selected)
            # and original positions (positions_selected)
            X_selected = X[points_in_extended_block_mask]
            y_selected = y[points_in_extended_block_mask]
            # Get original indices of selected points
            positions_selected = torch.nonzero(points_in_extended_block_mask).squeeze(1).tolist()

            if X_selected.shape[0] > 0:
                block.append_points(X_selected, y_selected, positions_selected)
                assigned_points_mask[points_in_extended_block_mask] = True

      
        # After iterating through all blocks, identify points that were *never* assigned
        unmapped_points_mask = ~assigned_points_mask
        
        # Log warnings for points that were not mapped to *any* block.
        # This restores the original intent of the warning for out-of-bounds points.
        if torch.any(unmapped_points_mask):
            unmapped_X = X[unmapped_points_mask]
            # Log each unmapped point (or a summary if many)
            for i, point_x in enumerate(unmapped_X):
                self.logger.warning(
                    f"Point {point_x.tolist()} at original index {torch.nonzero(unmapped_points_mask).squeeze(1)[i].item()} "
                    "could not be mapped to any block."
                )

    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """   

        self._update_block_arrangement(X)
        self._map_points(X, y) # Sends points to their respective blocks
        self._prepare_block_targets() # Adjust the y 
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block

    def _prepare_block_targets(self):
        """
        Processes y values within each active block by calculating amplitude and target values.
        Ensures target values are in the correct 2D shape (n_samples_in_block, output_dim).
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if block.is_active: # Only process if block has points
                # PartitionBlock.calculate_amplitude_and_target handles y processing,
                # amplitude calculation, and ensuring self.target is 2D.
                block.calculate_amplitude_and_target()

                
    def init_sparse_coding_per_block(self,
                                     config: SparseCodingConfig,
                                     evaluation_func: Callable[[TensorBatch, TensorBatch], TensorBatch]):

        """
        Initializes a sparse coding layer for each block.

        Args:
            config (SparseCodingConfig): Configuration for sparse coding.
            evaluation_func (Callable): The function to use for dictionary-h combination.
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if block.is_active:
                block.sparse_coding_layer = SparseCodingFactory.create(
                    config = config,
                    logger = self.logger,
                    parameter_hook=self.sparse_coding_layer_hook,
                    evaluation_func=evaluation_func
            )

    def retrieve_active_blocks(self):
        """
        Retrieves all active blocks in the partition. An active block is a block with at least one point mapped

        Returns:
            List[PartitionBlock]: A list of active blocks.
        """
        return [
            self.blocks[index]
            for index in np.ndindex(self.blocks.shape)
            if self.blocks[index].is_active(self.activity_threshold)
        ]


    def _create_test_block_structure(self) -> np.ndarray:
        """
        Creates a new NumPy array of PartitionBlock instances, copying only
        the spatial properties, the learned sparse_coding_layer, and amplitude
        from the original training blocks. Clears data points (X, y, target, normalized_X).
        """
        test_blocks_array = np.empty_like(self.blocks, dtype=object)
        for index in np.ndindex(self.blocks.shape):
            original_block = self.blocks[index]
            new_pb = PartitionBlock(
                space_origin=original_block.space_origin,
                block_index=original_block.block_index,
                block_size=original_block.block_size,
                device=original_block.device
            )
            # Transfer the learned sparse_coding_layer and amplitude from original training block
            new_pb.sparse_coding_layer = original_block.sparse_coding_layer
            new_pb.amplitude = original_block.amplitude
            test_blocks_array[index] = new_pb
        return test_blocks_array

    def _prepare_test_block_targets(self, blocks_array: np.ndarray):
        """
        Iterates through the given blocks array and calls prepare_target_for_inference
        on each active block, utilizing its pre-set amplitude.
        """
        for idx in np.ndindex(blocks_array.shape):
            block = blocks_array[idx]
            if block.is_active: # Only process active blocks
                block.prepare_target_for_inference()


    
    def _retrieve_blocks_generic(self, X: torch.Tensor, y: torch.Tensor = None, for_inference: bool = False):
        """
        Generic method to retrieve blocks for both training and inference.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor, optional): Target data. If None, creates dummy targets.
            for_inference (bool): If True, clears target data after mapping.
            
        Returns:
            List[PartitionBlock]: A list of active blocks.
        """
        X = X.to(self.device)
        
        # Handle y parameter
        if y is not None:
            y = y.to(self.device)
        else:
            # Create dummy y for spatial mapping
            y = torch.zeros(X.shape[0], 1, device=self.device)
        
        # Create the block structure and transfer learned properties
        temp_blocks = self._create_test_block_structure()
            
        # Temporarily use temp_blocks for mapping
        original_blocks = self.blocks
        self.blocks = temp_blocks

        # Map points into blocks
        self._map_points(X, y, expand_scope=False)

        if for_inference:
            # Clear target data for inference (we only needed y for spatial mapping)
            for idx in np.ndindex(self.blocks.shape):
                block = self.blocks[idx]
                if block.is_active:
                    block.y = []
                    block.target = None
        else:
            # Prepare targets for training/validation
            self._prepare_test_block_targets(self.blocks)

        # Normalize X coordinates
        self._vectorized_normalization(self.blocks)

        # Retrieve active blocks and restore original blocks
        active_blocks = self.retrieve_active_blocks()
        self.blocks = original_blocks
 
        return active_blocks
