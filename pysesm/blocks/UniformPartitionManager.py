'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Abstract class for all Block Managers

Authors: The SESM Team 

License: 
'''


from ..sparse_coding.SparseCodingBaseLayer import SparseCodingConfig
from ..factories.SparseCodingFactory import SparseCodingFactory
from ..factories.BlockManagerFactory import BlockManagerFactory
from ..enums.DeviceTargetEnum import DeviceTarget # Assuming this is in pysesm.enums
from ..device_manager.DeviceManager import DeviceManager # Assuming this is in pysesm.device_manager

# Update BlockManager import:
from .BlockManager import BlockManager, BlockManagerConfig # Import both class and config
from .PartitionBlock import PartitionBlock

from dataclasses import dataclass # Ensure dataclass is imported
from typing import Union, Callable, Iterator, Dict, Optional, List # Import List for type hints
import logging
import numpy as np
import torch


DEFAULT_BLOCKS_PER_DIM = 4

@dataclass
class UniformPartitionConfig(BlockManagerConfig):
    """Configuration specific to UniformPartitionManager."""
    T: Union[torch.Tensor, int] = DEFAULT_BLOCKS_PER_DIM    # Blocks per dimension
    initial_bounds: Optional[np.ndarray] = None # Initial space bounds
    threshold: float = 0

@BlockManagerFactory.register("uniform_partition_manager")
class UniformPartitionManager(BlockManager):
    """
    A class to manage a uniform partitioning of the input space into
    blocks.

    The UniformPartitionManager divides the space into uniformly sized
    blocks, assigns points to these blocks, and configures or adjusts
    local models within each block.

    Args:
        logger (logging.Logger): Logger instance for recording messages and warnings.
        config (UniformPartitionConfig): A configuration object for this manager
        initial_bounds (np.ndarray, optional): Initial bounds for the partitioning, shaped as (2, n_features).
            - The first row contains the lower bounds.
            - The second row contains the upper bounds.
            If not provided, it is automatically calculated from the input data.
        threshold (float, optional): Threshold for determining block activity (default is 0).
        device_manager:

    """

    CONFIG_CLASS = UniformPartitionConfig 

    
    def __init__(
        self,
        config: UniformPartitionConfig,
        logger: logging.Logger,
        device_manager: Optional[DeviceManager] = None,
        sparse_coding_layer_hook = None
    ):
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
        super().__init__(config=config, logger=logger, device_manager=device_manager)
        
        self.T = config.T
        self.initial_bounds = config.initial_bounds
        self.threshold = config.threshold

        self.blocks = None
        self.block_size = None
        self.X = None
        self.y = None
        
        self.sparse_coding_layer_hook = sparse_coding_layer_hook

        # Helper for normalizing X coordinates in each block.
        self._vectorized_normalization = np.vectorize(lambda x: x.normalize_points())

    def _find_block(self, x: torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the block corresponding to a given point.

        Args:
            X (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """

        for index in np.ndindex(self.blocks.shape):
            block: PartitionBlock = self.blocks[index]
            if block.is_point_in_block(x):
                return block

        self.logger.warning("Could not find a block for point {}", x)
        return None

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        """
        Updates the arrangement of blocks based on the input data.

        This method initializes or adjusts the block arrangement and sizes, ensuring coverage of the input space.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        self.initial_bounds = self.initial_bounds.to(device)

        # If no T is given, create a T with a default size
        if type(self.T) is int:
            num_features = X.shape[1] if X.dim() > 1 else 1
            self.T = torch.tensor([self.T for _ in range(num_features)], device=device)

        # Ensure T is a tensor for subsequent operations
        self.T = self.T.to(device)

        # When no points and no blocks have been created
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.vstack(
                    [torch.min(X, dim=0).values, torch.max(X, dim=0).values]
                ).to(device)   # Calculates the range covered by the X vector
                self.logger.warning(
                    f"[{self.__class__.__name__}] No initial bounds provided, using calculated one {self.initial_bounds}"
                )

            # Space to be partitioned
            delta = self.initial_bounds[1] - self.initial_bounds[0]
            self.block_size = torch.div(delta, self.T.float()).to(device)

            
            self.blocks = np.empty(self.T.cpu().numpy(), dtype=object) 

            for index in np.ndindex(self.blocks.shape):
                self.blocks[index] = PartitionBlock(space_origin = self.initial_bounds[0].to(device), 
                                                    block_index = index, 
                                                    block_size = self.block_size.to(device),
                                                    device=device)
        else:
            new_max_x = torch.max(X).to(device)
            new_min_x = torch.min(X).to(device)

            # TODO: We need to fix this and reset all blocks with the new box if the new limits are large enough
            self.logger.debug(f"[{self.__class__.__name__}] Blocks already exist. Skipping re-arrangement for new points outside bounds.")
            

    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Maps input points to their respective sub-blocks.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target data corresponding to the input points.
        """
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        X = X.to(device)
        
        # y should be tensor already, convert to list of tensors if needed for new_point
        # Ensure y is a list of individual point tensors for new_point
        y_list = [yi.to(device) for yi in y.split(1, dim=0)] if y.dim() > 0 else [y.to(device)]


        for i in range(X.shape[0]):
            selected_block = self._find_block(X[i])
            if selected_block is not None:
                selected_block.new_point(X[i], y_list[i], i)
            else:
                self.logger.warning(f"Point {X[i]} at index {i} could not be mapped to any block. "
                                    "Consider adjusting initial bounds or partition strategy.")

        # After mapping points, normalize X and calculate amplitude/target for each active block.
        for idx in np.ndindex(self.blocks.shape):
            block = self.blocks[idx]
            if block.is_active: # Only process active blocks
                # Find the squeeze factor for the y's in the block
                block.calculate_amplitude_and_target() 


    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """   

        self._update_block_arrangement(X)
        self._map_points(X, y) # Sends points to their respective blocks
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block


    def init_sparse_coding_per_block(self,
                                     config: SparseCodingConfig,
                                     evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):

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
                    device= self.device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER),
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
            if self.blocks[index].is_active
        ]

    def retrieve_test_active_blocks(self, X, y):
        """
        Retrieves active blocks for testing purposes.

        Args:
            X (torch.Tensor): Test input data of shape (n_samples, n_features).
            y (torch.Tensor): Test target data of shape (n_samples,).

        Returns:
            List[PartitionBlock]: A list of active blocks corresponding to the test data.
        """
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        X = X.to(device)
        y = y.to(device)

        # Create new blocks array that can be modified without affecting original training blocks
        test_blocks = np.empty_like(self.blocks, dtype=object)
        for index in np.ndindex(self.blocks.shape):
            # Create a shallow copy of PartitionBlock, but clear its points data.
            # This reuses the spatial definition (scope, size, origin) but clears X, y, etc.
            # The clone_test / __deepcopy__ method from PartitionBlock.py is gone now.
            # We explicitly create a new PartitionBlock with same spatial properties.
            original_block = self.blocks[index]
            new_pb = PartitionBlock(
                space_origin=original_block.space_origin,
                block_index=original_block.block_index,
                block_size=original_block.block_size,
                device=original_block.device
            )
            # Crucially, we need to assign the *existing sparse_coding_layer* from the original block
            # to the new test block. This is what makes it a 'test' block with trained h.
            new_pb.sparse_coding_layer = original_block.sparse_coding_layer
            test_blocks[index] = new_pb
            
        # Temporarily use the test_blocks for mapping
        temp_current_blocks = self.blocks
        self.blocks = test_blocks

        # Map and normalize points into test blocks
        # This will populate X, y, normalized_X, amplitude, target in the new test_blocks
        self._map_points(X, y)

        # This works because it just adjusts coordinates to the block relative position
        self._vectorized_normalization(self.blocks) # Calls normalize_points on each block

        # Retrieve mapped test blocks and return to usual blocks
        test_active_blocks = self.retrieve_active_blocks()
        self.blocks = temp_current_blocks # Restore original training blocks

        return test_active_blocks
