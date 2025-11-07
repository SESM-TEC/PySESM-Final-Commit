'''
Copyright (C) 2025 Tecnológico de Costa Rica

Adaptive Partition Manager Class

Provides a partition manager based on a kd-tree space partition, which
adapts itself depending on the data it has.

Author: Hender Valdivia
'''
from dataclasses import dataclass
from collections.abc import Callable

import logging
import torch
import numpy as np

from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.PartitionStrategy import PartitionStrategy
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.base_types import TensorProxy
from .BlockManager import BlockManagerConfig
from ..sparse_coding.SparseCodingBaseLayer import SparseCodingConfig
from ..factories.SparseCodingFactory import SparseCodingFactory

@dataclass(kw_only=True)
class AdaptativePartitionConfig(BlockManagerConfig):
    """Configuration for AdaptativePartitionManager.
    
    This class defines the configuration parameters needed to set up adaptative
    partitioning of the input space into blocks with optional overlap between
    adjacent blocks for smooth transitions.
    """
    overlap_ratio: float = 0.1
    partition_strategy: PartitionStrategy = None
    
class AdaptativePartitionManager(BlockManager):
    
    CONFIG_CLASS = AdaptativePartitionConfig 
    
    def __init__(
        self,
        config: AdaptativePartitionConfig,
        logger: logging.Logger,
        sparse_coding_layer_hook=None
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
        super().__init__(config=config, logger=logger)
        
        self.logger = logger
        self.blocks: np.ndarray = np.empty(0, dtype=object)
        self.strategy=config.partition_strategy
        self.sparse_coding_layer_hook = sparse_coding_layer_hook
        self.X: torch.Tensor = None
        self.y: torch.Tensor = None
        self.total_blocks: int = 0
        self.initial_bounds = None
    

    def _find_block(self, x: torch.Tensor) -> PartitionBlock | None:
        """
        Finds the block corresponding to a given point.

        Args:
            x (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """
        return self.strategy.find_partition_for_point(x)

    def _update_block_arrangement(self, X: torch.Tensor, y: torch.Tensor = None) -> None:
        """
        Updates the block arrangement by delegating to the partitioning strategy.

        This method initializes the strategy if it's the first run, or adds new
        points to the existing strategy. It then refreshes its own list of blocks.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        # 1. Check if the strategy needs to be built for the first time.
        #    (We check an internal attribute of the strategy, like 'kdtree' in this case)
        if not self.strategy.built:
            self.strategy.build(X, y)
            self.strategy.built=True
            needs_refresh = True
        else:
            # 2. Otherwise, add points and check if a major rebuild happened.
            #    The strategy tells us if we need to refresh the blocks.
            restarted = self.strategy.add_points(X, y)
            needs_refresh = restarted

        # 3. If it's the first build OR a restart happened, get the new blocks.
        if needs_refresh:
            new_partitions = self.strategy.get_partitions()
            self.blocks = np.array([partition.block for partition in new_partitions], dtype=object)
            self.total_blocks = len(self.blocks)
            needs_refresh=False
            # 4. Handle overlap logic here.
            self._apply_overlap_to_blocks()

    def _apply_overlap_to_blocks(self):
        """Applies the overlap ratio to all current blocks."""
        if self.config.overlap_ratio is None:
            return 

        overlap_ratio = torch.tensor(self.config.overlap_ratio, device=self.device)
        for block in self.blocks:
            block.overlap = block.block_size * overlap_ratio

    def _vectorized_normalization(self, x: np.ndarray = None):
        """
        Applies normalization while preserving aspect ratio for dimensions.
        
        Args:
            x: Array of blocks containing X data to normalize
        """
        if x is None:
            return
        
        vectorized_normalize = np.vectorize(lambda block: block.normalize_points())
        vectorized_normalize(x)


    def _map_points(self, X: torch.Tensor = None, y: torch.Tensor = None, expand_scope: bool = False):
        """
        Synchronizes the data points from the strategy's partitions into the manager's blocks.

        Args:
            X (torch.Tensor): Data
            y (torch.Tensor): Data
            expand_scope (bool): If True, blocks will map points from a wider area
                                defined by their scope plus their overlap.
        """
        # Clear all points from all blocks to ensure a fresh start.
        for block in self.blocks:
            block.clear_points()

        if not expand_scope:
            partition_data_list = self.strategy.get_partitions()

            if len(self.blocks) != len(partition_data_list):
                self.logger.error("Mismatch between manager blocks and strategy partitions!")
                return

            for i, block in enumerate(self.blocks):
                partition_data = partition_data_list[i]
                if partition_data.X is not None and isinstance(partition_data.X, torch.Tensor):
                    for j in range(partition_data.X.shape[0]):
                        block.new_point(partition_data.X[j], partition_data.y[j], pos=j)
        else:
            all_X, all_y = self.strategy.get_all_points()

            if all_X.shape[0] == 0:
                return 

            for block in self.blocks:
                lower_bound = block.block_scope[0] - block.overlap
                upper_bound = block.block_scope[1] + block.overlap

                # Create a mask to find all points within the expanded bounds
                mask = (all_X >= lower_bound) & (all_X <= upper_bound)
                mask = mask.all(dim=1)

                X_selected = all_X[mask]
                y_selected = all_y[mask]

                # Find original indices if needed, otherwise use a counter
                original_indices = torch.where(mask)[0].tolist()

                if X_selected.shape[0] > 0:
                    # Again, an append_points would be ideal.
                    for i in range(X_selected.shape[0]):
                        block.new_point(X_selected[i], y_selected[i], pos=original_indices[i])

        for block in self.blocks:
            if block.is_active():
                block.y = [yi.unsqueeze(0) if yi.dim() == 0 else yi for yi in block.y]

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

    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the kdtree and maps them to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """

        self._update_block_arrangement(X, y)
        self._map_points()
        self._prepare_block_targets()
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

    def _create_test_block_structure(self) -> np.ndarray:
        """
        Creates a new NumPy array of PartitionBlock instances, copying only
        the spatial properties, the learned sparse_coding_layer, and amplitude
        from the original training blocks. Clears data points (X, y, target, normalized_X).
        """
        treeLeaves=self.kdtree.get_leaves()
        test_blocks = np.empty_like(self.blocks, dtype=object)  
        
        #Clone blocks and map test data to each test_block    
        pos=0
        for index in np.ndindex(self.blocks.shape):
            node=treeLeaves[index[0]]
            if node.Data.test_data is not None:
                node.Data.block.clear_points()
                new_pb = PartitionBlock(    
                    space_origin=node.Data.block.space_origin,
                    block_index=node.Data.block.block_index,
                    block_size=node.Data.block.block_size,
                    device=node.Data.block.device
                )
                # Transfer the learned sparse_coding_layer and amplitude from original training block
                new_pb.sparse_coding_layer = node.Data.block.sparse_coding_layer
                new_pb.amplitude = node.Data.block.amplitude
                test_blocks[index] = new_pb
                for i, _ in enumerate(node.Data.test_data):
                    test_blocks[index].new_point(node.Data.test_data[i], node.Data.test_y[i],pos)
                    pos+=1
                

        for idx in np.ndindex(test_blocks.shape):
            block = test_blocks[idx]
            if block is not None and len(block.y) > 0:  # Only if block has points
                block.y = [yi.unsqueeze(0) if yi.dim() == 0 else yi for yi in block.y]

        # Selects blocks that weren't assigned any test data
        test_blocks = np.array(
            [test_blocks[index] for index in np.ndindex(test_blocks.shape) 
            if test_blocks[index] is not None and test_blocks[index].is_active()],
            dtype=object)
            
        return test_blocks


    def _retrieve_blocks_generic(self, X: torch.Tensor, 
                                 y: torch.Tensor = None, 
                                 for_inference: bool = False):

        """
        Generic method to retrieve blocks for both training and inference.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor, optional): Target data. If None, creates dummy targets.
            for_inference (bool): If True, clears target data after mapping.

        Returns:
            List[PartitionBlock]: A list of active blocks.
        """
        device = self.device
        X = X.to(device)

        # Handle y parameter
        if y is not None:
            y = y.to(device)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
        else:
            # Create dummy y for spatial mapping
            y = torch.zeros(X.shape[0], 1, device=device)

        # Configure data and start splitting without changing the tree
        self.kdtree.root.Data.test_data=X
        self.kdtree.root.Data.test_y=y
        self.kdtree._splitDataInNodes_test(self.kdtree.root)
              
        blocks_structure=self._create_test_block_structure()

        assert blocks_structure.size > 0
        for block in blocks_structure:
            assert block.X!=[]

        #Normalizes data
        self._vectorized_normalization(blocks_structure)

        # Configure blocks
        if for_inference:
            # For inference, clear target data (we only needed y for spatial mapping)
            for index in np.ndindex(blocks_structure.shape):
                block = blocks_structure[index]
                if block.is_active:
                    block.y = []
                    block.target = None
        else:
            # For training/validation, prepare targets
            for index in np.ndindex(self.blocks.shape):
                block = self.blocks[index]
                if len(block.y) != 0:
                    block.target = torch.stack([value * block.amplitude for value in block.y])
                    if block.target.dim() == 1:
                        block.target = block.target.unsqueeze(-1)
                    block.target = block.target.detach()

        # Get active blocks
        active_blocks = [
            blocks_structure[index]
            for index in np.ndindex(blocks_structure.shape)
            if blocks_structure[index].is_active
            ]
        
        # Reconfigure kdtree and blocks to leave them as they were
        self._map_points()

        return active_blocks
