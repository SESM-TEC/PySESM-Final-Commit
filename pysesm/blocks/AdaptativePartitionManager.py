'''
Copyright (C) 2025 Tecnológico de Costa Rica

Adaptive Partition Manager Class

Provides a partition manager based on a kd-tree space partition, which
adapts itself depending on the data itself.

Author: Hender Valdivia
'''

from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.KDTree import KDTree
from pysesm.blocks.Node import Node
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from copy import deepcopy

import logging
import torch
import numpy as np
from typing import Union, Callable, Iterator, Type

def squeeze_factor(y: np.ndarray):
    """
    Calculates a squeezing factor for a given set of values.

    Args:
        y (np.ndarray): An array containing numeric values.

    Returns:
        float: The squeezing factor. If the maximum value in y exceeds 1, returns 1 / max(y).
        Otherwise, returns 1.0.
    """
    e_f = 0.0
    max_y = torch.stack(y).abs().max()
    if max_y > 1:
        e_f = 1 / max_y
    else:
        e_f = 1.0
    return e_f

class AdaptativePartitionManager(BlockManager):
    def __init__(
        self,
        logger: logging.Logger,
        n_functions,
        maxNodeSize: int,
        maxSplitsBeforeRestart: int = 5,
        device_manager=None
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
        super().__init__()
        
        self.n_functions = n_functions
        self.logger = logger
        self.maxNodeSize = maxNodeSize
        self.maxSplitsBeforeRestart = maxSplitsBeforeRestart
        self.blocks = []
        self.X = None
        self.y = None
        self.total_blocks=0
        self.splits=0
        self._vectorized_normalization = np.vectorize(lambda x: x.normalize())
        self.kdtree=None
        self.device_manager=device_manager
        self.device="cpu"
        if self.device_manager is not None:
            self.device=device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

    def _find_block(self, x: torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the block corresponding to a given point.

        Args:
            x (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """
        node = self.kdtree.find_block(x)
        if node is not None:
            return node.block
        else:
            return None #Check type, should be PartitionBlock its node

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        """
        Updates the arrangement of blocks based on the input data.

        This method initializes or adjusts the block arrangement and sizes, ensuring coverage of the input space.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        X=X.to(self.device)
        if self.kdtree is None:
            self.splits=0
            self.kdtree = KDTree(X, self.maxNodeSize, device=self.device)
            X=X[:,:-1]
            treeNodes=self.kdtree.get_leaves()

            self.blocks = np.empty(len(treeNodes), dtype=object)  

            for i, node in enumerate(treeNodes):
                 block_size=node.bounds[0]-node.bounds[1]
                 self.total_blocks+=1
                 self.blocks[(i,)] = PartitionBlock(
                    node.bounds[1],  
                    (i,), 
                    block_size,
                    device=self.device
                 )

                 node.block=self.blocks[(i,)] #Define node blocks

        else:  #logic to add points to the existing tree
            for row in X:
                self.kdtree.add_point(row[:-1],row[-1:])
            treeNodes=self.kdtree.get_leaves()
            if len(treeNodes)>self.total_blocks:    #If a node was split, update self.blocks
                self.splits=len(treeNodes)-self.total_blocks    #How many nodes were split
                self.blocks = np.empty(len(treeNodes), dtype=object)  
                
                for i, node in enumerate(treeNodes):
                    node.block.block_index=(i,)
                    self.blocks[(i,)]=node.block
                self.total_blocks=len(treeNodes)
        if self.splits > self.maxSplitsBeforeRestart:
            treeNodes=self.kdtree.get_leaves()
            X=torch.Tensor()
            y=torch.Tensor()
            for node in treeNodes: 
                X=torch.cat((X,node.data),dim=0)
                y=torch.cat((y,node.y),dim=0)
            Xy=torch.cat((X,y), dim=1)
            self.kdtree = None
            self.splits = 0
            self._update_block_arrangement(Xy)

    def _configure_blocks(self, init_h: bool = True):
        """
        Configures each block with its expected squeeze factor and initializes sparse vectors if required.
        Internally, the .y attribute will hold the raw original y data, and .target the normalized version.

        Args:
            init_h (bool, optional): Whether to initialize the sparse vector `h` for each block (default is True).
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if len(block.y) != 0:
                
                if init_h:
                    # Squeeze should be computed only with training data
                    block.amplitude = squeeze_factor(block.y)

                    block.h = torch.nn.Parameter(
                        torch.rand(self.n_functions,1), requires_grad=True
                    )

                    block.h.data /= block.h.data.sum()

                    # self.logger.debug(
                    #     f"Created random vector for block at index {index}, created sparse vector h: {block.h}"
                    # )

                block.target = torch.stack([value * block.amplitude for value in block.y])
                if block.target.dim() == 1:
                    block.target = block.target.unsqueeze(-1)
                block.target = block.target.detach()

    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Maps kdtree leaf nodes data to their blocks.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target data corresponding to the input points.
        """
        treeNodes=self.kdtree.get_leaves()
        for _ , node in enumerate(treeNodes):
            node.block.clear_points()
            for i, _ in enumerate(node.data):
                node.block.new_point(node.data[i],node.y[i],i)
        for idx in np.ndindex(self.blocks.shape):
            block = self.blocks[idx]
            if len(block.y) > 0:  # Only if block has points
                block.y = [yi.unsqueeze(0) if yi.dim() == 0 else yi for yi in block.y]


    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """

        Xy=torch.cat((X,y), dim=1)
        self._update_block_arrangement(Xy)
        self._map_points(X, y)
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block
        self._configure_blocks() # Normalize y value and initialize h in each block 

    def init_ista_per_block(
        self,
        n_functions: int,       # self._update_block_arrangement(Xy)

        ista_alpha: float,
        ista_lambd: float,
        evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ista_optimizer: Callable[[Iterator[torch.nn.Parameter],float], torch.optim.Optimizer]
    ):
        """
        Initializes an ISTA layer for each block.

        Args:
            n_functions (int): Number of functions or features for the ISTA layer.
            ista_alpha (float): Learning rate for the ISTA layer.
            ista_lambd (float): Regularization parameter for the ISTA layer.
            evaluation_func (Callable): Function for evaluating the ISTA layer.
        """
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            block.ista_layer = ISTALayer(
                n_functions=n_functions,
                alpha=ista_alpha,
                lambd=ista_lambd,
                evaluation_func=evaluation_func,
                logger=self.logger,
                optimizer=ista_optimizer,
                device= self.device_manager.get_device(DeviceTarget.ISTA_LAYER),
                # parameter_hook=self._ista_hook if self.hook_manager and self.hook_manager.active_hooks[HookType.ISTALAYER] else None
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
        y = y.to(device).unsqueeze(-1)

        #Configures test data and starts splitting the data without changing the tree
        self.kdtree.root.test_data=X
        self.kdtree.root.test_y=y
        self.kdtree._splitDataInNodes_test(self.kdtree.root)
        
        treeLeaves=self.kdtree.get_leaves()
        test_blocks = np.empty(len(treeLeaves), dtype=object)  
        
        #Clone blocks and map test data to each test_block    
        pos=0
        for j, node in enumerate(treeLeaves):
            if node.test_data is not None:
                node.block.clear_points()
                test_blocks[(j,)] = node.block.clone_test()
                for i, _ in enumerate(node.test_data):
                    test_blocks[(j,)].new_point(node.test_data[i], node.test_y[i],pos)
                    pos+=1

        for idx in np.ndindex(test_blocks.shape):
            block = test_blocks[idx]
            if len(block.y) > 0:  # Only if block has points
                block.y = [yi.unsqueeze(0) if yi.dim() == 0 else yi for yi in block.y]

        # Selects blocks that weren't assigned any test data
        test_blocks = np.array(
            [test_blocks[index] for index in np.ndindex(test_blocks.shape) 
            if test_blocks[index].is_active],
            dtype=object)

        assert test_blocks.size > 0
        for block in test_blocks:
            assert block.X!=[]

        #Normalizes test data
        self._vectorized_normalization(test_blocks)
        # intento de nueva normalización
        # for block in test_blocks:
         #    node=self.kdtree._find_node(block.X[0])
          #   block.normalized_X=(torch.stack(block.X) - node.bounds[1])/block.block_size
        
        # Configure blocks:
        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if len(block.y) != 0:
                block.target = torch.stack([value * block.amplitude for value in block.y])
                if block.target.dim() == 1:
                    block.target = block.target.unsqueeze(-1)
                block.target = block.target.detach()

        #retrieve_active_test_blocks
        test_active_blocks = [
            test_blocks[index]
            for index in np.ndindex(test_blocks.shape)
            if test_blocks[index].is_active
            ]
        
        #Reconfigure kdtree and blocks to leave them as they were
        self._map_points(None, None)

        return test_active_blocks
