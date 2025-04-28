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
        initial_bounds: np.ndarray = None,
        threshold: float = 0,
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
        self.initial_bounds = initial_bounds
        self.threshold = threshold
        self.logger = logger
        self.blocks = []
        self.X = None
        self.y = None
        self.total_blocks=0
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
            X (torch.Tensor): A point in the input space.

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

            self.kdtree = KDTree(X, device=self.device)
            X=X[:,:-1]
            treeNodes=self.kdtree.get_leaves()

            self.blocks = np.empty(len(treeNodes), dtype=object)  

            for index in range(len(treeNodes)):
                 block_size=treeNodes[index].bounds[0]-treeNodes[index].bounds[1]
                 self.total_blocks+=1
                 self.blocks[(index,)] = PartitionBlock(
                    treeNodes[index].bounds[1],  
                    (index,), 
                    block_size,
                    device=self.device
                 )

                 treeNodes[index].block=self.blocks[(index,)] #Define node blocks

        else:  #logic to add points to the tree
            for row in X:
                self.kdtree.add_point(row[:-1],row[-1:])
            treeNodes=self.kdtree.get_leaves()
            if len(treeNodes)>self.total_blocks:    #If a node was split, update self.blocks
                self.blocks = np.empty(len(treeNodes), dtype=object)  
                
                for index in range(len(treeNodes)):
                    treeNodes[index].block.block_index=(index,)
                    self.blocks[(index,)]=treeNodes[index].block
                self.total_blocks=len(treeNodes)


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

                    self.logger.debug(
                        f"Created random vector for block at index {index}, created sparse vector h: {block.h}"
                    )

                block.target = torch.stack([value * block.amplitude for value in block.y])
                if block.target.dim() == 1:
                    block.target = block.target.unsqueeze(-1)
                block.target = block.target.detach()

    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Maps input points to their respective sub-blocks.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target data corresponding to the input points.
        """
        treeNodes=self.kdtree.get_leaves()
        for i in range(len(treeNodes)):
            if all(torch.equal(a, b) for a, b in zip(treeNodes[i].block.X,list(treeNodes[i].data.unbind(dim=0)))): 
                treeNodes[i].block.X=list(treeNodes[i].data.unbind(dim=0))
            if all(torch.equal(a, b) for a, b in zip(treeNodes[i].block.X,list(treeNodes[i].data.unbind(dim=0)))): 
                treeNodes[i].block.y=list(treeNodes[i].y.unbind(dim=0))

    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """
        print("AGREGANDO:",X,y)
        
        

        Xy=torch.cat((X,y), dim=1)
        self._update_block_arrangement(Xy)
        self._map_points(X, y)
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block
        self._configure_blocks() # Normalize y value and initialize h in each block 

    def init_ista_per_block(
        self,
        n_functions: int,
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
        y = y.to(device)

        # Copy blocks without X and y, nor their normalized versions, positions, etc.
        test_blocks = deepcopy(self.blocks) # "deepcopy" is not that deep...
        test_kdtree = deepcopy(self.kdtree) # "deepcopy" is not that deep...
 
        # Save temporarily current blocks
        temp_current_blocks = self.blocks
        temp_current_kdtree = self.kdtree
        self.blocks = test_blocks
        self.kdtree = test_kdtree
        y = y.unsqueeze(1)
        self.add_points(X, y)

        # Retrieved mapped test blocks and return to usual blocks
        test_active_blocks = self.retrieve_active_blocks()
        self.blocks = temp_current_blocks
        self.kdtree = temp_current_kdtree

        return test_active_blocks
