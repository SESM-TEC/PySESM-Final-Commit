from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.KDTree import KDTree
from pysesm.blocks.Node import Node
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer
from pysesm.enums.DeviceTargetEnum import DeviceTarget

import logging
import torch
import numpy as np
from typing import Union, Callable, Iterator, Type

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

    def _find_block(self, x: torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the block corresponding to a given point.

        Args:
            X (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """
        return self.kdtree.find_block(x) #Check type, should be PartitionBlock its node

    def _update_block_arrangement(self, X: torch.Tensor) -> None:
        """
        Updates the arrangement of blocks based on the input data.

        This method initializes or adjusts the block arrangement and sizes, ensuring coverage of the input space.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        #device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        if self.kdtree is None:

            self.kdtree = KDTree(X)
            treeNodes=self.kdtree.get_leaves()

            self.blocks = np.empty(len(treeNodes), dtype=object)  # Creates an empty array of size 5

            for index in range(len(treeNodes)):
                 block_size=treeNodes[index].bounds[0]-treeNodes[index].bounds[1]
                 self.total_blocks+=1
                 self.blocks[(index,)] = PartitionBlock(
                    treeNodes[index].bounds[1], 
                    (index,), 
                    block_size
                 )

        else:  #Add logic to add points to the tree
            for row in X:
                self.kdtree.add_point(row)
                treeNodes=self.kdtree.get_leaves()
                if len(treeNodes)>self.total_blocks:
                    self.total_blocks=len(treeNodes)
                    for index in range(len(treeNodes)):
                        treeNodes[index].block.block_index=(index,)
                        self.blocks[(index,)]=treeNodes[index].block


    def _configure_blocks(self, init_h: bool = True):
        """
        Configures each block with its expected squeeze factor and initializes sparse vectors if required.
        Internally, the .y attribute will hold the raw original y data, and .target the normalized version.

        Args:
            init_h (bool, optional): Whether to initialize the sparse vector `h` for each block (default is True).
        """


    def _map_points(self, X: torch.Tensor, y: np.ndarray):
        """
        Maps input points to their respective sub-blocks.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target data corresponding to the input points.
        """
        treeNodes=self.kdtree.get_leaves()
        for i in range(len(treeNodes)):
            print(treeNodes[i].block)
            treeNodes[i].block.X=list(treeNodes[i].data.unbind(dim=0))


    def add_points(self, X: torch.Tensor, y: torch.Tensor):
        """
        Adds points to the blocks, updating the partitioning and configuration as needed.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
        """
        self._update_block_arrangement(X)
        self._map_points(X, y)
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block

        #self._vectorized_normalization(self.blocks) # Normalize X coords in each block
        #self._configure_blocks() # Normalize y value and initialize h in each block


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
        
    def retrieve_active_blocks(self):
        """
        Retrieves all active blocks in the partition. An active block is a block with at least one point mapped

        Returns:
            List[PartitionBlock]: A list of active blocks.
        """


    def retrieve_test_active_blocks(self, X, y):
        """
        Retrieves active blocks for testing purposes.

        Args:
            X (torch.Tensor): Test input data of shape (n_samples, n_features).
            y (torch.Tensor): Test target data of shape (n_samples,).

        Returns:
            List[PartitionBlock]: A list of active blocks corresponding to the test data.
        """
        return