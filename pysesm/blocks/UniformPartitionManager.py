from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.models.ISTALayer import ISTALayer
from copy import deepcopy
from typing import Union, Callable, Iterator, Dict
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.enums.HookTypeEnum import HookType
import logging
import numpy as np
import torch
DEFAULT_BLOCKS_PER_DIM = 4

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

class UniformPartitionManager(BlockManager):
    """
    A class to manage a uniform partitioning of the input space into blocks.

    The UniformPartitionManager divides the space into uniformly sized blocks, assigns points to these blocks,
    and configures or adjusts local models within each block.

    Args:
        logger (logging.Logger): Logger instance for recording messages and warnings.
        T (torch.Tensor): A tensor defining the number of blocks per dimension.
        n_functions (int): Number of functions or features of interest in each block.
        initial_bounds (np.ndarray, optional): Initial bounds for the partitioning, shaped as (2, n_features).
            - The first row contains the lower bounds.
            - The second row contains the upper bounds.
            If not provided, it is automatically calculated from the input data.
        threshold (float, optional): Threshold for determining block activity (default is 0).
    """

    def __init__(
        self,
        logger: logging.Logger,
        T: Union[torch.Tensor, int, None],
        n_functions,
        initial_bounds: np.ndarray = None,
        threshold: float = 0,
        device_manager=None,
        hook_manager=None
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
        #torch.set_default_device(device)
        self.T = T
        self.n_functions = n_functions
        self.initial_bounds = initial_bounds
        self.threshold = threshold
        self.logger = logger
        self.blocks = None
        self.block_size = None
        self.X = None
        self.y = None
        self.device_manager = device_manager
        self.hook_manager = hook_manager
        self._vectorized_normalization = np.vectorize(lambda x: x.normalize())
    
    def _ista_hook(self, info: Dict) -> None:
        """
        Hook for ISTALayer to log or store data.
        """
        if self.hook_manager:
            self.hook_manager.log_hook_data(HookType.ISTALAYER, info)

    def _find_block(self, x: torch.Tensor) -> Union[PartitionBlock, None]:
        """
        Finds the block corresponding to a given point.

        Args:
            X (torch.Tensor): A point in the input space.

        Returns:
            PartitionBlock or None: The block containing the point, or None if not found.
        """
        # TODO: Find efficient way to find block (currently being worked on by Joshua)
        for index in np.ndindex(self.blocks.shape):
            block: PartitionBlock = self.blocks[index]
            if block.is_point_in_block(x):
                return block

        logging.warning("Could not find a block for point {}", x)
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
        if self.T is None:
            self.T = torch.tensor([DEFAULT_BLOCKS_PER_DIM for _ in range(X.dim())], device=device)
        elif type(self.T) is int:
            self.T = torch.tensor([self.T for _ in range(X.dim())], device=device)

        # When no points and no blocks have been created
        if self.blocks is None:
            # Check for user given initial bounds
            if self.initial_bounds is None:
                self.initial_bounds = torch.vstack(
                    [torch.min(X, dim=0).values, torch.max(X, dim=0).values]
                ).to(device)   # Calculates the range covered by the X vector
                logging.warning(
                    "[UniformPartitionManager] No initial bounds provided, using calculated one {}".format(
                        self.initial_bounds
                    )
                )

            # Space to be partitioned
            delta = self.initial_bounds[1] - self.initial_bounds[0]
            self.block_size = torch.div(delta, self.T).to(device)
            self.blocks = np.empty(self.T.cpu().numpy(), dtype=PartitionBlock)#----------

            for index in np.ndindex(self.blocks.shape):
                self.blocks[index] = PartitionBlock(
                    self.initial_bounds[0].to(device), 
                    index, 
                    self.block_size.to(device),
                    device=device
                )
        else:
            new_max_x = torch.max(X).to(device)
            new_min_x = torch.min(X).to(device)

    def _configure_blocks(self, init_h: bool = True):
        """
        Configures each block with its expected squeeze factor and initializes sparse vectors if required.
        Internally, the .y attribute will hold the raw original y data, and .target the normalized version.

        Args:
            init_h (bool, optional): Whether to initialize the sparse vector `h` for each block (default is True).
        """
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)

        for index in np.ndindex(self.blocks.shape):
            block = self.blocks[index]
            if len(block.y) != 0:
                
                if init_h:
                    # Squeeze should be computed only with training data
                    block.amplitude = squeeze_factor(block.y)

                    block.h = torch.nn.Parameter(
                        torch.rand(self.n_functions,1,device=device), requires_grad=True
                    )
                    block.h.data /= block.h.data.sum()

                    self.logger.debug(
                        f"Created random vector for block at index {index}, created sparse vector h: {block.h}"
                    )

                block.target = torch.stack([value * block.amplitude for value in block.y]).to(device)
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
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        X = X.to(device)
        y = [yi.to(device) for yi in y]

        for i in range(X.shape[0]):
            selected_block = self._find_block(X[i])
            if selected_block is not None:
                selected_block.new_point(X[i], y[i], i)

        # Fix dimensions at the end
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
        device = self.device_manager.get_device(DeviceTarget.PARTITION_MANAGER)
        X = X.to(device)
        y = y.to(device)

        self._update_block_arrangement(X)
        self._map_points(X, y) # Sends points to their respective blocks
        self._vectorized_normalization(self.blocks) # Normalize X coords in each block
        self._configure_blocks() # Normalize y value and initialize h in each block


    def init_ista_per_block(
        self,
        n_functions: int,
        ista_alpha: float,
        ista_lambd: float,
        evaluation_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ista_optimizer: Callable[[Iterator[torch.nn.Parameter],float], torch.optim.Optimizer],
        initial_h: torch.Tensor = None
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
                parameter_hook=self._ista_hook if self.hook_manager and self.hook_manager.active_hooks[HookType.ISTALAYER] else None
            )
            if initial_h is not None:
                block.ista_layer.setup(initial_h) 

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

        # Save temporarily current blocks
        temp_current_blocks = self.blocks
        self.blocks = test_blocks

        # Map and normalize points into test blocks
        self._map_points(X, y)

        # This works because it just adjust coordinates to the block relative position
        self._vectorized_normalization(self.blocks)

        # This only applies the already computed squeeze factor.
        self._configure_blocks(init_h=False)

        # Retrieved mapped test blocks and return to usual blocks
        test_active_blocks = self.retrieve_active_blocks()
        self.blocks = temp_current_blocks

        return test_active_blocks