from pysesm.blocks.PartitionBlock import PartitionBlock

from abc import ABC, abstractmethod
from typing import Union, Callable
from torch import Tensor
from numpy.typing import NDArray


class BlockManager(ABC):
    """
    Abstract base class for managing blocks within the SESM (Sparse-Encoded Surrogate Model) architecture.

    The BlockManager is responsible for maintaining and manipulating the blocks required by the model.
    It oversees the creation, modification, and organization of these blocks, which define partitions or regions
    of the input space used in the SESM framework.

    For example, a uniform partition manager might divide the input space evenly across dimensions, ensuring
    that all blocks are of equal size.

    Attributes:
        blocks (NDArray[PartitionBlock]):
            An array of `PartitionBlock` objects representing the individual blocks managed by this class.
            Each block corresponds to a distinct partition of the input space, with its structure and properties
            determined by the specific implementation.
    """

    blocks: Union[NDArray[PartitionBlock], None]

    @abstractmethod
    def __init__(self):
        """
        Abstract initializer for the BlockManager.

        This method must be implemented by subclasses to initialize the necessary attributes
        and ensure the proper configuration of the block manager. For example, subclasses may
        define specific block structures, partitioning methods, or data required for managing
        blocks.

        Note: This method should initialize any required attributes like `blocks` to prepare
        the subclass for its intended functionality.
        """
        pass

    @abstractmethod
    def _find_block(self, x: Tensor) -> Union[PartitionBlock, None]:
        """
        Abstract method to locate the block corresponding to the given point `x`.

        This method determines which block, if any, contains the point specified by `x` based on
        the current block configuration. If no matching block is found, the method returns `None`.

        Args:
            x (Tensor):
                A tensor representing the point for which the corresponding block needs to be identified.
                The dimensions and format of `x` must align with the block manager's configuration.

        Returns:
            Union[PartitionBlock, None]:
                The block containing the point `x`, or `None` if no matching block exists.

        Note:
            The implementation of this method is specific to the subclass and may depend on the
            block partitioning logic used (e.g., uniform partitioning, KD-tree partitioning, etc.).
        """
        pass

    @abstractmethod
    def _update_block_arrangement(self, X: Tensor) -> None:
        """
        Abstract method to revise and update the current block configuration based on the given set of points `X`.

        This method adjusts the block arrangement to ensure it can accommodate the provided data points, `X`,
        if the current SESM configuration permits such updates. The specific logic for updating the blocks
        (e.g., splitting, merging, or re-partitioning) is determined by the subclass implementation.

        Args:
            X (Tensor):
                A tensor containing the set of points or observations that may require adjustments
                to the block configuration. Each point should conform to the input feature space.

        Returns:
            None:
                This method does not return a value; it directly modifies the internal state
                of the block arrangement as needed.

        Note:
            The implementation of this method is dependent on the SESM configuration and the
            block partitioning logic used in the subclass. It may involve complex operations
            like adaptive partitioning or rebalancing based on the dataset distribution.
        """
        pass

    @abstractmethod
    def _map_points(self, X: Tensor, y: Tensor):
        """
        Abstract method that processes each data point `(x, y)` to assign it to the appropriate block.

        This method iterates over all points in `X` and their corresponding labels or outputs in `y`,
        locating the block to which each point belongs and assigning it to that block. If a block cannot
        be found for a specific point, the behavior depends on the prediction type set in the SESM
        configuration. Based on this configuration, the method may either:
        - Raise an error indicating the issue, or
        - Dynamically update the block arrangement to accommodate the point.

        Args:
            X (Tensor):
                A tensor of shape `(n_samples, n_features)` containing the input points to be mapped to blocks.
                Each row corresponds to one of the `n_samples` data points, with `n_features` dimensions per point.
                For example, a dataset with 100 samples, each having 5 features, would have a shape of `(100, 5)`.

            y (Tensor):
                A tensor of shape `(n_samples,)` or `(n_samples, output_dim)` containing the corresponding labels,
                outputs, or additional data associated with each point in `X`. The number of rows in `y` must match
                the number of samples in `X`. For instance, for a regression task with scalar outputs, `y` would
                have a shape of `(100,)`, while for a multi-output task with 3 outputs per sample, `y` would have
                a shape of `(100, 3)`.

        Returns:
            None:
                This method modifies the internal state of the block manager by assigning points to
                blocks or updating the block arrangement if necessary.

        Note:
            The implementation of this method depends on the SESM configuration, including how it
            handles unmapped points. Custom error handling or block adjustment logic should be
            implemented in the subclass.
        """
        pass

    @abstractmethod
    def add_points(self, X: Tensor, y: Tensor):
        pass

    @abstractmethod
    def init_ista_per_block(
        self,
        n_functions: int,
        ista_alpha: float,
        ista_momentum: float,
        ista_lambd: float,
        evaluation_func: Callable[[Tensor, Tensor], Tensor],
    ):
        pass

    @abstractmethod
    def retrieve_active_blocks(self):
        pass

    @abstractmethod
    def retrieve_test_active_blocks(self, X: Tensor, y: Tensor):
        pass
