from pysesm.blocks.PartitionBlock import PartitionBlock

from abc import ABC, abstractmethod
from typing import Union, Callable
from torch import Tensor
from numpy.typing import NDArray


class BlockManager(ABC):

    blocks: NDArray[PartitionBlock]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _find_block(self, X: Tensor) -> Union[PartitionBlock, None]:
        """
        This is a test
        Args:
            X:

        Returns:

        """
        pass

    @abstractmethod
    def _update_block_arrangement(self, X: Tensor) -> None:
        pass

    @abstractmethod
    def _configure_blocks(self, init_h: bool = True):
        pass

    @abstractmethod
    def _map_points(self, X: Tensor, y: Tensor):
        pass

    @abstractmethod
    def add_points(self, X: Tensor, y: Tensor):
        pass

    @abstractmethod
    def init_ista_per_block(self, n_functions: int, seed: int, ista_alpha: float, ista_lambd: float,
                            weight_decay: float,
                            evaluation_func: Callable[[Tensor, Tensor], Tensor]):
        pass

    @abstractmethod
    def retrieve_active_blocks(self):
        pass

    @abstractmethod
    def retrieve_test_active_blocks(self, X: Tensor, y: Tensor):
        pass
