"""
Data Container Interface.

Defines the abstract interface for data payloads managed by the KD-Tree.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.
"""
from abc import ABC, abstractmethod
import torch

class DataContainer(ABC):
    """
    Abstract base class for data wrappers used in KDTree nodes.
    Decouples the topology logic from the specific data tensor operations.
    """

    @abstractmethod
    def size(self) -> int:
        """Returns the number of samples in this container."""

    @abstractmethod
    def append(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Adds new data points to the container."""

    @abstractmethod
    def split(self) -> tuple["DataContainer", "DataContainer"]:
        """Splits the data into two new containers (left, right)."""

    @abstractmethod
    def clear_payload(self) -> None:
        """Clears heavy tensors from memory after splitting."""

    @abstractmethod
    def push_test_data_to_children(self, left_child: "DataContainer", right_child: "DataContainer") -> None:
        """Distributes testing/inference data to children nodes."""
        
