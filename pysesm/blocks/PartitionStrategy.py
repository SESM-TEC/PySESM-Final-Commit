"""
Copyright (C) 2025 Tecnológico de Costa Rica

KD-Tree-based Partition Strategy Implementation

Defines a partitioning strategy using a KD-tree for spatial subdivision of
data into balanced regions. The KDPartitionStrategy provides efficient
insertion and lookup of points and partitions using recursive splitting
based on data variance.

Author: Hender Valdivia
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from pysesm.base_types import BaseConfig


@dataclass(kw_only=True)
class PartitionStrategyConfig(BaseConfig):
    """
    Base configuration class for all partition strategies.

    Serves as the parent for specific configurations (e.g., KD-tree, uniform, adaptive),
    providing a consistent structure for parameter validation and inheritance.
    """

class PartitionStrategy(ABC):
    """
    Abstract base class defining the interface for all partitioning strategies.

    Partition strategies control how data is divided into subregions (partitions)
    that can be used for distributed learning, local modeling, or adaptive sampling.
    """

    def __init__(self, config: PartitionStrategyConfig):
        """
        Initialize the partition strategy with a given configuration.

        Args:
            config (PartitionStrategyConfig): The configuration object defining
                                              partition parameters and constraints.
        """
        self.config = config
        self.built=False

    @abstractmethod
    def build(self, X, y):
        """Builds the partitioning structure based on the provided data."""

    @abstractmethod
    def add_points(self, X, y):
        """Adds new data points to the partitioning structure."""

    @abstractmethod
    def get_partitions(self):
        """Returns the current partitions as a list of PartitionBlock objects."""

    @abstractmethod
    def find_partition_for_point(self, x):
        """Finds and returns the partition corresponding to a given data point."""
    
    @abstractmethod
    def get_all_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns all data points (X, y) held by the strategy."""