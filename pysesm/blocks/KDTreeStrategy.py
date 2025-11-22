"""
KD-Tree Partition Strategy Interface.

Implements a partitioning strategy based on a KD-tree spatial decomposition to
subdivide the input space into balanced regions/blocks.

Author: Hender Valdivia
Copyright (c) 2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from dataclasses import dataclass
import torch

from pysesm.blocks.PartitionStrategy import PartitionStrategy, PartitionStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks.KDTree import KDTree, dummyData


@dataclass(kw_only=True)
class KDTreeStrategyConfig(PartitionStrategyConfig):
    """
    Configuration class for KDPartitionStrategy.

    Attributes:
        maxNodeSize (int): Maximum number of points per KD-tree node before splitting.
        data_wrapper (callable): Class used to encapsulate data in each KD-tree node.
        device (torch.device or None): Device (CPU/GPU) for tensor storage and computation.
    """
    maxNodeSize: int = 500
    maxSplitsBeforeRestart: int = 5
    data_wrapper: callable = SESMData
    device: torch.device = None


class KDTreeStrategy(PartitionStrategy):
    """
    Partitioning strategy based on a KD-tree spatial decomposition.

    Uses a KDTree structure to recursively split data points along the dimension
    of greatest variance. Each leaf node represents a partition that can store
    and manage data independently.
    """

    CONFIG_CLASS = KDTreeStrategyConfig

    def __init__(self, config: KDTreeStrategyConfig):
        """
        Initializes the KDPartitionStrategy and builds the underlying KD-tree.

        Args:
            X (torch.Tensor): Data tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target tensor of shape (n_samples, 1).
            config (KDPartitionStrategyConfig): KD-tree-specific configuration parameters.
        """
        super().__init__(config)
        self.split_count: int = 0
        
    def build(self, X: torch.Tensor, y: torch.Tensor, test: bool = False):
        """
        Constructs the KD-tree using the input data.

        Args:
            X (torch.Tensor): Data tensor.
            y (torch.Tensor): Target tensor.
            test (bool): Indicates if the structure must be build for testing.
        """
        if not test:
            self.kdtree = KDTree(X, y, self.config.maxNodeSize, self.config.data_wrapper, self.config.device)
            for i, partition in enumerate(self.get_partitions()):
                partition.block.block_index=(i,)
        else:
            self.kdtree.root.Data.test_data=X
            self.kdtree.root.Data.test_y=y
            self.kdtree.root.Data.test_indices=torch.arange(X.size(0), device=X.device)
            return self.kdtree.splitDataInNodes_test()

    def add_points(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """
        Adds multiple points to the KD-tree incrementally.

        Each row from the combined (X, y) tensor is passed to the KDTree.add_point method.

        Args:
            X (torch.Tensor): New input data of shape (n_new_samples, n_features).
            y (torch.Tensor): Corresponding targets of shape (n_new_samples, 1).
        Returns:
            bool: Indicates whether the kdtree was split after adding more points or not
        """
        for row in torch.cat((X, y), dim=1):
            self.kdtree.add_point(row[:-1], row[-1:])

        if self.kdtree.split_after_add:
            self.kdtree.split_after_add=False
            self.split_count+=1

            if self.split_count>self.config.maxSplitsBeforeRestart:
                self.split_count = 0
                self._restart_kdtree()
            all_X, all_y = self.get_all_points()
            print(f"[KDTreeStrategy.add_points] get_all_points => {all_X.shape[0]} X points, {all_y.shape[0]} y points")
            return True
        all_X, all_y = self.get_all_points()
        print(f"[KDTreeStrategy.add_points-nonsplit] get_all_points => {all_X.shape[0]} X points, {all_y.shape[0]} y points")
        return False

    def _restart_kdtree(self):
        """
        Restarts the kdtree but keeps the same data.
        """
        treeNodes = self.kdtree.get_leaves()
        X = torch.cat([n.Data.X for n in treeNodes], dim=0)
        y = torch.cat([n.Data.y for n in treeNodes], dim=0)

        self.kdtree = None

        self.build(X, y)

    def get_partitions(self) -> list:
        """
        Retrieves all leaf partitions (blocks) from the KD-tree.

        Returns:
            list: List of partition blocks corresponding to KD-tree leaves.
        """
        leaves = self.kdtree.get_leaves()
        partitions = []
        for leaf in leaves:
            partitions.append(leaf.Data)
        return partitions

    def find_partition_for_point(self, x: torch.Tensor) -> PartitionBlock:
        """
        Finds the partition block corresponding to a given point.

        Args:
            x (torch.Tensor): 1D tensor representing the point coordinates.

        Returns:
            PartitionBlock: The partition block associated with the point.
        """
        return self.kdtree._find_node(x).Data.block

    def get_all_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregates and returns all points from the leaf nodes."""
        if not self.kdtree:
            return torch.empty(0, device=self.config.device), torch.empty(0, device=self.config.device)

        all_X = []
        all_y = []
        for leaf in self.kdtree.get_leaves():
            if leaf.Data.X is not None:
                all_X.append(leaf.Data.X)
                all_y.append(leaf.Data.y)
        
        if not all_X:
            # Handle case where tree is built but has no points (should not happen)
            return torch.empty(0, device=self.config.device), torch.empty(0, device=self.config.device)

        return torch.cat(all_X, dim=0), torch.cat(all_y, dim=0)

