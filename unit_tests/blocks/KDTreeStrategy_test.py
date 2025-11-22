"""
KD-Tree Strategy Tests.

Tests for the KD-Tree partitioning strategy, ensuring correct integration
between the strategy interface and the underlying tree structure.

Author: Hender Valdivia
Copyright (c) 2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import pytest
import torch

from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.KDTree import KDTree
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.PartitionBlock import PartitionBlock

@pytest.fixture(scope="module")
def common_device():
    """Shared device fixture for consistency across tests."""
    return "cpu"


@pytest.fixture
def create_kd_strategy(common_device):
    """Factory fixture for KDPartitionStrategy instances."""
    def _creator(X, y, maxNodeSize=5):
        config = KDTreeStrategyConfig(
            maxNodeSize=maxNodeSize,
            data_wrapper=SESMData,
            device=common_device
        )
        strategy=KDTreeStrategy(config)
        strategy.build(X,y)
        return strategy
    return _creator


def test_build_initialization(create_kd_strategy):
    """
    Ensures that KDPartitionStrategy builds a valid KDTree during initialization.
    """
    torch.manual_seed(0)
    X = torch.randn(50, 4)
    y = torch.randn(50, 1)
    strategy = create_kd_strategy(X, y)

    # Tree exists and structure is valid
    assert isinstance(strategy.kdtree, KDTree)
    assert strategy.kdtree.root is not None
    assert strategy.kdtree.maxNodeSize == strategy.config.maxNodeSize


def test_add_points_integration(create_kd_strategy):
    """
    Verifies that adding new points updates the KDTree structure correctly.
    """
    torch.manual_seed(1)
    X = torch.randn(40, 3)
    y = torch.randn(40, 1)
    strategy = create_kd_strategy(X, y)

    X_new = torch.rand(5, 3)
    y_new = torch.rand(5, 1)

    # Add points to strategy
    strategy.add_points(X_new, y_new)

    # Ensure data appears in KDTree leaves
    leaves = strategy.kdtree.get_leaves()
    total_points = sum([leaf.Data.X.size(0) for leaf in leaves])
    assert total_points == X.size(0) + X_new.size(0)


def test_get_partitions(create_kd_strategy):
    """
    Checks that get_partitions returns blocks consistent with KDTree leaves.
    """
    torch.manual_seed(2)
    X = torch.randn(30, 2)
    y = torch.randn(30, 1)
    strategy = create_kd_strategy(X, y)

    partitions = strategy.get_partitions()
    leaves = strategy.kdtree.get_leaves()

    assert isinstance(partitions, list)
    assert len(partitions) == len(leaves)
    for p in partitions:
        assert isinstance(p, strategy.config.data_wrapper)


def test_find_partition_for_point(create_kd_strategy):
    """
    Ensures that a query point maps to a valid KDTree leaf partition.
    """
    torch.manual_seed(3)
    X = torch.randn(60, 3)
    y = torch.randn(60, 1)
    strategy = create_kd_strategy(X, y)

    x_query = torch.rand(3)
    block = strategy.find_partition_for_point(x_query)

    assert block is not None
    # Optional: block should match a known leaf’s block
    node = strategy.kdtree._find_node(x_query)
    assert block == node.Data.block
