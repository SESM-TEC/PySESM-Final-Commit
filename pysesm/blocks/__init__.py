"""
Block Partitioning Module.

Provides classes and utilities for managing the partitioning of the input space
into manageable sub-regions (blocks), using strategies like uniform grids or
KD-trees (adaptive).

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from pysesm.blocks.BlockManager import BlockManager
from pysesm.blocks.UniformPartitionManager import UniformPartitionManager, UniformPartitionConfig
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionManager, AdaptativePartitionConfig
from pysesm.blocks.KDTree import KDTree 
from pysesm.blocks.Node import Node 
from pysesm.blocks.PartitionBlock import PartitionBlock

__all__ = ["PartitionBlock", "BlockManager", "UniformPartitionManager", "Node", "KDTree", "AdaptativePartitionManager"]

