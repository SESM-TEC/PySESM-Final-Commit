"""
Block Manager Factory.

Provides a factory to produce block manager instances, such as the Uniform
Partition Manager or the Adaptive Partition Manager, using the generic factory pattern.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
from .GenericFactory import GenericFactory
from ..blocks.BlockManager import BlockManager, BlockManagerConfig

class BlockManagerFactory(GenericFactory[BlockManager, BlockManagerConfig]):
    """Factory for creating BlockManager instances."""
    def __init__(self):
        super().__init__(product_name="block_manager", config_name="block manager config")
