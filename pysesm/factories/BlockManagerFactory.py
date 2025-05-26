'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Block Manager Factory

Provides a factory to produce block managers, like the uniform partition or
the adaptive partition.

Authors: The SESM Team 

License: 
'''

from .GenericFactory import GenericFactory
from ..blocks.BlockManager import BlockManager, BlockManagerConfig

class BlockManagerFactory(GenericFactory[BlockManager, BlockManagerConfig]):
    """Factory for creating BlockManager instances."""
    def __init__(self):
        super().__init__(product_name="block_manager", config_name="block manager config")
