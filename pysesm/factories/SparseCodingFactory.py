'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Sparse Coding Factory

Provides a factory to produce sparse coding layers using the generic factory pattern.

Authors: The SESM Team 

License: 
'''

from .GenericFactory import GenericFactory
from ..sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig


class SparseCodingFactory(GenericFactory[SparseCodingBaseLayer, SparseCodingConfig]):
    """
    Factory class for creating sparse coding layer instances.
    
    This factory provides a centralized way to create different sparse coding layers
    with flexible registration and instantiation methods. Layers can be created either
    by a string identifier or directly from a configuration object.
    
    Usage examples:
    
    # Registration (in layer implementation file):
    @SparseCodingFactory.register("classic_ista")
    class ISTALayer(SparseCodingBaseLayer):
        CONFIG_CLASS = ISTAConfig
        # ...implementation...
    
    # Creation by ID:
    sparse_layer = SparseCodingFactory.create("classic_ista", config=ista_config)
    
    # Creation by config type:
    sparse_layer = SparseCodingFactory.create(ista_config)
    """
    
    def __init__(self):
        super().__init__(product_name="sparse_coding_layer", config_name="sparse coding config")
