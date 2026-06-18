"""
Sparse Coding Factory.

Provides a factory to produce sparse coding layer instances, such as ISTA,
FISTA, or ADMM layers, using the generic factory pattern.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
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
