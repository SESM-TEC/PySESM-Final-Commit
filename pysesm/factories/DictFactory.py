"""
Dictionary Layer Factory.

Provides a factory to produce dictionary layer instances, such as the Gaussian
Dictionary Layer, using the generic factory pattern.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
from .GenericFactory import GenericFactory
from ..dictionaries.DictBaseLayer import DictBaseLayer, DictConfig


class DictFactory(GenericFactory[DictBaseLayer, DictConfig]):
    """
    Factory class for creating dictionary layer instances.
    
    This factory provides a centralized way to create different dictionary
    layers with flexible registration and instantiation methods. Dictionaries
    can be created either by a string identifier or directly from a
    configuration object.
    
    Usage examples:
    
    # Registration (in dictionary implementation file):
    @DictFactory.register("gaussian")
    class GaussianDictLayer(DictBaseLayer):
        CONFIG_CLASS = GaussianDictConfig
        # ...implementation...
    
    # Creation by ID:
    dict_layer = DictFactory.create("gaussian", config=gaussian_config)
    
    # Creation by config type:
    dict_layer = DictFactory.create(gaussian_config)
    """
    
    def __init__(self):
        super().__init__(product_name="dictionary",
                         config_name="dictionary config")
