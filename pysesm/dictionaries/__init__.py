"""
Dictionaries Module.

Provides the base classes and concrete implementations for dictionary layers
(like GaussianDictLayer) that learn the basis functions within the SESM
framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from .DictBaseLayer import DictBaseLayer, DictConfig
from .GaussianDictLayer import GaussianDictLayer, GaussianDictConfig
