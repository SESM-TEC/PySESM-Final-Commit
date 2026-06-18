"""
Factory Module.

Provides factory classes for dynamically creating and managing instances of
Dictionary Layers, Sparse Coding Layers, and Block Managers based on
configuration or string identifiers.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from .DictFactory import DictFactory
from .SparseCodingFactory import SparseCodingFactory
from .BlockManagerFactory import BlockManagerFactory

__all__ = [
    "DictFactory",
    "SparseCodingFactory",
    "BlockManagerFactory"]
