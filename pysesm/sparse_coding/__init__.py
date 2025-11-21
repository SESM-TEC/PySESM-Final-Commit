"""
Sparse Coding Module.

Provides base classes and concrete implementations of sparse coding algorithms
(ISTA, FISTA, ADMM) used to learn the sparse coefficients (h) within the SESM
framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from .SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig

from .ISTALayer import ISTALayer, ISTAConfig
from .FISTALayer import FISTALayer, FISTAConfig
from .ADMMLayer import ADMMLayer, ADMMConfig
from .sparse_coding_utils import StepSizeMethod, soft_threshold, calculate_step_size
