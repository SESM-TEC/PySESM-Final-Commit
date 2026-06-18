"""
Sparse Coding Factory Registration.

Registers the available sparse coding algorithms (ISTA, FISTA, ADMM) with the
SparseCodingFactory to enable dynamic instantiation via configuration or ID.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

# Import the concrete classes
from .ISTALayer import ISTALayer
from .FISTALayer import FISTALayer
from .ADMMLayer import ADMMLayer

# Import the factory
from ..factories import SparseCodingFactory

# Perform registrations
SparseCodingFactory.register("classic_ista", ISTALayer)
SparseCodingFactory.register("fista", FISTALayer)
SparseCodingFactory.register("admm", ADMMLayer)
