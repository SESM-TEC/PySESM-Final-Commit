"""
Dictionary Factory Registration.

Registers the available dictionary types (e.g., Gaussian) with the DictFactory
to enable dynamic instantiation via configuration or ID.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

# Import the concrete classes
from .GaussianDictLayer import GaussianDictLayer

# Import the factory
from ..factories import DictFactory

# Perform registrations
DictFactory.register("gaussian", GaussianDictLayer)
