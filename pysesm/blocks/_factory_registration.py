"""
Block Manager Factory Registration.

Registers the available Block Manager strategies (Uniform, Adaptive) with the
BlockManagerFactory to enable dynamic instantiation via configuration or ID.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

# Import the concrete classes
from .UniformPartitionManager import UniformPartitionManager
from .AdaptativePartitionManager import AdaptativePartitionManager
# Import the factory
from ..factories import BlockManagerFactory

# Perform registrations
BlockManagerFactory.register("uniform_partition_manager", UniformPartitionManager)
BlockManagerFactory.register("adaptative_partition_manager", AdaptativePartitionManager)
