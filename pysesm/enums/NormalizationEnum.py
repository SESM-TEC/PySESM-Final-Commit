"""
Normalization Enumerations.

Defines the enumeration `NormalizationEnum` which specifies the available
normalization strategies (e.g., STANDARD, BLOCK_SCOPE, LP_NORM) used within
the SESM framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from enum import Enum


class NormalizationEnum(Enum):
    STANDARD = "standard"
    BLOCK_SCOPE = "block_scope"
    LP_NORM = "lp_norm"
