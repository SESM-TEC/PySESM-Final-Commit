"""
Model Architectures.

Provides the main SESM model classes, including the base SESM implementation
and its specialized variants: Sequential (SSESM) and Batched (BSESM).

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from .SESM import SESM, SESMConfig
from .SSESM import SSESM
from .BSESM import BSESM
