"""
SESM Validation Utilities.

Provides validation functions to ensure data consistency and correctness before
training or inference, checking dimensions and compatibility between inputs and
model configurations.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
from pysesm.validation.sesm_validation import validate_sesm_partial_fit

__all__ = ["validate_sesm_partial_fit"]
