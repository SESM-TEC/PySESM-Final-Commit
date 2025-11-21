"""
Surrogate Function Module.

Provides classes and utilities for defining and managing surrogate functions,
such as Gaussian basis functions, used for approximation within the SESM
framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

from pysesm.functions.GaussianFunction import GaussianFunction
from pysesm.functions.SurrogateFunction import SurrogateFunction

__all__ = ["GaussianFunction", "SurrogateFunction"]
