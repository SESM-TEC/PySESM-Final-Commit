"""
PySESM Library Initialization.

Exposes the main sub-packages, base classes, and factory registrations for the
Sparse-Encoded Surrogate Model (SESM) framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

__version__ = "0.1.0" # You can update this version number as needed

# Import common sub-packages
from . import models
from . import utils
from . import blocks
from . import factories
from . import enums
from . import functions
from . import sparse_coding
from . import dictionaries

# Import common base types
from .base_types import BaseConfig

# --- Factory Registrations (dedicated files to break import cycles) ---
# These imports are for their side-effect of running the registration code
# contained within them. They don't typically add names to pysesm's namespace.
from .sparse_coding import _factory_registration as _
from .dictionaries import _factory_registration as _
from .blocks import _factory_registration as _

# Define __all__ for explicit export on 'from pysesm import *'
__all__ = [
    "models",
    "utils",
    "blocks",
    "factories",
    "enums",
    "functions",
    "sparse_coding",
    "dictionaries",
    "BaseConfig",
    "__version__",
]
