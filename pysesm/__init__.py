# pysesm/__init__.py

__version__ = "0.1.0" # You can update this version number as needed

# Import common sub-packages
from . import models
from . import utils
from . import blocks
from . import factories
from . import device_manager
from . import enums
from . import functions
from . import sparse_coding
from . import dictionaries

# Import common base types
from .base_types import BaseConfig

# Define __all__ for explicit export on 'from pysesm import *'
__all__ = [
    "models",
    "utils",
    "blocks",
    "factories",
    "device_manager",
    "enums",
    "functions",
    "sparse_coding",
    "dictionaries",
    "BaseConfig",
    "__version__",
]
