# pysesm/base_types.py

from dataclasses import dataclass
from typing import Union, List
import torch

@dataclass
class BaseConfig:
    """Base class for all configuration objects in pysesm."""
    pass

# Data in most functions is exchanged as:
# - torch.Tensor (dim=2 for sigle blocks or 3 for batched blocks)
# - nested_tensor (for irregular sized batches)
# - List[torch.Tensor] as an alternative to nested_tensor
TensorBatch = Union[torch.Tensor, torch.nested.nested_tensor, List[torch.Tensor]]
