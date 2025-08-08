# pysesm/base_types.py

from dataclasses import dataclass
from typing import Union, Sequence, TypeAlias
import torch

@dataclass
class BaseConfig:
    """Base class for all configuration objects in pysesm."""

# Data in most functions is exchanged as:
# - torch.Tensor (dim=2 for sigle blocks or 3 for batched blocks)
# - nested_tensor (for irregular sized batches)
# - List[torch.Tensor] as an alternative to nested_tensor
TensorBatch: TypeAlias = Union[torch.Tensor, Sequence[torch.Tensor]]
