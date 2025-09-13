# pysesm/base_types.py

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TypeAlias
import torch

@dataclass
class BaseConfig:
    """Base class for all configuration objects in pysesm."""
    device: str | None = None

# Data in most functions is exchanged as:
# - torch.Tensor (dim=2 for sigle blocks or 3 for batched blocks)
# - nested_tensor (for irregular sized batches)
# - List[torch.Tensor] as an alternative to nested_tensor
TensorBatch: TypeAlias = torch.Tensor | Sequence[torch.Tensor]

class TensorProxy:
    """
    A proxy for a single tensor that manages its placement across different
    devices, caching them for efficiency.

    This class holds a reference to a source tensor and lazily creates
    and caches copies on other devices as requested. It also delegates
    common attributes to the source tensor to act as a partial drop-in replacement.
    """
    _source: torch.Tensor
    _cache: dict[torch.device, torch.Tensor]

    def __init__(self, tensor: torch.Tensor):
        """
        Initializes the proxy with the source tensor.
        """
        self._source = tensor
        self._cache = {}

    def get_for_device(self, device: str | torch.device) -> torch.Tensor:
        """
        Returns the tensor on the specified device.
        """
        if isinstance(device, str):
            device = torch.device(device)
            
        if device == self._source.device:
            return self._source
        
        if device not in self._cache:
            self._cache[device] = self._source.to(device)
            
        return self._cache[device]

    @property
    def shape(self):
        return self._source.shape


    def dim(self):
        return self._source.dim()