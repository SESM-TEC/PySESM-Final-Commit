"""
Batch Operations Utilities.

Provides helper functions for manipulating and processing batches of tensors
and nested structures (like lists or dictionaries of tensors) within the SESM
framework.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import torch
from pysesm.base_types import TensorBatch

def recursive_cat(tb_list: list[TensorBatch]):
    """
    Recursively concatenate a list of TensorBatch objects along dim=0 (batch dim).
    Works for dicts, sequences, or single tensors.
    """
    tb0 = tb_list[0]
    if isinstance(tb0, torch.Tensor):
        return torch.cat(tb_list, dim=0)
    elif isinstance(tb0, dict):
        return {k: recursive_cat([tb[k] for tb in tb_list]) for k in tb0}
    elif isinstance(tb0, (list, tuple)):
        # Preserve type (list or tuple)
        return type(tb0)(recursive_cat([tb[i] for tb in tb_list]) for i in range(len(tb0)))
    else:
        raise TypeError(f"Unsupported type for concatenation: {type(tb0)}")
