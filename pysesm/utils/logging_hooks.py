"""
Logging Hooks.

Provides helper callback functions to log metrics to the console, standard
python loggers, and Weights & Biases (WandB). These are intended to be used
as hooks within the SESM model training loop.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import logging
from typing import Any

try:
    import wandb
except ImportError:
    # Handle case where wandb is not installed (e.g. core install)
    wandb = None


def log_to_console(layer_name: str, info: dict[str, Any]) -> None:
    """
    Hook to log metrics directly to stdout.

    Args:
        layer_name (str): Name of the layer/component reporting data.
        info (dict): Dictionary of metrics to log.
    """
    log_message = f"{layer_name} - " + ", ".join(f"{k}: {v}" for k, v in info.items())
    print(log_message)


def log_to_WB(
    layer_name: str,
    info: dict[str, Any],
    logger: logging.Logger | None = None,
    project_name: str | None = None
) -> None:
    """
    Hook to log metrics to a Python logger and/or Weights & Biases.

    This function is designed to be wrapped in a lambda and passed to
    SESM hooks (e.g., `sesm_hook` or `parameter_hook`).

    Args:
        layer_name (str): Name of the layer/component reporting data.
        info (dict): Dictionary of metrics to log.
        logger (logging.Logger, optional): Logger instance to write text logs to.
        project_name (str, optional): Name of the project in Weights & Biases
                                      (used for init if run doesn't exist).
    """
    # Format message for text logs
    log_message = f"{layer_name} - " + ", ".join(f"{k}: {v}" for k, v in info.items())

    # Python logger output
    if logger:
        logger.info(log_message)

    # Weights & Biases output
    if wandb:
        # Initialize run if not already active
        if wandb.run is None:
            wandb.init(project=project_name)
        
        # Prefix metrics with layer name for better organization in W&B UI
        wandb_metrics = {f"{layer_name}/{k}": v for k, v in info.items()}
        wandb.log(wandb_metrics)
