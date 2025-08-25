from __future__ import annotations

import logging
import wandb



def log_to_console(layer_name: str, info: str):
    """
    Handles logging to console
    
    Args:
        layer_name (str): Name of the layer being logged
        info (str): Dictionary of metrics to log
    """
    log_message = f"{layer_name} - " + ", ".join(f"{k}: {v}" for k, v in info.items())
    print(log_message)

def log_to_WB(layer_name: str, info: str, logger: logging.Logger=None, project_name: str=None):
    """
    Generic logging hook that can output to logger and/or wandb
    
    Args:
        layer_name (str): Name of the layer being logged
        info (str): Dictionary of metrics to log
        logger (logging.Logger): Logger instance from setup_logger()
        project_name (str): Name of the project in Weights & Biases
    """
    # Console output
    log_message = f"{layer_name} - " + ", ".join(f"{k}: {v}" for k, v in info.items())

    # Python logger output
    if logger:
        logger.info(log_message)
    
    # Weights & Biases output
    if not wandb.run:
        wandb.init(project=project_name)
    wandb.log({f"{layer_name}/{k}": v for k, v in info.items()})
