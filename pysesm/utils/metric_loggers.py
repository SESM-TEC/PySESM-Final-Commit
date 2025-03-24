import wandb
def generic_hook(layer_name, info, logger=None, project_name=None, use_wandb=False):
    """
    Generic logging hook that can output to console, logger, and/or wandb
    
    Args:
        layer_name: Name of the layer being logged
        info: Dictionary of metrics to log
        logger: Logger instance from setup_logger()
        use_wandb: Whether to use Weights & Biases
        project_name: Name of the project in Weights & Biases
    """
    # Console output
    log_message = f"{layer_name} - " + ", ".join(f"{k}: {v}" for k, v in info.items())
    print(log_message)
    
    # Python logger output
    if logger:
        logger.info(log_message)
    
    # Weights & Biases output
    if use_wandb:
        if not wandb.run:
            wandb.init(project=project_name)
        wandb.log({f"{layer_name}/{k}": v for k, v in info.items()})