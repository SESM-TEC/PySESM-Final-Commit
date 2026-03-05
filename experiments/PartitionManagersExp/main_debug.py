"""Entry point for PartitionManagers experiments.

Debug variant with memory profiling support.
"""

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import fun
from src.core_debug import train_stream_experiment

import wandb

try:
    from memory_profiler import profile
except ImportError:
    # Keep script executable even when memory_profiler is not installed.
    def profile(func):
        return func


FUNCTIONS = {
    "function_zhou": fun.function_zhou,
    "function_zakharov": fun.function_zakharov,
    "function_styblinski_tang": fun.function_styblinski_tang,
}


@profile
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Run a single experiment using the provided Hydra configuration."""
    logger = logging.getLogger("BSESM")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.dataset.name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {cfg.dataset.name}")
    func_obj = FUNCTIONS[cfg.dataset.name]

    run_name = f"Stream_{cfg.dataset.name}_D{cfg.dim}"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit="finish_previous",
        group=f"Func_{cfg.dataset.name}",
        job_type=f"Dim_{cfg.dim}",
    )

    try:
        train_stream_experiment(cfg, logger, func_obj)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Critical failure in %s: %s", run_name, e)
        wandb.finish(exit_code=1)
        raise e

    wandb.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
