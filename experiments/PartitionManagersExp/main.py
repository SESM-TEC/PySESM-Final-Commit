"""Entry point for PartitionManagers experiments.

This module runs a single experiment using a Hydra configuration located in
`conf/config.yaml` (hydra decorator handles injecting the runtime config).
"""

import logging

# Third-party imports
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Local imports
import fun
from src.core import train_stream_experiment

import wandb




FUNCTIONS = {
    "function_zhou": fun.function_zhou,
    "function_zakharov": fun.function_zakharov,
    "function_styblinski_tang": fun.function_styblinski_tang,
}


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Run a single experiment using the provided configuration `cfg`.

    This function is executed via the Hydra decorator so it's intentionally
    declared with a single `cfg` parameter (Hydra will provide it at runtime).
    """

    logger = logging.getLogger("BSESM")

    # Reproducibilidad
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Validar función objetivo
    if cfg.dataset.name not in FUNCTIONS:
        raise ValueError(f"Función {cfg.dataset.name} desconocida")
    func_obj = FUNCTIONS[cfg.dataset.name]

    # Nombre del Run: Solo Dimensión y Dataset (los métodos van dentro)
    run_name = f"Stream_{cfg.dataset.name}_D{cfg.dim}"

    # Inicializamos WandB para este Dataset+Dim
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        group=f"Func_{cfg.dataset.name}",
        job_type=f"Dim_{cfg.dim}"
    )

    try:
        train_stream_experiment(cfg, logger, func_obj)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Fallo crítico en %s: %s", run_name, e)
        wandb.finish(exit_code=1)
        raise e

    wandb.finish()


if __name__ == "__main__":
    # The Hydra-decorated `main` is invoked without arguments at runtime.
    # Pylint cannot see Hydra's injection, so silence the specific warning.
    main()  # pylint: disable=no-value-for-parameter
