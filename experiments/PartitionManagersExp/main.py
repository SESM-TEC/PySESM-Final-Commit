"""Entry point for PartitionManagers experiments.

This module runs a single experiment using a Hydra configuration located in
`conf/config.yaml` (hydra decorator handles injecting the runtime config).
"""

import logging

# Third-party imports
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import wandb

# Local imports (project)
from src.core import train_one_run
import fun


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
        raise ValueError(f"Función {cfg.dataset.name} no existe en fun.py")

    func_obj = FUNCTIONS[cfg.dataset.name]

    # WandB: inicializar
    run_name = f"{cfg.method.name}_{cfg.dataset.name}_D{cfg.dim}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        group="BSESM_Sweep",
    )

    try:
        logger.info("=== RUN: %s ===", run_name)
        res = train_one_run(cfg, logger, func_obj)
        logger.info("Done. MSE: %.5f", res["MSE"])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Use logger.exception to include traceback and keep lazy formatting
        logger.exception("Fallo: %s", e)
        wandb.finish(exit_code=1)

    # finalizar WandB cuando todo salió bien
    wandb.finish()


if __name__ == "__main__":
    # The Hydra-decorated `main` is invoked without arguments at runtime.
    # Pylint cannot see Hydra's injection, so silence the specific warning.
    main()  # pylint: disable=no-value-for-parameter
