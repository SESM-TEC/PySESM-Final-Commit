import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
import wandb
import fun
from src.core import train_one_run

FUNCTIONS = {
    "function_zhou": fun.function_zhou,
    "function_zakharov": fun.function_zakharov,
    "function_styblinski_tang": fun.function_styblinski_tang
}

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    # Logger
    logger = logging.getLogger("BSESM")

    # Reproducibilidad
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Validar función
    if cfg.dataset.name not in FUNCTIONS:
        raise ValueError(f"Función {cfg.dataset.name} no existe en fun.py")

    func_obj = FUNCTIONS[cfg.dataset.name]

    # WandB
    run_name = f"{cfg.method.name}_{cfg.dataset.name}_D{cfg.dim}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        group="BSESM_Sweep"
    )

    try:
        logger.info(f"=== RUN: {run_name} ===")
        res = train_one_run(cfg, logger, func_obj)
        logger.info(f"Done. MSE: {res['MSE']:.5f}")
    except Exception as e:
        logger.error(f"Fallo: {e}")
        wandb.finish(exit_code=1)

    wandb.finish()

if __name__ == "__main__":
    main()
