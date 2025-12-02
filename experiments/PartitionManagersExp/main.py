import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
import wandb
import fun
from src.core import train_stream_experiment  # <--- CAMBIO AQUÍ

FUNCTIONS = {
    "function_zhou": fun.function_zhou,
    "function_zakharov": fun.function_zakharov,
    "function_styblinski_tang": fun.function_styblinski_tang
}

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    
    logger = logging.getLogger("BSESM_Stream")
    
    if cfg.dataset.name not in FUNCTIONS:
        raise ValueError(f"Función {cfg.dataset.name} desconocida")
    func_obj = FUNCTIONS[cfg.dataset.name]

    # Nombre del Run (Ya no lleva n_samples porque es dinámico)
    run_name = f"{cfg.method.name}_{cfg.dataset.name}_D{cfg.dim}"
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        group="Stream_Study"
    )

    
    train_stream_experiment(cfg, logger, func_obj) # <--- LLAMADA
    # except Exception as e:
    #     logger.error(f"Fallo en {run_name}: {e}")
    #     wandb.finish(exit_code=1)
    #     raise e # Re-raise para ver el traceback si falla
    
    wandb.finish()

if __name__ == "__main__":
    main()