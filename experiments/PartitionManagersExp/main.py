import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import fun
from src.core import train_stream_experiment

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
    except Exception as e:
        logger.error(f"Fallo crítico en {run_name}: {e}")
        wandb.finish(exit_code=1)
        raise e
    
    wandb.finish()

if __name__ == "__main__":
    main()