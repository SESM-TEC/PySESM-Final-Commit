"""Entry point del experimento Ring Oscillator — SESM.

Flujo:
    1. Verifica que el CSV de validación (cfg.validation.csv_path) exista.
       Si no, sugiere correr `generate_validation_dataset.py`.
    2. Lanza el barrido n_runs × training_sizes × methods con BSESM.
       Cada run genera/usa SU PROPIO CSV de entrenamiento (seed distinto),
       garantizando independencia estadística entre runs.

Uso (desde experiments/DoE):
    python main.py
    python main.py n_runs=3 training_sizes=[200,500,1000] wandb.mode=disabled
"""

import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

from src.core import train_ring_osc_experiment


def _resolve(p: str) -> str:
    if os.path.isabs(p):
        return p
    try:
        base = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base = os.getcwd()
    return os.path.join(base, p)


def _check_validation_csv(cfg: DictConfig, logger: logging.Logger) -> None:
    path = _resolve(cfg.validation.csv_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nNo se encontró el CSV de validación en {path}\n"
            f"Genéralo primero ejecutando (desde experiments/DoE):\n"
            f"    python generate_validation_dataset.py\n"
            f"Para una corrida rápida con menos muestras:\n"
            f"    python generate_validation_dataset.py validation.n_samples=2000\n"
        )
    logger.info("Validación OK: %s", path)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("RingOscSESM")
    logger.info("Configuración:\n%s", OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    _check_validation_csv(cfg, logger)

    run_name = f"RingOsc_{cfg.target}_nRuns{cfg.n_runs}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit="finish_previous",
        group=f"Target_{cfg.target}",
        job_type="ring_osc",
    )

    try:
        train_ring_osc_experiment(cfg, logger)
    except Exception as exc:
        logger.exception("Fallo en %s: %s", run_name, exc)
        wandb.finish(exit_code=1)
        raise

    wandb.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter