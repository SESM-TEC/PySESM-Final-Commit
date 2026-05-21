"""Hydra entry point: generate the validation CSV for the ring oscillator
experiment (100k muestras por defecto).

Uso (desde experiments/DoE):
    python generate_validation_dataset.py

Sobrescribir por línea de comando, p. ej. para una corrida rápida:
    python generate_validation_dataset.py validation.n_samples=500 \
                                          validation.csv_path=val_small.csv

El generador es resumible: si el CSV ya existe, continúa donde quedó usando
exactamente los mismos parámetros (mismo seed → mismas filas).
"""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from generate_ring_osc_dataset import generate_ring_osc_dataset


def _resolve_csv_path(csv_path: str) -> str:
    """Hydra cambia el cwd → resolvemos el CSV contra el directorio original."""
    if os.path.isabs(csv_path):
        return csv_path
    try:
        base = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base = os.getcwd()
    return os.path.join(base, csv_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("RingOscValidation")
    logger.info("Configuración:\n%s", OmegaConf.to_yaml(cfg))

    osc = cfg.oscillator
    val = cfg.validation

    csv_path = _resolve_csv_path(val.csv_path)
    plot_dir = _resolve_csv_path(val.plot_dir) if val.get("plot_every", 0) else val.plot_dir

    logger.info("Generando validación: %d muestras -> %s", val.n_samples, csv_path)

    generate_ring_osc_dataset(
        n_samples=int(val.n_samples),
        csv_path=csv_path,
        seed=int(val.seed),
        plot_every=int(val.get("plot_every", 0)),
        plot_dir=plot_dir,
        n_stages=int(osc.n_stages),
        nmos_params=OmegaConf.to_container(osc.nmos, resolve=True),
        pmos_params=OmegaConf.to_container(osc.pmos, resolve=True),
        ranges=OmegaConf.to_container(osc.ranges, resolve=True),
        resume=True,
        flush_every=10,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
