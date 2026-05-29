"""Hydra entry point: generate the validation CSV for the ring oscillator
experiment (100k muestras por defecto).

Uso (desde experiments/DoE):
    python generate_validation_dataset.py

Sobrescribir por línea de comando, p. ej. para una corrida rápida:
    python generate_validation_dataset.py validation.n_samples=500 \
                                          validation.csv_path=val_small.csv

El generador es resumible: si el CSV ya existe, continúa donde quedó usando
exactamente los mismos parámetros (mismo seed → mismas filas).

Seguimiento en W&B: cada fila simulada se loguea como métrica escalar (para
gráficas en vivo) y se acumula en una tabla incremental `validation_samples`,
de modo que el avance se puede monitorear desde otros dispositivos. Para
desactivarlo: `wandb.mode=disabled`.
"""

import logging
import math
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

from generate_ring_osc_dataset import generate_ring_osc_dataset

_TABLE_KEY = "validation_samples"
_TABLE_COLUMNS = [
    "index", "W_n", "W_p", "L", "Vdd", "C_load", "f_osc", "P_avg", "t_rise",
]


def _resolve_csv_path(csv_path: str) -> str:
    """Hydra cambia el cwd → resolvemos el CSV contra el directorio original."""
    if os.path.isabs(csv_path):
        return csv_path
    try:
        base = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base = os.getcwd()
    return os.path.join(base, csv_path)


def _build_wandb_logger(cfg, n_samples, logger):
    """Init W&B and return (row_callback, finish_fn).

    Returns (None, noop) when W&B is disabled. The callback logs per-row scalar
    metrics (live charts) and appends to an incremental table that can be
    watched from any device. It must never raise — the generator wraps it in a
    try/except, but we also guard internally so a single bad row is harmless.
    """
    def _noop():
        pass

    wcfg = cfg.wandb
    if str(wcfg.get("mode", "online")).lower() == "disabled":
        logger.info("W&B deshabilitado (wandb.mode=disabled).")
        return None, _noop

    val = cfg.validation
    table_every = max(int(val.get("wandb_table_every", 50)), 1)

    wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        mode=wcfg.mode,
        name=f"ValGen_n{n_samples}",
        config=OmegaConf.to_container(cfg, resolve=True),
        group="validation_dataset",
        job_type="dataset_generation",
        reinit="finish_previous",
    )

    # Incremental tables (wandb >= 0.18) only ship new rows on each re-log, so
    # we can refresh live. On older wandb we fall back to a single log at the end.
    try:
        table = wandb.Table(columns=_TABLE_COLUMNS, log_mode="INCREMENTAL")
        incremental = True
    except TypeError:
        table = wandb.Table(columns=_TABLE_COLUMNS)
        incremental = False
        logger.warning(
            "wandb sin tablas incrementales: la tabla se subirá solo al final. "
            "Actualiza wandb (>=0.18) para verla crecer en vivo."
        )

    state = {"done": 0, "failed": 0}

    def row_callback(i, row):
        state["done"] += 1
        if any(math.isnan(row[k]) for k in ("f_osc", "P_avg", "t_rise")):
            state["failed"] += 1

        table.add_data(
            i + 1, row["W_n"], row["W_p"], row["L"], row["Vdd"], row["C_load"],
            row["f_osc"], row["P_avg"], row["t_rise"],
        )

        log_dict = {
            "progress/index": i + 1,
            "progress/done": state["done"],
            "progress/failed": state["failed"],
            "sample/W_n": row["W_n"],
            "sample/W_p": row["W_p"],
            "sample/L": row["L"],
            "sample/Vdd": row["Vdd"],
            "sample/C_load": row["C_load"],
            "sample/f_osc": row["f_osc"],
            "sample/P_avg": row["P_avg"],
            "sample/t_rise": row["t_rise"],
        }
        if incremental and (i + 1) % table_every == 0:
            log_dict[_TABLE_KEY] = table
        wandb.log(log_dict)

    def finish_fn():
        try:
            wandb.log({_TABLE_KEY: table})  # final flush (incl. non-incremental)
        finally:
            wandb.finish()

    return row_callback, finish_fn


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("RingOscValidation")
    logger.info("Configuración:\n%s", OmegaConf.to_yaml(cfg))

    osc = cfg.oscillator
    val = cfg.validation

    csv_path = _resolve_csv_path(val.csv_path)
    plot_dir = _resolve_csv_path(val.plot_dir) if val.get("plot_every", 0) else val.plot_dir

    logger.info("Generando validación: %d muestras -> %s", val.n_samples, csv_path)

    row_callback, finish_fn = _build_wandb_logger(cfg, int(val.n_samples), logger)

    try:
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
            row_callback=row_callback,
        )
    finally:
        finish_fn()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
