"""Hydra entry point: generate the validation CSV for the ring oscillator
experiment (100k muestras por defecto).

Uso (desde experiments/DoE):
    python generate_validation_dataset.py

Sobrescribir por línea de comando, p. ej. para una corrida rápida:
    python generate_validation_dataset.py validation.n_samples=500 \
                                          validation.csv_path=val_small.csv

El generador es resumible: si el CSV ya existe, continúa donde quedó usando
exactamente los mismos parámetros (mismo seed → mismas filas).

Seguimiento en W&B: cada fila simulada (entradas y salidas) se acumula en una
tabla incremental `validation_samples`, de modo que el avance se puede
monitorear desde otros dispositivos. No se generan métricas escalares ni
gráficas — el dataset se analiza luego con las funciones de EDA. Para
desactivar W&B: `wandb.mode=disabled`.
"""

import logging
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

    Returns (None, noop) when W&B is disabled. The callback appends each row
    (inputs AND outputs) to a single incremental table `validation_samples`
    that can be watched from any device. It logs NO scalar metrics/charts —
    the dataset is meant for downstream EDA, not for W&B plotting.
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
    # the table can be refreshed live without resending everything. On older
    # wandb we re-log the full table each flush (heavier — raise wandb_table_every).
    try:
        table = wandb.Table(columns=_TABLE_COLUMNS, log_mode="INCREMENTAL")
    except TypeError:
        table = wandb.Table(columns=_TABLE_COLUMNS)
        logger.warning(
            "wandb sin tablas incrementales: se re-sube la tabla completa en cada "
            "flush. Sube validation.wandb_table_every o actualiza wandb (>=0.18)."
        )

    def _cell(x):
        """Float for a W&B table cell. NaN is kept as a float (not None) so the
        column always has a numeric type and is never dropped — W&B hides table
        columns whose values are all None (e.g. when every sim in a batch fails).
        """
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    def row_callback(i, row):
        # One table with inputs AND outputs; no scalar metrics / charts.
        table.add_data(
            i + 1,
            _cell(row["W_n"]), _cell(row["W_p"]), _cell(row["L"]),
            _cell(row["Vdd"]), _cell(row["C_load"]),
            _cell(row["f_osc"]), _cell(row["P_avg"]), _cell(row["t_rise"]),
        )
        if (i + 1) % table_every == 0:
            wandb.log({_TABLE_KEY: table})

    def finish_fn():
        try:
            wandb.log({_TABLE_KEY: table})  # final flush of any remaining rows
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
