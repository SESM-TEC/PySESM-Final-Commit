"""
Analysis of the ring oscillator DoE experiment results.

Generates three families of plots from `ring_osc_experiment_results.csv`:

  Scaling analysis      -> 01_scaling_<metric>.png
      Metric vs n_samples per method, mean over runs with 95% CI bands.

  Paired comparison     -> 02_paired_<metric>.png  (+ 02_paired_stats.txt)
      uniform vs kdtree on the SAME (run, size) data. Per-size paired
      differences (boxplot) and a Wilcoxon signed-rank test per size.

  Computational cost    -> 03_cost_<resource>.png
      train/test time and peak GPU/RAM vs n_samples per method.

Reads `ring_osc_experiment_results.csv` from the same directory as this
script. All figures are written under plots/analysis/.

Usage (from experiments/DoE):
    python analyze_results.py
    python analyze_results.py --target f_osc
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE     = Path(__file__).resolve().parent
CSV_PATH = HERE / 'ring_osc_experiment_results.csv'
PLOT_DIR = HERE / 'plots' / 'analysis'

# Métricas de error (espacio normalizado log-z-score y espacio original).
ERROR_METRICS = {
    'mse_norm': 'MSE (normalized)',
    'mae_norm': 'MAE (normalized)',
    'mae_orig': 'MAE (original units)',
}
# Recursos de cómputo: columna -> (etiqueta eje, escala y).
COST_METRICS = {
    'train_time':            ('Train time [s]', 'linear'),
    'test_time':             ('Test time [s]', 'linear'),
    'torch_peak_alloc_mb':   ('Torch peak alloc [MB]', 'linear'),
    'ram_used_mb_mean':      ('RAM used mean [MB]', 'linear'),
}
# Colores estables por método (cualquier método extra cae al ciclo por defecto).
METHOD_COLORS = {'uniform': 'tab:blue', 'kdtree': 'tab:orange'}


def _color(method, fallback_idx):
    return METHOD_COLORS.get(method, f'C{fallback_idx}')


def _mean_ci(values, confidence=0.95):
    """Media y semiancho del IC. Con <2 muestras el IC es 0."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    sem = stats.sem(values)
    half = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, float(half)


# --------------------------------------------------------------------------
# 1) Scaling analysis: métrica vs n_samples, media ± IC95% sobre runs.
# --------------------------------------------------------------------------
def scaling_analysis(df, methods, sizes, out_dir):
    for metric, label in ERROR_METRICS.items():
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, method in enumerate(methods):
            sub = df[df['method'] == method]
            means, halves, xs = [], [], []
            for size in sizes:
                vals = sub[sub['n_samples'] == size][metric].dropna().values
                if len(vals) == 0:
                    continue
                m, h = _mean_ci(vals)
                xs.append(size)
                means.append(m)
                halves.append(h)
            if not xs:
                continue
            xs = np.array(xs)
            means = np.array(means)
            halves = np.array(halves)
            c = _color(method, i)
            ax.plot(xs, means, marker='o', color=c, label=method)
            ax.fill_between(xs, means - halves, means + halves, color=c, alpha=0.2)

        # mse_norm = 1.0 equivale a "predecir la media" en espacio normalizado.
        if metric == 'mse_norm':
            ax.axhline(1.0, ls='--', color='gray', lw=1,
                       label='baseline (predict mean)')

        ax.set_xlabel('Training samples (n_samples)')
        ax.set_ylabel(label)
        ax.set_title(f'Scaling: {label} vs training size')
        ax.set_xscale('log')
        if metric in ('mse_norm', 'mae_orig'):
            ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f'01_scaling_{metric}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  wrote {path.name}')


# --------------------------------------------------------------------------
# 2) Paired comparison: uniform vs kdtree sobre los MISMOS (run, size).
# --------------------------------------------------------------------------
def paired_comparison(df, methods, sizes, out_dir, ref='uniform', alt='kdtree'):
    if ref not in methods or alt not in methods:
        print(f'  paired: faltan métodos {ref}/{alt} — omitido')
        return

    stats_lines = [f'Paired comparison: {ref} (ref) vs {alt}',
                   'diff = metric[ref] - metric[alt]  (diff > 0 => alt mejor)',
                   '']

    for metric, label in ERROR_METRICS.items():
        if metric not in df.columns:
            continue
        # Emparejar por (run_id, n_samples) usando pivote por método.
        pivot = df.pivot_table(index=['run_id', 'n_samples'],
                               columns='method', values=metric)
        if ref not in pivot.columns or alt not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=[ref, alt])
        pivot['diff'] = pivot[ref] - pivot[alt]

        # Boxplot de diferencias pareadas por tamaño.
        per_size, labels = [], []
        for size in sizes:
            d = pivot.xs(size, level='n_samples')['diff'].values \
                if size in pivot.index.get_level_values('n_samples') else []
            if len(d):
                per_size.append(d)
                labels.append(str(size))
        if not per_size:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(per_size, tick_labels=labels, showmeans=True)
        ax.axhline(0.0, ls='--', color='gray', lw=1)
        ax.set_xlabel('Training samples (n_samples)')
        ax.set_ylabel(f'{label}: {ref} - {alt}')
        ax.set_title(f'Paired difference ({ref} - {alt})\n>0 favors {alt}')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        path = out_dir / f'02_paired_{metric}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  wrote {path.name}')

        # Wilcoxon signed-rank por tamaño (no asume normalidad).
        stats_lines.append(f'[{metric}] {label}')
        for size, d in zip(labels, per_size):
            d = np.asarray(d, dtype=float)
            n = len(d)
            wins = int(np.sum(d > 0))  # veces que alt gana
            if n >= 1 and np.any(d != 0):
                try:
                    _, p = stats.wilcoxon(d)
                    p_str = f'p={p:.4f}'
                except ValueError:
                    p_str = 'p=n/a'
            else:
                p_str = 'p=n/a (all zero)'
            stats_lines.append(
                f'  n_samples={size:>5s}: pairs={n:2d}  '
                f'{alt}_wins={wins:2d}/{n:<2d}  '
                f'mean_diff={np.mean(d):+.4e}  {p_str}'
            )
        stats_lines.append('')

    txt_path = out_dir / '02_paired_stats.txt'
    txt_path.write_text('\n'.join(stats_lines), encoding='utf-8')
    print(f'  wrote {txt_path.name}')


# --------------------------------------------------------------------------
# 3) Computational cost: tiempo y memoria vs n_samples.
# --------------------------------------------------------------------------
def cost_analysis(df, methods, sizes, out_dir):
    for metric, (label, yscale) in COST_METRICS.items():
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for i, method in enumerate(methods):
            sub = df[df['method'] == method]
            means, halves, xs = [], [], []
            for size in sizes:
                vals = sub[sub['n_samples'] == size][metric].dropna().values
                if len(vals) == 0:
                    continue
                m, h = _mean_ci(vals)
                xs.append(size)
                means.append(m)
                halves.append(h)
            if not xs:
                continue
            xs = np.array(xs)
            means = np.array(means)
            halves = np.array(halves)
            c = _color(method, i)
            ax.plot(xs, means, marker='o', color=c, label=method)
            ax.fill_between(xs, means - halves, means + halves, color=c, alpha=0.2)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel('Training samples (n_samples)')
        ax.set_ylabel(label)
        ax.set_title(f'Computational cost: {label} vs training size')
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f'03_cost_{metric}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  wrote {path.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=CSV_PATH,
                        help='Ruta al CSV de resultados.')
    parser.add_argument('--target', type=str, default=None,
                        help='Filtra a un solo target (p.ej. f_osc).')
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'No se encontró el CSV: {args.csv}')

    df = pd.read_csv(args.csv)
    if args.target is not None:
        df = df[df['target'] == args.target]
    if df.empty:
        raise ValueError('El CSV (tras filtrar) no tiene filas.')

    targets = sorted(df['target'].unique())
    if len(targets) > 1:
        print(f'Aviso: múltiples targets {targets}. Usa --target para separar; '
              f'se grafican mezclados.')

    methods = sorted(df['method'].unique())
    sizes = sorted(df['n_samples'].unique())
    print(f'Filas: {len(df)} | métodos: {methods} | sizes: {sizes} | '
          f'runs: {sorted(df["run_id"].unique())}')

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print('Scaling analysis:')
    scaling_analysis(df, methods, sizes, PLOT_DIR)
    print('Paired comparison:')
    paired_comparison(df, methods, sizes, PLOT_DIR)
    print('Computational cost:')
    cost_analysis(df, methods, sizes, PLOT_DIR)

    print(f'\nListo. Figuras en {PLOT_DIR}')


if __name__ == '__main__':
    main()
