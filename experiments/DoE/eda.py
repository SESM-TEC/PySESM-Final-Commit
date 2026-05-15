"""
Exploratory data analysis of the ring oscillator DoE dataset.

Outputs (written under plots/eda/):
  00_summary.txt                — sample/NaN counts and descriptive stats
  01_factor_histograms.png      — factor distributions vs uniform expectation
  02_response_histograms.png    — response distributions on a log axis
  03_factor_pairplot.png        — 5x5 pair plot of factors (design space coverage)
  04_marginal_effects.png       — 5 factors x 3 responses, log y, Spearman rho
  05_response_relationships.png — f_osc vs t_rise, P_avg vs f*C*V^2, P_avg vs f_osc

Reads `ring_osc_dataset.csv` from the same directory as this script.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

HERE     = Path(__file__).resolve().parent
CSV_PATH = HERE / 'ring_osc_dataset.csv'
PLOT_DIR = HERE / 'plots' / 'eda'

FACTORS   = ['W_n', 'W_p', 'L', 'Vdd', 'C_load']
RESPONSES = ['f_osc', 'P_avg', 't_rise']

FACTOR_LABELS = {
    'W_n':    'W_n [m]',
    'W_p':    'W_p [m]',
    'L':      'L [m]',
    'Vdd':    'Vdd [V]',
    'C_load': 'C_load [F]',
}
RESPONSE_LABELS = {
    'f_osc':  'f_osc [Hz]',
    'P_avg':  'P_avg [W]',
    't_rise': 't_rise [s]',
}


def _spearman(a, b):
    """Spearman rho between two pandas Series, NaN-safe."""
    sub = pd.concat([a, b], axis=1).dropna()
    if len(sub) < 4:
        return np.nan
    return sub.corr(method='spearman').iloc[0, 1]


def section_1_dataset_health(df, out_dir):
    """Counts, NaN counts, factor and response distributions."""
    n_total = len(df)
    lines = [f"Dataset: {n_total} samples", ""]
    lines.append("NaN counts per response:")
    for r in RESPONSES:
        n_nan = df[r].isna().sum()
        lines.append(f"  {r:<8s}: {n_nan:4d} NaN  ({100 * n_nan / n_total:5.1f}%)")
    lines += ["", "Factor stats:", df[FACTORS].describe().to_string()]
    lines += ["", "Response stats (valid only):", df[RESPONSES].describe().to_string()]
    (out_dir / '00_summary.txt').write_text("\n".join(lines), encoding='utf-8')
    print("  -> 00_summary.txt")

    # Factor histograms with uniform-expectation reference line.
    n_bins = 15
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.2))
    for ax, f in zip(axes, FACTORS):
        ax.hist(df[f], bins=n_bins, color='C0', edgecolor='black', alpha=0.85)
        ax.axhline(n_total / n_bins, color='red', ls='--', lw=1,
                   label='uniform exp.')
        ax.set_xlabel(FACTOR_LABELS[f])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('count')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('1a - Factor distributions (red = uniform expectation per bin)')
    fig.tight_layout()
    fig.savefig(out_dir / '01_factor_histograms.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  -> 01_factor_histograms.png")

    # Response histograms on a log x-axis.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, r in zip(axes, RESPONSES):
        vals = df[r].dropna()
        vals = vals[vals > 0]
        if len(vals) > 1:
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 20)
            ax.hist(vals, bins=bins, color='C1', edgecolor='black', alpha=0.85)
        ax.set_xscale('log')
        ax.set_xlabel(RESPONSE_LABELS[r])
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('count')
    fig.suptitle('1b - Response distributions (log x)')
    fig.tight_layout()
    fig.savefig(out_dir / '02_response_histograms.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  -> 02_response_histograms.png")


def section_2_design_coverage(df, out_dir):
    """5x5 factor pair plot to inspect coverage / gaps / clusters.

    NaN samples (any response NaN) are overlaid in red so the failure
    pattern in factor space is visible alongside coverage.
    """
    n = len(FACTORS)
    nan_mask = df[RESPONSES].isna().any(axis=1)
    valid    = df[~nan_mask]
    failed   = df[nan_mask]

    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                vals = df[FACTORS[i]]
                bins = np.linspace(vals.min(), vals.max(), 13)
                ax.hist(valid[FACTORS[i]], bins=bins, color='C0',
                        edgecolor='black', alpha=0.75, label='valid')
                if len(failed) > 0:
                    ax.hist(failed[FACTORS[i]], bins=bins, color='red',
                            edgecolor='black', alpha=0.6, label='NaN')
            else:
                ax.scatter(valid[FACTORS[j]], valid[FACTORS[i]], s=6,
                           alpha=0.5, color='C0', label='valid')
                if len(failed) > 0:
                    ax.scatter(failed[FACTORS[j]], failed[FACTORS[i]], s=28,
                               color='red', marker='x', linewidths=1.3,
                               label='NaN')
            if i == n - 1:
                ax.set_xlabel(FACTORS[j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(FACTORS[i], fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right',
                   bbox_to_anchor=(0.995, 0.995), fontsize=10,
                   frameon=True)
    fig.suptitle(
        f'2 - Factor pair plot (design space coverage; '
        f'{len(failed)}/{len(df)} NaN highlighted)',
        y=0.995, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])
    fig.savefig(out_dir / '03_factor_pairplot.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  -> 03_factor_pairplot.png")


def section_3_marginal_effects(df, out_dir):
    """5 factors x 3 responses scatter, log y, Spearman rho per panel."""
    n_f, n_r = len(FACTORS), len(RESPONSES)
    fig, axes = plt.subplots(n_f, n_r, figsize=(12, 14), sharey='col')

    for j, r in enumerate(RESPONSES):
        axes[0, j].set_title(RESPONSE_LABELS[r], fontsize=11, pad=10)

    for i, f in enumerate(FACTORS):
        for j, r in enumerate(RESPONSES):
            ax = axes[i, j]
            sub = df[[f, r]].dropna()
            sub = sub[sub[r] > 0]
            ax.scatter(sub[f], sub[r], s=12, alpha=0.55, color='C0')
            ax.set_yscale('log')
            rho = _spearman(sub[f], sub[r]) if len(sub) >= 4 else np.nan
            if not np.isnan(rho):
                ax.text(0.05, 0.95, f"rho={rho:+.2f}",
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

    fig.suptitle('3 - Marginal effects (log y, Spearman rho per panel)',
                 y=0.99, fontsize=12)
    fig.tight_layout(rect=[0.05, 0.0, 1, 0.97])
    # Row labels (factor names) on the left, after tight_layout so positions are final
    for i, f in enumerate(FACTORS):
        bbox = axes[i, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.005, y_center, FACTOR_LABELS[f], rotation=90,
                 va='center', ha='left', fontsize=11, fontweight='bold')
    fig.savefig(out_dir / '04_marginal_effects.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  -> 04_marginal_effects.png")


def section_4_response_relationships(df, out_dir):
    """f_osc vs t_rise, P_avg vs f*C*V^2 (physics check), P_avg vs f_osc."""
    valid = df.dropna(subset=RESPONSES).copy()
    valid = valid[(valid['f_osc'] > 0) & (valid['P_avg'] > 0) & (valid['t_rise'] > 0)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: f_osc vs t_rise -> expected inverse relationship
    ax = axes[0]
    rho_s = _spearman(valid['t_rise'], valid['f_osc'])
    p_log = np.corrcoef(np.log(valid['t_rise']), np.log(valid['f_osc']))[0, 1] \
        if len(valid) >= 2 else np.nan
    ax.scatter(valid['t_rise'], valid['f_osc'], s=14, alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('t_rise [s]')
    ax.set_ylabel('f_osc [Hz]')
    ax.set_title(f"f_osc vs t_rise\nPearson(log-log)={p_log:+.2f}  Spearman={rho_s:+.2f}",
                 fontsize=10)
    ax.grid(alpha=0.3, which='both')

    # Panel B: dynamic power scaling P_dyn ~ alpha * f * C * V^2
    ax = axes[1]
    fcv2 = valid['f_osc'] * valid['C_load'] * valid['Vdd'] ** 2
    rho_s = _spearman(fcv2, valid['P_avg'])
    p_log = np.corrcoef(np.log(fcv2), np.log(valid['P_avg']))[0, 1] \
        if len(valid) >= 2 else np.nan
    ax.scatter(fcv2, valid['P_avg'], s=14, alpha=0.6)
    lo = min(fcv2.min(), valid['P_avg'].min())
    hi = max(fcv2.max(), valid['P_avg'].max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='P = f*C*V^2  (alpha=1)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('f * C * V^2  [W]')
    ax.set_ylabel('P_avg [W]')
    ax.set_title(f"Dynamic power scaling\nPearson(log-log)={p_log:+.2f}  Spearman={rho_s:+.2f}",
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, which='both')

    # Panel C: P_avg vs f_osc directly
    ax = axes[2]
    rho_s = _spearman(valid['f_osc'], valid['P_avg'])
    p_log = np.corrcoef(np.log(valid['f_osc']), np.log(valid['P_avg']))[0, 1] \
        if len(valid) >= 2 else np.nan
    ax.scatter(valid['f_osc'], valid['P_avg'], s=14, alpha=0.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('f_osc [Hz]')
    ax.set_ylabel('P_avg [W]')
    ax.set_title(f"P_avg vs f_osc\nPearson(log-log)={p_log:+.2f}  Spearman={rho_s:+.2f}",
                 fontsize=10)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle('4 - Response relationships (log-log)', y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / '05_response_relationships.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("  -> 05_response_relationships.png")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            f"Run generate_ring_osc_dataset.py first."
        )
    df = pd.read_csv(CSV_PATH)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {CSV_PATH}")
    print(f"Output:  {PLOT_DIR}\n")

    print("Section 1 - Dataset health")
    section_1_dataset_health(df, PLOT_DIR)
    print("\nSection 2 - Design space coverage")
    section_2_design_coverage(df, PLOT_DIR)
    print("\nSection 3 - Marginal effects")
    section_3_marginal_effects(df, PLOT_DIR)
    print("\nSection 4 - Response relationships")
    section_4_response_relationships(df, PLOT_DIR)

    print(f"\nDone. Outputs under {PLOT_DIR}/")


if __name__ == '__main__':
    main()
