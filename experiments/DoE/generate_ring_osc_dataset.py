"""
Ring oscillator dataset generator using PySpice with CMOS Level 1 models.

Circuit topology (N-stage CMOS ring oscillator, N odd):
  [INV1(nN)] --n1-- [INV2(n1)] --n2-- ... --nN--+
                                                  |
  +-----------------------------------------------+
  Each inverter: PMOS(W_p) + NMOS(W_n), channel length L, load cap C_load

Inputs to SESM:  W_n, W_p, L, Vdd, C_load
Outputs:         f_osc [Hz], P_avg [W], t_rise [s]

This module is fully parameterizable so it can be driven either standalone
(the __main__ block below) or from a Hydra-configured caller that passes the
oscillator parameters / sampling ranges from a YAML.
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import CircuitSimulator

logging.getLogger('PySpice').setLevel(logging.CRITICAL)
sys.unraisablehook = lambda _: None

NMOS_MODEL_NAME = 'NMOS_L1'
PMOS_MODEL_NAME = 'PMOS_L1'

DEFAULT_NMOS_PARAMS = dict(LEVEL=1, KP=120e-6, VTO=0.5,  LAMBDA=0.01)
DEFAULT_PMOS_PARAMS = dict(LEVEL=1, KP=60e-6,  VTO=-0.5, LAMBDA=0.01)
DEFAULT_N_STAGES    = 21

DEFAULT_RANGES = {
    'W_n':    (0.5e-6,  5.0e-6),
    'W_p':    (1.0e-6, 10.0e-6),
    'L':      (0.5e-6,  2.0e-6),
    'Vdd':    (2.85,    3.75),
    'C_load': (10e-15, 500e-15),
}

FEATURE_COLUMNS  = ['W_n', 'W_p', 'L', 'Vdd', 'C_load']
RESPONSE_COLUMNS = ['f_osc', 'P_avg', 't_rise']
CSV_COLUMNS      = FEATURE_COLUMNS + RESPONSE_COLUMNS


def simulate_ring_osc(
    W_n, W_p, L, Vdd, C_load,
    n_stages=DEFAULT_N_STAGES,
    nmos_params=None,
    pmos_params=None,
    plot_path=None,
):
    """Transient simulation of an N-stage CMOS ring oscillator.

    Parameters (SI units):
        W_n, W_p, L, Vdd, C_load: device / supply / load values
        n_stages    : odd integer >= 3 (number of inverter stages)
        nmos_params : dict of NMOS Level-1 model parameters
        pmos_params : dict of PMOS Level-1 model parameters
        plot_path   : if set, save a V(n1) vs time plot to this path

    Returns:
        (f_osc, P_avg, t_rise) — np.nan on simulation failure.
    """
    if n_stages % 2 == 0 or n_stages < 3:
        raise ValueError(f"n_stages must be an odd integer >= 3 (got {n_stages})")

    nmos_params = dict(nmos_params) if nmos_params is not None else dict(DEFAULT_NMOS_PARAMS)
    pmos_params = dict(pmos_params) if pmos_params is not None else dict(DEFAULT_PMOS_PARAMS)

    f_osc, P_avg, t_rise = np.nan, np.nan, np.nan
    time_arr, v_n1 = None, None

    try:
        circuit = Circuit('RingOsc')
        circuit.model(NMOS_MODEL_NAME, 'NMOS', **nmos_params)
        circuit.model(PMOS_MODEL_NAME, 'PMOS', **pmos_params)

        circuit.V('DD', 'vdd', circuit.gnd, Vdd)

        stage_nodes = [f'n{i+1}' for i in range(n_stages)]

        for i in range(n_stages):
            in_node  = stage_nodes[(i - 1) % n_stages]
            out_node = stage_nodes[i]
            circuit.MOSFET(f'P{i+1}', out_node, in_node, 'vdd', 'vdd',
                           model=PMOS_MODEL_NAME, w=W_p, l=L)
            circuit.MOSFET(f'N{i+1}', out_node, in_node, circuit.gnd, circuit.gnd,
                           model=NMOS_MODEL_NAME, w=W_n, l=L)
            circuit.C(f'{i+1}', out_node, circuit.gnd, C_load)

        ic_terms = ' '.join(
            f'V({node})={(Vdd if i % 2 == 0 else 0):.6g}'
            for i, node in enumerate(stage_nodes)
        )
        circuit.raw_spice = f'.IC {ic_terms}'

        # tau ~ C*Vdd / I_drive (Level-1 saturation current, very rough)
        kp_nmos     = nmos_params.get('KP', 120e-6)
        vto_nmos    = nmos_params.get('VTO', 0.5)
        I_drive     = kp_nmos * (W_n / L) * max((Vdd - vto_nmos) ** 2, 1e-6) / 2
        tau_est     = C_load * Vdd / max(I_drive, 1e-12)
        t_period_est = 2 * n_stages * tau_est

        t_step    = max(t_period_est / 200, 1e-13)
        t_end_cap = max(200e-9, 50e-9 * n_stages)
        t_end     = min(max(30 * t_period_est, 5e-9), t_end_cap)

        sim      = CircuitSimulator.factory(circuit, temperature=25, nominal_temperature=25)
        analysis = sim.transient(step_time=t_step, end_time=t_end,
                                 use_initial_condition=True)

        time_arr = np.array(analysis.time)
        v_n1 = np.array(analysis['n1'])

        ss_idx = int(len(time_arr) * 0.30)
        v_ss   = v_n1[ss_idx:]
        v_mid  = Vdd / 2

        rising_cross = np.where(
            (v_ss[:-1] < v_mid) & (v_ss[1:] >= v_mid)
        )[0]

        if len(rising_cross) >= 3:
            periods = np.diff(time_arr[ss_idx + rising_cross])
            f_osc   = 1.0 / np.mean(periods)

            try:
                i_vdd = np.array(analysis.branches['vdd'])
                P_avg = float(np.mean(np.abs(i_vdd[ss_idx:])) * Vdd)
            except Exception:
                P_avg = np.nan

            v10, v90 = 0.1 * Vdd, 0.9 * Vdd
            for cross_idx in rising_cross[:5]:
                abs_idx  = ss_idx + cross_idx
                below10  = np.where(v_n1[:abs_idx] < v10)[0]
                if not len(below10):
                    continue
                i_10    = below10[-1]
                above90 = np.where(v_n1[i_10:] >= v90)[0]
                if len(above90):
                    t_rise = time_arr[i_10 + above90[0]] - time_arr[i_10]
                    break

    except Exception:
        pass

    if plot_path is not None and time_arr is not None:
        _plot_waveform(
            time_arr, v_n1,
            dict(W_n=W_n, W_p=W_p, L=L, Vdd=Vdd, C_load=C_load),
            dict(f_osc=f_osc, P_avg=P_avg, t_rise=t_rise),
            plot_path,
        )

    return f_osc, P_avg, t_rise


def _plot_waveform(time_arr, v_n1, params, metrics, plot_path):
    """Save a V(n1) vs time plot for a single simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_arr * 1e9, v_n1, lw=0.8)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('V(n1) [V]')
    ax.grid(alpha=0.3)
    title = (
        f"W_n={params['W_n']*1e6:.2f}um  W_p={params['W_p']*1e6:.2f}um  "
        f"L={params['L']*1e6:.2f}um  Vdd={params['Vdd']:.2f}V  "
        f"C={params['C_load']*1e15:.1f}fF\n"
        f"f_osc={metrics['f_osc']:.3e} Hz   "
        f"P_avg={metrics['P_avg']:.3e} W   "
        f"t_rise={metrics['t_rise']:.3e} s"
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def _ranges_to_dict(ranges):
    """Coerce ranges (dict / OmegaConf) into a plain {name: (lo, hi)} dict."""
    if ranges is None:
        return dict(DEFAULT_RANGES)
    out = {}
    for name in FEATURE_COLUMNS:
        if name not in ranges:
            raise KeyError(f"Missing range for '{name}' in ranges config")
        lo, hi = ranges[name]
        out[name] = (float(lo), float(hi))
    return out


def _sample_params(rng, ranges):
    return [rng.uniform(ranges[name][0], ranges[name][1]) for name in FEATURE_COLUMNS]


def _count_existing_rows(csv_path):
    """Return number of data rows already present in the CSV (0 if missing/empty)."""
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = sum(1 for _ in reader)
        return max(rows - 1, 0)  # subtract header
    except Exception:
        return 0


def generate_ring_osc_dataset(
    n_samples=100,
    csv_path='ring_osc_dataset.csv',
    seed=42,
    plot_every=0,
    plot_dir='plots',
    n_stages=DEFAULT_N_STAGES,
    nmos_params=None,
    pmos_params=None,
    ranges=None,
    resume=True,
    flush_every=1,
    progress_log=True,
):
    """Generate (or resume) a CSV with random parameters + simulated metrics.

    Args:
        n_samples   : total number of rows in the final CSV
        csv_path    : output path
        seed        : RNG seed (deterministic samples → resume-safe)
        plot_every  : if > 0, dump a waveform plot every N simulations
        plot_dir    : directory for waveform plots
        n_stages    : odd integer >= 3
        nmos_params : NMOS L1 params (defaults to DEFAULT_NMOS_PARAMS)
        pmos_params : PMOS L1 params (defaults to DEFAULT_PMOS_PARAMS)
        ranges      : dict of feature → (lo, hi) sampling ranges
        resume      : if True and csv_path exists, continue from last row count
        flush_every : write to disk every N completed rows
        progress_log: print per-sample line if True
    """
    ranges = _ranges_to_dict(ranges)
    nmos_params = dict(nmos_params) if nmos_params is not None else dict(DEFAULT_NMOS_PARAMS)
    pmos_params = dict(pmos_params) if pmos_params is not None else dict(DEFAULT_PMOS_PARAMS)

    rng = np.random.default_rng(seed)

    # Pre-generate all parameter sets so a resumed run reproduces the same rows.
    sampled = np.array(
        [_sample_params(rng, ranges) for _ in range(n_samples)],
        dtype=float,
    )

    csv_path = str(csv_path)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    start_idx = _count_existing_rows(csv_path) if resume else 0
    if start_idx >= n_samples:
        print(f"[generate_ring_osc_dataset] CSV {csv_path} already has "
              f"{start_idx} rows >= n_samples ({n_samples}). Nothing to do.")
        return

    file_exists = os.path.exists(csv_path) and start_idx > 0

    plot_enabled = plot_every and plot_every > 0
    if plot_enabled:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
    width = max(len(str(n_samples)), 3)

    t_start = time.perf_counter()
    pending = []

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        for i in range(start_idx, n_samples):
            W_n, W_p, L, Vdd, C_load = sampled[i]

            plot_path = None
            if plot_enabled and (i + 1) % plot_every == 0:
                plot_path = str(Path(plot_dir) / f'sim_{i+1:0{width}d}.png')

            f_osc, P_avg, t_rise = simulate_ring_osc(
                W_n, W_p, L, Vdd, C_load,
                n_stages=n_stages,
                nmos_params=nmos_params,
                pmos_params=pmos_params,
                plot_path=plot_path,
            )

            pending.append([W_n, W_p, L, Vdd, C_load, f_osc, P_avg, t_rise])

            if progress_log:
                print(f"[{i+1:6d}/{n_samples}]  f={f_osc:.3e} Hz  "
                      f"P={P_avg:.3e} W  tr={t_rise:.3e} s")

            if len(pending) >= max(flush_every, 1):
                writer.writerows(pending)
                f.flush()
                pending.clear()

        if pending:
            writer.writerows(pending)
            f.flush()

    elapsed = time.perf_counter() - t_start
    done = n_samples - start_idx
    rate = elapsed / done if done > 0 else 0.0
    print(f"\nDataset guardado en {csv_path}")
    print(f"Filas nuevas: {done}  (total: {n_samples})")
    print(f"Tiempo: {elapsed:.2f} s  ({rate:.3f} s/muestra)")


def load_clean_dataset(csv_path, drop_nan=True, columns=None):
    """Helper for downstream code: load a CSV produced by this module.

    Args:
        csv_path : path to the CSV
        drop_nan : drop rows where any response is NaN
        columns  : optional list of response columns to validate (default: all)

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    if drop_nan:
        check_cols = columns if columns is not None else RESPONSE_COLUMNS
        df = df.dropna(subset=check_cols).reset_index(drop=True)
    return df


if __name__ == '__main__':
    # Standalone usage — kept compatible with the previous behavior.
    generate_ring_osc_dataset(
        n_samples=500,
        csv_path='ring_osc_dataset.csv',
        seed=42,
        plot_every=0,
        plot_dir='plots',
    )
