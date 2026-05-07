"""
Ring oscillator dataset generator using PySpice with CMOS Level 1 models.

Circuit topology (3-stage CMOS ring oscillator):
  [INV1(n3)] --n1-- [INV2(n1)] --n2-- [INV3(n2)] --n3--+
                                                          |
  +-------------------------------------------------------+
  Each inverter: PMOS(W_p) + NMOS(W_n), channel length L, load cap C_load

Inputs to SESM:  W_n, W_p, L, Vdd, C_load
Outputs:         f_osc [Hz], P_avg [W], t_rise [s]
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import CircuitSimulator

logging.getLogger('PySpice').setLevel(logging.CRITICAL)
sys.unraisablehook = lambda _: None

NMOS_MODEL  = 'NMOS_L1'
PMOS_MODEL  = 'PMOS_L1'
NMOS_PARAMS = dict(LEVEL=1, KP=120e-6, VTO=0.5,  LAMBDA=0.01)
PMOS_PARAMS = dict(LEVEL=1, KP=60e-6,  VTO=-0.5, LAMBDA=0.01)

N_STAGES = 101  # must be odd; larger → longer critical path

if N_STAGES % 2 == 0 or N_STAGES < 3:
    raise ValueError(f"N_STAGES must be an odd integer >= 3 (got {N_STAGES})")


def simulate_ring_osc(W_n, W_p, L, Vdd, C_load):
    """
    Transient simulation of an N-stage CMOS ring oscillator.

    Parameters (SI units):
        W_n    : NMOS channel width [m]
        W_p    : PMOS channel width [m]
        L      : channel length [m]
        Vdd    : supply voltage [V]
        C_load : load capacitance per stage [F]

    Returns:
        f_osc  (float): oscillation frequency [Hz], NaN on failure
        P_avg  (float): average power [W],         NaN on failure
        t_rise (float): rise time 10%→90% [s],     NaN on failure
    """
    try:
        circuit = Circuit('RingOsc')
        circuit.model(NMOS_MODEL, 'NMOS', **NMOS_PARAMS)
        circuit.model(PMOS_MODEL, 'PMOS', **PMOS_PARAMS)

        circuit.V('DD', 'vdd', circuit.gnd, Vdd)

        # Stage nodes: n1..nN — last stage feeds back to input of stage 1
        stage_nodes = [f'n{i+1}' for i in range(N_STAGES)]

        for i in range(N_STAGES):
            in_node  = stage_nodes[(i - 1) % N_STAGES]
            out_node = stage_nodes[i]
            circuit.MOSFET(f'P{i+1}', out_node, in_node, 'vdd', 'vdd',
                           model=PMOS_MODEL, w=W_p, l=L)
            circuit.MOSFET(f'N{i+1}', out_node, in_node, circuit.gnd, circuit.gnd,
                           model=NMOS_MODEL, w=W_n, l=L)
            circuit.C(f'{i+1}', out_node, circuit.gnd, C_load)

        # Break symmetry: alternate Vdd / 0 across all stage nodes
        ic_terms = ' '.join(
            f'V({node})={(Vdd if i % 2 == 0 else 0):.6g}'
            for i, node in enumerate(stage_nodes)
        )
        circuit.raw_spice = f'.IC {ic_terms}'

        # Estimate propagation delay per stage: tau ~ C*Vdd / I_drive
        I_drive     = 120e-6 * (W_n / L) * max((Vdd - 0.5) ** 2, 1e-6) / 2
        tau_est     = C_load * Vdd / max(I_drive, 1e-12)
        t_period_est = 2 * N_STAGES * tau_est

        t_step  = max(t_period_est / 200, 1e-13)            # no finer than 0.1 ps
        t_end_cap = max(200e-9, 50e-9 * N_STAGES)            # scale cap with N_STAGES
        t_end   = min(max(30 * t_period_est, 5e-9), t_end_cap)

        sim      = CircuitSimulator.factory(circuit, temperature=25, nominal_temperature=25)
        analysis = sim.transient(step_time=t_step, end_time=t_end,
                                 use_initial_condition=True)

        time = np.array(analysis.time)
        v_n1 = np.array(analysis['n1'])

        # --- Frequency: rising zero-crossings at Vdd/2 in steady-state window ---
        ss_idx = int(len(time) * 0.30)
        v_ss   = v_n1[ss_idx:]
        v_mid  = Vdd / 2

        rising_cross = np.where(
            (v_ss[:-1] < v_mid) & (v_ss[1:] >= v_mid)
        )[0]

        if len(rising_cross) < 3:
            return np.nan, np.nan, np.nan

        periods = np.diff(time[ss_idx + rising_cross])
        f_osc   = 1.0 / np.mean(periods)

        # --- Average power from supply branch current ---
        try:
            i_vdd = np.array(analysis.branches['vdd'])
            P_avg = float(np.mean(np.abs(i_vdd[ss_idx:])) * Vdd)
        except Exception:
            P_avg = np.nan

        # --- Rise time (10%→90%) at n1, first valid rising edge in steady state ---
        v10, v90 = 0.1 * Vdd, 0.9 * Vdd
        t_rise   = np.nan
        for cross_idx in rising_cross[:5]:
            abs_idx  = ss_idx + cross_idx
            below10  = np.where(v_n1[:abs_idx] < v10)[0]
            if not len(below10):
                continue
            i_10    = below10[-1]
            above90 = np.where(v_n1[i_10:] >= v90)[0]
            if len(above90):
                t_rise = time[i_10 + above90[0]] - time[i_10]
                break

        return f_osc, P_avg, t_rise

    except Exception:
        return np.nan, np.nan, np.nan


def generate_ring_osc_dataset(n_samples=100, csv_path='ring_osc_dataset.csv', seed=42):
    """Generate a CSV dataset with random parameters and simulated ring oscillator metrics.

    Parameters:
        n_samples (int): number of samples to generate
        csv_path  (str): output CSV path
        seed      (int|None): RNG seed for reproducibility; pass None for non-deterministic runs
    """
    rng = np.random.default_rng(seed)
    data = []
    t_start = time.perf_counter()

    for i in range(n_samples):
        W_n    = rng.uniform(0.5e-6,  5e-6)
        W_p    = rng.uniform(1e-6,   10e-6)
        L      = rng.uniform(0.5e-6,  2e-6)
        Vdd    = rng.uniform(1.0,     3.3)
        C_load = rng.uniform(10e-15, 500e-15)

        f_osc, P_avg, t_rise = simulate_ring_osc(W_n, W_p, L, Vdd, C_load)
        data.append([W_n, W_p, L, Vdd, C_load, f_osc, P_avg, t_rise])
        print(f"[{i+1:3d}/{n_samples}]  f={f_osc:.3e} Hz  P={P_avg:.3e} W  tr={t_rise:.3e} s")

    df = pd.DataFrame(data, columns=['W_n', 'W_p', 'L', 'Vdd', 'C_load',
                                     'f_osc', 'P_avg', 't_rise'])
    df.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - t_start
    print(f"\nDataset guardado en {csv_path}")
    print(f"Tiempo total de generación: {elapsed:.2f} s  ({elapsed/n_samples:.3f} s/muestra)")


if __name__ == '__main__':
    n_samples = 100
    csv_path  = 'ring_osc_dataset.csv'
    seed      = 42
    generate_ring_osc_dataset(n_samples=n_samples, csv_path=csv_path, seed=seed)
