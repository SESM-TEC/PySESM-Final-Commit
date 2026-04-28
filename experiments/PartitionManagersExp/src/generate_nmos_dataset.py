"""
NMOS dataset generator/filler using PySpice with built-in Level 1 NMOS model.

Modes:
  --fill      : reads the existing CSV and fills the empty Id column
  (default)   : generates a new dataset from random parameters

Circuit topology (resistive load):
  Vdd --[Rd]-- drain --[M1(Vgs)]-- gnd
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import CircuitSimulator

# Suppress ngspice-shared noise: .cm load errors and cffi callback TypeErrors
logging.getLogger('PySpice').setLevel(logging.CRITICAL)
sys.unraisablehook = lambda _: None

# Level 1 NMOS model parameters — no external file needed
NMOS_MODEL = 'NMOS_L1'
NMOS_PARAMS = dict(LEVEL=1, KP=120e-6, VTO=0.5, LAMBDA=0.01)


def simulate_nmos(W, L, Vgs, Vdd, Rd):
    """
    Operating-point simulation of the resistive-load NMOS circuit.

    Parameters (SI units):
        W   : channel width  [m]
        L   : channel length [m]
        Vgs : gate-source voltage [V]
        Vdd : supply voltage      [V]
        Rd  : drain resistor      [Ω]

    Returns:
        Id (float): drain current in amperes, or NaN on simulation failure.
    """
    try:
        circuit = Circuit('NMOS OP')
        circuit.model(NMOS_MODEL, 'NMOS', **NMOS_PARAMS)

        circuit.V('DD', 'vdd',  circuit.gnd, Vdd)
        circuit.V('GS', 'gate', circuit.gnd, Vgs)
        circuit.R('D',  'vdd',  'drain',     Rd )
        circuit.MOSFET('1', 'drain', 'gate', circuit.gnd, circuit.gnd,
                       model=NMOS_MODEL, w=W, l=L)

        sim      = CircuitSimulator.factory(circuit, temperature=25, nominal_temperature=25)
        analysis = sim.operating_point()

        v_drain = float(analysis['drain'][0])
        return (Vdd - v_drain) / Rd

    except Exception:
        return np.nan

def generate_nmos_dataset(n_samples=100, csv_path="nmos_dataset.csv"):
    """Generate a new CSV dataset with random parameters and simulated Id."""
    np.random.seed(42)
    data = []
    t_start = time.perf_counter()

    for i in range(n_samples):
        W   = np.random.uniform(1e-4,  10e-4)
        L   = np.random.uniform(1e-6,   2e-6)
        Vgs = np.random.uniform(1.0,    3.3)
        Vds = np.random.uniform(0.1,    3.3)   # stored for reference; not a sim input
        Vdd = np.random.uniform(1.0,    3.3)
        Rd  = np.random.uniform(100,  10000)
        Id  = simulate_nmos(W, L, Vgs, Vdd, Rd)
        data.append([W, L, Vgs, Vds, Vdd, Rd, Id])
        print(f"[{i+1}/{n_samples}]  Id = {Id:.6e} A")

    df = pd.DataFrame(data, columns=["W", "L", "Vgs", "Vds", "Vdd", "Rd", "Id"])
    df.to_csv(csv_path, index=False)
    elapsed = time.perf_counter() - t_start
    print(f"\nDataset guardado en {csv_path}")
    print(f"Tiempo total de generación: {elapsed:.2f} s  ({elapsed/n_samples:.3f} s/muestra)")


if __name__ == "__main__":
    n_samples = 100
    csv_path  = 'nmos_dataset.csv'

    generate_nmos_dataset(n_samples=n_samples, csv_path=csv_path)