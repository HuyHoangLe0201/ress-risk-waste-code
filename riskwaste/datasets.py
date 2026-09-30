"""Readers for public run-to-failure data, and a synthetic fleet for trying the method.

The data are not redistributed. C-MAPSS is available from the NASA Prognostics Data
Repository (data set 6, "Turbofan Engine Degradation Simulation Data Set").
"""
import os

import numpy as np


def load_cmapss(directory, subset="FD002"):
    """Training units of one C-MAPSS subset, each run to failure.

    Returns
    -------
    sequences : list of (cycles, channels) arrays
        Operating settings and sensor channels, unstandardised.
    lifetimes : array
        Number of cycles to failure of each unit.
    """
    d = np.loadtxt(os.path.join(directory, "train_%s.txt" % subset))
    unit = d[:, 0].astype(int)
    cycle = d[:, 1].astype(int)
    X = d[:, 2:]
    sequences, lifetimes = [], []
    for u in np.unique(unit):
        rows = unit == u
        order = np.argsort(cycle[rows])
        sequences.append(X[rows][order])
        lifetimes.append(float(rows.sum()))
    return sequences, np.array(lifetimes)


def synthetic_fleet(n_units=150, n_channels=6, seed=0):
    """A small fleet with increasing-hazard lifetimes and noisy degrading sensors.

    Each unit's health falls along a convex path to failure, with a unit-specific rate,
    and every channel reads a noisy linear function of it. Useful for trying the method
    without downloading anything; nothing in the paper depends on it.
    """
    rng = np.random.default_rng(seed)
    life = np.maximum(rng.weibull(3.0, n_units) * 180.0, 40.0).round()
    mix = rng.normal(size=(1, n_channels))
    sequences = []
    for T in life:
        t = np.arange(1, int(T) + 1) / T
        health = t ** rng.uniform(1.5, 3.0)
        drift = np.cumsum(rng.normal(0, 0.01, len(t)))
        sequences.append((health + drift)[:, None] * mix
                         + rng.normal(0, 0.15, (len(t), n_channels)))
    return sequences, life
