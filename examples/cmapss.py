"""The method on a C-MAPSS subset.

Download the "Turbofan Engine Degradation Simulation Data Set" from the NASA Prognostics
Data Repository, unpack it, and run

    python examples/cmapss.py path/to/CMAPSSData FD002

Channels are standardised on the training units only, and the threshold is searched over
the exact candidate set.
"""
import sys

import numpy as np

import riskwaste as rw
from riskwaste.datasets import load_cmapss
from riskwaste.predictors import RidgeRUL, ridge_fit_predict

directory = sys.argv[1] if len(sys.argv) > 1 else "CMAPSSData"
subset = sys.argv[2] if len(sys.argv) > 2 else "FD002"
KAPPA = 9.0

sequences, lifetimes = load_cmapss(directory, subset)
n = len(sequences)
print("%s: %d engines, mean life %.0f cycles" % (subset, n, lifetimes.mean()))

cal = np.random.default_rng(0).permutation(n)[: n // 2]
model = RidgeRUL().fit([sequences[u] for u in cal])
policy = rw.ThresholdPolicy([model.predict(sequences[u]) for u in cal])
print("kappa* = %.2f on the calibration half; %.1f%% of cycles visible"
      % (policy.kappa_star(), 100 * policy.visible_fraction()))
print(rw.choose_threshold(policy.paths, KAPPA))

result = rw.paired_bootstrap(lifetimes, ridge_fit_predict(sequences), KAPPA, n_boot=200)
print(result)
