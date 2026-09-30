"""The method on a synthetic fleet, in about a minute and with nothing to download.

Swap ``synthetic_fleet`` for your own run-to-failure sequences and ``RidgeRUL`` for your
own predictor; nothing else changes.
"""
import numpy as np

import riskwaste as rw
from riskwaste.datasets import synthetic_fleet
from riskwaste.predictors import RidgeRUL, ridge_fit_predict

KAPPA = 9.0                      # c_f / c_p = 10

sequences, lifetimes = synthetic_fleet(n_units=150, seed=0)
n = len(sequences)
rng = np.random.default_rng(0)
order = rng.permutation(n)
cal, val = order[: n // 2], order[n // 2:]

# 1. Any predictor. Here, ridge on sensors and trailing means, trained on half the fleet.
model = RidgeRUL().fit([sequences[u] for u in cal])
predicted = [model.predict(s) for s in sequences]

# 2. Every threshold rule is a point (omega, P_f); the calibration half fixes where the
#    optimum lies before any policy is committed to.
policy = rw.ThresholdPolicy([predicted[u] for u in cal])
print("calibration half: %d units, %.0f%% of monitored cycles visible to a threshold rule"
      % (policy.n_units, 100 * policy.visible_fraction()))
print("zero-failure threshold ell_0 = %.1f cycles, kappa* = %.2f"
      % (policy.ell0, policy.kappa_star()))
ell, omega, pf, L = policy.optimum(KAPPA)
print("optimum at kappa = %g: threshold %.1f, omega %.3f, P_f %.3f%s"
      % (KAPPA, ell, omega, pf,
         "  (the zero-failure boundary)" if ell == policy.ell0 else ""))

# 3. Which operating points are ever optimal, and over which cost ratios.
th, om, pfs = policy.frontier()
for j, lo, hi in rw.optimal_intervals(om, pfs)[-4:]:
    print("   optimal for kappa in [%7.3f, %7s): omega %.3f, P_f %.3f"
          % (lo, "inf" if hi == np.inf else "%.3f" % hi, om[j], pfs[j]))

# 4. Algorithm 2: adopt the boundary only if kappa clears an upper quantile of kappa*.
print(rw.choose_threshold([predicted[u] for u in cal], KAPPA, n_boot=300))

# 5. Algorithm 1: how precisely can this fleet compare monitoring with a schedule?
result = rw.paired_bootstrap(lifetimes, ridge_fit_predict(sequences), KAPPA, n_boot=100)
print(result)
print("interval %s parity" % ("excludes" if result.excludes_parity else "contains"))
