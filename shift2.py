"""The shift that matters is in the tail, not in the distribution.

shift.py bounds the risk under a change of law by the Kolmogorov distance Delta and
finds the bound holds everywhere and is sharp.  It also finds something the first
version did not predict: on FD002 a RANDOM split of the fleet produced larger
deviations than the paper's covariate split, even though the covariate split is the
one that changes the population.  That is not noise, it is the bound being loose in
the right way.  The risk only ever evaluates the two laws at ell_(k), which sits far
out in the upper tail, while Delta is a supremum over the whole line.  A shift in the
middle of the distribution inflates Delta and does nothing to the risk.

So the useful statement localises.  For any tau, splitting on whether ell_(k) has
fallen below the tau-quantile x_tau of P,

    | risk - k/(n+1) |  <=  Delta_tau + P(Bin(n, 1-tau) < k),

with Delta_tau the supremum of |F_P - F_Q| over [x_tau, infinity) only.  The second
term is the chance that the k-th largest of n draws falls below a quantile it has no
business reaching, and it is negligible for 1-tau a few multiples of k/n.

Everything here is exact.  Treating the two groups as the two populations, the risk
E[Qbar(ell_(k))] is a finite sum against the exact law of the k-th order statistic
under sampling from a discrete P, so there is no Monte Carlo error to hide in and a
violated bound would be a real violation.

Run: python shift2.py
"""
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from shift import prep_with_cov                                  # noqa: E402


def exact_risk(A, B, k, n):
    """E[Qbar(ell_(k))] exactly, P and Q the empirical laws of A and B."""
    v = np.unique(A)
    pa = np.array([np.mean(A > x) for x in v])          # P(draw > v)
    qb = np.array([np.mean(B > x) for x in v])          # Qbar(v)
    # P(ell_(k) <= v) = P(fewer than k of n draws exceed v)
    G = stats.binom.cdf(k - 1, n, pa)
    w = np.diff(np.concatenate([[0.0], G]))
    return float(np.sum(w * qb)), float(np.sum(w * pa))


def tail_delta(A, B, tau):
    """sup |F_P - F_Q| over the upper tail beyond P's tau-quantile."""
    x0 = np.quantile(A, tau)
    grid = np.unique(np.concatenate([A, B]))
    grid = grid[grid >= x0]
    if len(grid) == 0:
        return 0.0
    d = [abs(np.mean(A <= x) - np.mean(B <= x)) for x in grid]
    return float(max(d))


def run(name, A, B, label, mult=8.0):
    n = min(len(A), len(B)) // 2
    ksg = float(stats.ks_2samp(A, B).statistic)
    print("   %-26s n=%-4d global Delta=%.4f" % (label, n, ksg))
    print("      k   nominal    realised   deviation   global   tail    "
          "+slack term   tail bound  holds?")
    for k in (1, 2, 3, 5):
        got, chk = exact_risk(A, B, k, n)
        nom = k / float(n + 1)
        assert abs(chk - nom) < 0.02, "self-check %.4f vs %.4f" % (chk, nom)
        dev = got - nom
        tau = max(0.0, 1.0 - mult * k / n)
        dt = tail_delta(A, B, tau)
        slack = float(stats.binom.cdf(k - 1, n, 1.0 - tau))
        bound = dt + slack
        print("      %-3d %-10.5f %-10.5f %+10.5f  %-8.4f %-7.4f %-13.5f %-11.4f %s"
              % (k, nom, got, dev, ksg, dt, slack, bound,
                 "yes" if abs(dev) <= bound + 1e-12 else "NO"))


def main():
    for name, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                     ("Severson", M.load_severson)):
        lu, grp = prep_with_cov(ld)
        A, B = lu[grp == 0], lu[grp == 1]
        print("=" * 100)
        print("%s: %d units, covariate groups of %d and %d"
              % (name, len(lu), len(A), len(B)))
        run(name, A, B, "covariate shift A->B")
        rng = np.random.default_rng(3)
        q = rng.permutation(len(lu))
        run(name, lu[q[:len(A)]], lu[q[len(A):]], "random split (control)")


if __name__ == "__main__":
    main()
