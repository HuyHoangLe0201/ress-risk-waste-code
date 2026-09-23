r"""Which secant actually attains kappa*?

Theorem 5 defines kappa* as a maximum over the whole failure branch,

    kappa* = max_{ell < ell_0} (omega_0 - omega(ell)) / (P_f(ell) (1 - omega_0)).

Equation (12) then replaces that maximum by the single point where exactly one calibration
unit fails, P_f = 1/n, and reads the entire n-dependence off the top spacing of the order
statistics. The manuscript asserts the substitution -- "the binding secant runs from the
kink to the last point of the failure branch" -- without a condition under which it holds.
It is a statement about the shape of the frontier, and an achievable set need not have that
shape.

So it is measured. For each fleet and split, find the argmax over the failure branch and
report how many calibration units fail there. If that count is 1 every time, the
substitution is an empirical regularity of these fleets and should be labelled as one; if
it is not, Equation (12) does not follow from Theorem 5 even here.
"""
import sys
from collections import Counter

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

SPLITS = 40
NGRID = 400


def branch(seq, life, seed):
    """omega and P_f along the threshold family on the calibration half."""
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    l0 = lu.max()
    g = np.linspace(np.quantile(lu, .01), l0 + .05 * Tb, NGRID)
    om, pf = [], []
    for e in g:
        per = np.array([M.unit_cost(i, e, R, rm, life) for i in cal])
        om.append(1 - per[:, 1].mean() / Tb)
        pf.append(per[:, 2].mean())
    return np.array(om), np.array(pf), len(cal)


def run(name, loader):
    seq, life = loader()
    counts, ratios = Counter(), []
    for seed in range(SPLITS):
        om, pf, ncal = branch(seq, life, seed)
        z = np.nonzero(pf <= 0)[0]
        if len(z) == 0:
            continue
        z0 = int(z[0])                      # the kink: first threshold with no failures
        b = np.arange(z0)
        b = b[pf[b] > 0]
        if len(b) == 0:
            continue
        om0 = om[z0]
        sec = (om0 - om[b]) / (pf[b] * (1 - om0))
        j = b[int(np.argmax(sec))]
        k = int(round(pf[j] * ncal))        # units failing at the binding point
        counts[k] += 1
        # what Eq. (12) would give instead: the k = 1 point
        one = b[np.argmin(np.abs(pf[b] * ncal - 1.0))]
        s_one = (om0 - om[one]) / (pf[one] * (1 - om0))
        ratios.append(s_one / sec.max())
    tot = sum(counts.values())
    print("\n%s  (%d usable splits, n_cal = %d)" % (name, tot, ncal))
    for k, c in sorted(counts.items()):
        print("   binding point has %3d failing unit(s): %2d splits (%4.0f%%)"
              % (k, c, 100 * c / tot))
    r = np.array(ratios)
    print("   Eq.(12) surrogate / true max:  median %.3f, 10th pct %.3f, min %.3f"
          % (np.median(r), np.percentile(r, 10), r.min()))


if __name__ == "__main__":
    run("FD002", M.FD002)
    run("FD004", M.FD004)
    run("Severson", M.load_severson)
