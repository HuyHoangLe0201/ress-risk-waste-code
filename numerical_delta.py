"""Numerical delta method against the percentile bootstrap, with a control.

Run 1 said the two intervals differ by a factor of 2 to 5 at the minimum.  Before
believing that, two checks.

NEGATIVE CONTROL.  Apply the identical machinery to a functional that IS fully
differentiable: the cost at a fixed grid point, phi(theta) = theta_k for fixed k.
There the numerical delta method must reproduce the percentile interval, because
the directional derivative is linear.  If it does not, the discrepancy at the
minimum is my arithmetic, not the estimand.

STABILITY.  Repeat over calibration/validation splits.  A ratio that swings with
the seed is a property of one draw, not of the fleet.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NBOOT = 3000
NGRID = 120
ALPHA = 0.05
SEEDS = (0, 1, 2, 3, 4)


def pieces(ld, seed):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    lu_cal = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu_cal, .05),
                    np.quantile(lu_cal, .999) + .28 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in val] for e in g])
    return C[:, :, 0], C[:, :, 1], Tb, len(val)


def curve(num, den, Tb, idx=None):
    if idx is None:
        return num.mean(1) / den.mean(1) * Tb
    return num[:, idx].mean(1) / den[:, idx].mean(1) * Tb


def intervals(theta, star, n, phi_fn, eps):
    """Percentile interval of phi(theta*) and the numerical-delta interval."""
    naive = np.array([phi_fn(s) for s in star])
    lo_n, hi_n = np.quantile(naive, [ALPHA / 2, 1 - ALPHA / 2])
    phi = phi_fn(theta)
    Z = np.sqrt(n) * (star - theta)
    D = np.array([(phi_fn(theta + eps * z) - phi) / eps for z in Z])
    lo = phi - np.quantile(D, 1 - ALPHA / 2) / np.sqrt(n)
    hi = phi - np.quantile(D, ALPHA / 2) / np.sqrt(n)
    return (lo_n, hi_n), (lo, hi), phi, naive.mean()


def run(name, ld):
    rows_min, rows_ctl, moved, optim = [], [], [], []
    for seed in SEEDS:
        num, den, Tb, n = pieces(ld, seed)
        theta = curve(num, den, Tb)
        rng = np.random.default_rng(100 + seed)
        B = rng.integers(0, n, size=(NBOOT, n))
        star = np.array([curve(num, den, Tb, b) for b in B])
        eps = n ** -0.25
        kstar = int(theta.argmin())

        (a, b_, phi, mstar) = intervals(theta, star, n, lambda t: t.min(), eps)
        rows_min.append(((b_[1] - b_[0]) / (a[1] - a[0]), a, b_, phi))
        moved.append((star.argmin(1) != kstar).mean())
        optim.append((phi - mstar) / phi)

        # control: the SAME grid point, treated as a fixed differentiable target
        (c, d, _, _) = intervals(theta, star, n, lambda t, k=kstar: t[k], eps)
        rows_ctl.append((d[1] - d[0]) / (c[1] - c[0]))

    r = np.array([x[0] for x in rows_min])
    print("=" * 76)
    print("%-9s n=%d   over %d splits" % (name, n, len(SEEDS)))
    print("   MIN over grid   : width ratio ndm/percentile  median %.2f   range %.2f-%.2f"
          % (np.median(r), r.min(), r.max()))
    print("   CONTROL fixed k : width ratio ndm/percentile  median %.2f   range %.2f-%.2f"
          "   <- must be ~1"
          % (np.median(rows_ctl), min(rows_ctl), max(rows_ctl)))
    print("   bootstrap argmin moves off the sample argmin on %.0f%% of draws"
          % (100 * np.mean(moved)))
    print("   sample minimum is optimistic by %.2f%% on average"
          % (100 * np.mean(optim)))
    a = rows_min[0][1]
    b_ = rows_min[0][2]
    print("   seed 0: percentile [%.3f, %.3f]   numerical delta [%.3f, %.3f]"
          % (a[0], a[1], b_[0], b_[1]))
    return np.median(r), np.median(rows_ctl)


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
