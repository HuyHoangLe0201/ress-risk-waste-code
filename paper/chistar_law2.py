r"""Does chi-star refuse to concentrate on the real fleets, as the limit law says?

chistar_law.py identifies the limit of the top spacing and shows, on simulated laws,
that its coefficient of variation converges to a constant rather than to zero, with a
control statistic that does shrink at the usual rate.  That is the mechanism.  This
tests the conclusion where it matters: on chi-star itself, computed from its own
definition

    chi* = max_{l < l_0} (omega_0 - omega(l)) / (P_f(l)(1 - omega_0)),

over sub-fleets of growing size drawn from each fleet.  If chi-star were consistently
estimable its spread would fall like n^{-1/2}; the claim is that it does not fall at
all, because the whole n-dependence sits in a spacing between the two largest units and
no fleet size makes that spacing better determined.

The same run reports the shape.  Divided by its own mean, chi-star should follow
W_gamma with the gamma implied by the fleet's own growth exponent, and in particular
should not look like a statistic with a shrinking spread.

The waste and failure tables are built once for all units and all grid points, so a
sub-fleet is a column selection; deriving the grid from each sub-fleet instead would
move the grid with n and confound the effect being measured, which is why the grid is
fixed outside the loop.

Run: python chistar_law2.py
"""
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402
from chistar_law import W                                        # noqa: E402


def tables(ld, ngrid=200):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    lu = np.asarray(lu, float)
    grid = np.linspace(np.quantile(lu, 0.02), lu.max(), ngrid)
    Wm = np.zeros((ngrid, N))
    Fm = np.zeros((ngrid, N))
    for u in range(N):
        for k, e in enumerate(grid):
            h = np.nonzero(paths[u] <= e)[0]
            if len(h) == 0 or Rs[u][h[0]] <= 0:
                Fm[k, u] = 1.0
            else:
                Wm[k, u] = float(Rs[u][h[0]])
    return Wm, Fm, np.asarray(lives, float), N


def chistar(Wm, Fm, lives, sub):
    om = Wm[:, sub].mean(1) / lives[sub].mean()
    pf = Fm[:, sub].mean(1)
    om0 = float(om.max())
    k = (pf > 0) & (om < om0)
    if not k.any() or om0 >= 1:
        return np.nan
    return float(np.max((om0 - om[k]) / (pf[k] * (1.0 - om0))))


def run(name, ld, gamma, reps=1200, seed=0):
    Wm, Fm, lives, N = tables(ld)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (25, 50, 100, 200) if s <= N]
    print("=" * 92)
    print("%s  N=%d   growth exponent implies gamma=%.2f" % (name, N, gamma))
    # The control is computed on the SAME sub-fleets: an ordinary fleet average,
    # omega at a fixed mid-grid threshold, which is consistently estimable.  Both
    # statistics then carry the same finite-population and grid artefacts, so the
    # comparison of how their spreads move with n is immune to those, which the
    # absolute size of either coefficient of variation is not.
    kmid = Wm.shape[0] // 2
    print("   n      median chi*  CV chi*   CV of omega (control)   ratio of the two"
          "   KS vs W_gamma")
    first = {}
    for n in sizes:
        vals, ctl = [], []
        for _ in range(reps):
            sub = rng.choice(N, size=n, replace=False)
            v = chistar(Wm, Fm, lives, sub)
            if np.isfinite(v):
                vals.append(v)
                ctl.append(Wm[kmid, sub].mean() / lives[sub].mean())
        v, c = np.array(vals), np.array(ctl)
        w = W(rng, gamma, reps=40000)
        cv_v, cv_c = v.std() / v.mean(), c.std() / c.mean()
        first.setdefault("v", cv_v)
        first.setdefault("c", cv_c)
        print("   %-6d %-12.4f %-9.3f %-23.4f %-18.2f %.4f"
              % (n, np.median(v), cv_v, cv_c, cv_v / cv_c,
                 float(stats.ks_2samp(v / v.mean(), w / w.mean()).statistic)))
    print("   over this range of n the control's spread fell by %.2f and chi*'s by "
          "%.2f;\n   a consistently estimable statistic would fall by %.2f"
          % (first["c"] / cv_c, first["v"] / cv_v,
             np.sqrt(sizes[-1] / float(sizes[0]))))


if __name__ == "__main__":
    # gamma = growth exponent - 1, from the exponents the paper fits
    for nm, ld, g in (("FD002", M.FD002, -0.18), ("FD004", M.FD004, -0.27),
                      ("Severson", M.load_severson, 0.70)):
        run(nm, ld, g)
