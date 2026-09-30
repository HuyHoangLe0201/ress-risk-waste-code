"""The optimal cost as a function of the cost ratio, and what it says about monitoring.

Each admissible point contributes to the cost the AFFINE function of chi

    L(chi) = (1 + chi P_f)/(1 - omega) = 1/(1-omega) + chi * P_f/(1-omega),

so the optimal cost of a family is a minimum of affine functions of chi: concave,
piecewise linear, increasing, with one piece per admissible vertex.  Its slope at
large chi is min P_f/(1-omega) over the family.

That last quantity separates the two families completely.  The threshold family
reaches P_f = 0 at ell_0 with finite waste omega_0, so its slope is zero and its
cost SATURATES at 1/(1-omega_0).  Age replacement can reach P_f = 0 only by
replacing at age zero, which wastes everything, so under a lifetime law with
support down to 0 its cost grows without bound.  Hence

    g_CBM / g_AGE  ->  0   as chi -> infinity,

i.e. the value of monitoring is not a number but a function of the cost ratio, and
it is unbounded.  The empirical age curve is the exception that proves it: on a
finite fleet the schedule acquires a spurious zero-failure point at t = min T_u,
with omega = 1 - T_min/Tbar, so the ratio tends to a positive constant instead.
That is exactly the degenerate optimum Section 3.2 already warns about, now with
a consequence attached.

Checked below on the fitted laws and on the empirical curves.
"""
import sys

import numpy as np
from scipy import stats

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

CHIS = np.geomspace(0.5, 3000.0, 60)


def threshold_family(ld, seed=0):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .30 * Tb, 300)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    return om, pf, Tb, np.asarray(life)[idx]


def age_fitted(T, m=200000):
    """(omega, P_f) for age replacement under a fitted lognormal law."""
    sh, loc, sc = stats.lognorm.fit(T, floc=0)
    d = stats.lognorm(sh, scale=sc)
    hi = d.ppf(1 - 1e-9)
    u = np.linspace(0.0, hi, m)
    R = d.sf(u)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(u))])
    Tb = I[-1]
    t = np.linspace(hi * 1e-4, hi, 4000)
    Im = np.interp(t, u, I)
    return 1.0 - Im / Tb, d.cdf(t)


def age_empirical(T):
    """The same, using the fleet's own lifetimes: this is the degenerate version."""
    Tb = T.mean()
    t = np.linspace(T.min() * 0.5, T.max(), 4000)
    Emin = np.array([np.minimum(T, x).mean() for x in t])
    pf = np.array([(T <= x).mean() for x in t])
    return 1.0 - Emin / Tb, pf


def L(om, pf, chi):
    return float(np.min((1.0 + chi * pf) / (1.0 - om)))


def run(name, ld):
    omT, pfT, Tb, T = threshold_family(ld)
    omA, pfA = age_fitted(T)
    omE, pfE = age_empirical(T)

    LT = np.array([L(omT, pfT, c) for c in CHIS])
    LA = np.array([L(omA, pfA, c) for c in CHIS])
    LE = np.array([L(omE, pfE, c) for c in CHIS])

    om0 = float(omT[np.nonzero(pfT <= 1e-12)[0].min()])
    print("=" * 78)
    print("%-9s Tbar=%.1f  omega_0(threshold)=%.3f  saturation 1/(1-w0)=%.3f"
          % (name, Tb, om0, 1 / (1 - om0)))
    print("   chi        :" + "".join("%9.0f" % c for c in CHIS[::12]))
    print("   L_CBM      :" + "".join("%9.3f" % v for v in LT[::12]))
    print("   L_AGE fit  :" + "".join("%9.3f" % v for v in LA[::12]))
    print("   ratio fit  :" + "".join("%9.3f" % v for v in (LT / LA)[::12]))
    print("   ratio emp  :" + "".join("%9.3f" % v for v in (LT / LE)[::12]))
    # concavity of L in chi, and the number of linear pieces
    for tag, Lv in (("CBM", LT), ("AGE fit", LA)):
        d2 = np.diff(Lv, 2)
        print("   %-8s concave in chi? %s   (max positive second difference %.2e)"
              % (tag, bool(np.all(d2 <= 1e-9)), float(d2.max())))
    print("   empirical age zero-failure point: omega = 1 - Tmin/Tbar = %.3f, "
          "cost %.3f" % (1 - T.min() / Tb, 1 / (1 - (1 - T.min() / Tb))))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
