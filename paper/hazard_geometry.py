"""Two structural facts about the age-replacement trajectory, and a check of both.

In the risk-waste plane the age family traces a curve parameterised by the
replacement age.  Differentiating the coordinates of Theorem 2,

    d(omega)/dz  = -nu R_Z(z),     d(P_f)/dz = f_Z(z)
    =>  d(omega)/d(P_f) = -nu / h_Z(z),

so the SLOPE of the age-replacement trajectory is minus the coefficient of
variation over the hazard rate.  Two consequences follow at once.

(1) OPTIMALITY.  Iso-cost lines have slope d(omega)/d(P_f) = -chi/L, so tangency
    gives h_Z(z*) = nu L* / chi, i.e. in the untransformed scale

        h_T(t*) = L* / (chi Tbar).

    This should reproduce the classical Barlow-Proschan condition exactly; if it
    does not, the coordinate system is wrong somewhere.

(2) CONVEXITY.  The slope -nu/h_Z increases with z exactly when h_Z increases, so
    the trajectory is convex in the plane if and only if the lifetime law is IFR.
    Under DFR it is concave and the tangency is a maximum, which is the classical
    statement that preventive age replacement cannot pay.

Both are checked numerically below against a direct search, on fitted lognormal
and Weibull laws and on the empirical fleets.
"""
import sys

import numpy as np
from scipy import optimize, stats

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def survival_grid(dist, m=200000):
    """A fine grid and the cumulative integral of the survival function.

    E[min(T,t)] = int_0^t R(u) du and E[(T-t)+] = Tbar - E[min(T,t)], so one
    cumulative trapezoid gives every quantity below without per-point quadrature.
    """
    hi = dist.ppf(1 - 1e-9)
    u = np.linspace(0.0, hi, m)
    R = dist.sf(u)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(u))])
    return u, R, I


def curves(dist, chi):
    """Normalised cost, hazard, P_f and omega along the replacement age."""
    u, R, I = survival_grid(dist)
    Tb = I[-1]                       # int_0^inf R = E[T]
    Pf = 1.0 - R
    om = 1.0 - I / Tb                # E[(T-t)+]/Tbar
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (1.0 + chi * Pf) / (I / Tb)
        h = dist.pdf(u) / np.maximum(R, 1e-300)
    return u, L, h, Pf, om, Tb


def check(name, dist, chi):
    u, L, h, Pf, om, Tb = curves(dist, chi)
    ok = np.isfinite(L) & (u > dist.ppf(1e-6))
    j = int(np.nanargmin(np.where(ok, L, np.inf)))
    edge = j <= 1 or j >= len(u) - 2
    return name, chi, h[j], L[j] / (chi * Tb), u[j], L[j], edge


print("PART 1  tangency condition   h_T(t*) =?= L* / (chi Tbar)")
print("%-26s %5s %12s %12s %8s" % ("law", "chi", "hazard", "L*/(chi Tb)", "ratio"))
laws = [
    ("lognormal  s=0.20", stats.lognorm(0.20, scale=200)),
    ("lognormal  s=0.45", stats.lognorm(0.45, scale=200)),
    ("Weibull    k=2.5",  stats.weibull_min(2.5, scale=200)),
    ("Weibull    k=4.0",  stats.weibull_min(4.0, scale=200)),
    ("gamma      a=6",    stats.gamma(6.0, scale=40)),
]
for nm, d in laws:
    for chi in (4.0, 9.0, 29.0):
        nm_, c, h, pred, ts, Ls, edge = check(nm, d, chi)
        if edge:
            print("%-26s %5.0f   boundary optimum (no interior tangency)" % (nm, chi))
        else:
            print("%-26s %5.0f %12.6f %12.6f %8.4f" % (nm, c, h, pred, h / pred))

print()
print("PART 2  convexity of the trajectory vs the IFR property")
print("%-26s %10s %10s %s" % ("law", "min dh", "convex?", "IFR?"))
for nm, d in laws + [("Weibull    k=0.8 (DFR)", stats.weibull_min(0.8, scale=200))]:
    u, L, h, Pf, om, Tb = curves(d, 9.0)
    m = (Pf > 0.01) & (Pf < 0.99)
    d1 = np.gradient(om[m], Pf[m])
    d2 = np.gradient(d1, Pf[m])
    ifr = bool(np.all(np.diff(h[m]) > -1e-12))
    conv = bool(np.mean(d2 > 0) > 0.95)
    print("%-26s %10.2e %10s %s" % (nm, np.diff(h[m]).min(), conv, ifr))

print()
print("PART 3  the same tangency on the fitted fleet laws, chi = 9")
FLEETS = [("FD001", lambda: M.load_cmapss("train_FD001.txt")),
          ("FD002", M.FD002),
          ("FD003", lambda: M.load_cmapss("train_FD003.txt")),
          ("FD004", M.FD004),
          ("Severson", M.load_severson)]
for nm, ld in FLEETS:
    seq, life = ld()
    T = np.asarray(life, float)
    sh, loc, sc = stats.lognorm.fit(T, floc=0)
    d = stats.lognorm(sh, scale=sc)
    _, _, h, pred, ts, Ls, edge = check(nm, d, 9.0)
    u, L, hz, Pf, om, Tb = curves(d, 9.0)
    m = (Pf > 0.01) & (Pf < 0.99)
    ifr = bool(np.all(np.diff(hz[m]) > -1e-12))
    d1 = np.gradient(om[m], Pf[m]); d2 = np.gradient(d1, Pf[m])
    tag = "boundary" if edge else "%.4f" % (h / pred)
    print("  %-9s n=%3d cv=%.3f t*=%6.1f L*=%.4f  h/pred=%-9s IFR=%-6s convex frac=%.2f"
          % (nm, len(T), d.std() / d.mean(), ts, Ls, tag, ifr, float(np.mean(d2 > 0))))
