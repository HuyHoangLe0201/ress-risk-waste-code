"""Which points of the risk-waste plane can ever be optimal?

Lemma 1 makes the cost (1 + chi P_f)/(1 - omega), so an iso-cost set is the
straight line 1 + chi P_f = L (1 - omega).  Minimising a cost whose level sets are
straight lines over an achievable set A is a linear-programming question, and
three things follow at once.

  (a) Randomising over rules mixes their coordinates linearly, so the achievable
      set of randomised rules is conv(A).  A linear objective over conv(A) attains
      its minimum at an extreme point, so randomisation never helps.

  (b) A point of A is optimal for SOME chi > 0 exactly when some line of negative
      slope supports A there, i.e. exactly when it is a vertex of the lower-left
      convex envelope of A.

  (c) By Proposition 1 the age-replacement curve is convex iff the law is IFR.
      Under IFR every age is therefore a vertex and is optimal for exactly one
      cost ratio.  Under non-IFR part of the curve lies strictly inside its own
      envelope, and those ages are optimal for NO cost ratio whatever.

(c) is the falsifiable part: the four turbofan fleets are IFR and should have an
empty inadmissible set, the battery fleet is not and should not.  Anything else
refutes the chain.
"""
import sys

import numpy as np
from scipy import stats

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def age_curve(dist, m=4000):
    """(omega, P_f) along the replacement age, plus the age grid."""
    hi = dist.ppf(1 - 1e-6)
    t = np.linspace(dist.ppf(1e-6), hi, m)
    u = np.linspace(0.0, hi, 400000)
    R = dist.sf(u)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(u))])
    Tb = I[-1]
    Imin = np.interp(t, u, I)
    om = 1.0 - Imin / Tb                      # E[(T-t)+]/Tbar
    pf = dist.cdf(t)
    return t, om, pf, Tb


def lower_envelope(om, pf):
    """Indices of the vertices of the lower-left convex envelope of the curve.

    Walk in increasing omega (decreasing P_f) and keep a lower convex chain: a
    point is dropped when its predecessor and successor support a line beneath it.
    """
    o = np.argsort(om)
    om, pf = om[o], pf[o]
    stack = []
    for i in range(len(om)):
        while len(stack) >= 2:
            a, b = stack[-2], stack[-1]
            cross = ((om[b] - om[a]) * (pf[i] - pf[a])
                     - (pf[b] - pf[a]) * (om[i] - om[a]))
            if cross <= 0:                    # b is above the chord a-i
                stack.pop()
            else:
                break
        stack.append(i)
    return o[np.array(stack)], om, pf


def run(name, dist):
    t, om, pf, Tb = age_curve(dist)
    idx, _, _ = lower_envelope(om, pf)

    o = np.argsort(om)
    xs, ys = om[o], pf[o]
    vo = np.argsort(om[idx])
    envy = np.interp(xs, om[idx][vo], pf[idx][vo])
    gap = ys - envy                       # how far the curve sits above its envelope

    hz = dist.pdf(t) / np.maximum(dist.sf(t), 1e-300)
    keep = (pf > 0.01) & (pf < 0.99)
    ifr = bool(np.all(np.diff(hz[keep]) > -1e-12))

    k = keep[o]
    tol = 1e-6 * max(1.0, float(np.ptp(ys)))
    frac = 100 * float((gap[k] > tol).mean())
    print("  %-11s %-8s inadmissible ages %5.1f%%   max gap in P_f %.2e"
          % (name, "IFR" if ifr else "not IFR", frac, float(gap[k].max())))
    return ifr, frac


print("Proposition 2(c): ages optimal for no cost ratio")
print("  parametric laws")
for nm, d in (("lognorm .20", stats.lognorm(0.20, scale=200)),
              ("lognorm .45", stats.lognorm(0.45, scale=200)),
              ("Weibull 2.5", stats.weibull_min(2.5, scale=200)),
              ("Weibull 0.8", stats.weibull_min(0.8, scale=200)),
              ("gamma 6", stats.gamma(6.0, scale=40))):
    run(nm, d)

print("  fitted fleet laws")
FLEETS = [("FD001", lambda: M.load_cmapss("train_FD001.txt")),
          ("FD002", M.FD002),
          ("FD003", lambda: M.load_cmapss("train_FD003.txt")),
          ("FD004", M.FD004),
          ("Severson", M.load_severson)]
for nm, ld in FLEETS:
    seq, life = ld()
    T = np.asarray(life, float)
    sh, loc, sc = stats.lognorm.fit(T, floc=0)
    run(nm, stats.lognorm(sh, scale=sc))
