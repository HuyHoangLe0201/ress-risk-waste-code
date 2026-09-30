"""What can a schedule reach, and is monitoring outside it?

Proposition 2 optimises over conv(A) without ever saying what A is.  For rules
that use no unit-level information the answer is exact.  If tau is independent of
T then

    P_f = E_tau[F(tau)],     omega = E_tau[ Ecal(tau) ] / Tbar,

both expectations over the law of tau of the DETERMINISTIC-age coordinates.  So
the achievable set of information-free rules is exactly the convex hull of the age
curve: contained in it because every point is a mixture, and equal to it because
every mixture is realised by randomising the age.  Nothing outside that hull is
reachable by any schedule, however clever, randomised or not.

That converts the paper's headline comparison from a tuning question into a
structural one.  If the threshold family's points lie strictly outside the hull,
then continuous monitoring is not beating a well-chosen age -- it is reaching
operating points no schedule can reach at all.  Tested here by measuring, for each
threshold policy, how far below the age hull it sits.
"""
import sys

import numpy as np
from scipy import stats

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def age_curve(T, m=200000, npts=3000):
    """(omega, P_f) of deterministic age replacement under the fitted law."""
    sh, loc, sc = stats.lognorm.fit(np.asarray(T, float), floc=0)
    d = stats.lognorm(sh, scale=sc)
    hi = d.ppf(1 - 1e-9)
    u = np.linspace(0.0, hi, m)
    R = d.sf(u)
    I = np.concatenate([[0.0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(u))])
    Tb = I[-1]
    t = np.linspace(hi * 1e-5, hi, npts)
    return 1.0 - np.interp(t, u, I) / Tb, d.cdf(t)


def lower_hull(x, y):
    """Vertices of the lower-left convex hull of the point set, sorted by x."""
    o = np.lexsort((y, x))
    x, y = x[o], y[o]
    st = []
    for i in range(len(x)):
        while len(st) >= 2:
            a, b = st[-2], st[-1]
            if (x[b]-x[a])*(y[i]-y[a]) - (y[b]-y[a])*(x[i]-x[a]) <= 0:
                st.pop()
            else:
                break
        st.append(i)
    return x[np.array(st)], y[np.array(st)]


def threshold_family(ld, seed=0):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, 120)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in cal] for e in g])
    return (1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1),
            np.asarray(life)[cal], Tb)


def run(name, ld):
    omC, pfC, T, Tb = threshold_family(ld)
    omA, pfA = age_curve(T)
    hx, hy = lower_hull(omA, pfA)                 # the schedule frontier

    # for each monitoring point, how far below the schedule hull does it sit?
    pf_on_hull = np.interp(omC, hx, hy, left=hy[0], right=hy[-1])
    below = pf_on_hull - pfC                      # >0 means unreachable by a schedule
    frac = float((below > 1e-9).mean())

    # and the reverse reading: at equal P_f, how much less waste?
    om_on_hull = np.interp(pfC[::-1], hy[::-1], hx[::-1])[::-1]
    saved = om_on_hull - omC

    print("=" * 78)
    print("%-9s %d monitoring points vs the schedule hull" % (name, len(omC)))
    print("   outside the hull (unreachable by ANY schedule) : %.0f%%" % (100 * frac))
    print("   at equal P_f, waste saved: median %.3f of mean life, max %.3f"
          % (float(np.median(saved)), float(np.max(saved))))
    print("   at equal waste, P_f reduced by: median %.4f, max %.4f"
          % (float(np.median(below)), float(np.max(below))))
    k0 = int(np.nonzero(pfC <= 1e-12)[0].min())
    print("   the zero-failure point (omega_0=%.3f) sits %.3f in waste below the "
          "hull's P_f=0 point (%.3f)" % (omC[k0], hx[-1] - omC[k0], hx[-1]))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
