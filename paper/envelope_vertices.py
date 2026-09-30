"""Theorem 5 as a special case of Proposition 2, and what that costs a grid search.

Proposition 2(ii): a rule is optimal at some cost ratio exactly when its point is
a vertex of the lower-left convex envelope of the family's achievable set.  Apply
it to the threshold family rather than the age family.

As ell grows, P_f falls and omega rises, and at ell_0 the curve reaches P_f = 0
and turns horizontal: every ell > ell_0 has the same P_f and a larger omega, so it
is dominated.  That is Theorem 5(i), obtained by dominance instead of calculus.
The turn itself is a vertex, and a vertex is exposed not by one supporting line
but by a cone of them, so it is optimal on an INTERVAL of cost ratios rather than
at a single one; chi* is the lower end of that interval and the upper end is
infinity because no vertex lies beyond.  That is Theorem 5(ii).

The sharp consequence is about counting.  On a finite fleet P_f only takes the
values k/n, so the achievable set is finite and its envelope has finitely many
vertices --- typically very few.  A grid search over hundreds of thresholds is
therefore really a search over a handful of candidates, and the empirical
minimiser can only ever land on one of them.  That predicts the instability
already measured in Appendix C: the argmin was seen to move between draws on
37-45% of resamples, and it can only move among vertices.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 260


def family(ld, seed=0):
    """The threshold family's (omega, P_f) on a grid, plus the grid and ell_0."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .30 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    om = 1.0 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    return g, om, pf, float(lu.max()), len(idx)


def envelope_vertices(om, pf):
    """Vertices of the lower-left convex envelope, as indices into the inputs."""
    o = np.lexsort((pf, om))
    x, y = om[o], pf[o]
    st = []
    for i in range(len(x)):
        while len(st) >= 2:
            a, b = st[-2], st[-1]
            cr = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cr <= 1e-15:
                st.pop()
            else:
                break
        st.append(i)
    v = o[np.array(st)]
    # keep only the Pareto-undominated part: nothing with a larger omega at equal P_f
    keep = [j for j in v if not np.any((om < om[j] - 1e-12) & (pf <= pf[j] + 1e-12))]
    return np.array(sorted(set(keep.__iter__()), key=lambda j: om[j]))


def chi_interval(om, pf, verts):
    """Cost-ratio interval on which each vertex is the optimum."""
    out = []
    for j in verts:
        L = (1.0 + 0.0) / (1.0 - om[j])          # placeholder, recomputed per chi
        out.append(j)
    return out


def run(name, ld):
    g, om, pf, l0, n = family(ld)
    v = envelope_vertices(om, pf)
    k0 = int(np.argmin(np.abs(g - l0)))
    zero = np.nonzero(pf <= 1e-12)[0]
    kink = int(zero.min()) if len(zero) else -1

    # which grid points are ever the empirical minimiser, over a sweep of chi?
    chis = np.geomspace(0.2, 300, 400)
    winners = set()
    for c in chis:
        L = (1.0 + c * pf) / (1.0 - om)
        winners.add(int(np.argmin(L)))

    print("=" * 76)
    print("%-9s n=%d  grid=%d" % (name, n, NGRID))
    print("   envelope vertices                     : %d of %d grid points (%.1f%%)"
          % (len(v), NGRID, 100 * len(v) / NGRID))
    print("   distinct minimisers over chi in [0.2,300] : %d" % len(winners))
    print("   kink index %d ; is it a vertex? %s ; is it ever the minimiser? %s"
          % (kink, kink in set(v.tolist()), kink in winners))
    ws = sorted(winners)
    print("   winners at grid indices %s" % ws[:12])
    # the cost ratio at which the kink takes over
    took = [c for c in chis if int(np.argmin((1.0 + c * pf) / (1.0 - om))) == kink]
    if took:
        print("   kink is optimal for chi >= %.2f (sweep upper end %.0f)"
              % (min(took), chis[-1]))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
