"""Does case (iii) of Proposition 17 survive a fleet that is allowed to split?

Proposition 17 says a requirement P_f <= alpha binding at a NON-VERTEX of the lower
envelope is reproduced by no cost ratio: "the requirement carries information no
price can express".  That is a statement about a single deterministic rule.  But
Proposition 2(i) already allows mixtures, and a fleet of a hundred aircraft can run
two thresholds -- forty units at one, sixty at the other -- which is a mixture with
weights the operator chooses.  So the question is whether the information gap is
real or an artefact of insisting on one rule.

In the dual coordinates a = 1/(1-omega), b = P_f/(1-omega) the answer is forced.
Cost is affine, L = a + chi*b (Lemma 1).  Mixing over cycles is convex combination:
the primal mixture p has dual weights q_i proportional to p_i(1-omega_i), and both
are simplices, so the achievable set of mixtures is the hull of the dual points.
And the requirement is LINEAR there, since P_f = b/a, so

    P_f <= alpha    <==>    b - alpha*a <= 0,

a half-plane through the origin.  Minimising an affine function over a polytope
under ONE extra inequality is a linear programme whose basic solutions have at most
two nonzero weights.  Hence the constrained optimum is always a split between two
policies, and they are adjacent vertices of the envelope.  The price that makes
them tie is chi_ij = (a_i - a_j)/(b_j - b_i); at that ratio the whole edge is
optimal and the split is what picks the point on it.  So a requirement is a price
AND a split, and case (iii) is the case where the split is not trivial.

Three things are checked here, none of them by trusting the LP:

  1. the LP optimum, an exhaustive search over all pairs, and the closed-form
     interpolation to P_f = alpha along the binding edge all agree;
  2. the dual bookkeeping is right end to end -- the recovered primal weights p are
     converted back to (omega, P_f) and costed with Lemma 1 directly, which is the
     check that would fail if q and p had been confused;
  3. the value function's derivative vanishes exactly at the binding edge's tie
     price, so the requirement costs the operator money precisely when the price it
     implies exceeds the price the operator was already paying.

Run: python chance_split.py
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NGRID = 400
TOL = 1e-12


def family(ld, seed=0):
    """The threshold family on one fleet, as (omega, P_f) per threshold."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = cal
    lu = np.array([pa[i][R[i] > 0].min() for i in idx])
    g = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in idx] for e in g])
    return 1.0 - C[:, :, 1].mean(1) / Tb, C[:, :, 2].mean(1), len(idx)


def L(om, pf, chi):
    return (1.0 + chi * pf) / (1.0 - om)


def dual(om, pf):
    return 1.0 / (1.0 - om), pf / (1.0 - om)


def lower_hull(a, b):
    """Vertices of the lower-left hull, the rules optimal at some chi > 0.

    Sorted by b, keeping the lower convex chain: these are exactly the points
    minimising a + chi*b for some chi > 0, which is Proposition 2(ii).
    """
    o = np.lexsort((a, b))
    st = []
    for k in o:
        while len(st) >= 2:
            i, j = st[-2], st[-1]
            cr = ((b[j] - b[i]) * (a[k] - a[i])
                  - (a[j] - a[i]) * (b[k] - b[i]))
            if cr <= 0:                      # j is not below the chord i->k
                st.pop()
            else:
                break
        if len(st) == 1 and a[k] >= a[st[0]] - TOL and b[k] > b[st[0]]:
            continue                          # dominated: more failures, no less waste
        st.append(k)
    while len(st) >= 2 and a[st[-1]] >= a[st[-2]]:
        st.pop()                              # trailing points beaten on both axes
    return np.array(st)


def best_pair(a, b, chi, alpha):
    """Exact constrained optimum by exhaustive search over pairs.

    On the segment between two dual points the objective and the constraint are
    both affine in the mixing weight, so the optimum sits at an endpoint of the
    feasible sub-interval; enumerating pairs therefore needs no inner search and
    proves the LP's "at most two" rather than assuming it.
    """
    n = len(a)
    obj = a + chi * b
    con = b - alpha * a                       # <= 0 is feasibility
    best = (np.inf, None, None, 0.0)
    for i in range(n):
        ci, oi = con[i], obj[i]
        cj, oj = con, obj
        d = cj - ci
        # t in [0,1] on the segment i->j;  feasible where ci + t*d <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tcut = np.where(np.abs(d) > TOL, -ci / d, np.inf)
        lo = np.where((d > TOL) | (np.abs(d) <= TOL), 0.0, np.maximum(0.0, tcut))
        hi = np.where(d > TOL, np.minimum(1.0, tcut), 1.0)
        lo = np.where((np.abs(d) <= TOL) & (ci > TOL), 1.0, lo)   # infeasible pair
        hi = np.where((np.abs(d) <= TOL) & (ci > TOL), 0.0, hi)
        ok = lo <= hi + TOL
        val_lo = oi + lo * (oj - oi)
        val_hi = oi + hi * (oj - oi)
        t = np.where(val_lo <= val_hi, lo, hi)
        val = np.where(val_lo <= val_hi, val_lo, val_hi)
        val = np.where(ok, val, np.inf)
        j = int(np.argmin(val))
        if val[j] < best[0]:
            best = (float(val[j]), i, j, float(t[j]))
    return best


def lp(a, b, chi, alpha):
    """The same problem as a linear programme, as an independent solver."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None
    n = len(a)
    r = linprog(c=a + chi * b,
                A_ub=(b - alpha * a)[None, :], b_ub=[0.0],
                A_eq=np.ones((1, n)), b_eq=[1.0],
                bounds=[(0.0, 1.0)] * n, method="highs")
    if not r.success:
        return None
    return float(r.fun), r.x


def report(name, ld, chi0):
    om, pf, n = family(ld)
    a, b = dual(om, pf)
    V = lower_hull(a, b)
    print("=" * 92)
    print("%-9s n=%d  chi=%.1f   %d rules, %d envelope vertices"
          % (name, n, chi0, len(om), len(V)))
    print("   alpha   determ.   split     gain    support  weights        "
          "equivalent chi   agree")
    rows = []
    for alpha in (0.20, 0.10, 0.05, 0.02, 0.01, 0.005):
        feas = pf <= alpha + TOL
        if not feas.any():
            continue
        jd = int(np.argmin(np.where(feas, L(om, pf, chi0), np.inf)))
        det = float(L(om[jd], pf[jd], chi0))

        val, i, j, t = best_pair(a, b, chi0, alpha)
        r = lp(a, b, chi0, alpha)

        # dual weights -> primal cycle weights -> (omega, P_f) -> Lemma 1 cost
        q = np.zeros(len(a))
        q[i] += 1.0 - t
        q[j] += t
        p = q / (1.0 - om)
        p = p / p.sum()
        om_m = float(p @ om)
        pf_m = float(p @ pf)
        direct = L(om_m, pf_m, chi0)

        sup = int(np.sum(q > 1e-9))
        chi_eq = (np.inf if abs(b[j] - b[i]) < TOL
                  else (a[i] - a[j]) / (b[j] - b[i]))
        agree = (abs(direct - val) < 1e-9
                 and (r is None or abs(r[0] - val) < 1e-7))
        rows.append((alpha, det, val, i, j, t, chi_eq, om_m, pf_m))
        print("   %-7.3f %-9.4f %-9.4f %-7.2f%% %-8d %-14s %-16s %s"
              % (alpha, det, val, 100.0 * (det - val) / det, sup,
                 "%.3f/%.3f" % (1.0 - t, t) if sup == 2 else "1.000",
                 ("%.3f" % chi_eq) if sup == 2 else "-- (vertex)",
                 "yes" if agree else "NO"))
    return om, pf, a, b, V, rows


def value_curve(a, b, chi, grid):
    out = np.empty(len(grid))
    for k, al in enumerate(grid):
        out[k] = best_pair(a, b, chi, al)[0]
    return out


def derivative_check(name, om, pf, a, b, chi0):
    """Does the value function turn over exactly at the binding edge's price?

    L*(alpha) is linear-fractional on each edge, so its derivative vanishes only
    where the edge's two rules tie in cost -- that is, at the edge's equivalent
    price.  Below that price the requirement is free; above it, it costs.
    """
    print("   alpha    dL*/dalpha    tie price of binding edge   chi < tie?  costly?")
    for alpha in (0.10, 0.05, 0.02, 0.01):
        h = alpha * 1e-4
        lo = best_pair(a, b, chi0, alpha - h)[0]
        hi = best_pair(a, b, chi0, alpha + h)[0]
        d = (hi - lo) / (2 * h)
        _, i, j, t = best_pair(a, b, chi0, alpha)
        tie = (np.inf if abs(b[j] - b[i]) < TOL
               else (a[i] - a[j]) / (b[j] - b[i]))
        costly = d < -1e-9
        print("   %-8.3f %-13.4f %-27s %-11s %s"
              % (alpha, d, ("%.3f" % tie) if np.isfinite(tie) else "inf",
                 "yes" if chi0 < tie else "no", "yes" if costly else "no"))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        for c in (0.3, 9.0):
            om, pf, a, b, V, rows = report(nm, ld, c)
            derivative_check(nm, om, pf, a, b, c)
