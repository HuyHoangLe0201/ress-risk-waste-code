"""The structure of the constrained optimum: which two rules, and why two.

chance_split.py establishes that a requirement P_f <= alpha is met by splitting the
fleet between two rules.  This asks the three structural questions that turn that
into a theorem rather than an observation.

  1. WHICH two.  The claim is that they are ADJACENT vertices of the lower-left
     envelope, the pair bracketing alpha.  Checked against the hull directly.

  2. WHY two, and how many when requirements pile up.  One inequality plus the
     simplex is two active constraints, so LP duality gives at most k+1 rules for k
     requirements.  That bound is not the truth here, and the first version of this
     test assumed it was: a second cap was added expecting three rules, the problem
     came back INFEASIBLE, and the table recorded that as agreement because -1 is
     less than 3.  The reason is geometric.  Every operational requirement is an
     upper bound on a quantity increasing in the direction the cost increases, so
     the optimum can never sit where two cuts cross inside the hull -- from there
     one drops onto the hull's own lower boundary and improves.  Two rules
     therefore suffice for ANY number of such requirements, which is checked with
     one, two and three imposed together.  The control is a pair of requirements
     written the wrong way round, as lower bounds: their corner is the lower-left
     of the feasible set, needs three hull points by Caratheodory, and must come
     back as three or the test is not sensitive enough to support the claim.

  3. HOW the price moves with alpha.  Within one edge the two rules and their tie
     price are fixed and only the split moves, so the equivalent price is a STEP
     function of alpha whose steps are the edges.  The count must therefore be
     stable under refining the alpha grid, which is what distinguishes a step
     function from one that merely looks flat at the resolution plotted.

Run: python chance_split2.py
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from chance_split import (family, dual, lower_hull, best_pair, L, TOL)  # noqa: E402


def tie(a, b, i, j):
    return (np.inf if abs(b[j] - b[i]) < TOL
            else (a[i] - a[j]) / (b[j] - b[i]))


def adjacency(name, om, pf, a, b, chi0):
    V = list(lower_hull(a, b))
    pos = {v: k for k, v in enumerate(V)}
    print("  %s chi=%.1f: are the two rules adjacent envelope vertices?" % (name, chi0))
    for alpha in (0.05, 0.02, 0.01, 0.005):
        val, i, j, t = best_pair(a, b, chi0, alpha)
        if min(t, 1 - t) < 1e-9:
            print("     alpha=%-6.3f single rule, vertex=%s" % (alpha, i in pos))
            continue
        both = i in pos and j in pos
        adj = both and abs(pos[i] - pos[j]) == 1
        brack = min(pf[i], pf[j]) <= alpha + 1e-9 <= max(pf[i], pf[j]) + 1e-9
        print("     alpha=%-6.3f both vertices=%-5s adjacent=%-5s "
              "bracket alpha=%-5s  P_f=%.4f,%.4f"
              % (alpha, both, adj, brack, pf[i], pf[j]))


def solve(a, b, chi, rows, n):
    """min a + chi*b over the hull subject to rows <= 0.  Returns support, cost."""
    from scipy.optimize import linprog
    r = linprog(c=a + chi * b,
                A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                A_eq=np.ones((1, n)), b_eq=[1.0],
                bounds=[(0.0, 1.0)] * n, method="highs")
    if not r.success:
        return None, np.nan
    return int(np.sum(r.x > 1e-9)), float(r.fun)


def how_many_rules(name, om, pf, a, b, chi0, alpha=0.01):
    """How many rules does a split need when requirements pile up?

    The LP bound is k+1 for k requirements, but that bound is not the truth here.
    Every operational requirement -- a cap on failures per cycle, on failures per
    unit time, on waste, on preventive replacements -- is an UPPER bound on a
    quantity that increases in the same direction the cost does.  The minimum of
    a+chi*b then cannot sit at a corner where two cuts meet inside the hull,
    because from such a corner one can always move down onto the hull's own lower
    boundary and improve.  So two rules should suffice however many requirements
    are imposed, and the count should not grow with k.

    The control is a pair of requirements written the wrong way round -- lower
    bounds, which no regulator writes but which the proof must exclude.  Their
    corner IS the lower-left of the feasible set, and expressing an interior point
    needs three hull points by Caratheodory.  If that case does not produce three,
    the test is not sensitive enough to support the claim about the other one.
    """
    try:
        import scipy.optimize                                    # noqa: F401
    except ImportError:
        print("     scipy missing; skipped")
        return
    n = len(a)
    val, i, j, t = best_pair(a, b, chi0, alpha)
    p = np.zeros(n)
    p[i] += (1 - t) / (1 - om[i])
    p[j] += t / (1 - om[j])
    p /= p.sum()
    a_star = 1.0 / (1.0 - float(p @ om))
    b_star = float(p @ pf) / (1.0 - float(p @ om))

    cuts = [("P_f<=%.3f" % alpha, b - alpha * a),
            ("failures per lifetime <= %.4f" % (1.15 * b_star), b - 1.15 * b_star),
            ("replacements per lifetime <= %.4f" % (1.15 * a_star), a - 1.15 * a_star)]
    print("     %s chi=%.1f: requirements imposed together" % (name, chi0))
    for k in (1, 2, 3):
        sup, v = solve(a, b, chi0, [c for _, c in cuts[:k]], n)
        print("       %d upper-bound requirement%s -> %s rules, cost %s"
              % (k, "s" if k > 1 else " ",
                 "infeasible" if sup is None else sup,
                 "--" if sup is None else "%.4f" % v))
    lo = [("replacements per lifetime >= %.4f" % (1.02 * a_star),
           1.02 * a_star - a),
          ("failures per lifetime >= %.4f" % (1.02 * b_star), 1.02 * b_star - b)]
    sup, v = solve(a, b, chi0, [c for _, c in lo], n)
    print("       control: 2 lower-bound requirements -> %s rules, cost %s   %s"
          % ("infeasible" if sup is None else sup,
             "--" if sup is None else "%.4f" % v,
             "corner inside the hull, as it must be"
             if sup == 3 else "control did not bite"))


def price_path(name, om, pf, a, b, chi0, lo=0.004, hi=0.06):
    """Is the equivalent price a step function whose steps are the edges?"""
    V = list(lower_hull(a, b))
    for m in (200, 800):
        grid = np.linspace(lo, hi, m)
        prices, edges = [], []
        for al in grid:
            val, i, j, t = best_pair(a, b, chi0, al)
            if min(t, 1 - t) < 1e-9:
                continue
            prices.append(round(float(tie(a, b, i, j)), 9))
            edges.append((min(i, j), max(i, j)))
        u = sorted(set(prices))
        ue = set(edges)
        in_range = [v for v in V if lo <= pf[v] <= hi]
        print("     %s chi=%.1f grid=%-4d  distinct prices=%d  distinct edges=%d"
              "  vertices with P_f in range=%d"
              % (name, chi0, m, len(u), len(ue), len(in_range)))
    print("       prices: %s" % ", ".join("%.3f" % v for v in u))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        om, pf, n = family(ld)
        a, b = dual(om, pf)
        print("=" * 92)
        adjacency(nm, om, pf, a, b, 0.3)
        print("  how many rules does a split need?")
        how_many_rules(nm, om, pf, a, b, 0.3)
        print("  price path")
        price_path(nm, om, pf, a, b, 0.3)
