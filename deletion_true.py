r"""The cost of deleting one admissible policy, with the substitute the fleet actually has.

An earlier version of this check replaced a deleted hull vertex by the CHORD joining its
two neighbours.  That is the envelope of the remaining VERTICES, not of the remaining
policies: a point that is not a vertex lies above the hull but can lie below that chord,
inside the triangle the chord cuts off, and when the vertex goes it is that point which
takes over.  Using the chord overstates the rise -- on FD002 by a factor of five -- and it
was that overstatement, not the paper, which said every vertex is necessary.

Done properly: delete the vertex from the full set of policies, rebuild the envelope from
what is left, and take the largest rise over cost ratios, each measured against the
lattice tolerance (chi/n) L*(chi) of Section 4 at the same chi.

Two readings follow, and they are different questions.  Against the remaining POLICIES a
vertex is cheap to lose, because the fleet carries near-substitutes.  Against the remaining
VERTICES it is not.  The first is what "no vertex is individually necessary" means and it
is what the covering formulation is built on; the second is the geometry of the hull, and
the gap between them is a measure of how thick the policy set is around its own envelope.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
import minimax as MX                                             # noqa: E402
import hullreal as HR                                            # noqa: E402
from deletion import dual                                        # noqa: E402
from hullscale import lower_hull                                 # noqa: E402

CHI = np.logspace(-3, 5, 1200)


def envelope(b, a, chi):
    return np.min(a[None, :] + b[None, :] * chi[:, None], axis=1)


def deletion_ratios(b, a, n, against_vertices=False):
    """For each hull vertex, the worst rise from dropping it, over the tolerance."""
    h = lower_hull(b, a)
    if len(h) < 3:
        return None
    base = envelope(b, a, CHI)
    tol = CHI / n * base
    out = []
    for j in h[1:-1]:
        if against_vertices:
            keep = np.array([k for k in h if k != j])
        else:
            keep = np.array([k for k in range(len(b)) if k != j])
        rise = envelope(b[keep], a[keep], CHI) - base
        out.append(float(np.max(rise / tol)))
    return np.array(out)


def report(name, b, a, n, tag):
    r = deletion_ratios(b, a, n)
    rv = deletion_ratios(b, a, n, against_vertices=True)
    print("%-9s %-10s vertices %3d | vs policies: worst %6.2f, necessary %d"
          "   | vs vertices: worst %6.2f, necessary %d"
          % (name, tag, len(lower_hull(b, a)), r.max(), int((r > 1).sum()),
             rv.max(), int((rv > 1).sum())))
    return r, rv


if __name__ == "__main__":
    print("ratio > 1 means the rise exceeds the fleet's own resolution: the policy "
          "is necessary")
    print("=" * 96)
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        n = len(HR.prep(ld)[0][0])
        for nthr in (50, 300, 800):
            A = MX.achievable(ld, n_thr=nthr)
            b, a = dual(A[:, 1], A[:, 0])
            report(nm, b, a, n, "grid %d" % nthr)
        tab, lives = HR.prep(ld)
        pf, om = HR.pareto(tab, lives, np.arange(n))
        b, a = dual(pf, om)
        report(nm, b, a, n, "Pareto")
        print("-" * 96)
