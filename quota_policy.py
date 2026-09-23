"""Report at a quota, but which point should actually be RUN?

There is a tension between two things this paper now says.  Theorem 5 prescribes the
zero-failure threshold whenever chi exceeds chi*, which on the turbofan fleets it does by a
wide margin.  Proposition 8 says that threshold is anchored at an extreme and cannot be
located reliably, and G9 says to quote the ceiling at a quota instead.  Reporting at a
quota while running an unlocatable point would be incoherent, so the question is which point
is cheaper to run, not merely which is cheaper to describe.

The comparison is out of sample and uses the paper's own protocol.  Both thresholds are
chosen on a calibration half -- the zero-failure one at the largest observed ell_u, the
quota one at the (1-alpha) quantile -- and both are scored on units neither has seen.  The
zero-failure threshold is chosen at an extreme of the calibration half, so it should
generalise badly; the quota threshold is an interior quantile and should not.  If the quota
policy is competitive, the recommendation is an operating one and not only a reporting one.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep, anchor                                   # noqa: E402


def cost_at(paths, Rs, lives, units, ell, chi):
    w, f = [], []
    for i in units:
        h = np.nonzero(paths[i] <= ell)[0]
        if len(h) == 0 or Rs[i][h[0]] <= 0:
            w.append(0.0); f.append(1.0)
        else:
            w.append(float(Rs[i][h[0]])); f.append(0.0)
    om = float(np.mean(w)) / float(lives[units].mean())
    return np.inf if om >= 1 else (1 + chi * float(np.mean(f))) / (1 - om)


def run(name, ld, reps=400, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (30, 60, 100) if 2 * s <= N]
    alphas = [0.02, 0.05, 0.10]

    print("=" * 80)
    print("%s   N=%d   thresholds chosen on n units, scored on a disjoint half"
          % (name, N))
    for chi in (9.0, 30.0):
        print("  chi=%.0f" % chi)
        print("%9s %12s" % ("n", "zero-failure")
              + "".join("%12s" % ("quota %.2f" % a) for a in alphas))
        for s in sizes:
            zc, qc = [], {a: [] for a in alphas}
            for _ in range(reps):
                p = rng.permutation(N)
                fit, score = p[:s], p[s:s + N // 2]
                zc.append(cost_at(paths, Rs, lives, score, lu[fit].max(), chi))
                for a in alphas:
                    qc[a].append(cost_at(paths, Rs, lives, score,
                                         anchor(lu[fit], a), chi))
            row = "%9d %12.4f" % (s, float(np.mean(zc)))
            for a in alphas:
                row += "%12.4f" % float(np.mean(qc[a]))
            print(row)


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
