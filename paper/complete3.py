"""Out-of-sample completeness, done efficiently.

Same question as complete2, same protocol, but the per-unit outcome of every rule
is computed ONCE and the split averages are taken over subsets afterwards.  A rule's
coordinates on any subset are just means of stored per-unit waste and failure
indicators, so two hundred half-splits cost no more than one.

The question: the in-sample sweep found two-parameter rules strictly inside the
constant-threshold frontier, by up to 0.096 in P_f.  A richer family does that
in-sample whether or not it is better.  Choosing the rule on one half and scoring it
on the other says which.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def outcomes(paths, grid, kind):
    """waste[r,u] and fail[r,u] for every rule r and unit u."""
    W = np.zeros((len(grid), len(paths)))
    Fl = np.zeros((len(grid), len(paths)))
    for r, (a, b) in enumerate(grid):
        for u, p in enumerate(paths):
            n = len(p)
            if kind == "line":
                thr = a + b * np.arange(n, dtype=float)
                hit = np.nonzero(p <= thr)[0]
            else:                                   # slope-augmented
                d = np.diff(p, prepend=p[0])
                hit = np.nonzero(p + b * d <= a)[0]
            if len(hit) == 0:
                Fl[r, u] = 1.0
            else:
                W[r, u] = n - 1 - hit[0]
    return W, Fl


def run(name, ld, reps=200, seed=0):
    seq, life = ld()
    R, rm, pa, cal0, val0, _ = M.prep(seq, life, seed=0)
    allu = np.concatenate([cal0, val0])
    paths = [np.asarray(pa[i], float) for i in allu]
    n_u = np.array([len(p) for p in paths], float)
    lo = min(p.min() for p in paths)
    hi = max(p.max() for p in paths)

    A_grid = [(a, 0.0) for a in np.linspace(lo, hi, 160)]
    B_grid = [(a, b) for a in np.linspace(lo, hi, 60)
              for b in np.linspace(-0.6, 0.6, 21)]
    C_grid = [(a, c) for a in np.linspace(lo, hi, 60)
              for c in np.linspace(0.0, 12.0, 21)]

    WA, FA = outcomes(paths, A_grid, "line")
    WB, FB = outcomes(paths, B_grid, "line")
    WC, FC = outcomes(paths, C_grid, "slope")

    chis = [2.0, 5.0, 9.0, 20.0, 50.0]
    rng = np.random.default_rng(seed)
    rec = {c: {"B": [], "C": []} for c in chis}

    for _ in range(reps):
        perm = rng.permutation(len(paths))
        ca, va = perm[:len(perm) // 2], perm[len(perm) // 2:]

        def L(W, Fl, s, chi):
            om = W[:, s].mean(1) / n_u[s].mean()
            pf = Fl[:, s].mean(1)
            return (1.0 + chi * pf) / (1.0 - om)

        for chi in chis:
            la_c, la_v = L(WA, FA, ca, chi), L(WA, FA, va, chi)
            ia = int(np.argmin(la_c))
            for tag, (W, Fl) in (("B", (WB, FB)), ("C", (WC, FC))):
                lc, lv = L(W, Fl, ca, chi), L(W, Fl, va, chi)
                rec[chi][tag].append((la_v[ia], lv[int(np.argmin(lc))]))

    print("=" * 78)
    print("%s   n=%d   %d half-splits, rule chosen on one half and scored on the other"
          % (name, len(paths), reps))
    print("%6s %6s %10s %10s %19s %9s %9s" %
          ("chi", "rich", "A 1-par", "rich", "rich - A (mean+-se)",
           "median", "rich wins"))
    for chi in chis:
        for tag in ("B", "C"):
            z = np.array(rec[chi][tag])
            d = z[:, 1] - z[:, 0]
            se = d.std(ddof=1) / np.sqrt(len(d))
            star = "  *" if abs(d.mean()) > 2 * se else "   "
            print("%6.0f %6s %10.4f %10.4f  %+8.4f +-%.4f%s %9.4f %6d/%d"
                  % (chi, tag, z[:, 0].mean(), z[:, 1].mean(), d.mean(), se,
                     star, np.median(d), int((d < -1e-9).sum()), len(d)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
