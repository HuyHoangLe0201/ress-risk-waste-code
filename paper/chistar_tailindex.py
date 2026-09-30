"""Does chi-star's growth exponent equal one plus the extreme-value index of ell_u?

chi* is attained at k=1 and scales as n times the gap between the largest and second
largest ell_u.  Under extreme-value theory the top spacing of a sample from a law with index
gamma normalises by a_n ~ n^gamma, so

    chi*  ~  n * (top spacing)  ~  n^{1+gamma} * W

for a non-degenerate W.  That is a falsifiable prediction connecting two quantities this
paper measures independently: the growth exponent of chi* with fleet size, and the tail
index of ell_u estimated in the extreme-value section.

It also carries a consequence worth stating plainly if it survives.  The exponent 1+gamma is
positive whenever gamma > -1, so the empirical chi* would diverge with fleet size and the
population chi* would be infinite -- the finite values this paper reports being finite-sample
artefacts throughout.

Two independent measurements, compared: the moment estimator of gamma on {ell_u}, and the
log-log slope of chi-star-hat against n from subsampling.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402


def moment_gamma(x, k):
    """Dekkers-Einmahl-de Haan moment estimator of the extreme-value index."""
    y = np.sort(x)
    y = y - y.min() + 1.0                       # shift to positive; index is invariant
    n = len(y)
    if k >= n:
        return np.nan
    lo = np.log(y[n - k - 1])
    d = np.log(y[n - k:]) - lo
    M1, M2 = float(np.mean(d)), float(np.mean(d ** 2))
    if M2 <= 0 or M1 <= 0:
        return np.nan
    return M1 + 1.0 - 0.5 / (1.0 - M1 * M1 / M2)


def chistar_k1(paths, Rs, lives, sub):
    """chi* as the k=1 secant, using exact order statistics rather than a grid."""
    Tb = float(lives[sub].mean())
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in sub])
    o = np.sort(lu)[::-1]

    def coords(e):
        w, f = [], []
        for i in sub:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        return float(np.mean(w)) / Tb, float(np.mean(f))

    om0, _ = coords(o[0])
    om1, pf1 = coords(o[1])
    if pf1 <= 0 or om1 >= om0:
        return np.nan
    return (om0 - om1) / (pf1 * (1.0 - om0))


def run(name, ld, reps=250, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (40, 70, 120, 200) if s <= N]

    xs, ys = [], []
    for s in sizes:
        v = [chistar_k1(paths, Rs, lives, rng.choice(N, s, replace=False))
             for _ in range(reps)]
        v = np.array([x for x in v if np.isfinite(x)])
        xs.append(s)
        ys.append(float(np.mean(v)))
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

    gam = [moment_gamma(lu, k) for k in (int(0.10 * N), int(0.15 * N), int(0.20 * N))]
    gam = [g for g in gam if np.isfinite(g)]

    print("=" * 74)
    print("%s   N=%d" % (name, N))
    print("   chi* at n = " + ", ".join("%d:%.2f" % (a, b) for a, b in zip(xs, ys)))
    print("   growth exponent of chi*      %+.2f" % slope)
    print("   moment gamma of ell_u        %s"
          % ", ".join("%+.2f" % g for g in gam))
    if gam:
        print("   predicted exponent 1+gamma   %s   <-- compare with %+.2f"
              % (", ".join("%+.2f" % (1 + g) for g in gam), slope))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
