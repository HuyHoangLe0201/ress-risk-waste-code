"""Why chi-star never settles: a leading-order formula that makes it obvious.

At the zero-failure point the secant ratio against the point admitting k failures is

    r_k = (omega_0 - omega_k) / ( (k/n) (1 - omega_0) ) = n (omega_0 - omega_k) / (k (1-omega_0)).

Removing the k units with the largest ell_u drops their waste out of the average, so to
leading order omega_0 - omega_k is about k * wbar / (n Tbar) with wbar the typical wasted
life of one of those extreme units.  Substituting, the k cancels and so does the n:

    r_k  ~  wbar / ( Tbar (1 - omega_0) ),

flat in k.  Two things follow at once.  The maximum over k is then decided by fluctuation
rather than by trend, which is why the maximising secant sits a couple of units from the
anchor at every fleet size; and chi* is essentially the wasted life of ONE extreme unit,
normalised, which is why it grows with the fleet and never settles -- a larger fleet simply
finds a more extreme unit.

The test: compare chi-star computed properly against w_max / (Tbar (1 - omega_0)), where
w_max is the wasted life at threshold ell_0 of the single unit attaining ell_0.  If the two
track, the paper's central constant has a one-unit explanation.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402


def chistar_and_formula(paths, Rs, lives, sub, grid):
    Tb = float(lives[sub].mean())
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in sub])
    ell0 = lu.max()

    om, pf = [], []
    for e in grid[grid <= ell0]:
        w, f = [], []
        for i in sub:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        om.append(float(np.mean(w)) / Tb)
        pf.append(float(np.mean(f)))
    om, pf = np.array(om), np.array(pf)
    if len(om) == 0:
        return None
    om0 = om[-1]
    k = (pf > 0) & (om < om0)
    if not k.any():
        return None
    chi = float(np.max((om0 - om[k]) / (pf[k] * (1.0 - om0))))

    # wasted life of the single unit that attains ell_0, at threshold ell_0
    j = sub[int(np.argmax(lu))]
    h = np.nonzero(paths[j] <= ell0)[0]
    w_max = float(Rs[j][h[0]]) if len(h) and Rs[j][h[0]] > 0 else 0.0
    return chi, w_max / (Tb * (1.0 - om0))


def run(name, ld, reps=200, seed=0):
    paths, Rs, lives, lu = prep(ld)
    N = len(paths)
    grid = np.linspace(np.quantile(lu, 0.02), lu.max(), 160)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (40, 80, 140, 220) if s <= N]

    print("=" * 70)
    print("%s   N=%d" % (name, N))
    print("%8s %12s %14s %10s" % ("n", "chi* direct", "one-unit form", "ratio"))
    for s in sizes:
        a, b = [], []
        for _ in range(reps):
            r = chistar_and_formula(paths, Rs, lives,
                                    rng.choice(N, s, replace=False), grid)
            if r:
                a.append(r[0]); b.append(r[1])
        A, B = float(np.mean(a)), float(np.mean(b))
        print("%8d %12.3f %14.3f %10.2f" % (s, A, B, A / B if B > 0 else np.nan))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
