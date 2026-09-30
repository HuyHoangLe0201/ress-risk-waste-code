"""Is the headline ratio's drift with fleet size ENTIRELY selection penalty?

Two measurements now exist separately.  The ratio g_CBM/g_AGE falls from 0.87 at n=40 to
0.68 at n=260, and each policy's selection penalty -- the cost of choosing it on n units
rather than on a large pool -- has been measured, the threshold rule's being four to thirty
times the age baseline's.

If the drift is nothing but selection, then subtracting each penalty from its own cost must
remove it:

    r_infinity(n) = [ g_CBM(n) - E_CBM(n) ] / [ g_AGE(n) - E_AGE(n) ]

should be flat in n, and its common value is the ratio the two policies would show if both
were chosen on a large fleet.  If it is not flat, something other than selection is moving
the ratio and the explanation in the paper is incomplete.

All four quantities are computed on the SAME draws, which matters: measuring the ratio and
the penalties on different draws would let sampling noise masquerade as a residual.
"""
import sys

import numpy as np
from scipy import stats

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def prep(ld):
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.minimum.accumulate(np.asarray(pa[i], float)) for i in idx]
    Rs = [np.asarray(R[i], float) for i in idx]
    lives = np.array([float(len(p)) for p in paths])
    return paths, Rs, lives


def cbm_pick(paths, Rs, lives, fit, chi, grid):
    best, arg = np.inf, grid[0]
    for e in grid:
        w, f = [], []
        for i in fit:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        om = float(np.mean(w)) / float(lives[fit].mean())
        if om < 1:
            c = (1 + chi * float(np.mean(f))) / (1 - om)
            if c < best:
                best, arg = c, e
    return arg


def cbm_score(paths, Rs, lives, units, e, chi):
    w, f = [], []
    for i in units:
        h = np.nonzero(paths[i] <= e)[0]
        if len(h) == 0 or Rs[i][h[0]] <= 0:
            w.append(0.0); f.append(1.0)
        else:
            w.append(float(Rs[i][h[0]])); f.append(0.0)
    om = float(np.mean(w)) / float(lives[units].mean())
    return np.inf if om >= 1 else (1 + chi * float(np.mean(f))) / (1 - om)


def age_pick(lives, fit, chi, ngrid=1200):
    T = lives[fit]
    sh, loc, sc = stats.lognorm.fit(T, floc=0)
    d = stats.lognorm(sh, loc, sc)
    ts = np.linspace(T.min() * 0.3, T.max() * 1.05, ngrid)
    S = d.sf(ts)
    L = np.concatenate([[0.0], np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(ts))]) + ts[0]
    return ts[int(np.argmin((1 + chi * d.cdf(ts)) / (L / float(d.mean()))))]


def age_score(lives, units, t, chi):
    S = lives[units]
    om = 1.0 - float(np.minimum(S, t).mean()) / float(S.mean())
    return np.inf if om >= 1 else (1 + chi * float(np.mean(S <= t))) / (1 - om)


def run(name, ld, chi=9.0, reps=250, seed=0):
    paths, Rs, lives = prep(ld)
    N = len(paths)
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(N)])
    GRID = np.linspace(lu.min(), lu.max(), 140)      # fixed across all n
    rng = np.random.default_rng(seed)
    sizes = [s for s in (15, 25, 40, 65, 100) if 2 * s <= N - N // 3]

    print("=" * 80)
    print("%s   N=%d, chi=%.0f" % (name, N, chi))
    print("%7s %10s %10s %10s %10s %14s" %
          ("n", "g_CBM", "g_AGE", "E_CBM", "E_AGE", "debiased ratio"))
    for s in sizes:
        gc, ga, ec, ea, rr = [], [], [], [], []
        for _ in range(reps):
            p = rng.permutation(N)
            fit, score = p[:s], p[s:s + N // 3]
            pool = p[:N - N // 3]                    # the large-fleet reference
            e_s, e_p = (cbm_pick(paths, Rs, lives, fit, chi, GRID),
                        cbm_pick(paths, Rs, lives, pool, chi, GRID))
            t_s, t_p = age_pick(lives, fit, chi), age_pick(lives, pool, chi)
            a = cbm_score(paths, Rs, lives, score, e_s, chi)
            b = cbm_score(paths, Rs, lives, score, e_p, chi)
            c = age_score(lives, score, t_s, chi)
            d = age_score(lives, score, t_p, chi)
            if all(np.isfinite(v) for v in (a, b, c, d)) and c > 0 and d > 0:
                gc.append(a); ga.append(c); ec.append(a - b); ea.append(c - d)
                rr.append(b / d)                     # both chosen on the large pool
        gc, ga = np.mean(gc), np.mean(ga)
        ec, ea = np.mean(ec), np.mean(ea)
        print("%7d %10.4f %10.4f %10.4f %10.4f %14.4f"
              % (s, gc, ga, ec, ea, float(np.mean(rr))))
        print("%7s %10s %10s %10s %10s   raw ratio %.4f"
              % ("", "", "", "", "", gc / ga))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004)):
        run(nm, ld)
