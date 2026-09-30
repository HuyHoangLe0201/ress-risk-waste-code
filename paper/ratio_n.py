"""Does the paper's one surviving claim move with fleet size?

The headline is g_CBM/g_AGE = 0.61-0.70 at c_f/c_p = 10 on the two turbofan fleets.  At
chi = 9 the optimum sits at the zero-failure point, since chi* is about 1.6 there, so the
numerator is evaluated at exactly the operating point whose waste Proposition 7 shows to
be biased low.  If that bias reaches the ratio, the reported value is a property of a
fleet of 260 units rather than of condition monitoring.

The protocol is the paper's own and self-corrects in part: the threshold is chosen on the
calibration half and scored on units it has not seen, so validation failures are counted
honestly.  Whether that is enough is an empirical question, and the way to ask it is to
vary n and watch.

Age replacement is fitted parametrically on the calibration half, as guideline G3
requires, and both policies are scored on the same held-out units.
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
    return paths, Rs, np.array([float(len(p)) for p in paths])


def cbm_cost(paths, Rs, lives, fit, score, chi, ngrid=140):
    lo = min(paths[i].min() for i in fit)
    hi = max(paths[i].max() for i in fit)

    def cost(units, e):
        w, f = [], []
        for i in units:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        om = float(np.mean(w)) / float(lives[units].mean())
        return np.inf if om >= 1 else (1 + chi * float(np.mean(f))) / (1 - om)

    grid = np.linspace(lo, hi, ngrid)
    e_star = grid[int(np.argmin([cost(fit, e) for e in grid]))]
    return cost(score, e_star)


def age_cost(lives, fit, score, chi, ngrid=1500):
    T = lives[fit]
    sh, loc, sc = stats.lognorm.fit(T, floc=0)
    d = stats.lognorm(sh, loc, sc)
    # E[min(X,t)] = int_0^t S(x) dx on one shared grid, cumulated once, instead of a
    # fresh numerical integration per candidate age
    ts = np.linspace(T.min() * 0.3, T.max() * 1.05, ngrid)
    S = d.sf(ts)
    L = np.concatenate([[0.0], np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(ts))])
    L += ts[0] * 1.0                      # S is essentially 1 below ts[0]
    mu = float(d.mean())
    cs = (1 + chi * d.cdf(ts)) / (L / mu)
    t_star = ts[int(np.argmin(cs))]
    S = lives[score]
    om = 1.0 - float(np.minimum(S, t_star).mean()) / float(S.mean())
    pf = float(np.mean(S <= t_star))
    return np.inf if om >= 1 else (1 + chi * pf) / (1 - om)


def run(name, ld, chi=9.0, reps=200, seed=0):
    paths, Rs, lives = prep(ld)
    N = len(paths)
    rng = np.random.default_rng(seed)
    sizes = [s for s in (40, 70, 120, 180, 249, 260) if s <= N]
    print("=" * 66)
    print("%s   N=%d" % (name, N))
    print("%8s %12s %12s %12s" % ("n", "g_CBM/g_AGE", "sd", "reps"))
    for s in sizes:
        acc = []
        for _ in range(reps):
            sub = rng.choice(N, s, replace=False)
            h = s // 2
            fit, score = sub[:h], sub[h:]
            try:
                a = cbm_cost(paths, Rs, lives, fit, score, chi)
                b = age_cost(lives, fit, score, chi)
            except Exception:
                continue
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                acc.append(a / b)
        acc = np.array(acc)
        print("%8d %12.4f %12.4f %12d"
              % (s, acc.mean(), acc.std(ddof=1), len(acc)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004)):
        run(nm, ld)
