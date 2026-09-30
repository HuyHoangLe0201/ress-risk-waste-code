r"""Two implementation objections, measured rather than argued.

LEAKAGE. Channels are standardised by the pooled mean and standard deviation of the whole
fleet, and the fleet is only then split into calibration and scoring halves. No label
crosses the split, but the scoring units' covariates have entered the preprocessing, which
is transductive. The fix is to standardise on the calibration half alone; the question is
whether the reported comparison moves when that is done.

GRID. Cost comparisons search 25 thresholds on [0, 0.4 Tbar]. On a finite fleet a threshold
policy changes only when the threshold crosses one of the per-unit statistics ell_u, so the
exact search is over those values. Theorem 5 is a statement about the continuous family;
the experiment minimises on a coarse grid. If the two disagree, the theory and the
measurement are about different objects.

Both are run on the same splits, changing one thing at a time, and report g_CBM/g_age so
the numbers sit beside the ones in the manuscript.
"""
import sys

import numpy as np
from scipy.stats import lognorm

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

CF = M.CF
NSPLIT = 20


def raw_cmapss(fn):
    """load_cmapss without the pooled standardisation, so it can be done per split."""
    d = np.loadtxt(M.CM + "\\" + fn)
    u, c, X = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2:]
    un = np.unique(u)
    life = np.array([float(c[u == k].max()) for k in un])
    keep = X.std(0) > 1e-8
    return [X[u == k][:, keep] for k in un], life


def standardise(seq, idx):
    """Mean and s.d. from the units in idx only."""
    S = np.vstack([seq[i] for i in idx])
    mu, sd = S.mean(0), S.std(0)
    sd[sd < 1e-12] = 1.0
    return [(a - mu) / sd for a in seq]


def run(name, fn):
    seq_raw, life = raw_cmapss(fn)
    n = len(seq_raw)
    Tb = life.mean()
    rows = {k: [] for k in ("pooled/25", "cal-only/25", "pooled/exact", "cal-only/exact")}

    for seed in range(NSPLIT):
        p = np.random.default_rng(seed).permutation(n)
        cal, val = p[:n // 2], p[n // 2:]
        Lc, Lv = life[cal], life[val]

        s_, _, sc = lognorm.fit(Lc, floc=0)
        t = np.linspace(1., 4 * Lc.max(), 4000)
        S = lognorm.sf(t, s_, scale=sc)
        cum = np.concatenate([[0.], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
        tR = t[10 + int(np.argmin(((S + CF * (1 - S)) / np.maximum(cum, 1e-12))[10:]))]
        g_age = np.where(Lv <= tR, CF, 1.).mean() / np.minimum(Lv, tR).mean() * Tb

        for scheme in ("pooled", "cal-only"):
            seq = standardise(seq_raw, range(n) if scheme == "pooled" else cal)
            X = M.design(seq)
            R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
            w = M.fit(np.vstack([X[i] for i in cal]),
                      np.concatenate([R[i] for i in cal]))
            pa = {i: X[i] @ w for i in range(n)}
            rm = {i: np.minimum.accumulate(pa[i]) for i in pa}

            def cost(ix, e):
                per = np.array([M.unit_cost(i, e, R, rm, life) for i in ix])
                return per[:, 0].mean() / per[:, 1].mean()

            coarse = np.linspace(0, .4 * Tb, 25)
            # the exact set of decision-changing thresholds on the calibration half
            lu = np.array([pa[i][R[i] > 0].min() for i in cal])
            exact = np.unique(np.clip(lu, 0, None))

            for gname, grid in (("25", coarse), ("exact", exact)):
                best = grid[int(np.argmin([cost(cal, e) for e in grid]))]
                rows["%s/%s" % (scheme, gname)].append(cost(val, best) * Tb / g_age)

    print("\n%s  (n = %d, %d splits)" % (name, n, NSPLIT))
    base = np.median(rows["pooled/25"])
    print("   %-16s %-22s %s" % ("standardisation/grid", "g_CBM/g_age", "vs pooled/25"))
    for k in ("pooled/25", "cal-only/25", "pooled/exact", "cal-only/exact"):
        v = np.array(rows[k])
        med = np.median(v)
        print("   %-16s %.3f [%.3f, %.3f]      %+.1f%%   (%d grid pts)"
              % (k, med, *np.percentile(v, [10, 90]),
                 100 * (med - base) / base, 25 if k.endswith("25") else -1))


if __name__ == "__main__":
    run("FD002", "train_FD002.txt")
    run("FD004", "train_FD004.txt")
