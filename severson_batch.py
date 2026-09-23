r"""Batch-aware analysis of the Severson fleet.

The cells come from three manufacturing batches, and ell_u has an intraclass correlation of
0.34 across them: 135 cells carry an effective sample size near 8. The manuscript answers
that by withdrawing the interval while keeping the point estimate. That is honest about the
uncertainty and silent about the thing the clustering actually threatens, which is whether
a policy calibrated on some cells transfers to cells made at another time.

Three analyses, each asking a different question.

  random split      the manuscript's protocol: cells shuffled, half calibrate, half score.
                    Batches are mixed across the split, so a cell's own batch is almost
                    always represented in calibration.

  leave one batch   calibrate the predictor, the threshold and the age baseline on two
                    batches, score on the third. This is the transfer question, and it is
                    the one a fleet operator faces when a new lot arrives.

  batch bootstrap   resample the three batches with replacement rather than the cells.
                    With three clusters the resulting interval is necessarily coarse; how
                    coarse is itself the answer.

Reported in every case is g_CBM/g_age on the scoring cells. Tbar cancels in that ratio, so
the normalisation is taken on the scoring set throughout.
"""
import os
import sys

import numpy as np
from scipy.stats import lognorm

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

TSP = os.path.join(
    r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\IEEE_Trans.representation",
    "TSP_segmented_projections", "data")
Z = np.load(os.path.join(TSP, "severson_cells.npz"), allow_pickle=True)
RAW = [np.asarray(a, float) for a in Z["data"]]
NAMES = [str(x) for x in Z["names"]]
# Cells are named b<batch>c<cell>: b1c3, b2c0, b3c45.
BATCH = np.array([int(n[1:n.index("c")]) for n in NAMES])
CF = M.CF


def prepare(keep):
    """Normalise on the retained cells, as load_severson does for the whole fleet."""
    cs = [RAW[i] for i in keep]
    S = np.vstack(cs)
    ok = S.std(0) > 1e-12
    mu, sd = S[:, ok].mean(0), S[:, ok].std(0)
    seq = [(a[:, ok] - mu) / sd for a in cs]
    return seq, np.array([float(len(a)) for a in cs])


def ratio(seq, life, cal, val):
    """g_CBM / g_age on the scoring cells, both trained on cal."""
    X = M.design(seq)
    R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
    Tb = life[val].mean()
    Lc, Lv = life[cal], life[val]

    # age baseline: lognormal fitted on the calibration lifetimes
    s_, _, sc = lognorm.fit(Lc, floc=0)
    t = np.linspace(1., 4 * Lc.max(), 4000)
    S = lognorm.sf(t, s_, scale=sc)
    cum = np.concatenate([[0.], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
    tR = t[10 + int(np.argmin(((S + CF * (1 - S)) / np.maximum(cum, 1e-12))[10:]))]
    g_age = np.where(Lv <= tR, CF, 1.).mean() / np.minimum(Lv, tR).mean() * Tb

    # condition-based: ridge on cal, threshold chosen on cal, scored on val
    w = M.fit(np.vstack([X[i] for i in cal]), np.concatenate([R[i] for i in cal]))
    pa = {i: X[i] @ w for i in range(len(seq))}
    rm = {i: np.minimum.accumulate(pa[i]) for i in pa}
    grid = np.linspace(0, .4 * life.mean(), 25)

    def cost(ix, e):
        per = np.array([M.unit_cost(i, e, R, rm, life) for i in ix])
        return per[:, 0].mean() / per[:, 1].mean()

    best = grid[int(np.argmin([cost(cal, e) for e in grid]))]
    return cost(val, best) * Tb / g_age


# ---------------------------------------------------------------- random split
keep = np.arange(len(RAW))
seq, life = prepare(keep)
rs = []
for seed in range(40):
    p = np.random.default_rng(seed).permutation(len(seq))
    rs.append(ratio(seq, life, p[:len(p) // 2], p[len(p) // 2:]))
rs = np.array(rs)
print("random split, cells shuffled   median %.3f   [%.3f, %.3f] over 40 splits"
      % (np.median(rs), *np.percentile(rs, [2.5, 97.5])))

# ---------------------------------------------------------------- leave one batch out
print("\nleave one batch out (calibrate on two, score on the third):")
lobo = []
for b in sorted(set(BATCH)):
    tr = np.nonzero(BATCH != b)[0]
    te = np.nonzero(BATCH == b)[0]
    seq, life = prepare(np.concatenate([tr, te]))
    cal = np.arange(len(tr))
    val = np.arange(len(tr), len(tr) + len(te))
    r = ratio(seq, life, cal, val)
    lobo.append(r)
    print("   held-out batch %d  (%3d train, %3d test)   g_CBM/g_age = %.3f%s"
          % (b, len(tr), len(te), r, "   <-- worse than parity" if r >= 1 else ""))
print("   spread across the three held-out batches: %.3f to %.3f"
      % (min(lobo), max(lobo)))

# ---------------------------------------------------------------- batch bootstrap
print("\nbatch-level bootstrap (resample the three batches, not the cells):")
bs, rng = [], np.random.default_rng(0)
bat = sorted(set(BATCH))
for _ in range(300):
    draw = rng.choice(bat, size=len(bat), replace=True)
    idx = np.concatenate([np.nonzero(BATCH == b)[0] for b in draw])
    if len(set(draw)) == 1:                 # one batch drawn three times
        continue
    seq, life = prepare(idx)
    p = rng.permutation(len(seq))
    try:
        bs.append(ratio(seq, life, p[:len(p) // 2], p[len(p) // 2:]))
    except Exception:
        continue
bs = np.array(bs)
print("   %d usable resamples   median %.3f   95%% interval [%.3f, %.3f]"
      % (len(bs), np.median(bs), *np.percentile(bs, [2.5, 97.5])))
print("   distinct batch multisets available with three clusters: 10")
