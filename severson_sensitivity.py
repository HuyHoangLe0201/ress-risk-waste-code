r"""How much do ell_0, omega_0 and kappa* depend on which Severson cells are in the fleet?

ell_0 is a maximum over units and kappa* is read off the envelope near it, so both are
extreme order statistics: a handful of cells entering or leaving the fleet can move them in
a way that a mean would absorb. The screening that produces the 135-cell fleet is therefore
not a housekeeping detail, and the manuscript states it without showing what it costs.

What can be varied here is bounded by what is stored. severson_cells.npz holds the fleet
after screening -- 135 cells, all with finite summaries, the shortest 168 cycles -- so the
capacity and fade criteria cannot be relaxed without returning to the raw batch files.
Three perturbations are available and are the ones that bear on the tail:

  length cuts       drop the shortest cells, which is the screening rule made stricter
  drop the longest  remove the cells that define ell_0 itself
  leave one batch   the three manufacturing batches, one at a time

For each fleet the same pipeline runs: split, ridge predictor, running minima, threshold
grid, envelope. Reported are ell_0 (the smallest threshold at which no calibration unit
fails), omega_0 (the waste it costs) and kappa* from Theorem 5.
"""
import os
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

TSP = os.path.join(
    r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\IEEE_Trans.representation",
    "TSP_segmented_projections", "data")
Z = np.load(os.path.join(TSP, "severson_cells.npz"), allow_pickle=True)
RAW = [np.asarray(a, float) for a in Z["data"]]
NAMES = [str(x) for x in Z["names"]]


def batch_of(name):
    """Cells are named b<batch>c<cell>, e.g. b1c3, b2c0, b3c45.

    Splitting on an underscore, as the first version did, returned the whole name and gave
    135 one-cell "batches" instead of three. The leave-one-batch-out rows were therefore
    leave-one-cell-out rows wearing the wrong label.
    """
    return int(name[1:name.index("c")])


BATCH = np.array([batch_of(n) for n in NAMES])


def fleet(mask):
    """Normalise on the retained cells only, as load_severson does for the full fleet."""
    cs = [RAW[i] for i in np.nonzero(mask)[0]]
    S = np.vstack(cs)
    ok = S.std(0) > 1e-12
    mu, sd = S[:, ok].mean(0), S[:, ok].std(0)
    return ([(a[:, ok] - mu) / sd for a in cs],
            np.array([float(len(a)) for a in cs]))


def stats(seq, life, seeds=range(20)):
    """Medians of ell_0, omega_0 and kappa* over several calibration splits."""
    l0s, w0s, ks = [], [], []
    for seed in seeds:
        R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
        lu = np.array([pa[i][R[i] > 0].min() for i in cal])
        l0 = float(lu.max())
        g = np.linspace(np.quantile(lu, .02), l0 + .3 * Tb, 170)
        om, pf = [], []
        for e in g:
            per = np.array([M.unit_cost(i, e, R, rm, life) for i in cal])
            om.append(1 - per[:, 1].mean() / Tb)
            pf.append(per[:, 2].mean())
        om, pf = np.array(om), np.array(pf)
        z0 = int(np.argmax(pf <= 0))          # first grid point with no failures
        if z0 == 0 or (pf[:z0] <= 0).all():
            continue
        l0s.append(l0 / Tb)
        w0s.append(om[z0])
        ks.append(M.critical_ratio(om, pf, z0) - 1.0)
    return (np.median(l0s), np.median(w0s), np.median(ks), len(l0s))


LENS = np.array([len(a) for a in RAW])
finite = np.array([bool(np.isfinite(a).all()) for a in RAW])

CASES = [("baseline, 135 cells", finite)]
for cut in (200, 300, 450):
    CASES.append(("shortest dropped, len > %d" % cut, finite & (LENS > cut)))
order = np.argsort(-LENS)
for k in (1, 3):
    m = finite.copy()
    m[order[:k]] = False
    CASES.append(("longest %d cell(s) dropped" % k, m))
for b in sorted(set(BATCH[BATCH >= 0])):
    CASES.append(("batch %d left out" % b, finite & (BATCH != b)))

print("%-28s %5s %10s %10s %10s %s"
      % ("fleet", "n", "ell_0/Tbar", "omega_0", "kappa*", "splits"))
base = None
for label, mask in CASES:
    seq, life = fleet(mask)
    l0, w0, ks, ns = stats(seq, life)
    if base is None:
        base = (l0, w0, ks)
    print("%-28s %5d %10.3f %10.3f %10.2f %6d   (kappa* %+.0f%%)"
          % (label, int(mask.sum()), l0, w0, ks, ns,
             100 * (ks - base[2]) / base[2]))
