"""How much of a predictor's output can any maintenance decision actually see?

Theorem 4 writes both coordinates through Lambda, the RUNNING MINIMUM of Rhat, and the
alarm fires at the first time Lambda reaches the threshold.  So the coordinates are
functionals of Lambda alone, and Lambda is determined by the path's RECORD MINIMA.
Everything the predictor says at a cycle where it sits above its own running minimum is
invisible to every threshold policy at every cost ratio.

Two things to measure.  First, the invariance itself: replace the predictor at every
non-record cycle by its running minimum plus arbitrary positive noise -- which preserves
the running-minimum process exactly -- and check that omega and P_f do not move at any
threshold.  Second, the size of the effect: what fraction of monitored cycles are records?
For a fluctuating path the number of lower records grows like log of the length, so the
visible fraction may be very small, and if so a large majority of a predictor's output
cannot influence any decision this framework can express.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def coords_at(paths, n_u, ells):
    Tb = float(n_u.mean())
    out = []
    for e in ells:
        w, f = [], []
        for p in paths:
            rm = np.minimum.accumulate(p)
            h = np.nonzero(rm <= e)[0]
            if len(h) == 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(len(p) - 1 - h[0])); f.append(0.0)
        out.append((float(np.mean(w)) / Tb, float(np.mean(f))))
    return np.array(out)


def run(name, ld, seed=0):
    seq, life = ld()
    R, rm0, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    base = [np.asarray(pa[i], float) for i in idx]
    n_u = np.array([len(p) for p in base], float)
    lo = min(p.min() for p in base)
    hi = max(p.max() for p in base)
    ells = np.linspace(lo, hi, 200)
    A0 = coords_at(base, n_u, ells)

    rg = np.random.default_rng(seed)
    tot = rec = 0
    pert = []
    for p in base:
        r = np.minimum.accumulate(p)
        is_rec = p <= r + 1e-12               # cycle achieves the running minimum
        tot += len(p)
        rec += int(is_rec.sum())
        q = p.copy()
        k = ~is_rec
        # anything at or above the running minimum leaves Lambda untouched
        q[k] = r[k] + rg.uniform(0, 500, int(k.sum()))
        assert np.allclose(np.minimum.accumulate(q), r)
        pert.append(q)

    A1 = coords_at(pert, n_u, ells)
    print("=" * 74)
    print("%s   n=%d, %d monitored cycles" % (name, len(base), tot))
    print("   record cycles: %d (%.1f%%)   invisible: %.1f%%"
          % (rec, 100.0 * rec / tot, 100.0 * (1 - rec / tot)))
    print("   after randomising every invisible cycle by U(0,500):")
    print("      max |d omega| = %.2e    max |d P_f| = %.2e"
          % (np.max(np.abs(A1[:, 0] - A0[:, 0])),
             np.max(np.abs(A1[:, 1] - A0[:, 1]))))
    strict = [int((np.diff(np.minimum.accumulate(p)) < 0).sum()) + 1 for p in base]
    print("   strict record minima per unit: median %d, over median length %d"
          % (int(np.median(strict)), int(np.median(n_u))))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
