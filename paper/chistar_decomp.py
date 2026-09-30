"""Where does chi-star's downward bias come from?

The paper reports two biases separately.  Guideline G6 says chi-star-hat is biased downward
"by the fleet's own resolution of the failure branch", and Proposition 7 says omega_0-hat
is biased downward because it is anchored at an extreme.  They are not independent, because
chi-star is DEFINED through omega_0:

    chi* = max_{l < l_0}  (omega_0 - omega(l)) / (P_f(l) (1 - omega_0)).

A downward bias in omega_0 shrinks the numerator and enlarges the denominator at once, so it
pushes chi-star down twice over.  The failure-branch resolution is a second, separate
channel.  Nobody has said which of the two carries the bias, and a practitioner correcting
the wrong one gains nothing.

The decomposition is direct.  For a subsample compute chi-star three ways: with its own
omega_0 and its own branch; with the whole fleet's omega_0 but the subsample's branch; and
on the whole fleet.  The middle one isolates how much of the gap closes when omega_0 alone
is repaired.
"""
import sys

import numpy as np

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


def branch(paths, Rs, lives, sub, grid=None, ngrid=200):
    """(omega, P_f) over a threshold grid.

    The grid must be supplied from outside when subsamples of different sizes are being
    compared.  Deriving it from the subsample's own quantiles makes both its range and its
    resolution move with n, which would be read as a sampling effect when it is an
    artefact of where the thresholds were placed.
    """
    if grid is None:
        lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in sub])
        grid = np.linspace(np.quantile(lu, 0.02), lu.max(), ngrid)
    Tb = float(lives[sub].mean())
    om, pf = [], []
    for e in grid:
        w, f = [], []
        for i in sub:
            h = np.nonzero(paths[i] <= e)[0]
            if len(h) == 0 or Rs[i][h[0]] <= 0:
                w.append(0.0); f.append(1.0)
            else:
                w.append(float(Rs[i][h[0]])); f.append(0.0)
        om.append(float(np.mean(w)) / Tb)
        pf.append(float(np.mean(f)))
    return np.array(om), np.array(pf)


def chistar(om, pf, om0):
    k = (pf > 0) & (om < om0)
    if not k.any():
        return np.nan
    return float(np.max((om0 - om[k]) / (pf[k] * (1.0 - om0))))


def run(name, ld, reps=250, seed=0):
    paths, Rs, lives = prep(ld)
    N = len(paths)
    lu_all = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(N)])
    GRID = np.linspace(np.quantile(lu_all, 0.02), lu_all.max(), 200)  # fixed for all n
    om_f, pf_f = branch(paths, Rs, lives, np.arange(N), GRID)
    om0_full = om_f.max()
    chi_full = chistar(om_f, pf_f, om0_full)

    rng = np.random.default_rng(seed)
    sizes = [s for s in (30, 60, 100, 160) if s <= N]
    print("=" * 78)
    print("%s   N=%d   full-fleet omega_0=%.4f  chi*=%.2f"
          % (name, N, om0_full, chi_full))
    print("%7s %11s %14s %11s %14s" %
          ("n", "chi* naive", "chi* om0 fixed", "chi* full", "gap closed"))
    for s in sizes:
        a, b = [], []
        for _ in range(reps):
            sub = rng.choice(N, s, replace=False)
            om, pf = branch(paths, Rs, lives, sub, GRID)
            om0 = om.max()
            a.append(chistar(om, pf, om0))
            b.append(chistar(om, pf, om0_full))
        a = np.array(a, float); b = np.array(b, float)
        a, b = a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)]
        gap = chi_full - a.mean()
        closed = b.mean() - a.mean()
        # only meaningful while the naive estimate is genuinely below the full-fleet one
        lab = ("%.0f%%" % (100.0 * closed / gap)) if gap > 0.05 * chi_full else "n/a"
        print("%7d %11.2f %14.2f %11.2f %14s"
              % (s, a.mean(), b.mean(), chi_full, lab))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
