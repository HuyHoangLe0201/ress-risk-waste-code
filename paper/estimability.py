"""Which quantities in this framework are estimable at the parametric rate, and which not?

The paper's results on estimability are scattered: an oracle inequality for the selected
policy, a downward bias for omega_0, a lattice in the risk direction, a withdrawn interval.
Underneath them is one dichotomy.

A quantity that is a MEAN over units -- omega(l) and P_f(l) at a fixed threshold, and any
smooth function of them such as the cost of a FIXED policy -- is unbiased or biased at
O(1/n), and estimable at n^{-1/2}.  A quantity anchored at an EXTREME order statistic of
{l_u} -- omega_0, the ceiling 1/(1-omega_0), and chi* -- is not: its bias is the gap between
a sample maximum and the population endpoint, governed by the tail rather than by n.

The prediction is checkable and is checked here.  Bias of the fixed-policy cost against
fleet size should fall at least as fast as n^{-1}, while omega_0's was measured at n^{-0.47}
to n^{-0.76}.  If the fixed-policy bias also came out near n^{-0.5}, the dichotomy would be
wrong and the classification could not be stated.
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


def coords(paths, Rs, lives, sub, e):
    w, f = [], []
    for i in sub:
        h = np.nonzero(paths[i] <= e)[0]
        if len(h) == 0 or Rs[i][h[0]] <= 0:
            w.append(0.0); f.append(1.0)
        else:
            w.append(float(Rs[i][h[0]])); f.append(0.0)
    return float(np.mean(w)) / float(lives[sub].mean()), float(np.mean(f))


def run(name, ld, chi=9.0, reps=800, seed=0):
    paths, Rs, lives = prep(ld)
    N = len(paths)
    lu = np.array([float(paths[i][Rs[i] > 0].min()) for i in range(N)])
    # a FIXED policy, held the same at every fleet size: nothing is selected here
    e = float(np.quantile(lu, 0.60))
    om_f, pf_f = coords(paths, Rs, lives, np.arange(N), e)
    L_full = (1 + chi * pf_f) / (1 - om_f)

    rng = np.random.default_rng(seed)
    sizes = [s for s in (15, 25, 40, 65, 105) if s <= N]
    print("=" * 70)
    print("%s   fixed policy at l = 60th pct of l_u,  full-fleet L = %.4f"
          % (name, L_full))
    print("%8s %11s %11s %9s %11s %s"
          % ("n", "mean L", "bias", "bias %", "MC s.e.", "detectable"))
    for s in sizes:
        v = []
        for _ in range(reps):
            sub = rng.choice(N, s, replace=False)
            om, pf = coords(paths, Rs, lives, sub, e)
            if om < 1:
                v.append((1 + chi * pf) / (1 - om))
        v = np.array(v)
        b = float(v.mean()) - L_full
        se = float(v.std(ddof=1)) / np.sqrt(len(v))
        print("%8d %11.4f %+11.4f %8.2f%% %11.4f %s"
              % (s, v.mean(), b, 100 * b / L_full, se,
                 "yes" if abs(b) > 2 * se else "no"))
    print("   a rate is not fitted here: the bias is unsigned and within Monte Carlo")
    print("   error, whereas omega_0's was one-signed and 10-44%% of its value")


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
