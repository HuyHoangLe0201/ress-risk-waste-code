"""What rate does policy selection converge at, and does it explain n = 450-550?

The paper reaches a fleet size of roughly 450-550 by three empirical routes and never
derives one.  The theory now available gives the rate directly.

Upper bound.  P_f(l) = Pr(l_u > l) is a survival function, so DKW bounds its uniform
error by sqrt(log(2/delta)/(2n)).  The waste w(l) = E[W_u(l)]/Tbar is the mean of a
family of monotone bounded functions of l, whose bracketing entropy is O(1/eps), giving
the same n^{-1/2} up to logs.  Where omega is bounded away from one, L = (1+chi P_f)/(1-w)
is Lipschitz in both coordinates, so sup_l |Lhat - L| = O(sqrt(log n / n)), and the
threshold chosen by minimising Lhat satisfies the oracle inequality

    L(lhat) - min_l L(l)  <=  2 sup_l |Lhat - L|.

Lower bound.  Two fleet laws differing by eps in total variation are indistinguishable at
sample size n ~ eps^{-2}, while their optimal costs can differ by order eps.  So no
procedure beats n^{-1/2}, and the fleet size for excess cost delta is Theta(delta^{-2}).

This script measures the exponent rather than assuming it.  Units are split into a pool
and an evaluation half; a threshold is chosen on n units drawn from the pool and scored on
the evaluation half against the best threshold there, so the excess is exactly the
oracle-inequality quantity and has no floor from evaluation noise.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402


def per_unit(ld, n_thr=240):
    """waste[t,u] and fail[t,u] over a threshold grid, computed once."""
    seq, life = ld()
    R, rm, pa, cal, val, _ = M.prep(seq, life, seed=0)
    idx = np.concatenate([cal, val])
    paths = [np.asarray(pa[i], float) for i in idx]
    n_u = np.array([len(p) for p in paths], float)
    lo = min(p.min() for p in paths)
    hi = max(p.max() for p in paths)
    ells = np.linspace(lo, hi, n_thr)
    W = np.zeros((n_thr, len(paths)))
    F = np.zeros((n_thr, len(paths)))
    for t, e in enumerate(ells):
        for u, p in enumerate(paths):
            h = np.nonzero(p <= e)[0]
            if len(h) == 0:
                F[t, u] = 1.0
            else:
                W[t, u] = len(p) - 1 - h[0]
    return W, F, n_u


def cost(W, F, n_u, sub, chi):
    om = W[:, sub].mean(1) / n_u[sub].mean()
    pf = F[:, sub].mean(1)
    return (1.0 + chi * pf) / (1.0 - om)


def run(name, ld, chis=(3.0, 9.0, 30.0), reps=1500, seed=0):
    W, F, n_u = per_unit(ld)
    N = W.shape[1]
    rg = np.random.default_rng(seed)
    perm = rg.permutation(N)
    pool, ev = perm[:N // 2], perm[N // 2:]

    # cap at a third of the pool: beyond that a draw is most of the pool, the chosen
    # threshold converges to the pool optimum, and the excess stops falling because the
    # pool/evaluation discrepancy is a fixed positive floor rather than sampling error
    sizes = [s for s in (8, 11, 15, 21, 29, 40) if s <= len(pool) // 3]
    print("=" * 74)
    print("%s   pool %d, evaluation %d" % (name, len(pool), len(ev)))
    print("%6s %8s %12s %12s" % ("chi", "n", "rel. excess", "x sqrt(n)"))
    for chi in chis:
        Lev = cost(W, F, n_u, ev, chi)
        best = Lev.min()
        xs, ys = [], []
        for s in sizes:
            acc = []
            for _ in range(reps):
                sub = rg.choice(pool, size=s, replace=False)
                t = int(np.argmin(cost(W, F, n_u, sub, chi)))
                acc.append((Lev[t] - best) / best)   # relative excess
            e = float(np.mean(acc))
            xs.append(s); ys.append(e)
            print("%6.0f %8d %12.5f %12.4f" % (chi, s, e, e * np.sqrt(s)))
        k = [i for i, y in enumerate(ys) if y > 1e-12]
        if len(k) >= 3:
            sl, ic = np.polyfit(np.log(np.array(xs)[k]), np.log(np.array(ys)[k]), 1)
            need = lambda d: float(np.exp((np.log(d) - ic) / sl))  # noqa: E731
            print("%6.0f  slope %+.3f | n for 5%% excess: %5.0f | for 2%%: %6.0f"
                  % (chi, sl, need(0.05), need(0.02)))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
