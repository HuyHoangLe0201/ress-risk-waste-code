r"""The risk axis of the frontier is a conformal level, and the empirical one is optimistic.

Theorem 11 prices the boundary policy at 1/(n+1) instead of 0.  That is the first case of an
exact statement covering every policy on the frontier, and it identifies the family.

Order the unit thresholds ell_(1) >= ... >= ell_(n) and alarm at e = ell_(j+1).  In sample
exactly the j units ell_(1..j) fail, so the reported risk is P_f = j/n.  A new unit fails
when its ell exceeds e, and among the n+1 exchangeable values the new one lands above the
(j+1)-th largest with probability exactly (j+1)/(n+1).  So

    reported  P_f = j/n          actual  P_f = (j+1)/(n+1),

distribution-free, with no tail model and nothing fitted.  This is the split-conformal
guarantee, and it says the paper's threshold family is a conformal predictor whose level is
read off the order statistic index.  Two consequences:

  - the empirical frontier is optimistic at every point, by
        (j+1)/(n+1) - j/n = (n-j)/(n(n+1)),
    which is 1/(n+1) at the boundary and decays to zero as j approaches n.  The bias is
    largest exactly where these fleets operate;
  - and a target out-of-sample risk can be hit exactly by choosing the index, j+1 =
    ceil(alpha (n+1)), rather than by searching a grid and hoping.

Both halves are checked here by splitting: fit the order statistic on one half, measure the
failure rate on the other, for every j and not only j = 0.  The waste coordinate is measured
the same way, because the threshold is chosen from the data and omega could inherit a
selection effect of its own; if it does, the correction is not confined to the risk axis.
Finally the corrected frontier is rebuilt and the envelope, the optimum and chi* are
recomputed to see what the correction is worth.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402
from hullscale import lower_hull                                 # noqa: E402


def unit_tables(ld):
    paths, Rs, lives, lu = prep(ld)
    lu = np.asarray(lu, float)
    acc = [np.minimum.accumulate(p) for p in paths]
    return paths, Rs, lives, lu, acc


def waste_at(acc, Rs, idx, e):
    """Mean wasted life over the given units at threshold e (0 for a failure)."""
    w = 0.0
    for i in idx:
        h = np.nonzero(acc[i] <= e)[0]
        if len(h) and Rs[i][h[0]] > 0:
            w += float(Rs[i][h[0]])
    return w / len(idx)


def split_check(ld, name, reps=1500, seed=0):
    """Out-of-sample risk and waste at the j-th order statistic, against (j+1)/(n+1)."""
    paths, Rs, lives, lu, acc = unit_tables(ld)
    N = len(lu)
    rng = np.random.default_rng(seed)
    k = N // 2
    js = [0, 1, 2, 4, 8, 16, 32]
    js = [j for j in js if j < k - 1]
    got = {j: [] for j in js}
    wtr = {j: [] for j in js}
    wte = {j: [] for j in js}
    for _ in range(reps):
        p = rng.permutation(N)
        tr, te = p[:k], p[k:]
        o = np.sort(lu[tr])[::-1]
        for j in js:
            e = float(o[j])                       # the (j+1)-th largest on train
            got[j].append(float(np.mean(lu[te] > e)))
            wtr[j].append(waste_at(acc, Rs, tr, e))
            wte[j].append(waste_at(acc, Rs, te, e))
    Tb = float(np.mean(lives))
    print("=" * 78)
    print("%s   N=%d, fitted on %d units" % (name, N, k))
    print("%4s %11s %13s %8s | %11s %11s %8s"
          % ("j", "in-sample", "held-out P_f", "predict", "waste tr", "waste te",
             "ratio"))
    for j in js:
        pred = (j + 1.0) / (k + 1.0)
        obs = float(np.mean(got[j]))
        a, b = float(np.mean(wtr[j])) / Tb, float(np.mean(wte[j])) / Tb
        print("%4d %11.5f %13.5f %8.2f | %11.5f %11.5f %8.3f"
              % (j, j / k, obs, obs / pred, a, b, b / a if a > 0 else np.nan))


def corrected_frontier(ld, name):
    """Rebuild the frontier with the conformal risk axis and see what moves."""
    paths, Rs, lives, lu, acc = unit_tables(ld)
    N = len(lu)
    Tb = float(np.mean(lives))
    o = np.sort(lu)[::-1]
    idx = np.arange(N)
    rows = []
    for j in range(0, N // 2):
        e = float(o[j])
        om = waste_at(acc, Rs, idx, e) / Tb
        if om < 1.0:
            rows.append((j / N, (j + 1.0) / (N + 1.0), om))
    R = np.array(rows)

    def env(pf, om, chi):
        return float(np.min((1.0 + chi * pf) / (1.0 - om)))

    def kink(pf, om):
        """chi* : the largest cost ratio at which a failing policy still wins."""
        a = 1.0 / (1.0 - om)
        b = pf / (1.0 - om)
        h = lower_hull(b, a)
        bb, aa = b[h], a[h]
        best = 0.0
        for i in range(len(bb) - 1):
            db = bb[i + 1] - bb[i]
            if db < 0:
                best = max(best, (aa[i] - aa[i + 1]) / (bb[i + 1] - bb[i]) * -1.0)
        # chi* is the last breakpoint: slope between consecutive hull vertices
        sl = [-(aa[i + 1] - aa[i]) / (bb[i + 1] - bb[i])
              for i in range(len(bb) - 1) if bb[i + 1] != bb[i]]
        sl = [s for s in sl if s > 0]
        return max(sl) if sl else np.nan

    print("=" * 78)
    print("%s   corrected risk axis: reported j/n -> actual (j+1)/(n+1)" % name)
    print("%8s %12s %12s %9s" % ("chi", "L* reported", "L* actual", "understated"))
    for chi in (1, 3, 9, 30, 100):
        lr = env(R[:, 0], R[:, 2], chi)
        la = env(R[:, 1], R[:, 2], chi)
        print("%8d %12.4f %12.4f %8.1f%%" % (chi, lr, la, 100 * (la - lr) / lr))
    print("   chi* reported %.2f   actual %.2f"
          % (kink(R[:, 0], R[:, 2]), kink(R[:, 1], R[:, 2])))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        split_check(ld, nm)
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        corrected_frontier(ld, nm)
