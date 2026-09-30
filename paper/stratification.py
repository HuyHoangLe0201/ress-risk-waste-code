"""Does stratifying the threshold by sub-population pay?

Every result so far treats the fleet as one population.  In the plane the question
of heterogeneity is exact.  If the fleet splits into groups with weights w_g, a
COMMON threshold puts the fleet at the mixture point sum_g w_g Phi_g(ell), a curve
traced by one parameter; GROUP-SPECIFIC thresholds put it at sum_g w_g Phi_g(ell_g)
with the ell_g chosen independently, which sweeps a set containing that curve.  So
stratifying can only help the target, with equality exactly when the groups share
an optimal threshold.

Out of sample it is not obvious at all, and the paper's own thesis says why: each
group's threshold is then chosen on n_g units instead of n, so the target improves
while the estimate worsens.  On Severson the three manufacturing batches give an
exogenous partition of 42, 47 and 46 cells, and a calibration half leaves about
sixteen cells per group.  Whichever way it comes out is informative -- either
heterogeneity matters and the plane sizes it, or policy stratification is one more
thing these fleet sizes cannot support.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

DRAWS = 200
NGRID = 120


def severson_with_batch():
    """The same cells load_severson returns, with their batch labels aligned."""
    z = np.load(M.TSP + "\\severson_cells.npz", allow_pickle=True)
    raw = [np.asarray(a, float) for a in z["data"]]
    names = [str(x) for x in z["names"]]
    keep = [i for i, a in enumerate(raw)
            if len(a) > 150 and np.isfinite(a).all()]
    cs = [raw[i] for i in keep]
    bat = np.array([names[i][:2] for i in keep])
    S = np.vstack(cs)
    ok = S.std(0) > 1e-12
    mu, sd = S[:, ok].mean(0), S[:, ok].std(0)
    seq = [(a[:, ok] - mu) / sd for a in cs]
    life = np.array([float(len(a)) for a in cs])
    return seq, life, bat


def cost_of(idx, ell, R, rm, life, Tb):
    per = np.array([M.unit_cost(i, ell, R, rm, life) for i in idx])
    return per[:, 0].mean(), per[:, 1].mean(), per[:, 2].mean()


def run():
    seq, life, bat = severson_with_batch()
    groups = sorted(set(bat))
    print("batches:", {g: int((bat == g).sum()) for g in groups})

    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    allu = np.concatenate([cal, val])
    lu = np.array([pa[i][R[i] > 0].min() for i in allu])
    grid = np.linspace(np.quantile(lu, .02), lu.max() + .25 * Tb, NGRID)

    rng = np.random.default_rng(7)
    com, str_ = [], []
    for _ in range(DRAWS):
        p = rng.permutation(len(allu))
        h = len(allu) // 2
        A, B = allu[p[:h]], allu[p[h:]]
        bA, bB = bat[A], bat[B]

        # common threshold: one ell chosen on A, scored on B
        cA = np.array([cost_of(A, e, R, rm, life, Tb) for e in grid])
        LA = cA[:, 0] / cA[:, 1]
        e_com = grid[int(np.argmin(LA))]
        nB, dB, _ = cost_of(B, e_com, R, rm, life, Tb)
        com.append(nB / dB * Tb)

        # stratified: one ell per batch, chosen on that batch's half
        num = den = 0.0
        ok = True
        for g in groups:
            Ag, Bg = A[bA == g], B[bB == g]
            if len(Ag) < 5 or len(Bg) < 3:
                ok = False
                break
            cg = np.array([cost_of(Ag, e, R, rm, life, Tb) for e in grid])
            eg = grid[int(np.argmin(cg[:, 0] / cg[:, 1]))]
            n_, d_, _ = cost_of(Bg, eg, R, rm, life, Tb)
            num += n_ * len(Bg)
            den += d_ * len(Bg)
        if ok:
            str_.append(num / den * Tb)
        else:
            com.pop()

    com, str_ = np.array(com), np.array(str_)
    print("=" * 74)
    print("out-of-sample cost over %d splits" % len(com))
    print("   common threshold     : %.4f  [%.4f, %.4f]"
          % (com.mean(), *np.quantile(com, [.25, .75])))
    print("   batch-specific       : %.4f  [%.4f, %.4f]"
          % (str_.mean(), *np.quantile(str_, [.25, .75])))
    d = str_ - com
    print("   stratified minus common: mean %+.4f (%+.1f%%), "
          "stratified wins on %.0f%% of splits"
          % (d.mean(), 100 * d.mean() / com.mean(), 100 * (d < 0).mean()))


if __name__ == "__main__":
    run()
