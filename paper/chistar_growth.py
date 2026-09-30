"""The chi* growth law, tested under the paper's own sub-fleet protocol.

Run 2 compared exponents but computed chi* on a whole sub-fleet with the
predictor frozen, whereas the paper draws a sub-fleet, refits the predictor on
its calibration half and evaluates chi* there.  Different protocol, incomparable
numbers, so run 2 could not be used to judge either the law or the paper's
exponents.  This run uses the paper's protocol exactly and adds two things to it:
the top-spacing proxy n D_n / (Tbar (1 - omega_0)), and the extreme-value index of
ell_u.

Three quantities that ought to line up if the proposed mechanism is right:
the exponent of chi* in log n, the exponent of the proxy, and 1 + gamma.
Two seeds, so the spread of each exponent is visible rather than assumed.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

REPS = 30
NGRID = 260


def one_draw(seq, life, pool, nprime, rng):
    """The paper's protocol: draw, split, refit on CAL, evaluate chi* on CAL."""
    sub = rng.choice(pool, size=nprime, replace=False)
    p = rng.permutation(len(sub))
    cal = sub[p[: nprime // 2]]
    X = M.design([seq[i] for i in sub])
    Xm = {i: X[j] for j, i in enumerate(sub)}
    R = {i: np.arange(len(seq[i]) - 1, -1, -1.0) for i in sub}
    w = M.fit(np.vstack([Xm[i] for i in cal]),
              np.concatenate([R[i] for i in cal]))
    pa = {i: Xm[i] @ w for i in sub}
    rm = {i: np.minimum.accumulate(pa[i]) for i in sub}
    Tb = life[cal].mean()
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(lu.min() - .08 * Tb, lu.max() + .35 * Tb, NGRID)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in cal] for e in g])
    pf = C[:, :, 2].mean(1)
    om = 1 - C[:, :, 1].mean(1) / Tb
    z = np.nonzero(pf == 0)[0]
    if not len(z) or z.min() == 0:
        return np.nan, np.nan, None
    k = int(z.min())
    b = np.arange(k)
    o0 = om[k]
    chis = float(((o0 - om[b]) / (pf[b] * (1 - o0))).max())
    s = np.sort(lu)
    proxy = len(cal) * float(s[-1] - s[-2]) / (Tb * (1 - o0))
    return chis, proxy, lu


def moment_evi(x, k):
    s = np.sort(x)[::-1]
    if s[k] <= 0:
        s = s - s.min() + 1e-9
    lg = np.log(s[:k]) - np.log(s[k])
    M1, M2 = lg.mean(), (lg ** 2).mean()
    return M1 + 1 - 0.5 / (1 - M1 ** 2 / M2)


def slope(ns, v):
    ok = np.isfinite(v) & (v > 0)
    x, y = np.log(np.asarray(ns)[ok]), np.log(v[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(c[0]), float(r2)


def run(name, ld, sizes):
    seq, life = ld()
    pool = np.arange(len(seq))
    print("=" * 78)
    print("%s   sizes %s   %d draws each" % (name, list(sizes), REPS))
    ex_c, ex_p, gams = [], [], []
    for seed in (1000, 2000):
        mc, mp, lus = [], [], []
        for nprime in sizes:
            rng = np.random.default_rng(seed + nprime)
            cs, ps = [], []
            for _ in range(REPS):
                c, p, lu = one_draw(seq, life, pool, nprime, rng)
                if np.isfinite(c) and c > 0:
                    cs.append(c); ps.append(p)
                    if lu is not None and nprime == sizes[-1]:
                        lus.append(lu)
            mc.append(np.median(cs) if cs else np.nan)
            mp.append(np.median(ps) if ps else np.nan)
        mc, mp = np.array(mc), np.array(mp)
        sc, r2c = slope(sizes, mc)
        sp, r2p = slope(sizes, mp)
        ex_c.append(sc); ex_p.append(sp)
        print("   seed %d: median chi* %s" % (seed, " ".join("%6.2f" % v for v in mc)))
        print("            exponent chi*  %+.2f (R2 %.2f)   proxy %+.2f (R2 %.2f)"
              % (sc, r2c, sp, r2p))
        if lus:
            x = np.concatenate(lus)
            gams += [moment_evi(x, max(8, int(f * len(x)))) for f in (.10, .15, .20)]
    g = np.array(gams)
    print("   chi* exponent across seeds : %+.2f, %+.2f" % tuple(ex_c))
    print("   proxy exponent across seeds: %+.2f, %+.2f" % tuple(ex_p))
    print("   1 + gamma (tail of ell_u)  : %+.2f to %+.2f" % (1 + g.min(), 1 + g.max()))


if __name__ == "__main__":
    for nm, ld, sz in (("FD002", M.FD002, (40, 60, 90, 130, 180, 260)),
                       ("FD004", M.FD004, (40, 60, 90, 130, 180, 249)),
                       ("Severson", M.load_severson, (40, 60, 90, 135))):
        run(nm, ld, sz)
