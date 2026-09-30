"""Is the frontier of ALL adapted rules an index rule, with the same constant?

Proposition 1 says the age optimum sits where the hazard equals L/(chi Tbar).
Theorem 5 says the threshold optimum sits at a kink.  If both are instances of one
condition, that condition should be: replace at the first time the CONDITIONAL
failure intensity, given everything observed, reaches L/(chi Tbar).  With no
information the conditional intensity is the unconditional hazard and the rule
reduces to Proposition 1; with a predictor it is a control limit on that
intensity, and thresholding predicted RUL is optimal exactly when the predicted
RUL is monotone in it.

That is checkable without estimating anything exotic.  Bin cycles by the running
minimum of predicted RUL, estimate the empirical one-cycle failure probability in
each bin, and ask two things:

  (a) is the conditional intensity MONOTONE in the running minimum?  If not, a
      threshold on predicted RUL cannot be the frontier rule and the paper's
      family is provably suboptimal among adapted rules.
  (b) at the cost-optimal threshold, does the intensity sit near L/(chi Tbar)?

Both are reported for what they are; (a) is the one that decides whether the
proposed unification is honest.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

NBIN = 12


def gather(ld, seed=0):
    """Per-cycle records: running-min of predicted RUL, and whether death is next."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = np.concatenate([cal, val])
    s, nxt = [], []
    for i in idx:
        r = R[i]
        live = r > 0
        m = rm[i][live]
        rr = r[live]
        s.append(m)
        nxt.append((rr <= 1.0).astype(float))     # fails at the next cycle
    return np.concatenate(s), np.concatenate(nxt), Tb, R, rm, pa, cal, life


def run(name, ld):
    s, nxt, Tb, R, rm, pa, cal, life = gather(ld)
    qs = np.quantile(s, np.linspace(0, 1, NBIN + 1))
    qs[0] -= 1e-9
    lam, mid, cnt = [], [], []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (s > a) & (s <= b)
        if m.sum() < 30:
            continue
        lam.append(float(nxt[m].mean()))
        mid.append(float(0.5 * (a + b) / Tb))
        cnt.append(int(m.sum()))
    lam, mid = np.array(lam), np.array(mid)

    order = np.argsort(mid)
    lam_o = lam[order]
    dec = np.all(np.diff(lam_o) <= 1e-12)
    inc = np.all(np.diff(lam_o) >= -1e-12)
    rho = float(np.corrcoef(mid, lam)[0, 1])

    print("=" * 78)
    print("%-9s conditional one-cycle failure probability by running-min bin" % name)
    print("   running-min/Tbar :" + "".join("%7.2f" % v for v in mid[order]))
    print("   intensity        :" + "".join("%7.4f" % v for v in lam_o))
    print("   monotone in the statistic? decreasing %s  increasing %s   corr %+.2f"
          % (dec, inc, rho))

    # (b) the intensity at the cost-optimal threshold, against L/(chi Tbar)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu, .05), lu.max() + .2 * Tb, 120)
    C = np.array([[M.unit_cost(i, e, R, rm, life) for i in cal] for e in g])
    om = 1 - C[:, :, 1].mean(1) / Tb
    pf = C[:, :, 2].mean(1)
    chi = M.CF - 1.0
    L = (1 + chi * pf) / (1 - om)
    j = int(np.argmin(L))
    target = L[j] / (chi * Tb)
    at_opt = float(np.interp(g[j] / Tb, mid[order], lam_o))
    print("   at the cost optimum: intensity %.5f   vs  L/(chi Tbar) = %.5f   "
          "ratio %.2f" % (at_opt, target, at_opt / target if target else np.nan))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
