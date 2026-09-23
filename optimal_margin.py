"""How far above the sample maximum should the threshold sit?

Section 4 recommends shifting the threshold conservatively above ell_0 and gives
no rule for how far.  The previous result supplies the missing ingredient: at the
sample maximum the out-of-sample failure probability is 1/(n+1), not zero, so
there is something to trade against the extra waste a shift costs.

Write p(m) for the out-of-sample exceedance probability at margin m above the
sample maximum.  Every surviving unit then wastes m more, so omega -> omega_0 +
m/Tbar and

    L(m) = (1 + chi p(m)) / (1 - omega_0 - m/Tbar).

Setting dL/dm = 0 and simplifying gives

    -p'(m*)  =  L* / (chi Tbar),

the density of the exceedance at the chosen margin equals L/(chi Tbar) -- the same
right-hand side as the tangency condition of Proposition 1, with the tail density
of ell_u in place of the lifetime hazard.  Two conditions of identical form, one
for choosing an age and one for choosing a margin.

Checked by sweeping the margin, scoring out of sample, and comparing the empirical
optimum against the first-order condition.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

DRAWS = 300


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    chi = M.CF - 1.0
    margins = np.linspace(0.0, 0.22 * Tb, 23)

    rng = np.random.default_rng(41)
    cost = np.zeros((DRAWS, len(margins)))
    pfs = np.zeros((DRAWS, len(margins)))
    for d in range(DRAWS):
        p = rng.permutation(len(pool))
        h = len(pool) // 2
        A, B = pool[p[:h]], pool[p[h:]]
        l0 = max(pa[i][R[i] > 0].min() for i in A)
        for j, m in enumerate(margins):
            per = np.array([M.unit_cost(i, l0 + m, R, rm, life) for i in B])
            cost[d, j] = per[:, 0].mean() / per[:, 1].mean() * Tb
            pfs[d, j] = per[:, 2].mean()
    mc = cost.mean(0)
    mp = pfs.mean(0)
    j = int(np.argmin(mc))

    print("=" * 78)
    print("%-9s Tbar=%.0f  n=%d per half  %d splits" % (name, Tb, len(pool) // 2, DRAWS))
    print("   margin/Tbar :" + "".join("%7.2f" % (m / Tb) for m in margins[::4]))
    print("   out-of-sample cost :" + "".join("%7.3f" % v for v in mc[::4]))
    print("   out-of-sample P_f  :" + "".join("%7.4f" % v for v in mp[::4]))
    print("   best margin = %.3f Tbar   cost %.4f  vs  %.4f at zero margin "
          "(gain %.1f%%)" % (margins[j] / Tb, mc[j], mc[0],
                             100 * (mc[0] - mc[j]) / mc[0]))
    print("   P_f falls %.4f -> %.4f" % (mp[0], mp[j]))

    # first-order condition: -p'(m*) against L*/(chi Tbar)
    if 0 < j < len(margins) - 1:
        dp = (mp[j + 1] - mp[j - 1]) / (margins[j + 1] - margins[j - 1])
        print("   -p'(m*) = %.6f   L*/(chi Tbar) = %.6f   ratio %.2f"
              % (-dp, mc[j] / (chi * Tb), -dp / (mc[j] / (chi * Tb))))
    else:
        print("   optimum at an endpoint of the swept range; condition not applicable")


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
