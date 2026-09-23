"""Is the "zero-failure" threshold zero-failure on units it has not seen?

ell_0 = max_u ell_u is a SAMPLE MAXIMUM, so it underestimates the population
endpoint, and a threshold set too low admits failures.  Exchangeability gives the
size exactly and without any distributional assumption: for i.i.d. continuous
draws, a new unit exceeds the maximum of n previous ones with probability

    P( ell_{n+1} > max(ell_1..ell_n) )  =  1 / (n+1),

so the realised failure probability of the zero-failure policy is 1/(n+1) in
expectation, not zero.  By the sensitivity relation (9) the realised cost then
exceeds its in-sample value by chi/(n+1) at leading order -- numerically the same
quantity as the variance step, and from the same source.

A population zero-failure threshold exists at all only if ell_u has a finite upper
endpoint, that is, extreme-value index gamma < 0.  Section 4 already measures
gamma: negative on the turbofan fleets, straddling zero on the battery fleet.
That predicts which fleet's boundary should fail out of sample.

Tested by splitting the fleet, setting the boundary on one half and scoring the
other, many times.
"""
import sys

import numpy as np

import os.path as _p; sys.path.insert(0, _p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402

DRAWS = 400


def run(name, ld):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=0)
    pool = np.concatenate([cal, val])
    lu = {i: pa[i][R[i] > 0].min() for i in pool}
    v = np.array([lu[i] for i in pool])

    rng = np.random.default_rng(29)
    rates, ns = [], []
    for _ in range(DRAWS):
        p = rng.permutation(len(pool))
        m = len(pool) // 2
        a, b = v[p[:m]], v[p[m:]]
        l0 = a.max()                     # the boundary, set on the calibration half
        rates.append(float((b > l0).mean()))
        ns.append(m)
    r = np.array(rates)
    n = ns[0]
    print("=" * 78)
    print("%-9s calibration n=%d, %d splits" % (name, n, DRAWS))
    print("   out-of-sample failure rate at the 'zero-failure' threshold:")
    print("      mean %.4f   median %.4f   fraction of splits with any failure %.2f"
          % (r.mean(), np.median(r), float((r > 0).mean())))
    print("   exchangeability predicts 1/(n+1) = %.4f          ratio %.2f"
          % (1.0 / (n + 1), r.mean() * (n + 1)))
    chi = M.CF - 1.0
    print("   implied relative cost penalty chi/(n+1) = %.1f%%   (measured %.1f%%)"
          % (100 * chi / (n + 1), 100 * chi * r.mean()))


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
