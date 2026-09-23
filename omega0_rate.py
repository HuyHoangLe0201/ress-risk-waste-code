"""How fast does the zero-failure waste converge, and is it the parametric rate?

omega_0 is the waste at the smallest threshold that fails nobody, so it is anchored at
ell_0 = max_u ell_u -- an extreme, not an average.  The paper uses it constantly: the cost
ceiling of Proposition 5 is 1/(1-omega_0), the value of monitoring is capped by it, and
Theorem 5's kink sits there.  Its estimation error has never been characterised.

A maximum of n draws does not converge at n^{-1/2}.  If the upper tail of ell_u has a finite
endpoint with extreme-value index gamma < 0, the sample maximum approaches that endpoint at
rate n^{gamma}, so omega_0-hat should converge at an EVT rate governed by the same tail index
the paper already estimates for chi* -- and it should be biased DOWNWARD, since a smaller
observed maximum means a smaller apparent zero-failure threshold and less apparent waste.
That would make the ceiling 1/(1-omega_0) optimistic on a finite fleet.

Measured here by subsampling: draw n units, compute the zero-failure point on them alone,
and compare against the whole fleet's.  Both the sign of the bias and the exponent are read
off rather than assumed.
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
    R = [np.asarray(R[i], float) for i in idx]
    return paths, R


def omega0(paths, R, sub):
    """Waste fraction at the smallest threshold failing nobody in `sub`."""
    ell0 = max(float(paths[i][R[i] > 0].min()) for i in sub)
    w, n_u = [], []
    for i in sub:
        hit = np.nonzero(paths[i] <= ell0)[0]
        n_u.append(len(paths[i]))
        w.append(float(R[i][hit[0]]) if len(hit) and R[i][hit[0]] > 0 else 0.0)
    return float(np.mean(w)) / float(np.mean(n_u)), ell0


def run(name, ld, reps=400, seed=0):
    paths, R = prep(ld)
    N = len(paths)
    full, _ = omega0(paths, R, range(N))
    rng = np.random.default_rng(seed)
    sizes = [s for s in (10, 16, 25, 40, 63, 100, 160) if s <= N // 2]

    print("=" * 70)
    print("%s   N=%d, full-fleet omega_0 = %.4f" % (name, N, full))
    print("%8s %12s %12s %10s" % ("n", "mean om0-hat", "bias", "|bias|"))
    xs, ys = [], []
    for s in sizes:
        vals = [omega0(paths, R, rng.choice(N, s, replace=False))[0]
                for _ in range(reps)]
        m = float(np.mean(vals))
        xs.append(s)
        ys.append(abs(full - m))
        print("%8d %12.4f %+12.4f %10.4f" % (s, m, m - full, abs(full - m)))
    sl = np.polyfit(np.log(xs), np.log(ys), 1)[0]
    print("   slope of log|bias| on log n: %+.3f   (parametric would be -0.5)" % sl)


if __name__ == "__main__":
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        run(nm, ld)
