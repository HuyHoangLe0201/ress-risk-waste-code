r"""Does the curvature explanation of the vertex exponent hold on the fleets?

hullscale.py derives the n^{1/3} count from a balance between the frontier's curvature and
the empirical fluctuation, and shows in simulation that a flat frontier gives H_n ~ log n
instead.  Section 4 measures n^{0.36} on FD002 and n^{0.19} on the battery fleet with a
poorer fit.  The derivation turns that discrepancy into a prediction rather than leaving it
as an anomaly: the battery frontier must be FLATTER relative to its own noise.

The comparison must not depend on fitting a curvature.  The dimensionless quantity the
balance actually involves is the sagitta -- how far the frontier bows below the chord
joining its endpoints -- measured in units of how far that sagitta moves when the fleet is
resampled:

    Lambda = sagitta / s.e.(sagitta),          both from the same unit-level bootstrap.

Lambda >> 1 is the curvature-dominated regime where the exponent is 1/3.  Lambda ~ 1 is the
noise-dominated regime where Sparre Andersen applies and the count is logarithmic.  This is
the paper's own resampling machinery, so it inherits the paper's floor, and it needs no
model of the frontier.

Reported per fleet: the vertex-count exponent from sub-fleets, and Lambda.  The prediction
is that they order the same way.  A fleet with the larger Lambda and the smaller exponent
would refute the mechanism.
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import makefigs6 as M                                            # noqa: E402
from hullscale import lower_hull, harmonic                       # noqa: E402


def prep(ld, seed=0):
    """Per-unit tables, computed once: the sub-fleet sweep below is run thousands of
    times, and recomputing a first crossing with nonzero() each time dominated it.

    For unit u the alarm index at threshold e is the first j with path[j] <= e, which is
    the first j with the running minimum <= e.  The running minimum is non-increasing, so
    that index is a searchsorted on its negation -- O(log T) instead of O(T), and
    vectorised over the whole threshold grid at once.
    """
    seq, life = ld()
    R, rm, pa, cal, val, Tb = M.prep(seq, life, seed=seed)
    idx = np.concatenate([cal, val])
    paths = [np.asarray(pa[i], float) for i in idx]
    negacc = [-np.minimum.accumulate(p) for p in paths]
    lives = np.array([len(p) for p in paths], float)
    return (paths, negacc), lives


def pareto(tab, lives, sub, ngrid=400):
    """The (P_f, omega) frontier over a sub-fleet: least waste at each failure count."""
    paths, negacc = tab
    Tb = float(np.mean([lives[i] for i in sub]))
    lo = min(paths[i].min() for i in sub)
    hi = max(paths[i].max() for i in sub)
    grid = np.linspace(hi, lo, ngrid)             # descending: P_f increases
    tot_w = np.zeros(ngrid)
    tot_f = np.zeros(ngrid, int)
    for i in sub:
        T = len(paths[i])
        j = np.searchsorted(negacc[i], -grid, side="left")
        hit = j < T
        tot_w += np.where(hit, (T - 1) - np.minimum(j, T - 1), 0)
        tot_f += ~hit
    om = tot_w / (len(sub) * Tb)
    ok = om < 1.0
    if not ok.any():
        return None, None
    best = {}
    for k, o in zip(tot_f[ok], om[ok]):
        k = int(k)
        if k not in best or o < best[k]:
            best[k] = float(o)
    if len(best) < 3:
        return None, None
    ks = np.array(sorted(best), float)
    return ks / len(sub), np.array([best[int(k)] for k in ks], float)


def vertices(pf, om):
    return len(lower_hull(pf, om)) - 1


def sagitta(pf, om):
    """How far the frontier bows below the chord joining its two ends."""
    if pf is None or len(pf) < 3:
        return np.nan
    chord = om[0] + (om[-1] - om[0]) * (pf - pf[0]) / (pf[-1] - pf[0])
    return float(np.max(chord - om))


def slope(sizes, draws, rng, B=400):
    """Exponent from the median counts, with a bootstrap s.e. over the draws.

    The count fluctuates by about a quarter with where the grid falls relative to the
    order statistics, so an exponent read off a handful of draws is not a number until
    it carries that spread.
    """
    med = np.array([np.median(d) for d in draws], float)
    lg, ls = np.log(med), np.log(sizes)
    sl, ic = np.polyfit(ls, lg, 1)
    pred = np.polyval([sl, ic], ls)
    r2 = 1.0 - np.sum((lg - pred) ** 2) / np.sum((lg - lg.mean()) ** 2)
    bs = []
    for _ in range(B):
        m = [np.median(rng.choice(d, len(d), replace=True)) for d in draws]
        bs.append(np.polyfit(ls, np.log(m), 1)[0])
    return float(sl), float(np.std(bs, ddof=1)), float(r2), med


def run(name, ld, reps=48, boots=200, seed=0):
    tab, lives = prep(ld)
    N = len(tab[0])
    rng = np.random.default_rng(seed)

    sizes = [s for s in (40, 70, 110, 170, 260) if s <= N]
    draws = []
    for s in sizes:
        v = []
        for _ in range(reps):
            sub = rng.choice(N, s, replace=False)
            pf, om = pareto(tab, lives, sub)
            if pf is not None:
                v.append(vertices(pf, om))
        draws.append(np.array(v, float))
    sl, sl_se, r2, med = slope(np.array(sizes, float), draws, rng)

    pf, om = pareto(tab, lives, np.arange(N))
    s0 = sagitta(pf, om)
    bs = []
    for _ in range(boots):
        sub = rng.choice(N, N, replace=True)          # resample UNITS
        p2, o2 = pareto(tab, lives, sub)
        if p2 is not None:
            bs.append(sagitta(p2, o2))
    se = float(np.std(bs, ddof=1))
    lam = s0 / se if se > 0 else np.inf
    lam_se = lam / np.sqrt(2.0 * (len(bs) - 1))       # delta method on a ratio to an s.e.

    print("=" * 76)
    print("%s   N=%d   (%d draws per size)" % (name, N, reps))
    print("   sub-fleet sizes   %s" % " ".join("%7d" % s for s in sizes))
    print("   hull vertices     %s" % " ".join("%7.1f" % c for c in med))
    print("   exponent %+.3f +- %.3f  (R2 %.3f)    H_N = %.2f, N^{1/3} = %.2f"
          % (sl, sl_se, r2, harmonic(N), N ** (1.0 / 3.0)))
    print("   sagitta %.5f  s.e. %.5f   Lambda = %.1f +- %.1f"
          % (s0, se, lam, lam_se))
    return sl, sl_se, lam, lam_se


if __name__ == "__main__":
    out = {}
    for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004),
                   ("Severson", M.load_severson)):
        out[nm] = run(nm, ld)
    print("=" * 76)
    print("%-10s %18s %16s" % ("fleet", "exponent", "Lambda"))
    for nm, (sl, sse, lam, lse) in out.items():
        print("%-10s %+12.3f +- %.3f %10.1f +- %.1f" % (nm, sl, sse, lam, lse))
    order_e = sorted(out, key=lambda k: out[k][0])
    order_l = sorted(out, key=lambda k: out[k][2])
    print("   by exponent: %s" % " < ".join(order_e))
    print("   by Lambda  : %s" % " < ".join(order_l))
    print("   mechanism predicts the two orderings agree: %s"
          % ("yes" if order_e == order_l else "NO -- refuted"))
