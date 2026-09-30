r"""The limit law of chi-star, not just its growth rate.

Equation (eq:spacing) carries the whole n-dependence of chi* in the top spacing of the
order statistics of ell_u,

    chi*  ~=  n D_n / (Tbar (1 - omega_0)),      D_n = ell_(n) - ell_(n-1),

and extreme-value theory is quoted for the RATE: D_n scales as n^gamma, so chi* grows
as n^{1+gamma}.  The paper stops there, writing "for a non-degenerate W" without saying
which W.  It is identifiable exactly, and what it says is stronger than a rate.

For a law in the domain of attraction with index gamma and norming a_n, the top order
statistics converge jointly to the points of a unit-rate Poisson process,

    (ell_(n-i+1) - b_n)/a_n  =>  (Gamma_i^{-gamma} - 1)/gamma,   Gamma_i = E_1 + ... + E_i,

so the top spacing has the non-degenerate limit

    D_n / a_n  =>  W_gamma = (Gamma_1^{-gamma} - Gamma_2^{-gamma})/gamma,

which at gamma = 0 is log(Gamma_2/Gamma_1) = log(1 + E_2/E_1).  That last quantity is
standard exponential exactly: E_2/E_1 has survival 1/(1+t), so 1 + E_2/E_1 is Pareto(1)
and its logarithm is Exp(1).  For the exponential law the statement is exact at every n
by Renyi's representation, not merely in the limit.

The consequence is not a refinement of the rate but a different kind of statement.  The
normalised chi* does not converge to a constant, so chi* is NOT consistently estimable
from one fleet however large: in the Gumbel domain its coefficient of variation stays at
one forever, and P(chi* > x) -> exp(-x Tbar (1-omega_0)/(n a_n)).  The growth exponent
the paper fits describes the movement of the quantiles, not a convergence.

Checked three ways: the exact exponential case, the limit across domains with known
gamma, and the coefficient of variation of chi* itself on the real fleets, which must
refuse to shrink with n if the claim is right.  A control law whose statistic IS
consistently estimable is run beside it, since a test that cannot see concentration
cannot testify to its absence.

Run: python chistar_law.py
"""
import numpy as np
from scipy import stats

REPS = 40000


def top_spacing(rng, draw, n, reps=REPS, budget=20_000_000):
    """Top spacing over `reps` fleets of size n, in chunks.

    Drawing all of it at once is reps*n doubles, which at n=1e5 asks for fifteen
    gigabytes; only the two largest values of each fleet are ever used, so the draws
    are taken in blocks sized to a fixed memory budget.
    """
    out = np.empty(reps)
    step = max(1, int(budget // max(n, 1)))
    for i in range(0, reps, step):
        m = min(step, reps - i)
        x = draw(rng, (m, n))
        x.partition(n - 2, axis=1)
        top = x[:, n - 2:]
        top.sort(axis=1)
        out[i:i + m] = top[:, 1] - top[:, 0]
    return out


def W(rng, gamma, reps=REPS):
    """Draws of the predicted limit W_gamma."""
    E = rng.exponential(size=(reps, 2))
    G1, G2 = E[:, 0], E[:, 0] + E[:, 1]
    if abs(gamma) < 1e-12:
        return np.log(G2 / G1)
    return (G1 ** (-gamma) - G2 ** (-gamma)) / gamma


def ks_shape(a, b):
    """Compare shapes after dividing each by its own mean, so norming cancels."""
    return float(stats.ks_2samp(a / a.mean(), b / b.mean()).statistic)


def main():
    rng = np.random.default_rng(0)

    print("exponential law: Renyi makes D_n exactly Exp(1) at every n")
    print("   n      mean D_n   sd D_n    KS vs Exp(1)   p")
    for n in (10, 50, 500, 5000):
        d = top_spacing(rng, lambda r, s: r.exponential(size=s), n)
        k = stats.kstest(d, "expon")
        print("   %-6d %-10.4f %-9.4f %-14.4f %.3f"
              % (n, d.mean(), d.std(), k.statistic, k.pvalue))

    print()
    print("other domains: does D_n/mean match W_gamma/mean?")
    print("   law                gamma   n      KS(D_n, W_gamma)  KS(D_n, Exp(1))  CV")
    laws = [("normal", 0.0, lambda r, s: r.normal(size=s)),
            ("lognormal", 0.0, lambda r, s: r.lognormal(0, 1, s)),
            ("Weibull(2)", 0.0, lambda r, s: r.weibull(2.0, s)),
            ("Pareto(2)", 0.5, lambda r, s: (1 - r.random(s)) ** -0.5),
            ("Pareto(1.25)", 0.8, lambda r, s: (1 - r.random(s)) ** -0.8),
            ("uniform", -1.0, lambda r, s: r.random(s))]
    for name, g, draw in laws:
        w = W(rng, g)
        for n in (200, 2000):
            d = top_spacing(rng, draw, n)
            print("   %-18s %-7.1f %-6d %-17.4f %-16.4f %.3f"
                  % (name, g, n, ks_shape(d, w),
                     ks_shape(d, rng.exponential(size=REPS)), d.std() / d.mean()))

    print()
    print("the point of it: the spread does not shrink with n")
    print("   law                n=100    n=1000   n=10000   n=100000   predicted CV")
    for name, g, draw in laws[:1] + laws[3:5]:
        w = W(rng, g)
        cvs = [top_spacing(rng, draw, n, reps=20000) for n in (100, 1000, 10000, 100000)]
        print("   %-18s " % name
              + "  ".join("%7.3f" % (c.std() / c.mean()) for c in cvs)
              + "     %.3f" % (w.std() / w.mean()))

    print()
    print("control: a statistic that IS consistently estimable, same machinery.")
    print("A coefficient of variation needs a mean bounded away from zero, so the")
    print("control statistics are positive; the normal median is not and its CV is")
    print("meaningless rather than informative.")
    print("   statistic              n=100    n=1000   n=10000   n=100000   sd ratio")
    ctl = [("median of lognormal", lambda r, s: r.lognormal(0, 1, s)),
           ("median of Pareto(2)", lambda r, s: (1 - r.random(s)) ** -0.5)]
    for name, draw in ctl:
        cvs, sds = [], []
        for n in (100, 1000, 10000, 100000):
            step = max(1, 20_000_000 // n)
            m = []
            for i in range(0, 4000, step):
                x = draw(rng, (min(step, 4000 - i), n))
                m.append(np.median(x, axis=1))
            m = np.concatenate(m)
            cvs.append(m.std() / abs(m.mean()))
            sds.append(m.std())
        print("   %-22s " % name
              + "  ".join("%7.4f" % c for c in cvs)
              + "   %.1f per decade (sqrt n predicts 3.2)"
              % (sds[0] / sds[-1]) ** (1 / 3.))


if __name__ == "__main__":
    main()
