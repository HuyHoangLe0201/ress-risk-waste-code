r"""Exploring by perturbing the threshold, instead of by running units to failure.

Theorem 21 prices exploration at Delta = (1+chi) - L*, the premium for running a unit to
failure rather than operating on it.  That is a fixed cost per explored unit and it is what
forces the m^{3/5} cumulative regret.  It is also not the only way to learn about the branch
below the current threshold, and it is the dearest.

At the alarm the unit has not failed; it has merely reached e.  Letting it run a little
further and replacing at e - delta reveals the indicator 1{ell_u > e - delta}, which is
exactly an observation of the frontier at e - delta, and it costs the frontier difference

    L(e - delta) - L(e)  ~  c delta^2 / 2

near an interior optimum.  Unlike Delta this vanishes with delta, so the exploration can be
made arbitrarily cheap at the price of a noisier signal, which is the classical trade of
Kiefer and Wolfowitz.  Balancing a finite-difference gradient error of sigma/(delta sqrt N)
against the perturbation cost gives

    delta ~ N^{-1/4},   per-unit regret ~ N^{-1/2},   cumulative regret ~ m^{1/2},

against the m^{3/5} of Theorem 21.  The two schemes are run head to head here on the same
population, with the regret of each measured the same way: the sum over units of the cost
actually incurred at the threshold actually operated, less the oracle cost.  The exponent is
what is being tested, so both are swept over the same range of m.
"""
import numpy as np


def make_pool(pool, seed):
    rng = np.random.default_rng(seed)
    return rng.random(pool), rng.random(pool) + 0.5, rng


def cost_at(lu, sc, e, chi):
    fail = lu > e
    om = float(np.mean(np.where(fail, 0.0, (e - lu) * sc)))
    return np.inf if om >= 1.0 else (1.0 + chi * float(np.mean(fail))) / (1.0 - om)


def cost_curve(lu, sc, es, chi):
    """Cost at every threshold at once, by cumulative sums rather than a loop.

    Sorting the units once turns each threshold into a searchsorted plus two prefix sums,
    so a grid of G thresholds costs O(N log N + G) instead of O(N G).  The refit inside the
    learning loop is run thousands of times and dominated everything otherwise.
    """
    o = np.argsort(lu)
    l, c = lu[o], sc[o]
    S1 = np.concatenate([[0.0], np.cumsum(c)])
    S2 = np.concatenate([[0.0], np.cumsum(l * c)])
    N = len(l)
    k = np.searchsorted(l, es, side="right")        # units with ell <= e
    om = (es * S1[k] - S2[k]) / N
    pf = (N - k) / N
    out = np.full(len(es), np.inf)
    ok = om < 1.0
    out[ok] = (1.0 + chi * pf[ok]) / (1.0 - om[ok])
    return out


def oracle(lu, sc, chi, es):
    cs = cost_curve(lu, sc, es, chi)
    j = int(np.argmin(cs))
    return float(es[j]), float(cs[j])


def perturb_run(m, chi, pl, ps, rng, Lstar, e0, es_true, cs_true, k=20, d=0.25,
                a=0.06, lo=0.03, hi=0.97):
    """Kiefer-Wolfowitz on the threshold: perturb, difference, step.  Cost is what it costs.

    Each round operates k units at e+delta and k at e-delta, so the perturbation cost is
    paid on every unit and is counted in the regret, as is the error of the current e.
    """
    e = float(e0)
    used, reg = 0, 0.0
    t = 0
    while used < m:
        t += 1
        dl = d * t ** -0.25
        tot = 0.0
        for sgn in (+1.0, -1.0):
            eo = float(np.clip(e + sgn * dl, lo, hi))
            i = rng.integers(0, len(pl), k)
            tot += sgn * cost_at(pl[i], ps[i], eo, chi)
            reg += k * (float(np.interp(eo, es_true, cs_true)) - Lstar)
            used += k
        g = tot / (2.0 * dl)
        e = float(np.clip(e - (a / t) * g, lo, hi))
    return reg, e


def failure_run(m, chi, pl, ps, rng, Lstar, e0, eps, es, es_true, cs_true,
                batch=40):
    """The Theorem 21 scheme: explore by running a fraction of units to failure."""
    e = float(e0)
    seen_l, seen_s = [], []
    used, reg, next_fit = 0, 0.0, 0
    while used < m:
        n_ex = int(rng.binomial(batch, eps))
        i = rng.integers(0, len(pl), n_ex)
        seen_l.extend(pl[i]); seen_s.extend(ps[i])
        reg += n_ex * ((1.0 + chi) - Lstar)                 # an explored unit fails
        reg += (batch - n_ex) * (float(np.interp(e, es_true, cs_true)) - Lstar)
        used += batch
        if len(seen_l) >= 20 and used >= next_fit:
            EL = np.array(seen_l); ES = np.array(seen_s)
            e = float(es[int(np.argmin(cost_curve(EL, ES, es, chi)))])
            next_fit = used * 1.3        # geometric refits: the estimate cannot
                                         # move materially between close ones
    return reg, e


def main(chi=1.0, pool=300000, reps=8, seed=0):
    pl, ps, rng = make_pool(pool, seed)
    es = np.linspace(0.03, 0.97, 191)
    e_star, Lstar = oracle(pl, ps, chi, es)
    print("=" * 78)
    print("chi=%.1f  oracle threshold %.3f  oracle cost %.4f  run-to-failure %.4f"
          % (chi, e_star, Lstar, 1.0 + chi))
    es_true = np.linspace(0.03, 0.97, 1401)
    cs_true = cost_curve(pl, ps, es_true, chi)
    ms = (2000, 8000, 32000, 128000)
    print("%9s %16s %16s %12s" % ("m", "perturb regret", "failure regret", "ratio"))
    P, F = [], []
    for m in ms:
        p = np.mean([perturb_run(m, chi, pl, ps, rng, Lstar, 0.97, es_true,
                                 cs_true)[0] for _ in range(reps)])
        # the failure scheme gets its own best eps at this m, which is generous to it
        f = min(np.mean([failure_run(m, chi, pl, ps, rng, Lstar, 0.97, ev, es,
                                     es_true, cs_true)[0] for _ in range(reps)])
                for ev in (0.005, 0.01, 0.02, 0.05, 0.10))
        P.append(p); F.append(f)
        print("%9d %16.1f %16.1f %12.2f" % (m, p, f, f / p))
    for nm, v, want in (("perturb", P, 0.5), ("failure", F, 0.6)):
        sl = float(np.polyfit(np.log(ms), np.log(v), 1)[0])
        print("   %-8s cumulative exponent %+.3f   (theory %+.3f)" % (nm, sl, want))


if __name__ == "__main__":
    main(chi=1.0)
    main(chi=0.5)
