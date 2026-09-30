"""Component-level replacement with a shared setup cost.  Does Lemma 1 have to be rebuilt?

The paper lists this as open on the grounds that the cycle is no longer one unit's life.
That is true, and it turns out not to matter, because the long-run rate can be written per
COMPONENT instead of per cycle.  Let nu be the rate of maintenance interventions, r_j the
replacement rate of component j and f_j = P_{f,j} r_j its failure rate.  Then

    g = S nu + sum_j [ c_j + Delta_j P_{f,j} ] / [ Tbar_j (1 - omega_j) ],

since r_j = 1 / (Tbar_j (1 - omega_j)) is exactly the Lemma 1 denominator for component j.
So the rate is a SUM of Lemma 1 terms, coupled only through the single scalar nu.

Two consequences.  With S = 0 the problem separates completely: m independent risk-waste
problems, solved componentwise.  With S > 0 the entire content of economic dependence is
one number, and coordinating replacements is worth it exactly when the setup saving
S(sum_j r_j - nu) exceeds the sum of the componentwise regrets it causes.

The claim to test is the identity itself: simulate a three-component system under an
uncoordinated policy and under group replacement, and check the formula reproduces the
directly simulated long-run rate in both.
"""
import numpy as np
from scipy.special import gamma as gammafn


def simulate(shapes, scales, cs, deltas, ages, S, group, horizon=200_000.0, seed=0):
    """Run the system to `horizon` and return the realised rate and the coordinates."""
    m = len(shapes)
    rng = np.random.default_rng(seed)

    # batch the lifetime draws: one scipy rvs per event is what made this intractable
    BATCH = 200_000
    pool = [rng.weibull(shapes[j], BATCH) * scales[j] for j in range(m)]
    ptr = [0] * m

    def draw(j):
        if ptr[j] >= len(pool[j]):
            pool[j] = rng.weibull(shapes[j], BATCH) * scales[j]
            ptr[j] = 0
        v = pool[j][ptr[j]]
        ptr[j] += 1
        return float(v)

    life = [draw(j) for j in range(m)]
    used = [0.0] * m
    cost = 0.0
    t = 0.0
    n_int = 0
    rep = [0] * m
    fail = [0] * m
    waste = [0.0] * m
    total_life = [0.0] * m

    while t < horizon:
        due = [min(life[j], ages[j]) - used[j] for j in range(m)]
        dt = min(due)
        who = [j for j in range(m) if due[j] <= dt + 1e-12]
        t += dt
        for j in range(m):
            used[j] += dt
        touched = list(range(m)) if group else who
        cost += S
        n_int += 1
        for j in touched:
            failed = used[j] >= life[j] - 1e-12
            cost += cs[j] + (deltas[j] if failed else 0.0)
            rep[j] += 1
            fail[j] += int(failed)
            waste[j] += max(life[j] - used[j], 0.0)
            total_life[j] += life[j]
            life[j] = draw(j)
            used[j] = 0.0

    g_sim = cost / t
    nu = n_int / t
    om = [waste[j] / total_life[j] for j in range(m)]
    pf = [fail[j] / rep[j] for j in range(m)]
    Tb = [float(scales[j] * gammafn(1.0 + 1.0 / shapes[j])) for j in range(m)]
    g_form = S * nu + sum((cs[j] + deltas[j] * pf[j]) / (Tb[j] * (1 - om[j]))
                          for j in range(m))
    return g_sim, g_form, nu, om, pf, [rep[j] / t for j in range(m)]


def main():
    shapes, scales = [2.0, 3.0, 1.6], [1.0, 1.4, 0.8]
    cs, deltas = [1.0, 1.2, 0.8], [6.0, 4.0, 9.0]
    ages = [0.55, 0.95, 0.42]

    print("%-26s %10s %10s %9s %8s" %
          ("policy", "g simulated", "g formula", "rel.err", "nu"))
    for S in (0.0, 0.5, 2.0):
        for group in (False, True):
            gs, gf, nu, om, pf, r = simulate(shapes, scales, cs, deltas, ages, S, group)
            lab = "%s  S=%.1f" % ("group   " if group else "separate", S)
            print("%-26s %10.5f %10.5f %9.2e %8.4f"
                  % (lab, gs, gf, abs(gs - gf) / gs, nu))
        gs0, _, nu0, _, _, r0 = simulate(shapes, scales, cs, deltas, ages, S, False)
        gs1, _, nu1, _, _, _ = simulate(shapes, scales, cs, deltas, ages, S, True)
        print("   setup saving S(sum r - nu) = %.5f   actual gain = %+.5f"
              % (S * (sum(r0) - nu1), gs0 - gs1))
        print()


if __name__ == "__main__":
    main()
