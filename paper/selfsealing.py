r"""Operating a threshold policy destroys the data that would test it.

Every result in this paper is estimated from run-to-failure benchmarks, in which each unit
is observed to the end.  A fleet running a threshold policy does not produce such data, and
the difference is not a matter of degree.

A unit alarms when its running minimum reaches the threshold e, so:

  - a unit that FAILS is observed to the end, and its ell_u is known exactly;
  - a unit the policy SAVES is stopped at the alarm, and all that is learned is ell_u <= e.

The observation is therefore left-censored at the operating threshold.  Two consequences
follow immediately and neither depends on the estimator used.

Thresholds ABOVE e remain evaluable: a censored unit has ell_u <= e < e', so it would not
have failed at e' either, and a failed unit has its ell known exactly.  Thresholds BELOW e
are not: whether a censored unit would have reached e' < e is exactly what was not observed.
So the identified part of the frontier is the branch at thresholds at or above the one being
run.

And the zero-failure policy is the LARGEST threshold.  Operating there identifies only that
more conservative policies are worse, which was never in doubt, and leaves the whole
interesting branch -- less waste at some risk, the branch that determines chi* and the kink
-- unidentified.  By Theorem 11 the boundary policy fails at rate 1/(n+1), so that is also
the rate at which a fleet operating there recovers any exact observation at all.

Two experiments.  The first is an identification statement and is exact: two populations
that agree above e but differ below produce identical observations, so no estimator can
separate them.  The second is what it costs: the cost ratio is moved below chi* midway
through, so the optimum should move off the boundary, and the question is whether a fleet
running its own optimum ever finds out.
"""
import numpy as np


def frontier(lu, sc, e):
    """(P_f, omega) at threshold e for a population with thresholds lu and scales sc."""
    fail = lu > e
    waste = np.where(fail, 0.0, (e - lu) * sc)
    return float(np.mean(fail)), float(np.mean(waste))


def cost(lu, sc, e, chi):
    pf, om = frontier(lu, sc, e)
    return np.inf if om >= 1.0 else (1.0 + chi * pf) / (1.0 - om)


def identification(n=200000, e=0.70, seed=0):
    """Two laws agreeing above e, differing below: the observations cannot tell them apart."""
    rng = np.random.default_rng(seed)
    # law A: ell uniform on (0,1).  law B: same above e, but mass below e reshaped
    a = rng.random(n)
    b = np.where(a > e, a, e * rng.random(n) ** 3)      # below e, a different shape
    print("=" * 78)
    print("identification: two laws that agree above e=%.2f and differ below" % e)
    for nm, x in (("law A", a), ("law B", b)):
        below = x <= e
        print("   %s: mass below e %.4f, mean of ell below e %.4f, "
              "P(ell>e) %.4f" % (nm, below.mean(), x[below].mean(), 1 - below.mean()))
    # what a fleet operating at e actually records
    print("   what operating at e records:")
    for nm, x in (("law A", a), ("law B", b)):
        fail = x > e
        print("      %s: failure rate %.4f, exact ell values only from failures, "
              "mean %.4f" % (nm, fail.mean(), x[fail].mean()))
    print("   the two records agree in distribution; the laws do not.")


def run(n_rounds=6000, pool=200000, chi_early=9.0, chi_late=1.0, explore=0.0,
        seed=1, grid=241):
    """A fleet that re-optimises its threshold from what it can actually identify.

    Only two kinds of record are informative about a threshold BELOW the one in force.  A
    unit deliberately run to failure gives its ell exactly and is a random draw, so a
    frontier built from those is unbiased.  A unit that failed on its own also gives its
    ell exactly, but it is selected for having a large one -- it failed precisely because
    its ell exceeded the threshold -- so those records describe the upper tail and nothing
    else.  A unit the policy saved gives only ell <= e.

    The learner therefore fits the frontier from the explored units alone.  With no
    exploration it has no unbiased record at any threshold below the one it is running, and
    cannot move.  Exploration is charged at what it costs: an explored unit runs to failure,
    so it pays 1 + chi rather than the policy's cost.
    """
    rng = np.random.default_rng(seed)
    lu_pool = rng.random(pool)
    sc_pool = rng.random(pool) + 0.5
    es = np.linspace(0.02, 0.99, grid)

    seen_l, seen_s = [], []
    e_cur = float(es[-1])                      # start conservative, at the boundary
    n_exp = 0
    for t in range(n_rounds):
        chi = chi_early if t < n_rounds // 2 else chi_late
        i = rng.integers(0, pool)
        if rng.random() < explore:
            seen_l.append(float(lu_pool[i])); seen_s.append(float(sc_pool[i]))
            n_exp += 1
        if t % 25 == 24 and len(seen_l) >= 20:
            EL = np.array(seen_l); ES = np.array(seen_s)
            cs = [cost(EL, ES, e, chi) for e in es]
            e_cur = float(es[int(np.argmin(cs))])

    out = {}
    for chi in (chi_early, chi_late):
        cs = [cost(lu_pool, sc_pool, e, chi) for e in es]
        j = int(np.argmin(cs))
        out[chi] = (float(es[j]), float(cs[j]))
    oe, oc = out[chi_late]
    op = cost(lu_pool, sc_pool, e_cur, chi_late)
    total = explore * (1.0 + chi_late) + (1.0 - explore) * op
    return dict(explore=explore, final_e=e_cur, oracle_e=oe, operating=op,
                total=total, oracle=oc, regret=total / oc - 1.0,
                explored=n_exp, early_e=out[chi_early][0])


if __name__ == "__main__":
    identification()
    print("=" * 78)
    print("a fleet re-optimising from what it observes; chi drops 9 -> 1 at the midpoint")
    r0 = run(explore=0.0)
    print("   oracle threshold at chi=9  %.3f   at chi=1  %.3f"
          % (r0["early_e"], r0["oracle_e"]))
    print("%9s %9s %9s %11s %11s %10s %9s"
          % ("explore", "final e", "oracle e", "operating", "total", "oracle",
             "regret"))
    for ex in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        r = run(explore=ex)
        print("%9.3f %9.3f %9.3f %11.4f %11.4f %10.4f %8.1f%%"
              % (ex, r["final_e"], r["oracle_e"], r["operating"], r["total"],
                 r["oracle"], 100 * r["regret"]))
