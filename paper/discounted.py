"""Does the plane survive discounting?

Every result rests on a long-run AVERAGE cost rate.  Capital-intensive assets are
evaluated on discounted totals instead, and the renewal-reward identity does not
apply there, so the objection "your objective is the wrong one" is open.

Work it out.  With discount rate r and renewal cycles, the expected discounted
cost over an infinite horizon is

    V = E[discounted cost in a cycle] / (1 - E[e^{-r * cycle length}]).

Write A = E[e^{-r tau}; the unit survives to the planned action] and
B = E[e^{-r T}; it fails first].  The cycle discount factor is e^{-r tau} on
survival and e^{-r T} on failure, so E[e^{-r cycle}] = A + B, and

    V / c_p  =  (A + (1 + chi) B) / (1 - A - B),

which is again LINEAR-FRACTIONAL in two coordinates, with chi entering linearly.
So the plane is not an artefact of the averaging: discounting replaces
(omega, P_f) by (A, B) and every consequence that was geometry in the plane --
admissibility at envelope vertices, concavity in chi, canonicity, two-price
identification -- carries over unchanged.

Checked by simulating the discounted total directly and comparing.
"""
import numpy as np

RNG = np.random.default_rng(5)
SHAPE, SCALE = 2.5, 200.0
CHI = 9.0
HORIZON = 60000.0


def simulate_V(tau, r, reps=4000):
    """Realised discounted total cost per unit c_p, averaged over long histories."""
    tot = np.zeros(reps)
    for i in range(reps):
        t = 0.0
        v = 0.0
        while t < HORIZON:
            T = SCALE * RNG.weibull(SHAPE)
            if T <= tau:
                t += T
                v += (1.0 + CHI) * np.exp(-r * t)
            else:
                t += tau
                v += 1.0 * np.exp(-r * t)
            if np.exp(-r * t) < 1e-12:
                break
        tot[i] = v
    return tot.mean()


def coordinates(tau, r, reps=4000000):
    """A and B by direct Monte Carlo on one cycle."""
    T = SCALE * RNG.weibull(SHAPE, reps)
    surv = T > tau
    A = np.exp(-r * tau) * surv.mean()
    B = (np.exp(-r * T[~surv])).sum() / reps
    return A, B


print("discounted objective: simulated V vs (A + (1+chi)B)/(1 - A - B)")
print("%8s %8s %14s %14s %8s %8s %8s" %
      ("r", "tau", "simulated", "identity", "ratio", "A", "B"))
for r in (0.02, 0.005, 0.001):
    for tau in (80.0, 140.0, 200.0):
        A, B = coordinates(tau, r)
        ident = (A + (1.0 + CHI) * B) / (1.0 - A - B)
        sim = simulate_V(tau, r)
        print("%8.3f %8.0f %14.5f %14.5f %8.4f %8.4f %8.4f"
              % (r, tau, sim, ident, sim / ident, A, B))

print()
print("r -> 0 limit: r*V should approach the average cost rate (1+chi P_f)/E[cycle]")
for r in (0.01, 0.003, 0.001, 0.0003):
    tau = 140.0
    A, B = coordinates(tau, r)
    V = (A + (1.0 + CHI) * B) / (1.0 - A - B)
    T = SCALE * RNG.weibull(SHAPE, 2000000)
    pf = (T <= tau).mean()
    cyc = np.minimum(T, tau).mean()
    print("   r=%.4f   r*V = %.6f   rate = %.6f   ratio %.3f"
          % (r, r * V, (1 + CHI * pf) / cyc, r * V / ((1 + CHI * pf) / cyc)))
