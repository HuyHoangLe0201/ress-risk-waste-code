r"""How fast can the fleet locate the optimal policy, as opposed to its cost?

Theorem 6 puts the admissible policies n^{-1/3} apart in P_f.  That spacing ought to be the
resolution with which the optimum can be identified at all, and if so the paper has been
reporting two quantities with different rates under one heading.

In dual coordinates the cost of a policy at cost ratio chi is a + b*chi, so the envelope is
a Legendre transform: with the achievable curve a = A(b) strictly convex, the optimum sits
at A'(b*) = -chi and the value is A(b*) + b* chi.  The empirical curve is A plus a walk of
size sqrt(v/n), which is the classical problem of minimising a convex function observed
with noise.  Near its minimum the objective is quadratic, so a displacement h costs c h^2/2
while the noise over that displacement is sqrt(v h / n); the two balance at the same
h ~ (v/(c^2 n))^{1/3} that set the vertex spacing.  Hence

    argmin:   b_hat - b*        = O_P(n^{-1/3})        -- Chernoff, cube root
    value:    L_hat - L*        = O_P(n^{-1/2}) with a downward bias O(n^{-2/3}).

The value is estimable at the parametric rate and the policy is not.  That is a sharper
statement than the paper's remark that the optimum's identity is unstable while its cost is
stable: it says the instability is a rate, it is the same n^{-1/3} as the vertex spacing,
and no amount of care in the search fixes it.

It also makes the paper's covering claim scale-dependent.  Deleting a hull vertex costs its
sagitta below the chord of its neighbours, which by the same balance is
c h^2/8 ~ n^{-2/3} -- and n^{-2/3} decays SLOWER than the lattice resolution chi/n against
which Section 4 checked it.  So "no vertex is individually necessary at tolerance chi/n"
cannot hold for every fleet size; it holds where the constants put the crossover, and the
crossover is computable.

Everything below is measured against a known truth, so the exponents are checked and not
assumed.
"""
import numpy as np

from hullscale import lower_hull

KAPPA = 4.0           # curvature of the achievable curve in dual coordinates
V = 1.0               # noise level of the empirical curve


def truth(chi):
    """A(b) = 1 + KAPPA (1-b)^2 / 2, so A'(b) = -KAPPA (1-b) and b* = 1 - chi/KAPPA."""
    bs = 1.0 - chi / KAPPA
    return bs, 1.0 + 0.5 * KAPPA * (1.0 - bs) ** 2 + bs * chi


def sample(n, rng):
    b = np.arange(n + 1) / n
    a = 1.0 + 0.5 * KAPPA * (1.0 - b) ** 2
    step = np.sqrt(V) / n * rng.standard_normal(n + 1)
    step[0] = 0.0
    return b, a + np.cumsum(step)


def rates(chi=1.0, reps=600):
    bs, Ls = truth(chi)
    print("=" * 78)
    print("chi = %.1f:  true b* = %.4f, true L* = %.5f" % (chi, bs, Ls))
    print("%8s %12s %12s %12s %12s %12s" %
          ("n", "|b_hat-b*|", "bias(L)", "sd(L)", "worst del", "median del"))
    ns = (100, 300, 1000, 3000, 10000, 30000)
    E, B, S, D, Dm = [], [], [], [], []
    rng = np.random.default_rng(0)
    for n in ns:
        eb, el, dl, dm = [], [], [], []
        for _ in range(reps):
            b, a = sample(n, rng)
            c = a + b * chi
            j = int(np.argmin(c))
            eb.append(abs(b[j] - bs))
            el.append(c[j] - Ls)
            h = lower_hull(b, a)
            if len(h) >= 3:
                # cost of dropping one vertex: its depth below the chord of its
                # neighbours, which is what the envelope becomes without it
                x, y = b[h], a[h]
                mid = (y[:-2] + (y[2:] - y[:-2]) * (x[1:-1] - x[:-2])
                       / (x[2:] - x[:-2]))
                dl.append(float(np.max(mid - y[1:-1])))
                dm.append(float(np.median(mid - y[1:-1])))
        E.append(np.mean(eb)); B.append(-np.mean(el))
        S.append(np.std(el, ddof=1)); D.append(np.mean(dl)); Dm.append(np.mean(dm))
        print("%8d %12.5f %12.6f %12.6f %12.6f %12.6f"
              % (n, E[-1], B[-1], S[-1], D[-1], Dm[-1]))
    ln = np.log(ns)
    for nm, y, want in (("|b_hat - b*|", E, -1.0 / 3), ("bias of L", B, -2.0 / 3),
                        ("sd of L", S, -0.5), ("worst delete", D, -2.0 / 3),
                        ("median delete", Dm, -2.0 / 3)):
        sl, ic = np.polyfit(ln, np.log(y), 1)
        pr = np.polyval([sl, ic], ln)
        r2 = 1 - np.sum((np.log(y) - pr) ** 2) / np.sum(
            (np.log(y) - np.mean(np.log(y))) ** 2)
        print("   %-14s exponent %+.3f   (theory %+.3f)   R2 %.3f"
              % (nm, sl, want, r2))
    return ns, D


def crossover(chi=1.0, reps=400):
    """Where does dropping a vertex stop being free at the fleet's own resolution?

    Section 4 checks each vertex against a RELATIVE tolerance of chi/n, so the absolute
    budget is (chi/n) L*(chi) and it decays as n^{-1}.  The worst deletion decays more
    slowly, so the two must cross; where they cross is set by the curvature and the noise
    and is not a universal fleet size.  The comparison must be made at a cost ratio whose
    optimum is interior -- at chi >= KAPPA the optimum is at the end of the curve, there
    is no interior vertex to delete, and the question does not arise.
    """
    _, Ls = truth(chi)
    print("=" * 78)
    print("worst deletion against the relative tolerance, chi = %.1f, L* = %.4f"
          % (chi, Ls))
    print("%8s %14s %14s %10s %10s"
          % ("n", "worst delete", "(chi/n) L*", "ratio", "necessary?"))
    rng = np.random.default_rng(1)
    for n in (100, 260, 600, 1500, 4000, 10000, 30000):
        dl = []
        for _ in range(reps):
            b, a = sample(n, rng)
            h = lower_hull(b, a)
            if len(h) >= 3:
                x, y = b[h], a[h]
                mid = (y[:-2] + (y[2:] - y[:-2]) * (x[1:-1] - x[:-2])
                       / (x[2:] - x[:-2]))
                dl.append(float(np.max(mid - y[1:-1])))
        d = float(np.mean(dl))
        tol = chi / n * Ls
        print("%8d %14.6f %14.6f %10.2f %10s"
              % (n, d, tol, d / tol, "yes" if d > tol else "no"))
    print("   the ratio grows without bound because the two decay at different")
    print("   rates; a fleet small enough always reports every vertex droppable.")


if __name__ == "__main__":
    rates(chi=1.0)
    crossover()
