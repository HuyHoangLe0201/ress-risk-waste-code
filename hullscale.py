r"""Where does the n^{1/3} vertex count come from, and when does it fail?

Section 4 measures the number of lower-hull vertices against fleet size, finds n^{0.36}
on FD002 with R^2 = 1.00, calls it close to the classical n^{1/3} for the convex hull of
random points in the plane, and reports the coincidence without deriving it.  The battery
fleet gives n^{0.19} on a poorer fit, which that reading leaves unexplained.

The convex-body analogy cannot be the mechanism: the dual points do not fill a
two-dimensional body, they lie along a one-dimensional curve indexed by the threshold.
The exponent has a different and more informative origin.

Write the empirical frontier as omega_hat(p) = omega_bar(p) + fluctuation, at
p = P_f = k/n.  Over a window of width h the frontier's own curvature c puts the curve
c h^2 / 8 above its chord, while the fluctuation is a sum over the ~nh units that switch
inside the window, hence a random walk of size sqrt(v h / n).  A vertex appears where the
two balance:

    sqrt(v h / n) ~ c h^2 / 8   =>   h ~ (v / (c^2 n))^{1/3},

so the vertex spacing is n^{-1/3} and the count is n^{1/3} times a constant that carries
the curvature as c^{2/3}.  This is Groeneboom's cube-root scaling for the convex minorant
of Brownian motion with parabolic drift, not the n^{1/3} of convex-body hulls; the two
exponents agree by accident.

The derivation immediately says when 1/3 is wrong.  If omega_bar is AFFINE on a stretch
there is no curvature to balance against, the local picture is a driftless walk, and
Sparre Andersen's cycle lemma applies instead: the number of faces of the convex minorant
of a walk with exchangeable increments is distributed as the number of cycles of a uniform
random permutation, with mean H_n = sum 1/k ~ log n.  So the count interpolates between
log n on a flat frontier and n^{1/3} on a curved one, and a fleet whose frontier is
flatter must show a smaller exponent -- which is what the battery fleet does.

Three experiments here, on a simulation where omega_bar is known exactly:

  (1) affine frontier.  The face count must have mean H_n.  This is also the check on the
      hull routine: H_n is a parameter-free prediction, so a hull bug cannot pass it.
  (2) strictly convex frontier.  The exponent must be 1/3, and the prefactor must scale
      as c^{2/3}.
  (3) the crossover.  Shrinking c at fixed n must walk the measured exponent from 1/3
      down toward 0.
"""
import numpy as np


def lower_hull(x, y):
    """Indices of the vertices of the convex minorant of the path (x, y), x sorted.

    Monotone chain: keep a vertex only while the turn stays counter-clockwise, so what
    survives is the lower boundary.
    """
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            i1, i2 = h[-2], h[-1]
            cross = ((x[i2] - x[i1]) * (y[i] - y[i1])
                     - (y[i2] - y[i1]) * (x[i] - x[i1]))
            if cross <= 0:          # i2 is on or above the chord i1 -> i
                h.pop()
            else:
                break
        h.append(i)
    return np.array(h)


def frontier(n, curvature, rng, v=1.0):
    """A path with a known population frontier and a random-walk fluctuation.

    omega_bar(p) = curvature * (1-p)^2 / 2 gives constant second derivative `curvature`;
    curvature = 0 is the affine case.  The fluctuation is a walk whose increments have
    standard deviation sqrt(v)/n, so that over a window of width h it is sqrt(v h / n),
    the size a sum over the ~nh units switching there actually has.
    """
    k = np.arange(n + 1)
    p = k / n
    bar = 0.5 * curvature * (1.0 - p) ** 2
    step = np.sqrt(v) / n * rng.standard_normal(n + 1)
    step[0] = 0.0
    return p, bar + np.cumsum(step)


def faces(n, curvature, rng, v=1.0):
    p, w = frontier(n, curvature, rng, v)
    return len(lower_hull(p, w)) - 1          # segments, not points


def harmonic(n):
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def exp_affine(reps=4000):
    print("=" * 74)
    print("(1) affine frontier: Sparre Andersen says E[faces] = H_n exactly")
    print("%8s %12s %12s %12s %12s" % ("n", "mean faces", "H_n", "sd", "sqrt(H_n)"))
    rng = np.random.default_rng(0)
    for n in (10, 30, 100, 300, 1000):
        f = np.array([faces(n, 0.0, rng) for _ in range(reps)], float)
        # Var = H_n - H_n^(2) for the cycle count of a random permutation
        h2 = float(np.sum(1.0 / np.arange(1, n + 1) ** 2))
        print("%8d %12.3f %12.3f %12.3f %12.3f"
              % (n, f.mean(), harmonic(n), f.std(ddof=1),
                 np.sqrt(harmonic(n) - h2)))


def exp_convex(reps=400):
    print("=" * 74)
    print("(2) strictly convex frontier: the exponent should be 1/3")
    ns = (40, 100, 260, 700, 2000, 6000)
    rng = np.random.default_rng(1)
    for c in (2.0, 16.0, 128.0):
        m = []
        for n in ns:
            f = np.array([faces(n, c, rng) for _ in range(reps)], float)
            m.append(f.mean())
        sl, ic = np.polyfit(np.log(ns), np.log(m), 1)
        pred = np.polyval([sl, ic], np.log(ns))
        r2 = 1.0 - np.sum((np.log(m) - pred) ** 2) / np.sum(
            (np.log(m) - np.mean(np.log(m))) ** 2)
        print("   c=%6.1f  faces %s   exponent %+.3f  R2 %.3f"
              % (c, " ".join("%6.1f" % t for t in m), sl, r2))
    print("   prefactor should scale as c^{2/3}: doubling c four-fold "
          "multiplies it by %.2f" % (4.0 ** (2.0 / 3.0)))


def exp_crossover(n=2000, reps=400):
    print("=" * 74)
    print("(3) crossover: shrinking the curvature at n=%d walks the count from" % n)
    print("    the cube-root regime down to the logarithmic one")
    rng = np.random.default_rng(2)
    print("%12s %12s %14s %14s" % ("curvature", "faces", "n^{1/3} scale", "H_n"))
    for c in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        f = np.array([faces(n, c, rng) for _ in range(reps)], float)
        print("%12.2f %12.2f %14.2f %14.2f"
              % (c, f.mean(), n ** (1.0 / 3.0), harmonic(n)))


if __name__ == "__main__":
    exp_affine()
    exp_convex()
    exp_crossover()
