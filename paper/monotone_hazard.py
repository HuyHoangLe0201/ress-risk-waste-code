"""The sufficient condition for Theorem 3's hypothesis, checked in both directions.

Theorem 3 is stated with an abstract hypothesis -- the region {(c_f-c_p)h >= g*} must be
absorbing.  A checkable sufficient condition is that the conditional hazard be
non-decreasing along paths, the filtered analogue of an increasing failure rate; under
trivial information it is exactly the IFR assumption under which the Barlow-Proschan
optimum is interior.

This checks the positive direction on a chain whose hazard is strictly increasing: the
one-step rule should select the cost-minimising threshold at every cost ratio.  The
negative direction is in frontier_quad.py, where a bump in the hazard breaks it.
"""
import numpy as np

K = 60
p = np.full(K, 0.25)
q = 0.0015 * (1.10 ** np.arange(K))          # strictly increasing
h = q.copy()
h[K - 1] += p[K - 1]                         # last state also fails by advancing

m = np.zeros(K + 1)
for i in range(K - 1, -1, -1):
    m[i] = (1.0 + p[i] * m[i + 1]) / (p[i] + q[i])
Tbar = m[0]


def stats(s):
    """(P_f, omega) for replacement on reaching state s."""
    u = 1.0
    for i in range(s):
        u *= p[i] / (p[i] + q[i])
    return 1.0 - u, u * m[s] / m[0]


def main():
    S = np.arange(1, K)
    U = np.array([stats(int(s)) for s in S])
    print("hazard non-decreasing in state:", bool(np.all(np.diff(h[:K - 1]) >= 0)))
    print("%8s %11s %11s %7s" % ("chi", "OLA state", "best state", "match"))
    ok = 0
    for chi in (1.0, 2.0, 3.0, 6.0, 9.0, 15.0, 30.0, 60.0):
        L = (1.0 + chi * U[:, 0]) / (1.0 - U[:, 1])
        jb = int(np.argmin(L))
        g = L[jb] / Tbar                     # optimal rate, normalised units
        cross = np.nonzero(chi * h[S] >= g)[0]
        jo = int(cross[0]) if len(cross) else len(S) - 1
        ok += (jo == jb)
        print("%8.0f %11d %11d %7s" % (chi, S[jo], S[jb], jo == jb))
    print("matched at %d of 8 cost ratios" % ok)


if __name__ == "__main__":
    main()
