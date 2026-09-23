r"""Figures for the section on what a fleet can learn at the boundary.

That section carries ten numbered results and had no figure.  Two are added, each drawn so
that the mechanism is visible rather than the conclusion restated.

The first is about measurement.  The bootstrap distribution of the boundary threshold is
drawn as it actually is -- an atom at the point estimate and nothing whatever above it --
beside the same resampling applied to a mean, which is symmetric and has no atom.  The
consequence, coverage that plateaus below nominal however large the fleet, is the second
panel, and the third shows the conformal relabelling holding across indices on all three
fleets.

The second is about operating.  Two populations that agree above the operating threshold and
differ below it are drawn with the unidentified region shaded, so the reader can see that
what the fleet records is the same for both.  Then the price: regret against the fraction of
units deliberately run to failure, and the two exploration schemes side by side.

Conventions follow makefigs_ress.py: the local style, 6.48in exactly, crop off.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from figstyle import (setup, clean, ring, COL, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND)                     # noqa: E402
import figstyle as _fs                                           # noqa: E402
# Identity, not width: pinned to 6.48 this fired on the legitimate move to the
# Elsevier 190 mm width rather than on the shadowing it exists to catch.
assert "RESS_paper" in os.path.abspath(_fs.__file__), \
    "IEEE figstyle shadowed the RESS one"

import os.path as _p; sys.path.append(_p.join(_p.dirname(_p.abspath(__file__)), "shared"))
import matplotlib.pyplot as plt                                  # noqa: E402
import makefigs6 as M                                            # noqa: E402
from quota import prep                                           # noqa: E402
from conformal import unit_tables, waste_at                      # noqa: E402

setup()
FLEETS = (("FD002", M.FD002, BLUE), ("FD004", M.FD004, ORANGE),
          ("Severson", M.load_severson, VIOLET))


def save(fig, path):
    fig.savefig(path)
    w, _ = fig.get_size_inches()
    assert abs(w - COL) < 1e-6, "%s authored %.3fin, not %.3f" % (path, w, COL)
    plt.close(fig)
    print(path, "written", flush=True)


def anno(ax, fx, fy, s, color=INK2, size=7.4, ha="left", va="top"):
    ax.text(fx, fy, s, transform=ax.transAxes, color=color, fontsize=size,
            ha=ha, va=va, zorder=6, linespacing=1.35)


def panel(ax, tag, title):
    ax.set_title("(%s)  %s" % (tag, title), loc="left", pad=4.0)


# ----------------------------------------------------------------- figure 12
def fig12(B=4000, seed=0):
    fig, ax = plt.subplots(1, 3, figsize=(COL, 2.35), layout="constrained")
    a, b, c = ax
    for q in ax:
        clean(q)

    # (a) the bootstrap of an extreme against the bootstrap of a mean
    panel(a, "a", "the bootstrap at the boundary")
    _, _, _, lu = prep(M.FD002)
    lu = np.asarray(lu, float)
    n = len(lu)
    rng = np.random.default_rng(seed)
    mx = np.array([lu[rng.integers(0, n, n)].max() for _ in range(B)])
    obs = float(lu.max())
    atom = float(np.mean(mx == obs))
    below = mx[mx < obs]
    a.hist(below, bins=28, color=BLUE, alpha=0.55, edgecolor="none",
           weights=np.full(len(below), 1.0 / B), zorder=3)
    lo = float(below.min()) if len(below) else obs
    # the atom is three times the tallest histogram bar, so it is drawn as its own
    # bar and the axis is set to hold both; an arrow to it would leave the frame
    a.bar([obs], [atom], width=0.030 * (obs - lo), color=INK, zorder=5)
    a.set_ylim(0, 1.12 * atom)
    a.axvspan(obs, obs + 0.55 * (obs - lo), color=BAND, lw=0, zorder=1)
    a.set_xlabel(r"bootstrap replicate of $\hat\ell_0$")
    a.set_ylabel("probability")
    anno(a, 0.03, 0.97,
         "atom %.3f at the estimate\n(predicted $1-(1-1/n)^n=%.3f$)\n"
         "nothing above it, ever" % (atom, 1 - (1 - 1 / n) ** n))
    a.annotate("the truth\nlies here", xy=(obs + 0.28 * (obs - lo), 0.30 * atom),
               ha="center", va="center", color=INK3, fontsize=6.9, zorder=6)
    print("   fig12(a) atom %.4f, mass above %.4f" % (atom, float(np.mean(mx > obs))))

    # (b) coverage that does not improve with n
    panel(b, "b", "and coverage stays short")
    ns = (100, 400, 1600, 6400)
    ALPHA, J = 0.05, 2.0
    rng2 = np.random.default_rng(3)
    for lbl, frac, col in (("$n$ out of $n$", 1.0, ORANGE),
                           ("$m=n^{0.7}$", None, BLUE)):
        cov = []
        for nn in ns:
            m = nn if frac else max(3, int(nn ** 0.7))
            hit = 0
            for _ in range(260):
                x = 1.0 - rng2.beta(1.0, 2.0, nn)
                o = float(x.max())
                d = np.array([x[rng2.integers(0, nn, m)].max() for _ in range(260)])
                dev = (o - d) * (m / nn) ** (1.0 / J)
                hit += (o + np.percentile(dev, 2.5) <= 1.0 <=
                        o + np.percentile(dev, 97.5))
            cov.append(hit / 260)
        b.plot(ns, cov, color=col, marker="o", label=lbl)
        print("   fig12(b) %-14s %s" % (lbl, " ".join("%.3f" % v for v in cov)))
    b.axhline(0.95, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)
    b.set_xscale("log")
    b.set_ylim(0.80, 1.02)
    b.set_xlabel("fleet size $n$")
    b.set_ylabel("coverage of a 95% interval")
    b.legend(loc="lower right", handlelength=1.2)

    # (c) the conformal relabelling, on the fleets
    panel(c, "c", "the risk axis is conformal")
    js = [0, 1, 2, 4, 8, 16, 32]
    for nm, ld, col in FLEETS:
        paths, Rs, lives, lu2, acc = unit_tables(ld)
        lu2 = np.asarray(lu2, float)
        N = len(lu2)
        k = N // 2
        rg = np.random.default_rng(0)
        nom, real = [], []
        for j in [x for x in js if x < k - 1]:
            v = []
            for _ in range(400):
                p = rg.permutation(N)
                o = np.sort(lu2[p[:k]])[::-1]
                v.append(float(np.mean(lu2[p[k:]] > float(o[j]))))
            nom.append((j + 1.0) / (k + 1.0))
            real.append(float(np.mean(v)))
        c.plot(nom, real, "o", ms=3.6, color=col, markeredgewidth=0, label=nm)
        print("   fig12(c) %-9s %s" % (nm, " ".join("%.4f" % v for v in real)))
    lim = [4e-3, 0.6]
    c.plot(lim, lim, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlim(*lim)
    c.set_ylim(*lim)
    c.set_xlabel(r"nominal $(j{+}1)/(n{+}1)$")
    c.set_ylabel("held-out risk")
    c.legend(loc="upper left", handlelength=0.8, handletextpad=0.4)

    save(fig, "fig12.pdf")





# ----------------------------------------------------------------- figure 13
def fig13():
    """What a fleet operating a policy can and cannot learn, and what learning costs."""
    import selfsealing as SS
    import perturb as PB

    fig, ax = plt.subplots(1, 3, figsize=(COL, 2.40), layout="constrained")
    a, b, c = ax
    for q in ax:
        clean(q)

    # (a) two laws the record cannot tell apart
    panel(a, "a", "what operating at $e$ hides")
    rng = np.random.default_rng(0)
    E = 0.70
    x = rng.random(200000)
    y = np.where(x > E, x, E * rng.random(200000) ** 3)
    bins = np.linspace(0, 1, 61)
    a.hist(x, bins=bins, density=True, histtype="step", color=BLUE, lw=1.5,
           label="law A", zorder=4)
    a.hist(y, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.5,
           ls=(0, (3, 2)), label="law B", zorder=4)
    a.axvspan(0, E, color=BAND, lw=0, zorder=1)
    a.axvline(E, color=INK, lw=1.0, zorder=5)
    a.set_xlabel(r"unit threshold $\ell_u$")
    a.set_ylabel("density")
    a.set_ylim(0, 3.4)
    anno(a, 0.03, 0.97, "shaded: censored, so\nunidentified")
    a.annotate("$e$", xy=(E, 1.55), xytext=(3, 0), textcoords="offset points",
               color=INK, fontsize=7.4)
    a.legend(loc="upper right", handlelength=1.4, framealpha=0.0)
    print("   fig13(a) mean below e: A %.4f  B %.4f; failure rate A %.4f B %.4f"
          % (x[x <= E].mean(), y[y <= E].mean(), (x > E).mean(), (y > E).mean()))

    # (b) the price of not exploring
    panel(b, "b", "the price of not exploring")
    exs = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    reg = []
    for ex in exs:
        R = [SS.run(explore=ex, seed=100 + k) for k in range(6)]
        reg.append(100.0 * (np.mean([r["total"] for r in R]) / R[0]["oracle"] - 1.0))
    b.plot(np.maximum(exs, 2e-3), reg, color=BLUE, marker="o", zorder=4)
    j = int(np.argmin(reg))
    ring(b, max(exs[j], 2e-3), reg[j], ORANGE, ms=5.0)
    b.set_xscale("log")
    b.set_xlabel("fraction run to failure on purpose")
    b.set_ylabel("regret against the oracle (%)")
    anno(b, 0.30, 0.97, "no exploration: the threshold\nnever moves, in every replicate")
    b.annotate("best %.0f%%" % (100 * exs[j]), xy=(max(exs[j], 2e-3), reg[j]),
               xytext=(0, 11), textcoords="offset points", ha="center",
               color=ORANGE, fontsize=7.0)
    # the leftmost point is exactly zero, which a log axis cannot place
    b.annotate("$0$", xy=(2e-3, reg[0]), xytext=(0, -11),
               textcoords="offset points", ha="center", color=INK2, fontsize=7.0)
    print("   fig13(b) regret %s" % " ".join("%.1f" % v for v in reg))

    # (c) the two ways of exploring
    panel(c, "c", "and a cheaper way to explore")
    pl, ps, rg = PB.make_pool(300000, 0)
    es = np.linspace(0.03, 0.97, 191)
    es_t = np.linspace(0.03, 0.97, 1401)
    chi = 1.0
    cs_t = PB.cost_curve(pl, ps, es_t, chi)
    _, Ls = PB.oracle(pl, ps, chi, es)
    ms = (4000, 16000, 64000, 256000)
    for lbl, col, fn in (("run to failure", ORANGE, "f"), ("perturb", BLUE, "p")):
        v = []
        for m in ms:
            if fn == "p":
                v.append(min(np.mean([PB.perturb_run(m, chi, pl, ps, rg, Ls, 0.97,
                                                     es_t, cs_t, a=aa)[0]
                                      for _ in range(3)])
                             for aa in (0.2, 0.4, 0.8)))
            else:
                v.append(min(np.mean([PB.failure_run(m, chi, pl, ps, rg, Ls, 0.97,
                                                     ev, es, es_t, cs_t)[0]
                                      for _ in range(3)])
                             for ev in (0.02, 0.05, 0.10)))
        sl = float(np.polyfit(np.log(ms), np.log(v), 1)[0])
        c.plot(ms, v, color=col, marker="o", label="%s, $m^{%.2f}$" % (lbl, sl))
        print("   fig13(c) %-15s %s  exponent %+.3f"
              % (lbl, " ".join("%.0f" % t for t in v), sl))
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlabel("units operated $m$")
    c.set_ylabel("cumulative regret")
    c.legend(loc="upper left", handlelength=1.2)

    save(fig, "fig13.pdf")


if __name__ == "__main__":
    fig12()
    fig13()
