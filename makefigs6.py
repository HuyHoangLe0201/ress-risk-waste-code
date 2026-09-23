"""Figures with a mechanism layer, not only a result layer.

Each panel earns a second and third reading:

  Fig 1  the cost curve carries its own +/-1 s.d. ribbon, so the variance regime
         is visible in the cost panel itself and not only asserted underneath;
         a third strip shows the per-unit statistic ell_u whose sample maximum
         IS the zero-failure boundary.  Theorem 4 becomes something you look at.
  Fig 2  the two forests are joined by a scaling panel: interval half-width
         against fleet size, with the n^{-1/2} law fitted and extrapolated, so
         the reader can read off the fleet size that would resolve the claim
         instead of being told one.
  Fig 3  a tangency diagram in the proper sense: the attainable frontier, the
         unattainable region under it, and a family of iso-cost lines
         P_f = (L(1-omega)-1)/chi.  The optimum is where the cheapest line still
         touches the frontier, so the tangency is seen rather than claimed.
  Fig 4  the window curves are filled against the schedule they are measured
         against, and each fleet's t_R is ticked on the axis, which puts the
         k* = t_R claim of Fig 2b onto the curve that produced it.
"""
import numpy as np
from scipy.stats import lognorm, gaussian_kde
from figstyle import (setup, clean, ring, note, COL, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND, SURFACE, COL1, COL2)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Wedge

setup()
CM = r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\CMAPSSData"
TSP = (r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\IEEE_Trans.representation"
       r"\TSP_segmented_projections\data")
CF = 10.0


def load_cmapss(fn):
    d = np.loadtxt(CM + "\\" + fn)
    u, c, X = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2:]
    un = np.unique(u)
    life = np.array([float(c[u == k].max()) for k in un])
    X = X[:, X.std(0) > 1e-8]; X = (X - X.mean(0)) / X.std(0)
    return [X[u == k] for k in un], life


def load_severson():
    z = np.load(f"{TSP}\\severson_cells.npz", allow_pickle=True)
    cs = [np.asarray(a, float) for a in z['data']]
    cs = [a for a in cs if len(a) > 150 and np.isfinite(a).all()]
    S = np.vstack(cs); ok = S.std(0) > 1e-12
    mu, sd = S[:, ok].mean(0), S[:, ok].std(0)
    return [(a[:, ok] - mu) / sd for a in cs], np.array([float(len(a)) for a in cs])


FD002 = lambda: load_cmapss("train_FD002.txt")
FD004 = lambda: load_cmapss("train_FD004.txt")


def design(seq):
    out = []
    for s in seq:
        f = [s]
        for w in (5, 20):
            c = np.cumsum(np.vstack([np.zeros((1, s.shape[1])), s]), 0)
            k = np.minimum(np.arange(1, len(s) + 1), w)
            lo = np.maximum(np.arange(1, len(s) + 1) - w, 0)
            f.append((c[np.arange(1, len(s) + 1)] - c[lo]) / k[:, None])
        out.append(np.hstack(f + [np.ones((len(s), 1))]))
    return out


def fit(A, y, lam=1e-1):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def agg(s, k):
    A = s[:k]; h = max(len(A) // 2, 1)
    return np.concatenate([A.mean(0), A[h:].mean(0) - A[:h].mean(0), [1.0]])


def prep(seq, life, seed=0):
    X = design(seq); R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
    n = len(seq); rg = np.random.default_rng(seed)
    p = rg.permutation(n); cal, val = p[:n // 2], p[n // 2:]
    w = fit(np.vstack([X[i] for i in cal]), np.concatenate([R[i] for i in cal]))
    pa = {i: X[i] @ w for i in range(n)}
    rm = {i: np.minimum.accumulate(pa[i]) for i in pa}
    return R, rm, pa, cal, val, life.mean()


def unit_cost(i, e, R, rm, life, cf=CF):
    h = np.nonzero(rm[i] <= e)[0]
    if len(h) == 0 or R[i][h[0]] <= 0:
        return cf, life[i], 1
    return 1.0, life[i] - R[i][h[0]], 0


def sweep(ld, nboot=200, ngrid=54):
    """Cost, failure rate, bootstrap s.d., and the per-unit statistic ell_u."""
    seq, life = ld()
    R, rm, pa, cal, val, Tb = prep(seq, life)
    lu_cal = np.array([pa[i][R[i] > 0].min() for i in cal])
    lu_val = np.array([pa[i][R[i] > 0].min() for i in val])
    g = np.linspace(np.quantile(lu_cal, .05),
                    np.quantile(lu_cal, .999) + .28 * Tb, ngrid)
    rg = np.random.default_rng(1)
    C = np.array([[unit_cost(i, e, R, rm, life) for i in val] for e in g])
    cost = C[:, :, 0].mean(1) / C[:, :, 1].mean(1) * Tb
    pf = C[:, :, 2].mean(1)
    B = rg.integers(0, len(val), size=(nboot, len(val)))
    bc = C[:, :, 0][:, B].mean(2) / C[:, :, 1][:, B].mean(2) * Tb
    sd = 100 * bc.std(1) / bc.mean(1)
    return g / Tb, cost, pf, sd, lu_val / Tb


# ───────────── Fig 1: cost with its own uncertainty, and the statistic behind it
def fig1():
    fig, ax = plt.subplots(3, 1, figsize=(COL, 3.64), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1.62, 0.86, 0.62],
                                        "hspace": 0.14})
    for nm, ld, col in (("FD002", FD002, BLUE), ("FD004", FD004, ORANGE)):
        x, cost, pf, sd, lu = sweep(ld)
        zero = x[pf == 0]
        x0 = zero.min() if len(zero) else x[-1]
        d, du = x - x0, lu - x0
        j = int(np.argmin(cost))
        lo, hi = cost * (1 - 2 * sd / 100), cost * (1 + 2 * sd / 100)
        ax[0].fill_between(d, lo, hi, color=col, alpha=0.22, lw=0, zorder=2)
        ax[0].plot(d, cost, "-", color=col, zorder=3, label=nm)
        ring(ax[0], d[j], cost[j], col)
        ax[1].plot(d, sd, "-", color=col, zorder=3)
        ring(ax[1], d[j], sd[j], col)
        k = gaussian_kde(du, bw_method=0.28)
        t = np.linspace(-0.45, du.max(), 300)     # never past the sample maximum
        y = k(t) / k(t).max()
        ax[2].fill_between(t, 0, y, color=col, alpha=0.22, lw=0, zorder=2)
        ax[2].plot(t, y, "-", lw=0.9, color=col, zorder=3)
        ax[2].plot(du, np.full_like(du, -0.20), "|", ms=2.6, mew=0.5,
                   color=col, zorder=3)
        print("fig1", nm, f"opt {d[j]:+.3f}  max ell_u {du.max():+.4f}  "
              f"sd {sd[:j].max():.1f}->{sd[j]:.1f}", flush=True)
    for a in ax:
        clean(a)
        a.axvspan(0, 0.42, color=BAND, lw=0, zorder=0)
        a.set_xlim(-0.42, 0.34)
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10]))
    ax[0].yaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "5", "10"]))
    ax[0].yaxis.set_minor_locator(FixedLocator([]))
    ax[0].set_ylim(0.93, 19)
    ax[0].set_ylabel("cost / oracle")
    ax[0].legend(loc="upper right", ncol=2, borderpad=0.15)
    note(ax[0], 0.014, 5.6, "zero-failure region", color=INK2, fontsize=8.8)
    note(ax[0], -0.405, 1.30, r"band: $\pm2$ bootstrap s.d.", color=INK2,
         fontsize=8.8, va="top")
    ax[1].set_ylim(0, 17.5); ax[1].set_yticks([0, 5, 10, 15])
    ax[1].set_ylabel("s.d. (%)")
    ax[2].set_ylim(-0.36, 1.62); ax[2].set_yticks([])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylabel(r"$\ell_u$", labelpad=11, rotation=0, va="center")
    note(ax[2], -0.405, 1.56, r"per-unit $\ell_u$: the sample maximum is the"
         " boundary", color=INK2, fontsize=8.8, va="top")
    ax[2].set_xlabel(r"threshold, offset from the zero-failure boundary"
                     "\n" r"$(\ell-\ell_0)/\bar{T}$", labelpad=1.0)
    # NOT the paper's Figure 4: makefigs7.fig1() draws that, and used to
    # write the same filename, so this one silently replaced it.
    fig.savefig("fig1_superseded.pdf")
    plt.close(fig)


# ───────────── Fig 2: two forests plus the scaling law that governs both
def fig2():
    fleets = ["FD001", "FD003", "Severson", "FD004", "FD002"]
    n = {"FD001": 100, "FD003": 100, "Severson": 135, "FD004": 249, "FD002": 260}
    cbm = {"FD001": (0.792, 0.533, 1.331), "FD003": (0.686, 0.457, 1.261),
           "FD004": (0.608, 0.511, 0.852), "FD002": (0.701, 0.601, 0.960)}
    # Point estimate only: batch clustering leaves 135 cells with an effective size near
    # 8, so this interval is withdrawn in the text and must not be drawn as evidence here
    # or fitted in panel (c).
    withdrawn = {"Severson": 0.599}
    ks = {"Severson": (1.147, 0.85, 1.56), "FD004": (1.013, 0.83, 1.15),
          "FD002": (1.038, 0.91, 1.17)}
    cv = np.array([0.241, 0.288, 0.340, 0.474, 0.483])
    kr = np.array([1.031, 1.099, 1.080, 1.073, 1.056])

    fig = plt.figure(figsize=(COL, 1.94), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.44, 1.0], width_ratios=[1.30, 1],
                          hspace=0.16, wspace=0.16)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, :])

    for ax_, dat, hi_x, rule in (
            (a, cbm, 1.44, lambda lo, hi: hi < 1.0),
            (b, ks, 1.72, lambda lo, hi: lo <= 1.0 <= hi)):
        clean(ax_, spines=("bottom",))
        ax_.grid(axis="y", visible=False)
        ax_.axvspan(1.0, hi_x, color=BAND, lw=0, zorder=0)
        ax_.axvline(1.0, color=INK2, lw=0.8, zorder=1)
        for i, f in enumerate(fleets):
            if f not in dat:
                continue
            m, lo, hi = dat[f]
            col = BLUE if rule(lo, hi) else INK3
            ax_.plot([lo, hi], [i, i], "-", lw=1.5, color=col, zorder=3,
                     solid_capstyle="butt")
            for q in (lo, hi):
                ax_.plot([q, q], [i - .16, i + .16], "-", lw=1.0, color=col, zorder=3)
            ring(ax_, m, i, col, ms=3.8)
        ax_.set_ylim(-0.7, len(fleets) - 0.15)
        ax_.set_yticks(range(len(fleets)))
    a.set_xlim(0.33, 1.44); a.set_xticks([0.5, 1.0])
    b.set_xlim(0.68, 1.72); b.set_xticks([1.0, 1.5])
    a.set_yticklabels([f"{f}  {n[f]}" for f in fleets], color=INK)
    b.set_yticklabels([]); b.tick_params(left=False)
    a.set_xlabel(r"$g_{\mathrm{CBM}}/g_{\mathrm{age}}$", labelpad=1.0)
    b.set_xlabel(r"$k^{*}/t_R$", labelpad=1.0)
    note(a, 0.35, len(fleets) - 0.38, "fleet   $n$", color=INK2, fontsize=8.8)
    note(a, 1.03, len(fleets) - 0.38, "worse", color=INK2, fontsize=8.8)
    note(b, 1.03, len(fleets) - 0.38, "longer", color=INK2, fontsize=8.8)
    for i, f in enumerate(fleets):
        if f not in ks:
            note(b, 1.0, i, "not reported", color=INK2, ha="center", fontsize=8.8)
        if f in withdrawn:
            a.plot([withdrawn[f]], [i], "o", ms=3.4, color=INK3, zorder=4,
                   markeredgewidth=0)
            note(a, withdrawn[f] + 0.05, i, "interval withdrawn", color=INK3,
                 fontsize=8.0, va="center")

    # (c) the scaling law: half-width against fleet size, fitted and extrapolated
    clean(c)
    shown = [f for f in fleets if f in cbm]
    nn = np.array([n[f] for f in shown], float)
    hw = np.array([(cbm[f][2] - cbm[f][1]) / 2 for f in shown])
    p = np.polyfit(np.log(nn), np.log(hw), 1)
    xs = np.linspace(np.log(85), np.log(760), 60)
    c.plot(np.exp(xs), np.exp(np.polyval(p, xs)), "-", lw=0.9, color=INK3, zorder=2)
    for f in shown:
        c.plot(n[f], (cbm[f][2] - cbm[f][1]) / 2, "o", ms=3.4, color=BLUE,
               zorder=3, markeredgewidth=0)
    need = float(np.exp((np.log(0.10) - p[1]) / p[0]))
    c.axhline(0.10, color=INK2, lw=0.7, zorder=1)
    ring(c, need, 0.10, ORANGE, ms=4.0, marker="D")
    c.set_xscale("log"); c.set_yscale("log")
    c.xaxis.set_major_locator(FixedLocator([100, 200, 400]))
    c.xaxis.set_major_formatter(FixedFormatter(["100", "200", "400"]))
    c.xaxis.set_minor_locator(FixedLocator([]))
    c.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.4]))
    c.yaxis.set_major_formatter(FixedFormatter(["0.1", "0.2", "0.4"]))
    c.yaxis.set_minor_locator(FixedLocator([]))
    c.set_xlim(85, 700); c.set_ylim(0.072, 0.60)
    c.set_xlabel("fleet size $n$", labelpad=1.0)
    c.set_ylabel("interval\nhalf-width", labelpad=1.0)
    note(c, 96, 0.115, f"slope ${p[0]:+.2f}$", color=INK2, fontsize=8.8,
         va="bottom")
    note(c, 660, 0.52, r"the four intervals of (a);" "\n"
         r"$\pm0.10$ needs $n\!\approx\!%d$" % int(round(need / 10) * 10),
         color=INK2, fontsize=8.8, ha="right", va="top")

    # A title reserves its own space; the tag placed by hand at (0.03, 1.05) hung back
    # into the axes and the spine of the panel above ran through "(a)" and "(c)".
    for ax_, tag in ((a, "(a)  against optimal age replacement"),
                     (b, "(b)  against the fleet-optimal age"),
                     (c, "(c)  half-width against fleet size")):
        ax_.set_title(tag, loc="left", color=INK, fontsize=8.8, pad=3.0)
    fig.savefig("fig2.pdf")
    plt.close(fig)
    print("fig2", f"halfwidth slope {p[0]:+.3f}, n for +/-0.10 = {need:.0f}",
          flush=True)


# ───────────── Fig 3: a tangency diagram, with iso-cost lines and the
#               unattainable region drawn
def frontier(ld, ngrid=170):
    seq, life = ld()
    R, rm, pa, cal, val, Tb = prep(seq, life)
    lu = np.array([pa[i][R[i] > 0].min() for i in cal])
    g = np.linspace(np.quantile(lu, .02), np.quantile(lu, .999) + .3 * Tb, ngrid)
    om, pf = [], []
    for e in g:
        per = np.array([unit_cost(i, e, R, rm, life) for i in cal])
        om.append(1 - per[:, 1].mean() / Tb); pf.append(per[:, 2].mean())
    return np.array(om), np.array(pf)


def opt_omega(om, pf, ratio):
    """Cost-rate minimiser on the frontier, computed on the frontier itself."""
    chi = ratio - 1.0
    L = (1 + chi * pf) / (1 - om)
    j = int(np.argmin(L))
    return j, float(L[j])


def critical_ratio(om, pf, z0):
    """1 + chi*, with chi* in closed form (Theorem 5): the kink is the global
    minimiser exactly when chi >= max over the failure branch of the secant
    (omega_0 - omega)/(P_f (1 - omega_0))."""
    b = np.arange(z0)
    om0 = om[z0]
    return 1.0 + float(((om0 - om[b]) / (pf[b] * (1 - om0))).max())


def fig3():
    """Theorem 5 as the near-optimal set, not as a thicket of iso-cost rays.

    Two earlier versions of this panel failed for opposite reasons. A tangency diagram made
    the reader compare the angles of five lines, on axes that squashed slopes by six. A
    filled cost field fixed the geometry but spent most of its ink on a large region meaning
    "expensive, do not go here".

    What a reader needs is the set of thresholds that are as good as the best one. Drawing
    that set as a band leaves one band, one trace and one reference line. The band narrows
    as failures get dearer and then locks onto omega_0 rather than approaching it, which is
    the discontinuity in Theorem 5. It also shows something the algebra states but no
    earlier version made visible: past chi* the optimum sits on the EDGE of its own
    tolerance band, not inside it, which is why the estimate at that point is unstable.

    Type is set for Elsevier. Mathtext subscripts render at about 0.7 of the base and a
    full-width figure is typeset at 190 mm, a scale of 1.154 from the authored width, so the
    8.0 pt base here reaches print at 5.6 pt authored and 6.5 pt final.
    """
    om2, pf2 = frontier(FD002)
    om4, pf4 = frontier(FD004)
    z2 = int(np.nonzero(pf2 == 0)[0][0])
    z4 = int(np.nonzero(pf4 == 0)[0][0])
    r2, r4 = critical_ratio(om2, pf2, z2), critical_ratio(om4, pf4, z4)
    RAT = np.exp(np.linspace(np.log(1.5), np.log(50), 260))
    FS = 8.0

    def surface(om, pf):
        chi = RAT - 1.0
        L = (1.0 + chi[None, :] * pf[:, None]) / (1.0 - om[:, None])
        return L, L / L.min(0)[None, :] - 1.0

    L2, E2 = surface(om2, pf2)
    near = E2 <= 0.01                              # "as good as the best, to one per cent"
    lo = np.array([om2[np.nonzero(c)[0][0]] for c in near.T])
    hi = np.array([om2[np.nonzero(c)[0][-1]] for c in near.T])

    fig = plt.figure(figsize=(COL, 2.06))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.28,
                          left=0.088, right=0.995, top=0.925, bottom=0.185)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    clean(a); clean(b)

    a.fill_between(RAT, lo, hi, color=BLUE, alpha=0.16, lw=0, zorder=1)
    a.plot(RAT, om2[np.argmin(L2, axis=0)], "-", color=ORANGE, lw=1.8, zorder=4)
    a.axhline(om2[z2], color=INK2, lw=0.8, ls="--", zorder=3)
    ring(a, r2, om2[z2], ORANGE, ms=4.2)
    note(a, 1.57, om2[z2] + 0.0035, r"$\omega_0$", color=INK2, fontsize=FS,
         va="bottom", ha="left")
    note(a, 3.4, 0.0755, "thresholds within\n1% of the best", color=BLUE,
         fontsize=FS, va="center", ha="left")
    note(a, 2.45, 0.0385, "cheapest\nthreshold", color=ORANGE, fontsize=FS,
         va="center", ha="left")
    a.set_xscale("log")
    a.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 30]))
    a.xaxis.set_major_formatter(FixedFormatter(["2", "3", "5", "10", "30"]))
    a.xaxis.set_minor_locator(FixedLocator([]))
    a.set_xlim(1.5, 50); a.set_ylim(0.030, 0.092)
    a.set_yticks([0.04, 0.06, 0.08])
    a.set_xlabel(r"cost ratio  $c_f/c_p$", labelpad=1.5)
    a.set_ylabel(r"wasted life  $\omega$", labelpad=2.5)

    for nm, om_, pf_, z_, rc, col, xa, ya in (
            # x=1.56 put each label on the steepest part of its own curve, and the
            # curve ran through the word; these sit right of where each reaches zero.
            ("FD002", om2, pf2, z2, r2, BLUE, 2.38, 0.46),
            ("FD004", om4, pf4, z4, r4, ORANGE, 3.40, 1.32)):
        L_, _ = surface(om_, pf_)
        sc = 100.0 * (L_[z_] / L_.min(0) - 1.0)
        b.fill_between(RAT, 0, sc, color=col, alpha=0.16, lw=0, zorder=2)
        b.plot(RAT, sc, "-", color=col, lw=1.8, zorder=3)
        ring(b, rc, 0.0, col, ms=4.0)
        note(b, xa, ya, nm, color=col, fontsize=FS, va="center", ha="left")
        print("fig3", nm, f"chi*+1 = {rc:.2f}; taking ell_0 anyway costs at most "
              f"{sc.max():.2f}% below it, {sc[RAT >= rc].max():.0e}% above", flush=True)
    b.axhline(0.0, color=INK2, lw=0.8, zorder=1)
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 30]))
    b.xaxis.set_major_formatter(FixedFormatter(["2", "3", "5", "10", "30"]))
    b.xaxis.set_minor_locator(FixedLocator([]))
    b.set_xlim(1.5, 50); b.set_ylim(-0.3, 3.3)
    b.set_yticks([0, 1, 2, 3])
    b.set_xlabel(r"cost ratio  $c_f/c_p$", labelpad=1.5)
    b.set_ylabel("cost of taking $\\ell_0$ anyway (%)", labelpad=2.5)
    note(b, 4.6, 0.42, "exactly zero\nabove $\\chi^{*}$", color=INK2, fontsize=FS,
         va="bottom", ha="left")

    for ax_, tag in ((a, "(a)  thresholds within 1% of the best"),
                     (b, "(b)  the excess from running the boundary anyway")):
        ax_.text(0.0, 1.005, tag, transform=ax_.transAxes, color=INK,
                 fontsize=FS + 0.6, ha="left", va="bottom")
    fig.savefig("fig3.pdf")
    plt.close(fig)


# ───────────── Fig 4: window curves filled against the schedule, with t_R ticked
def window_curve(seq, life, nsplit=8):
    X = design(seq); R = [np.arange(len(s) - 1, -1, -1.0) for s in seq]
    n = len(seq); Tb = life.mean()
    ks = np.unique(np.round(np.arange(0.10, 0.801, 0.06) * Tb).astype(int))
    XA = {k: np.array([agg(s, k) for s in seq]) for k in ks}
    acc = {k: [] for k in ks}; sv = {k: [] for k in ks}; fl = []; cb = []; tr = []
    for seed in range(nsplit):
        rg = np.random.default_rng(seed); p = rg.permutation(n)
        cal, val = p[:n // 2], p[n // 2:]
        Lc, Lv = life[cal], life[val]
        s_, _, sc = lognorm.fit(Lc, floc=0)
        t = np.linspace(1., 4 * Lc.max(), 4000); S = lognorm.sf(t, s_, scale=sc)
        cum = np.concatenate([[0.], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
        tR = t[10 + int(np.argmin(((S + CF * (1 - S)) / np.maximum(cum, 1e-12))[10:]))]
        tr.append(tR / Tb)
        fl.append(np.where(Lv <= tR, CF, 1.).mean() / np.minimum(Lv, tR).mean() * Tb)
        w = fit(np.vstack([X[i] for i in cal]), np.concatenate([R[i] for i in cal]))
        pa = {i: X[i] @ w for i in np.concatenate([cal, val])}
        rmn = {i: np.minimum.accumulate(pa[i]) for i in pa}
        gl = np.linspace(0, .4 * Tb, 25)

        def cc(idx, e):
            per = np.array([unit_cost(i, e, R, rmn, life) for i in idx])
            return per[:, 0].mean() / per[:, 1].mean()
        cb.append(cc(val, gl[int(np.argmin([cc(cal, e) for e in gl]))]) * Tb)
        nfix = int((Lc > ks.max()).sum())
        if nfix < 8:
            continue
        for k in ks:
            pool = cal[Lc > k]
            if len(pool) < 8:
                acc[k].append(np.nan); continue
            idx = rg.choice(pool, size=min(nfix, len(pool)), replace=False)
            Xc, yc = XA[k][idx], life[idx]
            wv = fit(Xc, yc); Th = np.zeros(len(idx))
            for f in np.array_split(np.arange(len(idx)), 5):
                trn = np.setdiff1d(np.arange(len(idx)), f)
                Th[f] = Xc[f] @ fit(Xc[trn], yc[trn])

            def cst(That, L, mm):
                d = np.maximum(That - mm, k); fa = (L <= k) | (L <= d)
                return np.where(fa, CF, 1.).mean() / np.where(fa, L, d).mean()
            mg = np.linspace(0, .8 * Tb, 30)
            mo = mg[int(np.argmin([cst(Th, yc, q) for q in mg]))]
            acc[k].append(cst(XA[k][val] @ wv, Lv, mo) * Tb)
            sv[k].append(np.mean(Lv > k))
    return (ks / Tb, np.array([np.nanmean(acc[k]) for k in ks]),
            np.array([np.mean(sv[k]) if sv[k] else np.nan for k in ks]),
            np.mean(fl), np.mean(cb), float(np.mean(tr)))


def fig4():
    fig, ax = plt.subplots(2, 1, figsize=(COL, 2.44), sharex=True, layout="constrained",
                           gridspec_kw={"height_ratios": [1.85, 1], "hspace": 0.13})
    clean(ax[0]); clean(ax[1])
    ax[0].axhspan(1.0, 2.20, color=BAND, lw=0, zorder=0)
    for nm, ld, col in (("FD002", FD002, BLUE), ("FD004", FD004, ORANGE),
                        ("Severson", load_severson, VIOLET)):
        seq, life = ld()
        x, m, s, F, C, tR = window_curve(seq, life)
        r = m / F
        j = int(np.nanargmin(r))
        ax[0].fill_between(x, 1.0, r, color=col, alpha=0.16, lw=0, zorder=2)
        ax[0].plot(x, r, "-", color=col, zorder=3, label=nm)
        ring(ax[0], x[j], r[j], col, ms=4.0)
        ax[0].plot([tR], [0.665], "^", ms=3.4, color=col, zorder=4,
                   markeredgewidth=0)
        ax[1].plot(x, s, "-", color=col, zorder=3)
        print("fig4", nm, f"min {r[j]:.2f} at k={x[j]:.2f}, tR={tR:.2f}", flush=True)
    ax[0].axhline(1.0, color=INK2, lw=0.8, zorder=3)
    ax[0].set_ylim(0.60, 2.10); ax[0].set_yticks([0.8, 1.2, 1.6, 2.0])
    ax[0].set_ylabel("cost / fleet\nschedule")
    ax[0].legend(loc="upper left", ncol=3, borderpad=0.1, columnspacing=0.8,
                 handletextpad=0.4)
    note(ax[0], 0.125, 1.52, "worse than the fleet schedule", color=INK2,
         va="bottom", fontsize=8.8)
    note(ax[0], 0.093, 0.665, r"$\blacktriangle$  fleet-optimal age $t_R$",
         color=INK2, va="center", fontsize=8.8)
    ax[1].set_ylim(0.60, 1.05); ax[1].set_yticks([0.6, 0.8, 1.0])
    ax[1].set_ylabel("fraction\nsurviving $k$")
    ax[1].set_xlabel(r"observation window  $k/\bar{T}$", labelpad=1.0)
    ax[1].set_xlim(0.07, 0.80)
    fig.savefig("fig4.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # fig1() is superseded by makefigs7.py and is not called here; see its
    # savefig comment.
    fig2(); fig3(); fig4()
