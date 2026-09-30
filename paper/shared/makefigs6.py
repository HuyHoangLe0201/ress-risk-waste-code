"""Figures with a mechanism layer, not only a result layer.

Each panel earns a second and third reading:

  Fig 1  the cost curve carries its own +/-1 s.d. ribbon, so the variance regime
         is visible in the cost panel itself and not only asserted underneath;
         a third strip shows the per-unit statistic ell_u whose sample maximum
         IS the zero-failure boundary.  Theorem 2 becomes something you look at.
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
from figstyle import (setup, clean, ring, note, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND, SURFACE, COL1, COL2)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    fig, ax = plt.subplots(3, 1, figsize=(COL1, 3.02), sharex=True,
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
    note(ax[0], 0.014, 5.6, "zero-failure region", color=INK3, fontsize=6.0)
    note(ax[0], -0.405, 1.30, r"band: $\pm2$ bootstrap s.d.", color=INK3,
         fontsize=5.8, va="top")
    ax[1].set_ylim(0, 17.5); ax[1].set_yticks([0, 5, 10, 15])
    ax[1].set_ylabel("s.d. (%)")
    ax[2].set_ylim(-0.36, 1.62); ax[2].set_yticks([])
    ax[2].grid(axis="y", visible=False)
    ax[2].set_ylabel(r"$\ell_u$", labelpad=11, rotation=0, va="center")
    note(ax[2], -0.405, 1.56, r"per-unit $\ell_u$: the sample maximum is the"
         " boundary", color=INK3, fontsize=5.8, va="top")
    ax[2].set_xlabel(r"threshold, offset from the zero-failure boundary"
                     "\n" r"$(\ell-\ell_0)/\bar{T}$", labelpad=1.0)
    fig.savefig("fig1.pdf")
    plt.close(fig)


# ───────────── Fig 2: two forests plus the scaling law that governs both
def fig2():
    fleets = ["FD001", "FD003", "Severson", "FD004", "FD002"]
    n = {"FD001": 100, "FD003": 100, "Severson": 135, "FD004": 249, "FD002": 260}
    cbm = {"FD001": (0.792, 0.533, 1.331), "FD003": (0.686, 0.457, 1.261),
           "Severson": (0.599, 0.385, 0.951), "FD004": (0.608, 0.511, 0.852),
           "FD002": (0.701, 0.601, 0.960)}
    ks = {"Severson": (1.147, 0.85, 1.56), "FD004": (1.013, 0.83, 1.15),
          "FD002": (1.038, 0.91, 1.17)}
    cv = np.array([0.241, 0.288, 0.340, 0.474, 0.483])
    kr = np.array([1.031, 1.099, 1.080, 1.073, 1.056])

    fig = plt.figure(figsize=(COL1, 2.00))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.44, 1.0], width_ratios=[1.30, 1],
                          hspace=0.80, wspace=0.16)
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
    note(a, 0.35, len(fleets) - 0.38, "fleet   $n$", color=INK3, fontsize=5.8)
    note(a, 1.03, len(fleets) - 0.38, "worse", color=INK3, fontsize=5.8)
    note(b, 1.03, len(fleets) - 0.38, "longer", color=INK3, fontsize=5.8)
    for i, f in enumerate(fleets):
        if f not in ks:
            note(b, 1.0, i, "not reported", color=INK3, ha="center", fontsize=5.6)

    # (c) the scaling law: half-width against fleet size, fitted and extrapolated
    clean(c)
    nn = np.array([n[f] for f in fleets], float)
    hw = np.array([(cbm[f][2] - cbm[f][1]) / 2 for f in fleets])
    p = np.polyfit(np.log(nn), np.log(hw), 1)
    xs = np.linspace(np.log(85), np.log(760), 60)
    c.plot(np.exp(xs), np.exp(np.polyval(p, xs)), "-", lw=0.9, color=INK3, zorder=2)
    for f in fleets:
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
    note(c, 96, 0.115, f"slope ${p[0]:+.2f}$", color=INK3, fontsize=5.8,
         va="bottom")
    note(c, 660, 0.52, r"the five fleets of (a);" "\n"
         r"$\pm0.10$ needs $n\!\approx\!%d$" % int(round(need / 10) * 10),
         color=INK2, fontsize=5.8, ha="right", va="top")

    for ax_, tag in ((a, "(a)"), (b, "(b)"), (c, "(c)")):
        ax_.text(-0.03, 1.19, tag, transform=ax_.transAxes, color=INK,
                 fontsize=6.6, ha="right", va="top")
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
    """1 + chi*, with chi* in closed form (Theorem 3): the kink is the global
    minimiser exactly when chi >= max over the failure branch of the secant
    (omega_0 - omega)/(P_f (1 - omega_0))."""
    b = np.arange(z0)
    om0 = om[z0]
    return 1.0 + float(((om0 - om[b]) / (pf[b] * (1 - om0))).max())


def fig3():
    om2, pf2 = frontier(FD002)
    om4, pf4 = frontier(FD004)
    z2 = int(np.nonzero(pf2 == 0)[0][0])
    z4 = int(np.nonzero(pf4 == 0)[0][0])
    r2, r4 = critical_ratio(om2, pf2, z2), critical_ratio(om4, pf4, z4)

    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.70),
                           gridspec_kw={"wspace": 0.44, "width_ratios": [1.1, 1]})
    a, b = ax
    clean(a); clean(b)

    # (a) the kink, and the cone of iso-cost slopes it supports
    xc, YT = om2[z2], 0.0092
    X0, X1 = xc - 0.0125, xc + 0.048
    smax = 1 / (1 - xc) / (r2 - 1)                # steepest supported slope
    xw = np.linspace(X0, xc, 40)
    a.fill_between(xw, 0, smax * (xc - xw), color=BAND, lw=0, zorder=0)
    keep = pf2 < YT * 1.4
    a.plot(om2[keep], pf2[keep], "-", color=BLUE, zorder=4, label="frontier")
    xs = np.array([X0, X1])
    for ratio, col, sty in ((2, INK3, ":"), (10, ORANGE, "-"), (30, VIOLET, "-")):
        j, L = opt_omega(om2, pf2, ratio)
        chi = ratio - 1.0
        a.plot(xs, (L * (1 - xs) - 1) / chi, sty, lw=0.9, color=col, zorder=2)
        if pf2[j] <= YT and X0 <= om2[j] <= X1:   # never mark outside the frame
            ring(a, om2[j], pf2[j], col, ms=4.0)
        print("fig3", f"cf/cp={ratio}: omega*={om2[j]:.4f} L={L:.4f} "
              f"{'kink' if j == z2 else 'off-kink'}", flush=True)
    a.set_xlim(X0, X1); a.set_ylim(-0.0006, YT)
    a.set_xticks([0.05, 0.07, 0.09])
    a.set_yticks([0.000, 0.004, 0.008])
    a.set_xlabel(r"wasted life  $\omega$", labelpad=1.0)
    a.set_ylabel(r"failure probability  $P_f$")
    a.plot([xc - .0075, xc + .0125], [0.0028, 0.0043], "-", lw=0.6, color=INK3,
           zorder=2)
    note(a, xc + .014, 0.0043, "cone of slopes\nthe kink supports",
         color=INK3, va="center", fontsize=5.6)
    note(a, xc + .0035, 0.0004, r"kink at $\omega_0$", color=INK2, fontsize=5.8,
         va="bottom")
    h = [Line2D([], [], color=BLUE, lw=1.4),
         Line2D([], [], color=ORANGE, lw=0.9),
         Line2D([], [], color=VIOLET, lw=0.9),
         Line2D([], [], color=INK3, lw=0.9, ls=":")]
    a.legend(h, ["frontier", "iso-cost, $10$", "iso-cost, $30$", "iso-cost, $2$"],
             loc="upper right", borderpad=0.1, handlelength=1.2,
             labelspacing=0.12, handletextpad=0.35, fontsize=5.6)

    # (b) the consequence: the optimum stops moving with the cost ratio
    ratios = np.exp(np.linspace(np.log(1.5), np.log(50), 90))
    for nm, om_, pf_, z_, rc, col in (("FD002", om2, pf2, z2, r2, BLUE),
                                      ("FD004", om4, pf4, z4, r4, ORANGE)):
        w = np.array([om_[opt_omega(om_, pf_, r)[0]] for r in ratios]) / om_[z_]
        b.plot(ratios, w, "-", color=col, zorder=3, label=nm)
        b.plot([rc], [1.0], "o", ms=3.4, color=col, zorder=4, markeredgewidth=0)
        print("fig3", nm, f"critical cost ratio {rc:.2f}", flush=True)
    b.axhline(1.0, color=INK2, lw=0.8, zorder=1)
    b.axvspan(max(r2, r4), 55, color=BAND, lw=0, zorder=0)
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator([2, 5, 10, 30]))
    b.xaxis.set_major_formatter(FixedFormatter(["2", "5", "10", "30"]))
    b.xaxis.set_minor_locator(FixedLocator([]))
    b.set_xlim(1.5, 50); b.set_ylim(0.55, 1.08)
    b.set_yticks([0.6, 0.8, 1.0])
    b.set_xlabel(r"cost ratio  $c_f/c_p$", labelpad=1.0)
    b.set_ylabel(r"$\omega^{*}/\omega_0$", labelpad=1.0)
    b.legend(loc="lower right", borderpad=0.1, handlelength=1.2,
             labelspacing=0.12, handletextpad=0.35, fontsize=5.8)
    note(b, max(r2, r4) * 1.12, 0.62, "optimum pinned\nat the kink",
         color=INK3, va="bottom", fontsize=5.6)
    for ax_, tag in ((a, "(a)"), (b, "(b)")):
        ax_.text(-0.03, 1.14, tag, transform=ax_.transAxes, color=INK,
                 fontsize=6.6, ha="right", va="top")
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
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 1.88), sharex=True,
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
    note(ax[0], 0.125, 1.52, "worse than the fleet schedule", color=INK3,
         va="bottom", fontsize=5.8)
    note(ax[0], 0.093, 0.665, r"$\blacktriangle$  fleet-optimal age $t_R$",
         color=INK3, va="center", fontsize=5.8)
    ax[1].set_ylim(0.60, 1.05); ax[1].set_yticks([0.6, 0.8, 1.0])
    ax[1].set_ylabel("fraction\nsurviving $k$")
    ax[1].set_xlabel(r"observation window  $k/\bar{T}$", labelpad=1.0)
    ax[1].set_xlim(0.07, 0.80)
    fig.savefig("fig4.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig2(); fig3(); fig1(); fig4()
