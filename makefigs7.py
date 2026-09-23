"""Figure 1 rebuilt as the anatomy of the estimand, read left to right.

The previous sets plotted results only; nothing showed the object being
estimated.  This figure runs (a) the raw predicted-RUL paths and one alarm, to
(b) the per-unit statistic those paths reduce to and the failure probability it
generates, to (c) the cost that statistic drives and its uncertainty, to (d) the
uncertainty as a number.  Panels (a) and (b) are the mechanism; (c) and (d) are
the result, and the reader can now trace one to the other.
"""
import numpy as np
import makefigs6 as M
from figstyle import (setup, clean, ring, note, COL, BLUE, ORANGE, VIOLET,
                      INK, INK2, INK3, BAND, SURFACE, COL1, COL2)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

setup()
RAMP = LinearSegmentedColormap.from_list("blues", ["#c3d9f2", "#123a63"])


def fig1():
    # two rows: (a)-(c) the FD002 mechanism, (d)-(e) the two-fleet consequence
    fig = plt.figure(figsize=(COL, 3.06))
    gs = fig.add_gridspec(2, 6, wspace=1.15, hspace=0.52)
    a = fig.add_subplot(gs[0, 0:2]); b = fig.add_subplot(gs[0, 2:4])
    w = fig.add_subplot(gs[0, 4:6])
    c = fig.add_subplot(gs[1, 0:3]); d = fig.add_subplot(gs[1, 3:6])
    ax = [a, b, w, c, d]
    for q in ax:
        clean(q)

    # ── the fleet the mechanism panels are drawn from
    seq, life = M.FD002()
    R, rm, pa, cal, val, Tb = M.prep(seq, life)
    lu = np.array([pa[i][R[i] > 0].min() for i in val]) / Tb
    l0 = lu.max()                                    # zero-failure boundary
    crit = val[int(np.argmax(lu))]                   # the unit that sets it
    # the second reference line is not an arbitrary offset: it is the threshold
    # at which the bootstrap s.d. of the cost peaks, i.e. the worst point of the
    # failure branch, so panels (b), (c) and (e) all refer to the same place.
    SW = {nm: M.sweep(ld) for nm, ld in (("FD002", M.FD002), ("FD004", M.FD004))}
    xs2, cs2, pfs2, sds2, _ = SW["FD002"]
    k02 = int(np.nonzero(pfs2 == 0)[0].min())
    ipk = int(np.argmax(sds2[:k02]))
    llow = float(xs2[ipk])
    pk_pf, pk_sd = float(pfs2[ipk]), float(sds2[ipk])
    print("fig1", f"variance peak at l/Tbar={llow:+.4f}  P_f={pk_pf:.4f}  "
          f"s.d.={pk_sd:.2f}%  vs optimum s.d.={sds2[k02]:.2f}%", flush=True)

    # (a) each path run against the true remaining life it is predicting, so it
    #     terminates on the x=0 axis at exactly its own ell_u
    XL, YL = 0.92, (-0.90, 0.97)   # headroom below the bundle for the legend
    order = np.argsort(life[val])
    for r, i in enumerate(val[order]):
        live = R[i] > 0
        a.plot(R[i][live] / Tb, np.minimum.accumulate(pa[i])[live] / Tb, "-",
               lw=0.55, color=RAMP(r / max(len(val) - 1, 1)), alpha=0.6, zorder=1)
    a.plot([0, XL], [0, XL], ":", lw=0.7, color=INK3, zorder=2)
    a.axhline(l0, color=INK2, lw=0.8, zorder=3)
    a.axhline(llow, color=INK3, lw=0.7, ls=":", zorder=3)
    live = R[crit] > 0
    xc, yc = R[crit][live] / Tb, np.minimum.accumulate(pa[crit])[live] / Tb
    a.plot(xc, yc, "-", lw=1.4, color=ORANGE, zorder=5)
    ring(a, xc[-1], yc[-1], ORANGE, ms=4.4)
    a.set_xlim(XL, -0.035); a.set_ylim(*YL)
    a.set_xticks([0.75, 0.5, 0.25, 0.0]); a.set_yticks([-0.5, 0, 0.5])
    a.set_xlabel(r"true remaining life  $R_u(t)/\bar{T}$", labelpad=1.0)
    a.set_ylabel(r"running min of $\hat{R}_u\,/\,\bar{T}$", labelpad=1.5)
    note(a, XL - .03, l0 + .055, r"$\ell_0$", color=INK2, fontsize=8.8)
    # mid-panel, not the left edge: the legend has the lower left and the two
    # collided once the panel was made shorter.
    note(a, XL - .42, llow - .055, r"peak variance", color=INK2, fontsize=8.8,
         va="top")
    ha = [Line2D([], [], color="#2c5f92", lw=0.9),
          Line2D([], [], color=INK3, lw=0.7, ls=":"),
          Line2D([], [], color=ORANGE, lw=1.4)]
    a.legend(ha, ["one unit", "perfect prediction", r"sets $\ell_0$"],
             loc="lower left", ncol=1, borderpad=0.1, labelspacing=0.12,
             handlelength=1.2, handletextpad=0.4, fontsize=8.8)

    # (b) Theorem 4: the failure probability is the upper tail of those endpoints
    lo_g, hi_g = float(lu.min()) - .05, l0 + .16
    grid = np.linspace(lo_g, hi_g, 400)
    pfs = np.array([(lu > z).mean() for z in grid])
    b.fill_between(grid, 0, pfs, color=BLUE, alpha=0.18, lw=0, zorder=1)
    b.plot(grid, pfs, "-", color=BLUE, zorder=3)
    b.plot(lu, np.full_like(lu, -0.085), "|", ms=3.4, mew=0.55, color=INK2,
           zorder=3)
    b.axvline(l0, color=INK2, lw=0.8, zorder=2)
    b.axvline(llow, color=INK3, lw=0.7, ls=":", zorder=2)
    ring(b, l0, 0.0, ORANGE, ms=4.2)
    plow = float((lu > llow).mean())
    ring(b, llow, plow, INK2, ms=3.4)
    b.set_xlim(lo_g, hi_g); b.set_ylim(-0.15, 1.06)
    b.set_yticks([0, 0.5, 1.0]); b.set_xticks([-0.4, -0.2, 0.0])
    b.set_xlabel(r"threshold  $\ell/\bar{T}$", labelpad=1.0)
    b.set_ylabel(r"$P_f=\Pr(\ell_u>\ell)$", labelpad=1.5)
    note(b, lo_g + .012, 0.055, r"one tick $=$ one $\ell_u$", color=INK2,
         fontsize=8.8, va="bottom")
    note(b, llow - .022, plow + .16, "%.0f%%" % (100 * plow), color=INK2,
         fontsize=8.8, ha="right")

    # (c) the other term of the cost, on the same threshold axis: wasted life.
    #     A failed unit runs to the end, so it wastes nothing and enters as zero.
    gw = np.linspace(lo_g, hi_g, 90)
    W = np.zeros((len(gw), len(val)))
    for gi, e in enumerate(gw):
        for uk, i in enumerate(val):
            hh = np.nonzero(rm[i] <= e * Tb)[0]
            if len(hh) and R[i][hh[0]] > 0:
                W[gi, uk] = R[i][hh[0]] / Tb
    wbar = W.mean(1)
    q1, q3 = np.percentile(W, [25, 75], axis=1)
    w.fill_between(gw, q1, q3, color=VIOLET, alpha=0.18, lw=0, zorder=1)
    w.plot(gw, wbar, "-", color=VIOLET, zorder=3)
    w.axvline(l0, color=INK2, lw=0.8, zorder=2)
    w.axvline(llow, color=INK3, lw=0.7, ls=":", zorder=2)
    w0 = float(np.interp(l0, gw, wbar))
    ring(w, l0, w0, VIOLET, ms=4.2)
    w.set_xlim(lo_g, hi_g); w.set_ylim(-0.012, max(q3.max(), wbar.max()) * 1.30)
    w.set_xticks([-0.4, -0.2, 0.0])
    w.set_xlabel(r"threshold  $\ell/\bar{T}$", labelpad=1.0)
    w.set_ylabel(r"wasted life  $W_u/\bar{T}$", labelpad=1.5)
    note(w, l0 - .015, w0 + .022, r"$\omega_0{=}%.3f$" % w0, color=VIOLET,
         fontsize=8.8, ha="right")
    note(w, lo_g + .012, max(q3.max(), wbar.max()) * 1.26,
         "mean and\ninterquartile band", color=INK3, fontsize=8.8, va="top")
    print("fig1", f"omega at l0 = {w0:.4f}", flush=True)

    # (d),(e) the consequence, both fleets on the shifted axis
    for nm, col in (("FD002", BLUE), ("FD004", ORANGE)):
        x, cost, pf, sd, _ = SW[nm]
        zero = x[pf == 0]
        sh = x - (zero.min() if len(zero) else x[-1])
        j = int(np.argmin(cost))
        c.fill_between(sh, cost * (1 - 2 * sd / 100), cost * (1 + 2 * sd / 100),
                       color=col, alpha=0.22, lw=0, zorder=2)
        c.plot(sh, cost, "-", color=col, zorder=3, label=nm)
        ring(c, sh[j], cost[j], col)
        d.plot(sh, sd, "-", color=col, zorder=3)
        ring(d, sh[j], sd[j], col)
        print("fig1", nm, f"opt {sh[j]:+.3f}  sd {sd[:j].max():.1f}->{sd[j]:.1f}",
              flush=True)
    for q in (c, d):
        q.axvspan(0, 0.42, color=BAND, lw=0, zorder=0)
        q.set_xlim(-0.40, 0.32)
        q.set_xticks([-0.4, -0.2, 0.0, 0.2])
        q.set_xlabel(r"$(\ell-\ell_0)/\bar{T}$", labelpad=1.0)
    c.set_yscale("log")
    c.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10]))
    c.yaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "5", "10"]))
    c.yaxis.set_minor_locator(FixedLocator([]))
    c.set_ylim(0.93, 21)
    c.set_ylabel("cost / oracle", labelpad=1.5)
    c.legend(loc="upper right", borderpad=0.12, labelspacing=0.15,
             handlelength=1.2, handletextpad=0.4, fontsize=8.8)
    note(c, -0.375, 17.5, r"band: $\pm2$ s.d.", color=INK2, fontsize=8.8,
         va="top")
    d.set_ylim(0, 16.5); d.set_yticks([0, 5, 10, 15])
    d.set_ylabel("s.d. of that\nestimate (%)", labelpad=1.5)
    note(d, 0.015, 15.4, "zero-failure\nregion", color=INK2, fontsize=8.8,
         va="top")

    for q, tag in zip(ax, ("(a)", "(b)", "(c)", "(d)", "(e)")):
        q.text(-0.04, 1.17, tag, transform=q.transAxes, color=INK,
               fontsize=8.8, ha="right", va="top")
    # This gridspec has 3 spanning panels over one row and 2 over the other, a
    # structure constrained layout collapses to zero width.  The margins are set
    # from the measured ink box instead, so the figure fills the 6.48in text
    # block the way the constrained-layout figures do.
    fig.subplots_adjust(left=0.073, right=0.995, top=0.931, bottom=0.115,
                        wspace=1.15, hspace=0.52)
    fig.savefig("fig1.pdf")
    plt.close(fig)
    print("fig1", f"l0={l0:.3f}  "
          f"P_f one step lower={(lu > llow).mean():.3f}", flush=True)


if __name__ == "__main__":
    fig1()
