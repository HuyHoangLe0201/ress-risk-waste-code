"""Shared figure style for the IEEE TR manuscript.

Palette validated with validate_palette.py (port of the dataviz validator):
  #2a78d6 blue, #eb6834 orange, #4a3aa7 violet
passes lightness band, chroma floor, CVD separation (worst 9.4), normal-vision
floor (worst 24.0) and contrast vs surface, on both the adjacent and the all
pairlists.

Design rules enforced here:
  - no dual axes anywhere (small multiples instead)
  - hairline SOLID grid, one shade off the surface, drawn under the data
  - thin marks, selective direct labels, legend whenever >= 2 series
  - text in ink tokens, never in a series colour
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
INK, INK2, INK3 = "#1a1a19", "#55544e", "#8a8880"
GRID = "#e6e5e1"
BAND = "#f2f1ed"          # regime shading, one step off the surface
SURFACE = "#ffffff"

COL1, COL2 = 3.45, 7.16   # IEEE single- and double-column widths, inches


def setup():
    plt.rcParams.update({
        "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
        "legend.fontsize": 6.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.edgecolor": INK3, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25,
        "lines.linewidth": 1.4, "lines.markersize": 3.6,
        "lines.solid_capstyle": "round",
    })


def clean(ax, spines=("left", "bottom")):
    """Recessive chrome: keep only the named spines."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
    ax.tick_params(length=2.2, pad=1.8)
    return ax


def ring(ax, x, y, color, ms=4.4, marker="o", z=5):
    """A mark with a 2-unit surface ring, for markers that overlap other marks."""
    ax.plot(x, y, marker, ms=ms + 1.6, color=SURFACE, zorder=z,
            markeredgewidth=0, clip_on=False)
    ax.plot(x, y, marker, ms=ms, color=color, zorder=z + 1,
            markeredgewidth=0, clip_on=False)


def note(ax, x, y, s, color=INK2, **kw):
    """A direct label in ink, never in a series colour."""
    kw.setdefault("fontsize", 6.4)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    return ax.annotate(s, (x, y), color=color, **kw)
