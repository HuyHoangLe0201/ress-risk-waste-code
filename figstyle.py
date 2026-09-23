"""Shared figure style for the RESS manuscript.

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
from matplotlib import patheffects

# palette
BLUE, ORANGE, VIOLET = "#1a5fb4", "#c2410c", "#4a148c"
INK, INK2, INK3 = "#1a1a19", "#33322e", "#5f5e58"
GRID = "#dbd9d3"
BAND = "#eae8e2"          # regime shading, one step off the surface
SURFACE = "#ffffff"

# cas-sc \linewidth measured from the class: 468.33pt = 6.48in.  Figures are
# authored at exactly that width and the tight crop is switched off, so
# \includegraphics[width=\linewidth] scales by 1.000 and the point sizes set
# below are the point sizes that reach the page.  The previous convention drew
# oversize and let the crop land near the text block; it landed at 6.22, 6.48,
# 6.29 and 6.37in instead, four different scale factors and four different
# effective type sizes in one paper.
# 190 mm, the Elsevier double-column width, in inches. Authored at the manuscript's
# 468.33pt \linewidth (6.48in = 164.6mm) the figures matched no width Elsevier
# typesets at, so production had to rescale them and the lettering rule was met or
# missed by luck. The manuscript includes at \linewidth and so scales by 0.866.
COL = COL1 = COL2 = 190.0 / 25.4


def setup():
    plt.rcParams.update({
        # Elsevier and IEEE both reject Type 3 fonts in artwork; 42 embeds TrueType.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # mathtext defaulted to DejaVu Sans, so every symbol clashed with the Times
        # labels around it.  STIX is metric-compatible with Times.
        "mathtext.fontset": "stix",
        "figure.dpi": 400, "savefig.dpi": 400,
        # None, not "tight": a crop to content discards the authored width.
        # Passing bbox_inches=None to savefig does NOT do this -- matplotlib
        # resolves that back to this rcParam.
        "savefig.bbox": None, "savefig.pad_inches": 0.0, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        # Elsevier wants 7pt normal and 6pt subscripts at the printed size, and mathtext
        # subscripts render at 0.7 of the base, so nothing that can carry a subscript may
        # sit below 8.6pt. The legend and the ticks were at 8.0, which put a subscripted
        # legend entry at 5.6pt -- the one figure that still failed after the base
        # annotation size was raised.
        "font.size": 8.8, "axes.labelsize": 8.8, "axes.titlesize": 9.2,
        "legend.fontsize": 8.8, "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
        # 0.6, matching the TikZ figures: at 0.5 the spines and ticks read as a whisper
        # beside 1.5pt data strokes, which is half of what "faint" meant. The grid stays at
        # 0.5 and one shade off the surface, because it is meant to be recessive.
        "axes.edgecolor": INK3, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25,
        "lines.linewidth": 1.5, "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
    })


def clean(ax, spines=("left", "bottom", "top", "right")):
    """Recessive chrome: keep the named spines, framed by default.

    The default used to be left and bottom only. An open pair of axes leaves the plot area
    unbounded, so the surrounding white reads as part of the figure and the panel looks
    sparser than it is -- measured, these panels carry as much ink per unit area as the
    reference journal's, and still looked emptier on the page. A closed frame is what that
    reference draws, and it is what tells the eye where the panel stops.

    The frame is drawn in ink, not black: at four spines a full-strength rule competes with
    the data it encloses.
    """
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_color(INK3)
    ax.tick_params(length=2.2, pad=1.8, color=INK3)
    return ax


def ring(ax, x, y, color, ms=4.4, marker="o", z=5):
    """A mark with a 2-unit surface ring, for markers that overlap other marks."""
    ax.plot(x, y, marker, ms=ms + 1.6, color=SURFACE, zorder=z,
            markeredgewidth=0, clip_on=False)
    ax.plot(x, y, marker, ms=ms, color=color, zorder=z + 1,
            markeredgewidth=0, clip_on=False)


def halo(width=1.9):
    """Stroke text in the surface colour underneath itself.

    An annotation inside the axes is drawn over the grid, and the grid then shows through
    the counters of the glyphs, which is what makes in-panel labels look dirty. A halo
    removes the grid from behind the letters only; a solid bbox would also remove whatever
    the label is meant to sit beside.
    """
    return [patheffects.withStroke(linewidth=width, foreground=SURFACE)]


def note(ax, x, y, s, color=INK2, **kw):
    """A direct label in ink, never in a series colour, haloed against the grid."""
    kw.setdefault("fontsize", 7.6)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "center")
    kw.setdefault("path_effects", halo())
    kw.setdefault("zorder", 7)
    return ax.annotate(s, (x, y), color=color, **kw)
