"""Faithful Python port of dataviz/scripts/validate_palette.js (node unavailable).
Same thresholds, same OKLab maths, same Machado-Oliveira-Fernandes 2009 severity-1.0
CVD transforms.  Prints the same six checks."""
import sys, math, itertools

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
}


def hex2srgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1, max(0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def dE(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(pal, mode="light", pairs="adjacent"):
    surf = SURFACE[mode]
    lo, hi = BAND[mode]
    rows = []
    Ls = [(c, round(oklch(c)[0], 3)) for c in pal]
    bad = [x for x in Ls if not (lo <= x[1] <= hi)]
    rows.append(("Lightness band", not bad,
                 f"outside [{lo},{hi}]: {bad}" if bad else f"all {len(pal)} in band"))
    Cs = [(c, round(oklch(c)[1], 3)) for c in pal]
    lowc = [x for x in Cs if x[1] < CHROMA_FLOOR]
    rows.append(("Chroma floor", not lowc,
                 f"below floor: {lowc}" if lowc else f"all {len(pal)} >= {CHROMA_FLOOR}"))
    prs = (list(zip(pal, pal[1:])) if pairs == "adjacent"
           else list(itertools.combinations(pal, 2)))
    worst_cvd, wc_pair = min(((min(dE(a, b, "protan"), dE(a, b, "deutan")), (a, b))
                              for a, b in prs), key=lambda t: t[0])
    st = "pass" if worst_cvd >= CVD_TARGET else ("floor" if worst_cvd >= CVD_FLOOR else "FAIL")
    rows.append((f"CVD separation ({pairs})", st != "FAIL",
                 f"worst {worst_cvd:.1f} on {wc_pair} [{st}, target {CVD_TARGET}]"))
    worst_nor, wn_pair = min(((dE(a, b), (a, b)) for a, b in prs), key=lambda t: t[0])
    rows.append(("Normal-vision floor", worst_nor >= NORMAL_FLOOR,
                 f"worst {worst_nor:.1f} on {wn_pair} [floor {NORMAL_FLOOR}]"))
    low = [(c, round(contrast(c, surf), 2)) for c in pal if contrast(c, surf) < CONTRAST_MIN]
    rows.append(("Contrast vs surface", not low,
                 f"below {CONTRAST_MIN}:1 {low}" if low else f"all {len(pal)} >= {CONTRAST_MIN}:1"))
    print(f"\n  palette={pal}  mode={mode}  pairs={pairs}  surface={surf}")
    ok = True
    for name, good, msg in rows:
        print(f"  {'PASS' if good else 'FAIL'}  {name:26s} {msg}")
        ok &= good
    print(f"  ==> {'ALL CHECKS PASS' if ok else 'FIX REQUIRED'}")
    return ok


if __name__ == "__main__":
    pal = sys.argv[1].split(",") if len(sys.argv) > 1 else ["#2a78d6", "#eb6834", "#1baf7a"]
    validate(pal, "light", "adjacent")
    validate(pal, "light", "all")
