r"""Renumber the results so they appear in increasing order.

The numbering grew with the argument rather than with the page, and a reader now meets
Theorem 4 before Theorem 1 and Proposition 20 before Proposition 3.  In file order the
theorems run 4, 1, 6, 5, 2, 7, 3 and the propositions 1, 2, 6, 20, 15, 21, 4, 3, 26, 14,
17, 19, 13, 12, 18, 11, 10, 9, 16, 8, 7, 5, 22-29.  Nothing about the mathematics depends
on it, but no journal will print it.

The results are numbered by hand, so the cross-references are text and not \ref, and a
sequential search-and-replace would collide the moment the map is not monotone (5 -> 7 and
7 -> 5 in the same pass).  This walks the file once and rewrites each reference from the
map, which makes the operation a permutation by construction.

References come in three shapes and all three carry the class from the head of the list:

    Proposition~14                      singular
    Propositions~2 and~3                a pair
    Propositions~2, 3 and~14            a list

so a continuation is consumed only when it directly follows a number already claimed.
Every occurrence found is checked against the defined results first; the file contains no
reference to another paper's theorem by number, which is what makes this safe.
"""
import io
import os
import re
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ress_cas.tex")

DEF = re.compile(r"\\textbf\{(Theorem|Proposition|Lemma) (\d+)\.\}")
HEAD = re.compile(r"\b(Theorems|Theorem|Propositions|Proposition|Lemmas|Lemma)([~ ])(\d+)")
CONT = re.compile(r"\A(,\s*|\s+and[~ ])(\d+)")


def build_map(s):
    """old -> new per class, new numbers following the order of the definitions."""
    seen = {}
    for m in DEF.finditer(s):
        cls, num = m.group(1), int(m.group(2))
        seen.setdefault(cls, [])
        assert num not in seen[cls], "%s %d defined twice" % (cls, num)
        seen[cls].append(num)
    return {c: {old: i + 1 for i, old in enumerate(v)} for c, v in seen.items()}


def rewrite(s, amap, report=None, where="the manuscript"):
    out, i, moved = [], 0, 0
    while True:
        m = HEAD.search(s, i)
        if not m:
            out.append(s[i:])
            break
        cls = m.group(1).rstrip("s")
        out.append(s[i:m.start()])
        num = int(m.group(3))
        if num not in amap[cls]:
            # A reference to a number with no definition left -- normally the
            # aftermath of merging two results without normalising the references
            # to the one that vanished.  Say which, because a bare KeyError does
            # not, and this runs over the side files where the context is lost.
            raise SystemExit(
                "%s %d is referenced in %s but no longer defined; "
                "normalise that reference before renumbering "
                "(defined: %s)" % (cls, num, where,
                                   ", ".join(map(str, sorted(amap[cls])))))
        new = amap[cls][num]
        moved += (new != num)
        out.append("%s%s%d" % (m.group(1), m.group(2), new))
        j = m.end()
        while True:                       # a list continues under the same class
            c = CONT.match(s[j:])
            if not c:
                break
            n2 = int(c.group(2))
            if n2 not in amap[cls]:
                # Stopping here is what a list like "Propositions 2 and 3, and 4
                # units" needs, but it is also how a merge does its damage: the
                # entry naming the result that was merged away keeps its old
                # number and silently comes to denote whichever result inherited
                # it.  That happened to three lists and cost a proposition a
                # citation of itself, so the ambiguous case is raised, not passed.
                raise SystemExit(
                    "in %s: the list \"%s\" continues with %s %d, which is not "
                    "defined -- if it named a result a merge removed, point it "
                    "at the merged result; if it is not a reference at all, "
                    "reword it (defined: %s)"
                    % (where, " ".join(s[m.start():j + c.end()].split()),
                       cls, n2, ", ".join(map(str, sorted(amap[cls])))))
            if report is not None:
                report.append((cls, n2, s[max(0, m.start() - 0):j + c.end()]))
            moved += (amap[cls][n2] != n2)
            out.append("%s%d" % (c.group(1), amap[cls][n2]))
            j += c.end()
        i = j
    return "".join(out), moved


def side_files():
    """Everything else that names a result by number.

    The scripts ship as the reproduction bundle and their docstrings carry the argument
    -- "Theorem 5's hypothesis", "the open case of Proposition 22".  Renumbering the
    manuscript without them would leave the bundle describing a paper that no longer
    exists, so they are rewritten from the same map.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    out = [os.path.join(here, "framework.tex")]
    out += [os.path.join(here, f) for f in sorted(os.listdir(here))
            if f.endswith(".py") and f != "renumber.py"]
    return [f for f in out if os.path.exists(f)]


def main(apply_it):
    s = io.open(P, encoding="utf-8").read()
    amap = build_map(s)
    for cls in sorted(amap):
        items = sorted(amap[cls].items(), key=lambda kv: kv[1])
        print("%s: %s" % (cls, " ".join("%d->%d" % kv for kv in items
                                        if kv[0] != kv[1]) or "already in order"))
    before = re.findall(r"(Theorem|Proposition|Lemma)s?[~ ](\d+)", s)

    report = []
    out, moved = rewrite(s, amap, report)
    print("\ncontinuations consumed (these inherit the class of the list head):")
    for cls, n, ctx in report:
        print("   %s %d   in  ...%s..." % (cls, n, " ".join(ctx.split())[-58:]))

    after = re.findall(r"(Theorem|Proposition|Lemma)s?[~ ](\d+)", out)
    assert len(before) == len(after), "reference count changed"
    exp = sorted((c, amap[c.rstrip('s')][int(n)]) for c, n in before)
    got = sorted((c, int(n)) for c, n in after)
    assert exp == got, "references are not the image of the map"
    order = [int(m.group(2)) for m in DEF.finditer(out)]
    per = {}
    for m in DEF.finditer(out):
        per.setdefault(m.group(1), []).append(int(m.group(2)))
    for cls, v in per.items():
        assert v == sorted(v) == list(range(1, len(v) + 1)), "%s still unordered" % cls

    print("\n%d references rewritten, %d unchanged" % (moved, len(before) - moved))
    print("definitions now ascending in every class: %s"
          % ", ".join("%s 1..%d" % (c, len(v)) for c, v in sorted(per.items())))

    side, scanned = [], 0
    for f in side_files():
        scanned += 1
        t = io.open(f, encoding="utf-8").read()
        t2, k = rewrite(t, amap, where=os.path.basename(f))
        if k:
            side.append((os.path.basename(f), k, t2))
    # k counts references MOVED, not references seen, so zero here means "every side
    # file already agrees with the manuscript" -- not "no side file was looked at".
    print("side files scanned: %d; references needing a rewrite: %d in %d files"
          % (scanned, sum(k for _, k, _ in side), len(side)))
    for nm, k, _ in side:
        print("   %-26s %d" % (nm, k))

    if apply_it:
        io.open(P, "w", encoding="utf-8").write(out)
        for f in side_files():
            t = io.open(f, encoding="utf-8").read()
            t2, k = rewrite(t, amap, where=os.path.basename(f))
            if k:
                io.open(f, "w", encoding="utf-8").write(t2)
        print("written")
    else:
        print("DRY RUN -- pass --apply to write")


if __name__ == "__main__":
    main("--apply" in sys.argv)
