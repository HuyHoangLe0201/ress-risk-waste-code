"""Rebuild Highlights.docx to the journal's stated spec: five bullets, <=85 chars.

The file had four bullets capped at 80 characters, built to a general Elsevier
recommendation of three to five.  The RESS guide asks for five at a maximum of
eighty-five, so a fifth is added rather than left to chance, and it carries the
result that answers the "simple extension" objection: the classical optimality
condition is the no-information case of the rule proved here.
"""
import docx

BULLETS = [
    "One identity places every replacement policy in a risk-waste plane",
    "Zero-failure threshold is optimal iff the cost ratio clears a constant",
    "That constant is 1.6 on turbofan fleets and 8.0 on a battery fleet",
    "Optimal rule thresholds conditional hazard at the optimal cost rate itself",
    "Unit-level bootstrap: intervals 3-6x wider, 93-94% is policy learning",
]
TITLE = "Optimality and Estimability of Condition-Based Replacement Policies"

P = (r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri"
     r"\IEEE_Trans.representation\RESS_paper\Highlights.docx")

over = [b for b in BULLETS if len(b) > 85]
assert not over, over
assert len(BULLETS) == 5, len(BULLETS)

d = docx.Document()
d.add_paragraph("Highlights")
d.add_paragraph(TITLE)
for b in BULLETS:
    d.add_paragraph(b)
d.save(P)

for b in BULLETS:
    print("%3d  %s" % (len(b), b))
print("\n%d bullets, longest %d chars (limit 85)" % (len(BULLETS), max(map(len, BULLETS))))
