"""
extract_lighting.py  —  Extract LIGHTING symbols from Legends.pdf
Outputs: Output/extracted/legends/LIGHTING/<LABEL>.png
"""

import re, fitz, cv2, numpy as np
from pathlib import Path
from collections import Counter

PDF_PATH = Path("Input/Legends.pdf")
OUT_DIR  = Path("Output/extracted/legends/LIGHTING")
DPI      = 150
SCALE    = DPI / 72.0


def safe_name(text, maxlen=120):
    text = re.sub(r'[<>:"/\\|?*\n\r]', '_', text)
    return re.sub(r'\s+', ' ', text).strip()[:maxlen]


# ── 1. render page ────────────────────────────────────────────────────────
doc   = fitz.open(str(PDF_PATH))
page  = doc[0]
mat   = fitz.Matrix(SCALE, SCALE)
pix   = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
img   = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
H, W  = color.shape[:2]
print(f"Page rendered: {W}×{H} px at {DPI} DPI")


# ── 2. extract all PDF words (scaled to pixels) ───────────────────────────
raw = page.get_text("words")
words = [
    (int(w[0]*SCALE), int(w[1]*SCALE),
     int(w[2]*SCALE), int(w[3]*SCALE), str(w[4]))
    for w in raw
]


# ── 3. find LIGHTING section Y bounds ─────────────────────────────────────
# The LIGHTING section header in column 1 appears at the TOP of the section
# (y < 400) — distinct from the word "LIGHTING" that also appears inside
# description labels like "TRACK LIGHTING WITH HEADS AS INDICATED".
light_hdr_y = None
for x1, y1, x2, y2, txt in words:
    if txt.strip().upper() == "LIGHTING" and x1 < 700 and y1 < 400:
        light_hdr_y = y1
        print(f"LIGHTING section header @ y={y1}, x={x1}")
        break

assert light_hdr_y is not None, "LIGHTING section header not found"

# SYMBOL/DESCRIPTION column sub-headers are at y≈241; first data row follows.
# Use the DESCRIPTION sub-header y as the reference for data_y_start.
desc_subhdr_y = next(
    (y1 for x1, y1, x2, y2, txt in words
     if txt.strip().upper() == "DESCRIPTION" and x1 < 700 and y1 < 400),
    light_hdr_y + 60
)
light_y1 = desc_subhdr_y   # data rows begin just after this sub-header
print(f"Data rows start  @ y>{desc_subhdr_y}")

# Section ends at the next section header in col 1 ("SWITCHING CONTROLS")
light_y2 = H
for x1, y1, x2, y2, txt in words:
    if txt.strip().upper() == "SWITCHING" and y1 > light_y1 and x1 < 700:
        light_y2 = y1
        print(f"LIGHTING ends    @ y={y1}  (SWITCHING CONTROLS)")
        break

print(f"LIGHTING data range: y=[{light_y1}, {light_y2}]")


# ── 4. find description-text left edge (modal x1 of label words) ──────────
sym_hdr_x1 = next(x1 for x1,y1,x2,y2,t in words
                  if t.strip().upper()=="SYMBOL" and x1<500 and y1<500)
sym_hdr_x2 = next(x2 for x1,y1,x2,y2,t in words
                  if t.strip().upper()=="SYMBOL" and x1<500 and y1<500)

section_words = [(x1,y1,x2,y2,txt) for x1,y1,x2,y2,txt in words
                 if y1 > light_y1 and y1 < light_y2 and x1 < 900]
desc_x_candidates = [x1 for x1,_,_,_,_ in section_words if x1 > sym_hdr_x2+30]
desc_text_x1 = Counter(desc_x_candidates).most_common(1)[0][0] if desc_x_candidates else 397
print(f"Description text starts @ x={desc_text_x1}")


# ── 5. measure symbol column X bounds from vertical lines ─────────────────
# Detect vertical column-border lines in the symbol area — same method used
# by the debug sheet so the crop X range exactly matches the debug boxes.
import cv2 as _cv2
_scan_x1 = max(0, sym_hdr_x1 - 150)
_scan_x2 = min(W, desc_text_x1 + 30)
_scan     = _cv2.cvtColor(color[light_y1:light_y2, _scan_x1:_scan_x2],
                          _cv2.COLOR_BGR2GRAY)
_, _bv    = _cv2.threshold(_scan, 220, 255, _cv2.THRESH_BINARY_INV)
_vk       = _cv2.getStructuringElement(_cv2.MORPH_RECT, (1, 60))
_vert     = _cv2.morphologyEx(_bv, _cv2.MORPH_OPEN, _vk)
_cs       = _vert.sum(axis=0).astype(float) / (_vert.shape[0] + 1e-9) / 255

def _cluster(vals, gap=12):
    if not vals: return []
    s = sorted(set(vals))
    out, cur = [], [s[0]]
    for v in s[1:]:
        if v - cur[-1] <= gap: cur.append(v)
        else: out.append(int(sum(cur)/len(cur))); cur = [v]
    out.append(int(sum(cur)/len(cur)))
    return out

_raw_vx  = [xi + _scan_x1 for xi, v in enumerate(_cs) if v > 0.06]
_vert_xs = _cluster(_raw_vx, gap=12)

_left_c  = [x for x in _vert_xs if x <= sym_hdr_x1 + 15]
_right_c = [x for x in _vert_xs if desc_text_x1-55 < x < desc_text_x1+15]
SYM_X1   = _left_c[-1]  if _left_c  else max(0, sym_hdr_x1 - 80)
SYM_X2   = _right_c[0]  if _right_c else desc_text_x1 - 3
print(f"Symbol column (from vert lines): x=[{SYM_X1}, {SYM_X2}]  "
      f"width={SYM_X2-SYM_X1}px")


# ── 6. gather description-column words in the LIGHTING section ────────────
data_y_start = light_y1 + 5    # just past the sub-header row
desc_words = [
    (x1, y1, x2, y2, txt)
    for (x1, y1, x2, y2, txt) in words
    if y1 >= data_y_start
    and y1 <  light_y2
    and x1 >= desc_text_x1 - 10   # description column ± small tolerance
    and x1 <  900                   # stay within column 1
]
print(f"Description words in section: {len(desc_words)}")


# ── 7. group words into text lines (same cy ± 6 px) ───────────────────────
line_buckets = {}
for w in desc_words:
    wcy = (w[1] + w[3]) // 2
    match = None
    for key in line_buckets:
        if abs(key - wcy) <= 6:
            match = key
            break
    if match is None:
        line_buckets[wcy] = []
        match = wcy
    line_buckets[match].append(w)

lines = sorted(
    [(cy, sorted(ws, key=lambda w: w[0])) for cy, ws in line_buckets.items()],
    key=lambda t: t[0]
)
print(f"Text lines found: {len(lines)}")


# ── 8. merge continuation lines → one label per legend row ────────────────
# Lines with gap < 18 px between bottom of one and top of next = same row
assembled = []
i = 0
while i < len(lines):
    cy0, ws0 = lines[i]
    parts  = [w[4] for w in ws0]
    row_y1 = min(w[1] for w in ws0)
    row_y2 = max(w[3] for w in ws0)

    j = i + 1
    while j < len(lines):
        _, ws_next = lines[j]
        next_y1 = min(w[1] for w in ws_next)
        if next_y1 - row_y2 < 18:           # continuation line
            parts.extend(w[4] for w in ws_next)
            row_y2 = max(row_y2, max(w[3] for w in ws_next))
            j += 1
        else:
            break

    assembled.append({
        "text": " ".join(parts),
        "y1":   row_y1,
        "y2":   row_y2,
        "cy":   (row_y1 + row_y2) // 2,
    })
    i = j

print(f"Assembled rows  : {len(assembled)}")
for r in assembled:
    print(f"  y={r['y1']:4d}–{r['y2']:4d}  {r['text'][:80]}")


# ── 9. crop & save ────────────────────────────────────────────────────────
# Row Y bounds: midpoints between consecutive CY values.
# No horizontal table lines exist in this legend — rows are whitespace-separated.
# CY midpoints correctly capture the full symbol cell including all sub-variants.
OUT_DIR.mkdir(parents=True, exist_ok=True)
saved = 0

for idx, row in enumerate(assembled):
    cy = row["cy"]
    if idx == 0:
        prev_cy = light_y1
    else:
        prev_cy = assembled[idx-1]["cy"]

    if idx == len(assembled) - 1:
        next_cy = light_y2
    else:
        next_cy = assembled[idx+1]["cy"]

    ry1 = max(0, (prev_cy + cy) // 2)
    ry2 = min(H, (cy + next_cy) // 2)

    if ry2 <= ry1 or SYM_X2 <= SYM_X1:
        continue

    crop = color[ry1:ry2, SYM_X1:SYM_X2].copy()
    if crop.size == 0:
        continue

    fname = safe_name(row["text"]) + ".png"
    cv2.imwrite(str(OUT_DIR / fname), crop)
    saved += 1

print(f"\nDone — {saved} crops saved to {OUT_DIR}")
