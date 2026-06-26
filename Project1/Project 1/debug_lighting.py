"""
debug_lighting.py  — Debug overlay for the LIGHTING section of Legends.pdf

Key findings from pixel analysis:
  - NO horizontal separator lines between rows (rows separated by whitespace only)
  - Vertical column borders at x=188, x=375
  - Row bounds = midpoints between consecutive text-label CY values
  - Symbol column: x=[188, 375], width=187px (31.7mm at 150 DPI)
  - Some rows intentionally show multiple sub-variants (e.g. BATTERY POWER = 3 symbols)

Output: Output/extracted/debug_lighting.png
"""

import re, fitz, cv2, numpy as np
from pathlib import Path
from collections import Counter

PDF_PATH  = Path("Input/Legends.pdf")
OUT_DEBUG = Path("Output/extracted/debug_lighting.png")
DPI       = 150
SCALE     = DPI / 72.0


# ─── 1. render ───────────────────────────────────────────────────────────────
doc  = fitz.open(str(PDF_PATH))
page = doc[0]
pix  = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), colorspace=fitz.csRGB)
img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
H, W  = color.shape[:2]
print(f"Page: {W}x{H}")


# ─── 2. PDF words ─────────────────────────────────────────────────────────────
words = [(int(w[0]*SCALE), int(w[1]*SCALE), int(w[2]*SCALE), int(w[3]*SCALE), str(w[4]))
         for w in page.get_text("words")]


# ─── 3. Section bounds ────────────────────────────────────────────────────────
SECT_Y1 = next(y1 for x1,y1,x2,y2,t in words
               if t.upper()=="DESCRIPTION" and x1<700 and y1<400)
SECT_Y2 = next((y1 for x1,y1,x2,y2,t in words
                if t.upper()=="SWITCHING" and y1>SECT_Y1 and x1<700), H)
print(f"Section: y=[{SECT_Y1}, {SECT_Y2}]")


# ─── 4. Measure symbol column width from vertical lines ──────────────────────
sym_hdr = next((x1,x2) for x1,y1,x2,y2,t in words
               if t.upper()=="SYMBOL" and x1<500 and y1<500)
sec_wds = [(x1,y1,x2,y2,t) for x1,y1,x2,y2,t in words
           if SECT_Y1<y1<SECT_Y2 and x1<900]
desc_x_cands = [x1 for x1,_,_,_,_ in sec_wds if x1 > sym_hdr[1]+30]
DESC_X1 = Counter(desc_x_cands).most_common(1)[0][0] if desc_x_cands else 397

# Detect vertical lines in the scan zone
scan_x1 = max(0, sym_hdr[0] - 150)
scan_x2 = min(W, DESC_X1 + 30)
scan    = gray[SECT_Y1:SECT_Y2, scan_x1:scan_x2]
_, bw_v = cv2.threshold(scan, 220, 255, cv2.THRESH_BINARY_INV)
vk      = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
vert    = cv2.morphologyEx(bw_v, cv2.MORPH_OPEN, vk)
cs      = vert.sum(axis=0).astype(float) / (vert.shape[0] + 1e-9) / 255

def cluster(vals, gap=12):
    if not vals: return []
    s = sorted(set(vals))
    out, cur = [], [s[0]]
    for v in s[1:]:
        if v - cur[-1] <= gap: cur.append(v)
        else: out.append(int(np.mean(cur))); cur = [v]
    out.append(int(np.mean(cur)))
    return out

raw_vx  = [xi + scan_x1 for xi, v in enumerate(cs) if v > 0.06]
vert_xs = cluster(raw_vx, gap=12)

# left border = last vert-line at or left of sym_hdr x1
# right border = first vert-line just before DESC_X1
left_c  = [x for x in vert_xs if x <= sym_hdr[0] + 15]
right_c = [x for x in vert_xs if DESC_X1-55 < x < DESC_X1+15]
SYM_L   = left_c[-1]  if left_c  else max(0, sym_hdr[0] - 80)
SYM_R   = right_c[0]  if right_c else DESC_X1 - 3
SYM_W   = SYM_R - SYM_L
print(f"Symbol column: x=[{SYM_L}, {SYM_R}]  width={SYM_W}px  ({SYM_W/DPI*25.4:.1f} mm)")
print(f"Vertical lines detected: {vert_xs}")


# ─── 5. Assemble text labels ─────────────────────────────────────────────────
dw = [(x1,y1,x2,y2,t) for x1,y1,x2,y2,t in words
      if SECT_Y1+5 <= y1 < SECT_Y2 and x1 >= DESC_X1-10 and x1 < 900]
lb = {}
for w in dw:
    cy = (w[1]+w[3])//2
    match = None
    for k in lb:
        if abs(k-cy) <= 6: match=k; break
    if match is None: lb[cy]=[]; match=cy
    lb[match].append(w)

ls = sorted([(cy, sorted(ws, key=lambda w:w[0])) for cy,ws in lb.items()],
            key=lambda t:t[0])
asm = []
i = 0
while i < len(ls):
    cy0, ws0 = ls[i]
    parts = [w[4] for w in ws0]
    r1 = min(w[1] for w in ws0); r2 = max(w[3] for w in ws0)
    j = i+1
    while j < len(ls):
        _, wn = ls[j]
        ny1 = min(w[1] for w in wn)
        if ny1-r2 < 18:
            parts.extend(w[4] for w in wn)
            r2 = max(r2, max(w[3] for w in wn)); j += 1
        else: break
    asm.append({"text":" ".join(parts),"y1":r1,"y2":r2,"cy":(r1+r2)//2})
    i = j
print(f"Labels assembled: {len(asm)}")


# ─── 6. Row bounds: midpoints between consecutive CY values ───────────────────
# No horizontal table lines exist — rows separated by whitespace only.
# Use CY midpoints which correctly reflect the visual row cell boundaries.
rows = []
for i, lbl in enumerate(asm):
    cy = lbl["cy"]
    if i == 0:
        ry1 = max(SECT_Y1, cy - (asm[1]["cy"] - cy) // 2) if len(asm) > 1 else SECT_Y1
    else:
        ry1 = (asm[i-1]["cy"] + cy) // 2

    if i == len(asm) - 1:
        ry2 = min(SECT_Y2, cy + (cy - asm[-2]["cy"]) // 2) if len(asm) > 1 else SECT_Y2
    else:
        ry2 = (cy + asm[i+1]["cy"]) // 2

    rows.append({**lbl, "ry1": ry1, "ry2": ry2})

# Print summary
print("\n  #   h    ry1–ry2   label")
for i, r in enumerate(rows):
    print(f"  {i+1:2d}  {r['ry2']-r['ry1']:3d}px  {r['ry1']:4d}–{r['ry2']:4d}  {r['text'][:60]}")


# ─── 7. Build debug image ─────────────────────────────────────────────────────
# Crop to LIGHTING section + enough right margin for labels
PAD = 15
CX1 = max(0, SYM_L - PAD)
CX2 = min(W, DESC_X1 + 900)
CY1 = max(0, SECT_Y1 - 30)
CY2 = min(H, SECT_Y2 + 15)

debug = color[CY1:CY2, CX1:CX2].copy()
dH, dW2 = debug.shape[:2]
dx = lambda x: x - CX1
dy = lambda y: y - CY1

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Draw detected vertical column lines (blue)
for xv in vert_xs:
    if CX1 <= xv <= CX2:
        cv2.line(debug, (dx(xv), 0), (dx(xv), dH-1), (200, 80, 0), 1)

# Draw row boxes and labels
COLORS = [(0, 110, 255), (0, 55, 200)]   # alternating orange shades
for i, row in enumerate(rows):
    bx1, bx2 = dx(SYM_L), dx(SYM_R)
    by1, by2  = dy(row["ry1"]), dy(row["ry2"])
    if by2 <= by1: continue

    col = COLORS[i % 2]
    # Bounding box around the symbol cell
    cv2.rectangle(debug, (bx1, by1), (bx2, by2), col, 1)
    # Row number (small, inside box top-left)
    cv2.putText(debug, f"{i+1}", (bx1+3, by1+11),
                FONT, 0.28, col, 1, cv2.LINE_AA)
    # Row separator tick on right side of box
    mid_y = (by1+by2)//2
    cv2.line(debug, (bx2, mid_y-3), (bx2+4, mid_y), col, 1)
    cv2.line(debug, (bx2, mid_y+3), (bx2+4, mid_y), col, 1)

    # Label text in description column
    label = row["text"][:75]
    cv2.putText(debug, label, (dx(DESC_X1)+4, mid_y+4),
                FONT, 0.31, (0, 165, 0), 1, cv2.LINE_AA)

# Horizontal lines at each row boundary (faint grey tick marks on left edge)
for row in rows:
    yy = dy(row["ry1"])
    cv2.line(debug, (0, yy), (6, yy), (160, 160, 160), 1)

# Header bar with measurements
hdr = (f"LIGHTING  |  symbol col x=[{SYM_L},{SYM_R}]  width={SYM_W}px ({SYM_W/DPI*25.4:.1f}mm)"
       f"  |  {len(rows)} rows  |  NO horizontal table lines — whitespace-separated rows")
cv2.rectangle(debug, (0, 0), (dW2, 22), (255, 255, 255), -1)
cv2.putText(debug, hdr, (4, 15), FONT, 0.34, (0, 0, 150), 1, cv2.LINE_AA)

OUT_DEBUG.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_DEBUG), debug)
print(f"\nSaved: {OUT_DEBUG}  ({dW2}x{dH}px)")
