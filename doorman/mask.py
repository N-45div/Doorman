"""Paint text out of a frame, for the text-blind second look."""

from __future__ import annotations

import io
from statistics import median

from PIL import Image, ImageDraw

Box = tuple[float, float, float, float]  # left, top, width, height as fractions of the image


def mask_text(image: bytes, boxes: list[Box], pad: float = 0.004) -> bytes:
    """Paint over every text region with the colour around it.

    A real object survives: its label goes blank, the box is still there. An
    object that only existed in words does not. The fill matters: a flat grey
    block over a note was itself read as a box on the door, so each region is
    filled with the median colour of a ring of pixels just outside it.
    Returns a JPEG.
    """
    img = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = img.size
    px = img.load()
    draw = ImageDraw.Draw(img)
    for left, top, width, height in boxes:
        x0, y0 = max(0, int((left - pad) * w)), max(0, int((top - pad) * h))
        x1, y1 = min(w - 1, int((left + width + pad) * w)), min(h - 1, int((top + height + pad) * h))
        if x1 <= x0 or y1 <= y0:
            continue
        ring = [(x, y) for x in range(x0, x1 + 1, 2) for y in (max(0, y0 - 3), min(h - 1, y1 + 3))]
        ring += [(x, y) for y in range(y0, y1 + 1, 2) for x in (max(0, x0 - 3), min(w - 1, x1 + 3))]
        fill = tuple(int(median(px[p][c] for p in ring)) for c in range(3))
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()
