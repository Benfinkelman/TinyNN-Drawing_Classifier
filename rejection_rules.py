from __future__ import annotations
import numpy as np

def ink_stats(x: np.ndarray) -> dict:
    """
    x: (H,W) numpy array. Ink is defined as x > 0.
    Returns: ink_sum, bbox_area, fill_ratio, aspect_ratio, bbox
    """
    x = np.asarray(x)
    ink = (x > 0)

    ink_sum = int(ink.sum())
    if ink_sum == 0:
        return {
            "ink_sum": 0,
            "bbox_area": 0,
            "fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "bbox": None,
        }

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])

    height = (r1 - r0 + 1)
    width = (c1 - c0 + 1)
    bbox_area = int(height * width)

    fill_ratio = float(ink_sum / bbox_area) if bbox_area > 0 else 0.0
    aspect_ratio = float(width / height) if height > 0 else 0.0

    return {
        "ink_sum": ink_sum,
        "bbox_area": bbox_area,
        "fill_ratio": fill_ratio,
        "aspect_ratio": aspect_ratio,
        "bbox": (r0, r1, c0, c1),
    }

def reject_reason(x: np.ndarray, *, min_ink: int = 60, max_ink: int = 2000, min_bbox_area: int = 120, min_fill_ratio: float = 0.12, aspect_ratio_limit: float = 8.0,) -> tuple[bool, str, dict]:
    """
    Returns (ok, reason, stats).
    ok=False => don't classify / don't save.
    """
    s = ink_stats(x)

    if s["ink_sum"] < min_ink:
        return False, f"Not enough ink (need ≥ {min_ink} ink pixels).", s

    if s["ink_sum"] > max_ink:
        return False, "Too much ink / too filled — try a cleaner sample.", s

    if s["bbox_area"] < min_bbox_area:
        return False, "Drawing is too small — make it larger.", s

    # Line-like: sparse fill inside bbox
    if s["fill_ratio"] < min_fill_ratio:
        return False, "Looks like a thin line / not a valid sample — draw a fuller shape.", s

    # Optional: extreme skinny bbox
    ar = s["aspect_ratio"]
    if ar > aspect_ratio_limit or (ar > 0 and (1 / ar) > aspect_ratio_limit):
        return False, "Too elongated (line-like). Draw a less skinny shape.", s

    return True, "", s
