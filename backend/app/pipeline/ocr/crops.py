from __future__ import annotations

from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


def polygon_to_bbox(polygon: List[List[float]]) -> Optional[Tuple[int, int, int, int]]:
    """Calculates the minimum bounding box for a given polygon."""
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = int(np.floor(min(xs)))
    y0 = int(np.floor(min(ys)))
    x1 = int(np.ceil(max(xs)))
    y1 = int(np.ceil(max(ys)))
    return x0, y0, x1, y1


def tight_crop_from_mask(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    fallback_polygon: List[List[float]],
    padding: int = 6,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extracts a tight, padded crop of text using a mask, with a polygon fallback.
    """
    h, w = image_bgr.shape[:2]

    def _expand(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
        """Applies padding to a bounding box, clamped to image dimensions."""
        return (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(w, x1 + padding),
            min(h, y1 + padding),
        )

    # Primary method: Use the in-memory mask for the tightest crop
    if mask is not None and mask.shape == (h, w):
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            x0, y0, x1, y1 = _expand(x, y, x + bw, y + bh)
            return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

    # Fallback method: Use the polygon if the mask fails
    bbox = polygon_to_bbox(fallback_polygon)
    if bbox is not None:
        x0, y0, x1, y1 = _expand(*bbox)
        return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

    # Final fallback: Return the whole image if all else fails
    return image_bgr.copy(), (0, 0, w, h)