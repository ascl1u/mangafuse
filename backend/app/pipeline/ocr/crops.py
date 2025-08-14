from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np


def polygon_to_bbox(polygon: List[List[float]]) -> Optional[Tuple[int, int, int, int]]:
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
    mask_path: Path,
    fallback_polygon: List[List[float]],
    padding: int = 6,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    h, w = image_bgr.shape[:2]

    def _expand(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
        return (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(w, x1 + padding),
            min(h, y1 + padding),
        )

    if mask is not None and mask.shape == (h, w):
        coords = cv2.findNonZero((mask > 0).astype(np.uint8))
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            x0, y0, x1, y1 = _expand(x, y, x + bw, y + bh)
            return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

    bbox = polygon_to_bbox(fallback_polygon)
    if bbox is not None:
        x0, y0, x1, y1 = _expand(*bbox)
        return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

    return image_bgr.copy(), (0, 0, w, h)


