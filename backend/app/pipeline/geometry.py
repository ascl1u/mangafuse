from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


def _largest_rectangle_in_histogram(heights: np.ndarray) -> Tuple[int, int, int]:
    """Return (area, left_idx_inclusive, right_idx_exclusive) for a histogram.

    Uses the classic monotonic stack algorithm in O(n).
    """
    stack: list[int] = []
    max_area = 0
    best_l = 0
    best_r = 0

    # Append sentinel zero to flush the stack
    extended = np.append(heights, 0)
    for i in range(len(extended)):
        h = int(extended[i])
        start = i
        while stack and extended[stack[-1]] > h:
            top = stack.pop()
            height = int(extended[top])
            # width is from the element after the new top to current i
            left = stack[-1] + 1 if stack else 0
            width = i - left
            area = height * width
            if area > max_area:
                max_area = area
                best_l = left
                best_r = i
            start = left
        stack.append(i)
    return max_area, best_l, best_r


def _rasterize_polygons_to_mask(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    crop_x0: int,
    crop_y0: int,
    crop_x1: int,
    crop_y1: int,
) -> np.ndarray:
    """Rasterize polygons into a binary mask for the specified crop region.

    Args:
        polygons: collection of polygons; each polygon is a sequence of (x, y).
        crop_x0..crop_y1: crop bounds in page coordinates.

    Returns:
        mask: uint8 array of shape (crop_h, crop_w) with values {0, 1}.
    """
    import cv2  # lazy import

    crop_w = max(0, int(crop_x1 - crop_x0))
    crop_h = max(0, int(crop_y1 - crop_y0))
    if crop_w == 0 or crop_h == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    pts_list = []
    for poly in polygons:
        if not poly:
            continue
        pts = np.array([[int(round(x - crop_x0)), int(round(y - crop_y0))] for x, y in poly], dtype=np.int32)
        if pts.shape[0] >= 3:
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)
    return mask


def compute_inner_rect_axis_aligned(
    polygon: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
    margin_px: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Compute the largest axis-aligned rectangle fully inside a polygon.

    The polygon is eroded by ``margin_px`` on all sides before computing the
    rectangle, effectively enforcing a margin of ~1.0×margin_px along borders.

    Args:
        polygon: list of [x, y] points in page coordinates.
        image_size: (width, height) of the page image.
        margin_px: integer margin (pixels) to shrink the usable region.

    Returns:
        (x0, y0, x1, y1) rectangle in page coordinates, or None when no area.
    """
    # Defensive checks
    if not polygon or len(polygon) < 3:
        return None
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        return None

    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Shapely is required for compute_inner_rect_axis_aligned. Install AI deps from backend/requirements-ai.txt"
        ) from exc

    # Build polygon and perform an inward buffer (erosion)
    poly = Polygon([(float(x), float(y)) for x, y in polygon])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None

    try:
        eroded = poly.buffer(-float(max(0, int(margin_px))), join_style=2)
    except Exception:
        eroded = poly

    if eroded.is_empty:
        return None

    # Collect polygons (handle MultiPolygon from erosion split)
    polys: Iterable[Polygon]
    if isinstance(eroded, MultiPolygon):
        polys = [p for p in eroded.geoms if not p.is_empty]
    else:
        polys = [eroded]
    if not polys:
        return None

    # Compute crop bounds from the union bbox, clamped to image bounds
    minx = min(int(np.floor(p.bounds[0])) for p in polys)
    miny = min(int(np.floor(p.bounds[1])) for p in polys)
    maxx = max(int(np.ceil(p.bounds[2])) for p in polys)
    maxy = max(int(np.ceil(p.bounds[3])) for p in polys)
    crop_x0 = max(0, min(width, minx))
    crop_y0 = max(0, min(height, miny))
    crop_x1 = max(0, min(width, maxx))
    crop_y1 = max(0, min(height, maxy))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return None

    # Rasterize eroded polygon(s) inside the crop
    mask = _rasterize_polygons_to_mask([list(p.exterior.coords) for p in polys], crop_x0, crop_y0, crop_x1, crop_y1)
    if mask.size == 0 or int(mask.sum()) == 0:
        return None

    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_coords = (0, 0, 0, 0)

    for row in range(h):
        # Update histogram of consecutive 1s
        heights = heights + mask[row, :].astype(np.int32)
        heights = heights * mask[row, :].astype(np.int32)

        area, l, r = _largest_rectangle_in_histogram(heights)
        if area > 0:
            # Height is the minimal height within [l, r)
            rect_height = int(np.min(heights[l:r])) if r > l else 0
            if rect_height > 0:
                # The rectangle's bottom is at current row (inclusive)
                y1 = row + 1
                y0 = y1 - rect_height
                x0 = l
                x1 = r
                # Map back to page coords
                px0 = crop_x0 + x0
                py0 = crop_y0 + y0
                px1 = crop_x0 + x1
                py1 = crop_y0 + y1
                cur_area = (px1 - px0) * (py1 - py0)
                if cur_area > best_area:
                    best_area = cur_area
                    best_coords = (px0, py0, px1, py1)

    if best_area <= 0:
        return None
    return best_coords


