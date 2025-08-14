from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from shapely.geometry import Polygon  # type: ignore

from app.pipeline.utils.geometry.poly_scanline import get_polygon_intervals_at_y


class CoreAreaResult:
    def __init__(
        self,
        rectangle_points: List[Tuple[float, float]],
        width: float,
        height: float,
        area: float,
        y_range: Tuple[float, float],
        x_interval: Tuple[float, float],
        params: Dict[str, float],
        error: Optional[str] = None,
    ) -> None:
        self.rectangle_points = rectangle_points
        self.width = width
        self.height = height
        self.area = area
        self.y_range = y_range
        self.x_interval = x_interval
        self.params = params
        self.error = error


def _normalize_polygon(poly: Polygon) -> Optional[Polygon]:
    if not poly.is_valid or poly.is_empty or poly.area <= 0:
        fixed = poly.buffer(0)
        try:
            from shapely.geometry import MultiPolygon as _MP  # type: ignore
        except Exception:
            return None
        if isinstance(fixed, Polygon):
            return fixed
        if isinstance(fixed, _MP):
            geoms = list(fixed.geoms)
            return max(geoms, key=lambda p: p.area) if geoms else None
        return None
    return poly


def compute_core_typable_area(polygon: Polygon, num_samples: int = 2400) -> CoreAreaResult:
    poly = _normalize_polygon(polygon)
    if poly is None:
        return CoreAreaResult([], 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), {"num_samples": float(num_samples)}, "Invalid or empty polygon")
    min_x, min_y, max_x, max_y = poly.bounds
    if num_samples < 2 or max_y <= min_y:
        return CoreAreaResult([], 0.0, 0.0, 0.0, (min_y, min_y), (min_x, min_x), {"num_samples": float(num_samples)}, "Insufficient sampling or degenerate vertical bounds")

    dy = (max_y - min_y) / float(max(1, num_samples - 1))
    y_values = [min_y + i * dy for i in range(num_samples)]

    y_valid: List[float] = []
    intervals_by_row: List[List[Tuple[float, float]]] = []
    for y in y_values:
        intervals = get_polygon_intervals_at_y(poly, y)
        if intervals:
            y_valid.append(y)
            intervals_by_row.append(intervals)
    n = len(y_valid)
    if n == 0:
        return CoreAreaResult([], 0.0, 0.0, 0.0, (min_y, min_y), (min_x, min_x), {"num_samples": float(num_samples)}, "No valid scanlines inside polygon")

    def intersect_sets(A: List[Tuple[float, float]], B: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        i = 0
        j = 0
        out: List[Tuple[float, float]] = []
        while i < len(A) and j < len(B):
            aL, aR = A[i]
            bL, bR = B[j]
            L = max(aL, bL)
            R = min(aR, bR)
            if R > L:
                out.append((L, R))
            if aR < bR:
                i += 1
            else:
                j += 1
        return out

    best_area = 0.0
    best_s = 0
    best_e = 0
    best_interval = (0.0, 0.0)
    for s in range(n):
        active = intervals_by_row[s]
        y_top = y_valid[s]
        for e in range(s, n):
            if e > s:
                active = intersect_sets(active, intervals_by_row[e])
                if not active:
                    break
            y_bottom = y_valid[e]
            height = y_bottom - y_top
            if height <= 0:
                continue
            widest = max(active, key=lambda t: t[1] - t[0])
            width = widest[1] - widest[0]
            area = width * height
            if area > best_area:
                best_area = area
                best_s = s
                best_e = e
                best_interval = widest

    if best_area <= 0:
        return CoreAreaResult([], 0.0, 0.0, 0.0, (min_y, min_y), (min_x, min_x), {"num_samples": float(num_samples)}, "Failed to find positive-area rectangle")

    y_top = y_valid[best_s]
    y_bottom = y_valid[best_e]
    x_left, x_right = best_interval
    rect_points = [(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]
    return CoreAreaResult(
        rectangle_points=rect_points,
        width=max(0.0, x_right - x_left),
        height=max(0.0, y_bottom - y_top),
        area=max(0.0, (x_right - x_left) * (y_bottom - y_top)),
        y_range=(y_top, y_bottom),
        x_interval=(x_left, x_right),
        params={"num_samples": float(num_samples)},
        error=None,
    )


