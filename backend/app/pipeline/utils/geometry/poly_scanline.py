from __future__ import annotations

from typing import List, Optional, Tuple

from shapely.geometry import LineString, MultiLineString, Polygon  # type: ignore


def get_polygon_bounds_at_y(polygon: Polygon, y: float) -> Tuple[Optional[float], Optional[float]]:
    min_x, _min_y, max_x, _max_y = polygon.bounds
    horizontal_line = LineString([(min_x - 10.0, y), (max_x + 10.0, y)])
    inter = polygon.intersection(horizontal_line)
    if inter.is_empty:
        return None, None
    if isinstance(inter, LineString):
        xs = [pt[0] for pt in inter.coords]
        return (min(xs), max(xs))
    if isinstance(inter, MultiLineString):
        xs: List[float] = []
        for ln in inter.geoms:
            xs.extend(pt[0] for pt in ln.coords)
        return (min(xs), max(xs)) if xs else (None, None)
    return None, None