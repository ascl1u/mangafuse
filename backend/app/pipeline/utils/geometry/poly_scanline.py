from __future__ import annotations

from typing import List, Optional, Tuple

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon  # type: ignore


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


def get_polygon_intervals_at_y(polygon: Polygon, y: float) -> List[Tuple[float, float]]:
    min_x, _min_y, max_x, _max_y = polygon.bounds
    horizontal_line = LineString([(min_x - 10.0, y), (max_x + 10.0, y)])
    inter = polygon.intersection(horizontal_line)
    intervals: List[Tuple[float, float]] = []

    def _push(line: LineString) -> None:
        coords = list(line.coords)
        if len(coords) < 2:
            return
        xs = [pt[0] for pt in coords]
        left = min(xs)
        right = max(xs)
        if right > left:
            intervals.append((left, right))

    if inter.is_empty:
        return []
    if isinstance(inter, LineString):
        _push(inter)
    elif isinstance(inter, MultiLineString):
        for ln in inter.geoms:
            _push(ln)
    elif isinstance(inter, GeometryCollection):
        for g in inter.geoms:
            if isinstance(g, LineString):
                _push(g)
            elif isinstance(g, MultiLineString):
                for ln in g.geoms:
                    _push(ln)

    if not intervals:
        return []
    intervals.sort(key=lambda t: t[0])
    merged: List[Tuple[float, float]] = []
    cur_l, cur_r = intervals[0]
    eps = 1e-9
    for l, r in intervals[1:]:
        if l <= cur_r + eps:
            cur_r = max(cur_r, r)
        else:
            if cur_r - cur_l > eps:
                merged.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    if cur_r - cur_l > eps:
        merged.append((cur_l, cur_r))
    return merged


