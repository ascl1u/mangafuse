from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import ImageFont  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from app.pipeline.typeset.layout import compute_optimal_layout
from app.pipeline.typeset.model import LayoutResult


def find_optimal_font_size(
    text: str,
    polygon: Polygon,
    font_path: Path,
    min_font_size: int = 6,
    max_font_size: int = 64,
) -> Tuple[int, Optional[LayoutResult]]:
    low = int(min_font_size)
    high = int(max_font_size)
    optimal_size = low
    optimal_layout: Optional[LayoutResult] = None
    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype(str(font_path), mid)
        layout = compute_optimal_layout(text, polygon, font)
        if layout is not None:
            optimal_size = mid
            optimal_layout = layout
            low = mid + 1
        else:
            high = mid - 1
    return optimal_size, optimal_layout


