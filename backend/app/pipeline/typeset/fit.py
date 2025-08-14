from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import ImageFont  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from app.pipeline.typeset.layout import compute_optimal_layout
from app.pipeline.typeset.model import LayoutResult


def find_optimal_font_size(
    text: str,
    polygon: Polygon,
    font_path: Path,
    font_cache: Dict[int, ImageFont.FreeTypeFont],
    min_font_size: int = 6,
    max_font_size: int = 64,
) -> Tuple[int, Optional[LayoutResult]]:
    """
    Finds the optimal font size using a binary search and a font cache.
    """
    low = int(min_font_size)
    high = int(max_font_size)
    optimal_size = low
    optimal_layout: Optional[LayoutResult] = None

    while low <= high:
        mid = (low + high) // 2
        if mid <= 0:
            break

        if mid not in font_cache:
            try:
                font_cache[mid] = ImageFont.truetype(str(font_path), mid)
            except IOError:
                # This can happen if a font doesn't support a specific size.
                # Treat it as a failure for this size.
                high = mid - 1
                continue
        font = font_cache[mid]

        layout = compute_optimal_layout(text, polygon, font)

        if layout is not None:
            # This size works, try a larger one
            optimal_size = mid
            optimal_layout = layout
            low = mid + 1
        else:
            # This size is too big, try a smaller one
            high = mid - 1

    return optimal_size, optimal_layout