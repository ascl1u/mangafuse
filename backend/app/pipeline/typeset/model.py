from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BubbleText:
    bubble_id: int
    polygon: List[List[float]]
    text: str
    font_size: Optional[int] = None


@dataclass
class LayoutResult:
    lines: List[str]
    baselines_y: List[float]
    centers_x: List[float]
    x_limits_per_line: List[Tuple[float, float]]
    y_limit: Tuple[float, float]
    demerit: float


