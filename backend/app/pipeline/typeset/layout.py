from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import ImageFont  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from app.pipeline.typeset.model import LayoutResult
from app.pipeline.utils.geometry.typable_area import compute_core_typable_area
from app.pipeline.utils.geometry.poly_scanline import (
    get_polygon_bounds_at_y,
    get_polygon_intervals_at_y,
)


def _tokenize_with_hyphens(text: str) -> Tuple[List[str], List[int], List[bool]]:
    tokens: List[str] = []
    token_word_ids: List[int] = []
    token_has_trailing_hyphen: List[bool] = []
    words = text.split()
    for word_idx, word in enumerate(words):
        parts = word.split("-")
        if len(parts) == 1:
            tokens.append(parts[0])
            token_word_ids.append(word_idx)
            token_has_trailing_hyphen.append(False)
        else:
            for p_idx, part in enumerate(parts):
                if p_idx < len(parts) - 1:
                    tokens.append(f"{part}-")
                    token_word_ids.append(word_idx)
                    token_has_trailing_hyphen.append(True)
                else:
                    tokens.append(part)
                    token_word_ids.append(word_idx)
                    token_has_trailing_hyphen.append(False)
    return tokens, token_word_ids, token_has_trailing_hyphen


def _measurements_for_tokens(font: ImageFont.FreeTypeFont, tokens: List[str]) -> Tuple[List[float], float]:
    token_widths: List[float] = []
    for tok in tokens:
        bbox = font.getbbox(tok)
        token_widths.append(float(bbox[2] - bbox[0]))
    space_bbox = font.getbbox(" ")
    space_width = float(space_bbox[2] - space_bbox[0])
    return token_widths, space_width


def _build_glue_widths(space_width: float, token_word_ids: List[int]) -> List[float]:
    glue: List[float] = []
    for i in range(len(token_word_ids) - 1):
        glue.append(0.0 if token_word_ids[i] == token_word_ids[i + 1] else space_width)
    return glue


def _prefix_sums(values: List[float]) -> List[float]:
    out = [0.0]
    for v in values:
        out.append(out[-1] + v)
    return out


def _segment_width(i: int, j: int, pref_tokens: List[float], pref_glue: List[float]) -> float:
    token_sum = pref_tokens[j] - pref_tokens[i]
    if j - i <= 1:
        return token_sum
    glue_sum = pref_glue[j - 1] - pref_glue[i]
    return token_sum + glue_sum


def _line_band_bounds(
    polygon: Polygon,
    baseline_y: float,
    top_offset: float,
    bottom_offset: float,
    x_limit: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    text_top_y = baseline_y + top_offset
    text_bottom_y = baseline_y + bottom_offset
    tl, tr = get_polygon_bounds_at_y(polygon, text_top_y)
    bl, br = get_polygon_bounds_at_y(polygon, text_bottom_y)
    if tl is None or bl is None or tr is None or br is None:
        return 0.0, 0.0
    eff_left = max(tl, bl)
    eff_right = min(tr, br)
    if x_limit is not None:
        eff_left = max(eff_left, x_limit[0])
        eff_right = min(eff_right, x_limit[1])
    if eff_right <= eff_left:
        return 0.0, 0.0
    return eff_left, eff_right


def compute_optimal_layout(
    text: str,
    polygon: Polygon,
    font: ImageFont.FreeTypeFont,
    *,
    line_spacing: float = 1.0,
    penalty_ragged: float = 100.0,
    penalty_hyphen_break: float = 500.0,
    penalty_widow: float = 2000.0,
) -> Optional[LayoutResult]:
    if not text or not text.strip():
        ascent, descent = font.getmetrics()
        return LayoutResult(lines=[], baselines_y=[], centers_x=[], x_limits_per_line=[], y_limit=(0.0, 0.0), demerit=0.0)

    ascent, descent = font.getmetrics()
    line_height = float(ascent + descent) * max(0.1, float(line_spacing))

    core = compute_core_typable_area(polygon)
    if core and not core.error and core.width > 0 and core.height > 0:
        y_top, y_bottom = core.y_range
        x_left, x_right = core.x_interval
        em_pad = max(0.0, float(getattr(font, "size", 0)) * 1.2)
        y_top_padded = y_top + em_pad
        y_bottom_padded = y_bottom - em_pad
        x_left_padded = x_left + em_pad
        x_right_padded = x_right - em_pad
        x_limit = (x_left_padded, x_right_padded)
    else:
        min_x, min_y, max_x, max_y = polygon.bounds
        y_top_padded, y_bottom_padded = float(min_y), float(max_y)
        x_limit = None

    available_height = y_bottom_padded - y_top_padded
    if available_height <= 0:
        return None

    rep_bbox = font.getbbox("Ay")
    top_offset = float(rep_bbox[1])
    bottom_offset = float(rep_bbox[3])

    max_lines_float = (available_height - ascent - bottom_offset) / line_height + 1.0
    max_lines = int(max_lines_float) if max_lines_float > 0 else 0
    if max_lines < 1:
        return None

    tokens, token_word_ids, token_has_trailing_hyphen = _tokenize_with_hyphens(text)
    N = len(tokens)
    if N == 0:
        return LayoutResult(
            lines=[],
            baselines_y=[],
            centers_x=[],
            x_limits_per_line=[],
            y_limit=(y_top_padded, y_bottom_padded),
            demerit=0.0,
        )
    token_widths, space_width = _measurements_for_tokens(font, tokens)
    glue = _build_glue_widths(space_width, token_word_ids)
    pref_tokens = _prefix_sums(token_widths)
    pref_glue = _prefix_sums(glue)

    longest = max(token_widths)
    probes = {0, max(0, max_lines // 2), max_lines - 1}
    can_fit = False
    for k in probes:
        baseline_y = y_top_padded + ascent + k * line_height
        left_k, right_k = _line_band_bounds(polygon, baseline_y, top_offset, bottom_offset, x_limit)
        if right_k - left_k >= longest:
            can_fit = True
            break
    if not can_fit:
        return None

    INF = 1e18
    dp = [[INF] * (N + 1) for _ in range(max_lines)]
    prev: List[List[int]] = [[-1] * (N + 1) for _ in range(max_lines)]
    dp[0][0] = 0.0

    def cost_line(i: int, j: int, k: int) -> float:
        lw = _segment_width(i, j, pref_tokens, pref_glue)
        baseline_y = y_top_padded + ascent + k * line_height
        left_k, right_k = _line_band_bounds(polygon, baseline_y, top_offset, bottom_offset, x_limit)
        Wk = right_k - left_k
        if Wk <= 0 or lw > Wk:
            return INF
        slack = Wk - lw
        rag = (slack / max(Wk, 1e-6)) ** 2 * penalty_ragged
        hyph = penalty_hyphen_break if token_has_trailing_hyphen[j - 1] else 0.0
        return rag + hyph

    for k in range(max_lines):
        for j in range(1, N + 1):
            best_cost = dp[k][j]
            best_i = prev[k][j]
            for i in range(j - 1, -1, -1):
                if k == 0 and i != 0:
                    continue
                base = 0.0 if (k == 0 and i == 0) else (dp[k - 1][i] if k > 0 else dp[k][0])
                if base >= INF / 2:
                    continue
                c = cost_line(i, j, k)
                if c >= INF / 2:
                    break
                candidate = base + c
                if candidate < best_cost:
                    best_cost = candidate
                    best_i = i
            dp[k][j] = best_cost
            prev[k][j] = best_i

    best_total = INF
    best_L = -1
    for L in range(1, max_lines + 1):
        total = dp[L - 1][N]
        if total >= INF / 2:
            continue
        i_last = prev[L - 1][N]
        if i_last is None or i_last < 0:
            continue
        last_word_ids = set(token_word_ids[i_last:N])
        if len(last_word_ids) == 1 and L >= 5:
            total += penalty_widow
        if total < best_total:
            best_total = total
            best_L = L
    if best_L == -1:
        return None

    cuts: List[int] = [N]
    k = best_L - 1
    j = N
    while k >= 0 and j > 0:
        i = prev[k][j]
        if i is None or i < 0:
            break
        cuts.append(i)
        j = i
        k -= 1
    cuts.sort()

    lines: List[str] = []
    for idx in range(len(cuts) - 1):
        i = cuts[idx]
        j = cuts[idx + 1]
        if i >= j:
            continue
        parts: List[str] = []
        parts.append(tokens[i])
        for t in range(i + 1, j):
            if token_word_ids[t - 1] != token_word_ids[t]:
                parts.append(" ")
            parts.append(tokens[t])
        lines.append("".join(parts))

    total_height = best_L * line_height
    y_center = (y_top_padded + y_bottom_padded) / 2.0
    y_start = y_center - total_height / 2.0
    baselines_y: List[float] = []
    centers_x: List[float] = []
    x_limits_per_line: List[Tuple[float, float]] = []

    for idx, line in enumerate(lines):
        baseline_y = y_start + ascent + idx * line_height
        lb = font.getbbox(line)
        line_top_offset = float(lb[1])
        line_bottom_offset = float(lb[3])
        left_k, right_k = _line_band_bounds(polygon, baseline_y, line_top_offset, line_bottom_offset, x_limit)
        if right_k - left_k <= 0:
            return None
        if not is_line_inside_polygon(line, baseline_y, font, polygon, (left_k, right_k), (y_top_padded, y_bottom_padded)):
            return None
        baselines_y.append(baseline_y)
        centers_x.append((left_k + right_k) / 2.0)
        x_limits_per_line.append((left_k, right_k))

    return LayoutResult(
        lines=lines,
        baselines_y=baselines_y,
        centers_x=centers_x,
        x_limits_per_line=x_limits_per_line,
        y_limit=(y_top_padded, y_bottom_padded),
        demerit=best_total,
    )


def is_line_inside_polygon(
    line_text: str,
    baseline_y: float,
    font: ImageFont.FreeTypeFont,
    polygon: Polygon,
    x_limit: Optional[Tuple[float, float]],
    y_limit: Optional[Tuple[float, float]],
) -> bool:
    if not line_text or not line_text.strip():
        return True
    bbox = font.getbbox(line_text)
    est_top_offset = float(bbox[1])
    est_bottom_offset = float(bbox[3])
    text_top_y = baseline_y + est_top_offset
    text_bottom_y = baseline_y + est_bottom_offset
    tl, tr = get_polygon_bounds_at_y(polygon, text_top_y)
    bl, br = get_polygon_bounds_at_y(polygon, text_bottom_y)
    if tl is None or bl is None or tr is None or br is None:
        return False
    if y_limit is not None:
        y_min, y_max = y_limit
        if text_top_y < y_min or text_bottom_y > y_max:
            return False
    eff_left = max(tl, bl)
    eff_right = min(tr, br)
    if x_limit is not None:
        eff_left = max(eff_left, x_limit[0])
        eff_right = min(eff_right, x_limit[1])
    eff_width = eff_right - eff_left
    if eff_width <= 0:
        return False
    line_width = float(bbox[2] - bbox[0])
    if line_width > eff_width:
        return False
    center_x = eff_left + eff_width / 2.0
    baseline_x = center_x - line_width / 2.0
    l = baseline_x + bbox[0]
    t = baseline_y + bbox[1]
    r = baseline_x + bbox[2]
    b = baseline_y + bbox[3]
    text_box = Polygon([(l, t), (r, t), (r, b), (l, b)])
    return polygon.contains(text_box)


