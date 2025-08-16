from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import ImageFont  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from app.pipeline.typeset.model import LayoutResult
from app.pipeline.utils.geometry.axis_rect import compute_inner_rect_axis_aligned
from app.pipeline.utils.geometry.poly_scanline import get_polygon_bounds_at_y


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
    token_widths = [float(font.getbbox(tok)[2] - font.getbbox(tok)[0]) for tok in tokens]
    space_width = float(font.getbbox(" ")[2] - font.getbbox(" ")[0])
    return token_widths, space_width


def _build_glue_widths(space_width: float, token_word_ids: List[int]) -> List[float]:
    glue: List[float] = []
    for i in range(len(token_word_ids) - 1):
        glue.append(0.0 if token_word_ids[i] == token_word_ids[i + 1] else space_width)
    return glue


def _prefix_sums(values: List[float]) -> List[float]:
    out = [0.0] * (len(values) + 1)
    for i, v in enumerate(values):
        out[i + 1] = out[i] + v
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

    return (eff_left, eff_right) if eff_right > eff_left else (0.0, 0.0)


def compute_optimal_layout(
    text: str,
    polygon: Polygon,
    font: ImageFont.FreeTypeFont,
    *,
    line_spacing: float = 1.0,
    penalty_ragged: float = 50.0,
    penalty_hyphen_break: float = 50.0,
    penalty_widow: float = 100.0,
) -> Optional[LayoutResult]:
    if not text or not text.strip():
        return LayoutResult(lines=[], baselines_y=[], centers_x=[], x_limits_per_line=[], y_limit=(0.0, 0.0), demerit=0.0)

    ascent, descent = font.getmetrics()
    line_height = float(ascent + descent) * line_spacing

    min_x, min_y, max_x, max_y = polygon.bounds
    em_pad = float(getattr(font, "size", 0)) * 0.5
    image_size_for_rect = (max_x, max_y)
    
    core_coords = compute_inner_rect_axis_aligned(list(polygon.exterior.coords), image_size_for_rect, margin_px=int(em_pad))
    
    if core_coords:
        x_left, y_top, x_right, y_bottom = core_coords
        y_top_padded, y_bottom_padded = float(y_top), float(y_bottom)
        x_limit = (float(x_left), float(x_right))
    else: # Fallback to polygon bounds
        y_top_padded, y_bottom_padded = float(min_y), float(max_y)
        x_limit = None

    available_height = y_bottom_padded - y_top_padded
    if available_height <= line_height:
        return None

    rep_bbox = font.getbbox("Ay")
    top_offset, bottom_offset = float(rep_bbox[1]), float(rep_bbox[3])
    max_lines = int((available_height - ascent - bottom_offset) / line_height) + 1
    if max_lines < 1:
        return None

    tokens, token_word_ids, token_has_trailing_hyphen = _tokenize_with_hyphens(text)
    N = len(tokens)
    if N == 0:
        return LayoutResult(lines=[], baselines_y=[], centers_x=[], x_limits_per_line=[], y_limit=(0.0, 0.0), demerit=0.0)

    token_widths, space_width = _measurements_for_tokens(font, tokens)
    glue = _build_glue_widths(space_width, token_word_ids)
    pref_tokens = _prefix_sums(token_widths)
    pref_glue = _prefix_sums(glue)

    # --- Dynamic Programming to find optimal line breaks ---
    INF = 1e18
    dp = [[INF] * (N + 1) for _ in range(max_lines)]
    prev = [[-1] * (N + 1) for _ in range(max_lines)]
    dp[0][0] = 0.0

    for k in range(max_lines):
        baseline_y = y_top_padded + ascent + k * line_height
        left_k, right_k = _line_band_bounds(polygon, baseline_y, top_offset, bottom_offset, x_limit)
        Wk = right_k - left_k
        if Wk <= 0: continue

        for j in range(1, N + 1):
            best_cost, best_i = INF, -1
            for i in range(j - 1, -1, -1):
                if k > 0 and dp[k - 1][i] >= INF / 2:
                    continue
                if k == 0 and i > 0:
                    break
                
                line_w = _segment_width(i, j, pref_tokens, pref_glue)
                if line_w > Wk:
                    if i > 0 and _segment_width(i-1, j, pref_tokens, pref_glue) > Wk:
                        break # Optimization: line is already too long
                    continue
                
                slack = Wk - line_w
                rag_cost = (slack / max(Wk, 1e-6)) ** 2 * penalty_ragged
                hyphen_cost = penalty_hyphen_break if token_has_trailing_hyphen[j - 1] else 0.0
                cost = rag_cost + hyphen_cost

                base_cost = dp[k - 1][i] if k > 0 else 0.0
                candidate = base_cost + cost

                if candidate < best_cost:
                    best_cost, best_i = candidate, i
            dp[k][j], prev[k][j] = best_cost, best_i

    # Find best solution and apply widow penalty
    best_total, best_L = INF, -1
    for L in range(1, max_lines + 1):
        total = dp[L - 1][N]
        if total >= INF / 2: continue
        
        i_last = prev[L - 1][N]
        if i_last == -1: continue

        last_word_ids = set(token_word_ids[i_last:N])
        if len(last_word_ids) <= 1 and L >= 2:
            total += penalty_widow

        if total < best_total:
            best_total, best_L = total, L
    
    if best_L == -1: return None

    # Backtrack to find line breaks
    cuts, k, j = [N], best_L - 1, N
    while k >= 0 and j > 0:
        i = prev[k][j]
        if i == -1: break
        cuts.append(i)
        j, k = i, k - 1
    cuts.sort()

    # Build final lines and calculate their positions
    lines: List[str] = []
    for idx in range(len(cuts) - 1):
        i, j = cuts[idx], cuts[idx + 1]
        line_tokens = tokens[i:j]
        line_word_ids = token_word_ids[i:j]
        parts = [line_tokens[0]]
        for t in range(1, len(line_tokens)):
            if line_word_ids[t - 1] != line_word_ids[t]:
                parts.append(" ")
            parts.append(line_tokens[t])
        lines.append("".join(parts))

    total_height = best_L * line_height
    y_center = (y_top_padded + y_bottom_padded) / 2.0
    y_start = y_center - total_height / 2.0 + (line_height - (ascent + descent))/2.0
    
    baselines_y, centers_x, x_limits_per_line = [], [], []
    for idx, line in enumerate(lines):
        baseline_y = y_start + ascent + idx * line_height
        lb = font.getbbox(line)
        line_top_offset, line_bottom_offset = float(lb[1]), float(lb[3])
        left_k, right_k = _line_band_bounds(polygon, baseline_y, line_top_offset, line_bottom_offset, x_limit)
        
        baselines_y.append(baseline_y)
        centers_x.append((left_k + right_k) / 2.0)
        x_limits_per_line.append((left_k, right_k))

    return LayoutResult(lines=lines, baselines_y=baselines_y, centers_x=centers_x, x_limits_per_line=x_limits_per_line, y_limit=(y_top_padded, y_bottom_padded), demerit=best_total)