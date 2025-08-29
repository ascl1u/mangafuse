from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont  # type: ignore
from shapely.geometry import Polygon, MultiPolygon  # type: ignore

from app.pipeline.typeset.fit import find_optimal_font_size
from app.pipeline.typeset.layout import compute_optimal_layout
from app.pipeline.typeset.model import BubbleText, LayoutResult


def _render_layout(
    canvas: Image.Image,
    font: ImageFont.FreeTypeFont,
    layout: LayoutResult,
    polygon_points: List[List[float]],
    text_layer: Optional[Image.Image],
    *,
    write_to_base: bool = True,
) -> None:
    ascent, _ = font.getmetrics()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    for line, base_y, center_x in zip(layout.lines, layout.baselines_y, layout.centers_x):
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = int(round(center_x - w / 2.0))
        y = int(round(base_y - ascent))
        draw_overlay.text((x, y), line, fill=(0, 0, 0, 255), font=font)

    mask = Image.new("L", canvas.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    pts = [(int(p[0]), int(p[1])) for p in polygon_points]
    if len(pts) >= 3:
        mask_draw.polygon(pts, fill=255)

    alpha = overlay.split()[3]
    combined_alpha = ImageChops.multiply(alpha, mask)
    overlay.putalpha(combined_alpha)
    
    if write_to_base:
        canvas.paste(overlay, (0, 0), overlay)
    if text_layer is not None:
        text_layer.paste(overlay, (0, 0), overlay)


def render_typeset(
    cleaned_image_bgr: np.ndarray,
    output_final_path: Path,
    records: List[BubbleText],
    font_path: Path,
    debug: bool = False,
    debug_overlay_path: Optional[Path] = None,
    text_layer_output_path: Optional[Path] = None,
    existing_text_layer_path: Optional[Path] = None,
    edited_bubble_ids: Optional[List[int]] = None,
) -> Tuple[dict, dict]:
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    base = Image.fromarray(cv2.cvtColor(cleaned_image_bgr, cv2.COLOR_BGR2RGB))

    debug_img = base.copy() if debug else None
    debug_draw = ImageDraw.Draw(debug_img) if debug_img else None
    # Initialize or load text layer if requested
    text_layer_img: Optional[Image.Image]
    if text_layer_output_path is not None or (existing_text_layer_path is not None and existing_text_layer_path.exists()):
        if existing_text_layer_path is not None and existing_text_layer_path.exists():
            text_layer_img = Image.open(str(existing_text_layer_path)).convert("RGBA")
        else:
            text_layer_img = Image.new("RGBA", base.size, (0, 0, 0, 0))
    else:
        text_layer_img = None

    font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

    used_sizes: Dict[int, int] = {}
    used_rects: Dict[int, Dict[str, int]] = {}

    # If edited_bubble_ids provided and we have a text layer, restrict to patch rendering
    patch_mode = text_layer_img is not None and edited_bubble_ids is not None and len(edited_bubble_ids) > 0
    target_records = [rec for rec in records if (not patch_mode or rec.bubble_id in edited_bubble_ids)]

    for rec in target_records:
        if not rec.polygon: continue
        
        try:
            shp = Polygon(rec.polygon).buffer(0)
            if shp.is_empty: continue
            # If buffer(0) creates multiple polygons from an invalid source shape,
            # we select the one with the largest area to be the main bubble.
            if isinstance(shp, MultiPolygon):
                shp = max(shp.geoms, key=lambda p: p.area)

        except Exception:
            continue

        size_used: int
        layout: Optional[LayoutResult]

        if isinstance(rec.font_size, int) and rec.font_size > 0:
            pref = int(rec.font_size)
            if pref not in font_cache:
                font_cache[pref] = ImageFont.truetype(str(font_path), pref)
            font = font_cache[pref]
            
            layout = compute_optimal_layout(rec.text, shp, font)
            if layout is None:
                size_used, layout = find_optimal_font_size(rec.text, shp, font_path, font_cache, max_font_size=pref)
            else:
                size_used = pref
        else:
            size_used, layout = find_optimal_font_size(rec.text, shp, font_path, font_cache)

        if layout is None or size_used <= 0:
            continue

        if size_used not in font_cache:
            font_cache[size_used] = ImageFont.truetype(str(font_path), size_used)
        font = font_cache[size_used]

        # In patch mode, clear the previous glyphs in the polygon from the text layer first
        if patch_mode and text_layer_img is not None:
            clear_mask = Image.new("L", base.size, 0)
            clr_draw = ImageDraw.Draw(clear_mask)
            pts = [(int(p[0]), int(p[1])) for p in rec.polygon]
            if len(pts) >= 3:
                clr_draw.polygon(pts, fill=255)
            transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))
            text_layer_img.paste(transparent, (0, 0), clear_mask)

        _render_layout(
            base,
            font,
            layout,
            rec.polygon,
            text_layer_img,
            write_to_base=(text_layer_img is None and not patch_mode),
        )

        used_sizes[rec.bubble_id] = int(size_used)
        if layout.x_limits_per_line:
            x0 = min(left for (left, _right) in layout.x_limits_per_line)
            x1 = max(right for (_left, right) in layout.x_limits_per_line)
            y0, y1 = layout.y_limit
            used_rects[rec.bubble_id] = {"x": int(x0), "y": int(y0), "w": int(max(1, x1 - x0)), "h": int(max(1, y1 - y0))}
            
            if debug and debug_draw:
                debug_draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0))
                pts = [(int(p[0]), int(p[1])) for p in rec.polygon]
                if len(pts) > 2:
                    debug_draw.line(pts + [pts[0]], fill=(0, 0, 255), width=1)

    # Compose final as base + text layer when a text layer is present; otherwise base already contains text
    final_img = base.copy()
    if text_layer_img is not None:
        final_img.paste(text_layer_img, (0, 0), text_layer_img)
    final_img.save(output_final_path)
    if debug and debug_overlay_path is not None and debug_img is not None:
        debug_img.save(debug_overlay_path)
    if text_layer_output_path is not None and text_layer_img is not None:
        text_layer_img.save(text_layer_output_path)
        
    return used_sizes, used_rects