from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops


@dataclass
class BubbleText:
    bubble_id: int
    polygon: List[List[float]]
    text: str
    # Optional font size override requested by the user. When provided, we will
    # attempt to honor it if it fits within the available region; otherwise we
    # will reduce the size to the largest that fits.
    font_size: Optional[int] = None


def _polygon_bbox(polygon: List[List[float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0 = int(np.floor(min(xs)))
    y0 = int(np.floor(min(ys)))
    x1 = int(np.ceil(max(xs)))
    y1 = int(np.ceil(max(ys)))
    return x0, y0, x1, y1


def _split_word_hard(word: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Split a single oversized word into chunks that each fit within max_width.

    Falls back to single-character chunks when necessary.
    """
    if not word:
        return [word]
    chunks: List[str] = []
    current = ""
    for ch in word:
        candidate = (current + ch) if current else ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = ch
            else:
                # Even a single char does not fit; force as single-char chunk
                chunks.append(ch)
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _wrap_text_to_fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    # Greedy word wrap with hard fallback for single oversized words
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = (" ".join(current + [word])).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current.append(word)
        else:
            # finalize current line if exists
            if current:
                lines.append(" ".join(current))
                current = []
            # place the word; split if it does not fit on an empty line
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_w = word_bbox[2] - word_bbox[0]
            if word_w <= max_width:
                current = [word]
            else:
                chunks = _split_word_hard(word, draw, font, max_width)
                if chunks:
                    lines.extend(chunks[:-1])
                    current = [chunks[-1]]
    if current:
        lines.append(" ".join(current))
    return lines if lines else [text]


def _compute_font_size_via_binary_search(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    max_width: int,
    max_height: int,
    min_size: int = 10,
    max_size: int = 64,
    line_spacing: float = 1.1,
) -> Tuple[int, List[str]]:
    best_size = min_size
    best_lines: List[str] = []
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(str(font_path), size=mid)
        lines = _wrap_text_to_fit(draw, text, font, max_width)
        total_h = 0
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            h = bbox[3] - bbox[1]
            total_h += h if i == 0 else int(h * line_spacing)
        if lines:
            # add some small margin for descenders
            total_h = int(total_h * 1.0)
        if total_h <= max_height:
            best_size = mid
            best_lines = lines
            lo = mid + 1
        else:
            hi = mid - 1
    return best_size, best_lines


def _wrap_with_preferred_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    max_width: int,
    max_height: int,
    preferred_size: int,
    min_size: int = 10,
    max_size: int = 64,
) -> Tuple[int, List[str]]:
    size = int(max(min_size, min(max_size, preferred_size)))
    font = ImageFont.truetype(str(font_path), size=size)
    lines = _wrap_text_to_fit(draw, text, font, max_width)
    total_h = 0
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        h = bbox[3] - bbox[1]
        total_h += h if i == 0 else int(h * 1.1)
    if lines:
        total_h = int(total_h * 1.0)
    if total_h <= max_height:
        return size, lines
    # Too big → fall back to binary search constrained by preferred size
    return _compute_font_size_via_binary_search(
        draw=draw,
        text=text,
        font_path=font_path,
        max_width=max_width,
        max_height=max_height,
        min_size=min_size,
        max_size=min(size, max_size),
    )


def _draw_centered_text(
    canvas: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    font_path: Path,
    text: str,
    polygon: List[List[float]],
    debug: bool = False,
    preferred_font_size: Optional[int] = None,
    text_layer: Optional[Image.Image] = None,
) -> Tuple[int, List[str]]:
    # Use overlay and mask to clip strictly to polygon
    max_width = max(1, x1 - x0)
    max_height = max(1, y1 - y0)

    # Use overlay for measurement and drawing
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Choose font size: honor preferred when possible
    if preferred_font_size is not None:
        font_size, lines = _wrap_with_preferred_size(
            draw_overlay, text, font_path, max_width, max_height, preferred_size=preferred_font_size
        )
    else:
        font_size, lines = _compute_font_size_via_binary_search(draw_overlay, text, font_path, max_width, max_height)
    font = ImageFont.truetype(str(font_path), size=font_size)

    # Measure lines with consistent spacing
    line_bboxes = [draw_overlay.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]
    content_height = 0
    for i, h in enumerate(line_heights):
        content_height += h if i == 0 else int(h * 1.1)

    # Vertical centering
    y = y0 + (max_height - content_height) // 2
    for i, line in enumerate(lines):
        bbox = draw_overlay.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = x0 + (max_width - w) // 2
        # Draw onto overlay with opaque alpha
        draw_overlay.text((x, y), line, fill=(0, 0, 0, 255), font=font)
        # Advance with consistent spacing
        y += (h if i == 0 else int(h * 1.1))

    # Build polygon mask
    mask = Image.new("L", canvas.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    pts = [(int(p[0]), int(p[1])) for p in polygon]
    if len(pts) >= 3:
        mask_draw.polygon(pts, fill=255)

    # Combine text alpha with polygon mask
    alpha = overlay.split()[3]
    combined_alpha = ImageChops.multiply(alpha, mask)
    overlay.putalpha(combined_alpha)

    # Composite onto canvas and optional text_layer
    canvas.paste(overlay, (0, 0), overlay)
    if text_layer is not None:
        text_layer.paste(overlay, (0, 0), overlay)
    return font_size, lines


def render_typeset(
    cleaned_path: Path,
    output_final_path: Path,
    records: List[BubbleText],
    font_path: Path,
    margin_px: int = 6,
    debug: bool = False,
    debug_overlay_path: Optional[Path] = None,
    text_layer_output_path: Optional[Path] = None,
) -> dict:
    """Render English text onto cleaned page image.

    Uses polygon bounding boxes minus a margin as fitting regions. Saves final image.
    Optionally saves a debug overlay with regions and outlines when debug is True.
    """
    if not cleaned_path.exists():
        raise FileNotFoundError(f"cleaned image not found: {cleaned_path}")
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    base = Image.open(cleaned_path).convert("RGB")
    text_layer_img = Image.new("RGBA", base.size, (0, 0, 0, 0)) if text_layer_output_path is not None else None
    debug_img = base.copy() if debug else None
    base_draw = ImageDraw.Draw(debug_img) if debug_img else None

    used_sizes: dict = {}
    for rec in records:
        if not rec.polygon:
            continue
        x0, y0, x1, y1 = _polygon_bbox(rec.polygon)
        x0 += margin_px
        y0 += margin_px
        x1 -= margin_px
        y1 -= margin_px
        if x1 <= x0 or y1 <= y0:
            continue
        size_used, _lines = _draw_centered_text(
            base,
            x0,
            y0,
            x1,
            y1,
            font_path,
            rec.text,
            rec.polygon,
            debug=debug,
            preferred_font_size=rec.font_size,
            text_layer=text_layer_img,
        )
        used_sizes[rec.bubble_id] = int(size_used)
        if debug and base_draw:
            base_draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0))
            # Outline the polygon
            pts = [(int(p[0]), int(p[1])) for p in rec.polygon]
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                base_draw.line([a, b], fill=(0, 0, 255), width=1)

    base.save(output_final_path)
    if debug and debug_overlay_path is not None and debug_img is not None:
        debug_img.save(debug_overlay_path)
    if text_layer_output_path is not None and text_layer_img is not None:
        text_layer_img.save(text_layer_output_path)
    return used_sizes


