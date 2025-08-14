from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2  # type: ignore
import numpy as np

from app.pipeline.ocr.crops import tight_crop_from_mask
from app.pipeline.ocr.preprocess import binarize_for_ocr


def build_text_inpaint_mask(
    image_bgr: np.ndarray,
    masks_dir: Path,
    bubbles: List[Dict],
    erode_border_px: int = 2,
    dilate_text_px: int = 1,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    union_text_mask = np.zeros((height, width), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for rec in bubbles:
        try:
            bubble_id = int(rec.get("id"))
        except Exception:
            continue
        polygon = rec.get("polygon") or []
        bubble_mask_path = masks_dir / f"{bubble_id}.png"

        mask_gray = cv2.imread(str(bubble_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            continue
        if mask_gray.shape != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)
        bubble_mask = (mask_gray > 0).astype(np.uint8)
        interior_mask = (
            cv2.erode(bubble_mask, kernel, iterations=max(0, int(erode_border_px))) if erode_border_px > 0 else bubble_mask
        )

        crop_bgr, (x0, y0, x1, y1) = tight_crop_from_mask(image_bgr, bubble_mask_path, polygon)
        if x1 <= x0 or y1 <= y0:
            continue

        try:
            bin_img = binarize_for_ocr(crop_bgr)
        except Exception:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_local = (bin_img < 128).astype(np.uint8) * 255
        if dilate_text_px > 0:
            text_local = cv2.dilate(text_local, kernel, iterations=int(dilate_text_px))

        canvas = np.zeros((height, width), dtype=np.uint8)
        h, w = y1 - y0, x1 - x0
        canvas[y0:y1, x0:x1] = text_local[:h, :w]
        bubble_text = cv2.bitwise_and(canvas, (interior_mask * 255).astype(np.uint8))

        union_text_mask = np.maximum(union_text_mask, bubble_text)

    _, union_text_mask = cv2.threshold(union_text_mask, 127, 255, cv2.THRESH_BINARY)
    return union_text_mask


