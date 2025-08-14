from __future__ import annotations
from typing import Dict, List

import cv2  # type: ignore
import numpy as np

from app.pipeline.ocr.crops import tight_crop_from_mask
from app.pipeline.ocr.preprocess import binarize_for_ocr


def build_text_inpaint_mask(
    image_bgr: np.ndarray,
    instance_masks: List[np.ndarray],
    bubbles: List[Dict],
    erode_border_px: int = 2,
    dilate_text_px: int = 1,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Builds a precise text-only mask from in-memory segmentation data.
    """
    height, width = image_bgr.shape[:2]
    union_text_mask = np.zeros((height, width), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i, rec in enumerate(bubbles):
        polygon = rec.get("polygon") or []

        if i >= len(instance_masks):
            continue # Safety check
        bubble_mask = instance_masks[i]
        
        # Erode bubble mask to avoid bubble outlines
        interior_mask = (
            cv2.erode(bubble_mask, kernel, iterations=max(0, int(erode_border_px)))
            if erode_border_px > 0
            else bubble_mask
        )

        crop_bgr, (x0, y0, x1, y1) = tight_crop_from_mask(image_bgr, bubble_mask, polygon)

        if x1 <= x0 or y1 <= y0:
            continue

        # Binarize the crop to find text pixels
        try:
            bin_img = binarize_for_ocr(crop_bgr)
        except Exception:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Isolate and dilate text to ensure full coverage
        text_local = (bin_img < 128).astype(np.uint8) * 255
        if dilate_text_px > 0:
            text_local = cv2.dilate(text_local, kernel, iterations=int(dilate_text_px))
            
        # Place the local text mask onto a full-size canvas
        canvas = np.zeros((height, width), dtype=np.uint8)
        h, w = y1 - y0, x1 - x0
        canvas[y0:y1, x0:x1] = text_local[:h, :w]
        
        # Combine with the interior mask to prevent spill-over
        bubble_text = cv2.bitwise_and(canvas, (interior_mask * 255).astype(np.uint8))
        union_text_mask = np.maximum(union_text_mask, bubble_text)

    _, union_text_mask = cv2.threshold(union_text_mask, 127, 255, cv2.THRESH_BINARY)
    return union_text_mask