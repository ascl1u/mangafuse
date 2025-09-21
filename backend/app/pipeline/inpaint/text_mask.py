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
    erode_border_px: int,
    dilate_text_px: int,
    kernel_size: int,
) -> np.ndarray:
    """
    Builds a precise text-only mask from in-memory segmentation data.
    """
    height, width = image_bgr.shape[:2]
    union_text_mask = np.zeros((height, width), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _rasterize_polygon(poly: List[List[float]]) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        if not poly:
            return canvas
        pts = np.array(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return canvas
        pts_i32 = np.round(pts).astype(np.int32)
        cv2.fillPoly(canvas, [pts_i32], 1)
        return canvas

    for i, rec in enumerate(bubbles):
        polygon = rec.get("polygon") or []

        # Prefer provided instance mask if present and correctly sized; otherwise rasterize polygon
        if i < len(instance_masks):
            candidate = instance_masks[i]
            if (
                isinstance(candidate, np.ndarray)
                and candidate.shape == (height, width)
                and np.any(candidate)
            ):
                bubble_mask = (candidate > 0).astype(np.uint8)
            else:
                bubble_mask = _rasterize_polygon(polygon)
        else:
            bubble_mask = _rasterize_polygon(polygon)
        
        # Erode bubble mask to avoid bubble outlines
        interior_mask = (
            cv2.erode(bubble_mask, kernel, iterations=max(0, int(erode_border_px)))
            if erode_border_px > 0
            else bubble_mask
        )

        crop_bgr, (x0, y0, x1, y1) = tight_crop_from_mask(image_bgr, bubble_mask, polygon)

        if x1 <= x0 or y1 <= y0:
            continue

        # Convert crop to HSV color space to isolate brightness from color.
        hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv_crop[:, :, 2]  # Value channel

        # Use Otsu's thresholding on the Value channel to reliably separate dark text
        # from the lighter, colored backgrounds.
        # This creates a mask where text is black (0) and background is white (255).
        _, text_mask_inv = cv2.threshold(
            v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Invert the mask so text is white (255), which is the standard for masks.
        text_local = cv2.bitwise_not(text_mask_inv)
        # Dilate text to ensure full coverage
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


def prepare_inpaint_regions(
    image_bgr: np.ndarray,
    instance_masks: List[np.ndarray],
    bubbles: List[Dict],
    *,
    erode_border_px: int = 6,
    dilate_text_px: int = 5,
    kernel_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Prepares canonical masks for the inpainting stage with a single source of morphology.

    Returns a dictionary with:
      - text_mask_total: union of text-only areas across bubbles
      - mask_white_union: union of class==0 bubble masks
      - mask_lama_union: union of class==1 bubble masks
      - interior_white_union: eroded union of white bubbles (protects borders)
      - white_text_mask: text_mask_total clipped to white bubbles
      - lama_text_mask: text_mask_total clipped to integrated bubbles
    """
    height, width = image_bgr.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    text_mask_total = build_text_inpaint_mask(
        image_bgr,
        instance_masks,
        bubbles,
        erode_border_px=erode_border_px,
        dilate_text_px=dilate_text_px,
        kernel_size=kernel_size,
    )

    mask_white_union = np.zeros((height, width), dtype=np.uint8)
    mask_lama_union = np.zeros((height, width), dtype=np.uint8)
    for i, rec in enumerate(bubbles):
        target = mask_white_union if rec.get("class", 0) == 0 else mask_lama_union
        if i < len(instance_masks):
            target[:] = np.maximum(target, (instance_masks[i] > 0).astype(np.uint8))

    # Eroded interior for white bubbles to keep a safety margin from outlines
    interior_white_union = (
        cv2.erode(mask_white_union, kernel, iterations=max(0, int(erode_border_px)))
        if erode_border_px > 0
        else mask_white_union
    )

    white_text_mask = cv2.bitwise_and(text_mask_total, text_mask_total, mask=mask_white_union * 255)
    lama_text_mask = cv2.bitwise_and(text_mask_total, text_mask_total, mask=mask_lama_union * 255)

    return {
        "text_mask_total": text_mask_total,
        "mask_white_union": mask_white_union,
        "mask_lama_union": mask_lama_union,
        "interior_white_union": interior_white_union,
        "white_text_mask": white_text_mask,
        "lama_text_mask": lama_text_mask,
    }