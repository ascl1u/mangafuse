from __future__ import annotations

import cv2  # type: ignore
import numpy as np


def binarize_for_ocr(
    crop_bgr: np.ndarray,
    block_size: int = 25,
    c_value: int = 15
) -> np.ndarray:
    """
    Binarizes an image crop using adaptive thresholding to prepare it for OCR.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding requires an odd block_size > 1
    if block_size <= 1:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )
    return th