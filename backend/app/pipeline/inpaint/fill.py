from __future__ import annotations
import cv2  # type: ignore
import numpy as np


def fill_white(
    image_bgr: np.ndarray,
    text_mask_gray: np.ndarray,
    bubble_mask_gray: np.ndarray,
) -> np.ndarray:
    """
    Fills the text area within a bubble with white using a robust, anti-aliased blend.

    This function adopts a conservative "clipping mask" strategy. Mask morphology
    (dilation/erosion) is handled upstream by prepare_inpaint_regions; here we only
    clip and softly blend to avoid haloing.

    Args:
        image_bgr: The source image in BGR format.
        text_mask_gray: A mask containing only the text pixels to be removed.
        bubble_mask_gray: A mask of the entire speech bubble, used as a clipping boundary.
        Note: Mask morphology must be handled upstream; this function will not
        dilate the text mask.

    Returns:
        A new image with the text area smoothly filled in white.
    """
    if text_mask_gray.dtype != np.uint8:
        text_mask_gray = text_mask_gray.astype(np.uint8)
    if bubble_mask_gray.dtype != np.uint8:
        bubble_mask_gray = bubble_mask_gray.astype(np.uint8)

    # Ensure masks are binary (0 or 255)
    _, text_mask = cv2.threshold(text_mask_gray, 0, 255, cv2.THRESH_BINARY)
    _, bubble_mask = cv2.threshold(bubble_mask_gray, 0, 255, cv2.THRESH_BINARY)

    # --- Robust Mask Refinement using Clipping ---
    # 1. Clip the provided text mask against the bubble's interior boundary.
    #    It prevents any part of the fill from touching or exceeding the bubble's line art.
    clipped_mask = cv2.bitwise_and(text_mask, bubble_mask)

    # Fast exit if there's nothing to do
    if not np.any(clipped_mask):
        return image_bgr.copy()

    # 2. Blur the final clipped mask to create soft edges for seamless blending.
    #    Keep a small fixed kernel to avoid parameter drift.
    ksize = 5
    alpha_mask_final = cv2.GaussianBlur(clipped_mask, (ksize, ksize), 0)

    # Re-clip after blur to guarantee the fill never crosses the clip boundary.
    alpha_mask_final = cv2.bitwise_and(alpha_mask_final, bubble_mask)

    # 4. Convert to a floating-point alpha mask (0.0 to 1.0) for blending.
    alpha_mask_float = alpha_mask_final.astype(np.float32) / 255.0

    # Prepare a solid white layer for blending.
    white_layer = np.full(image_bgr.shape, (255, 255, 255), dtype=image_bgr.dtype)

    # Convert the single-channel alpha mask to a 3-channel mask to blend BGR images.
    alpha_mask_3ch = cv2.cvtColor(alpha_mask_float, cv2.COLOR_GRAY2BGR)

    # Alpha blend the original image and the white layer.
    blended_bgr = image_bgr * (1 - alpha_mask_3ch) + white_layer * alpha_mask_3ch

    return np.clip(blended_bgr, 0, 255).astype(np.uint8)

