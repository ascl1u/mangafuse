from __future__ import annotations
import cv2  # type: ignore
import numpy as np


def fill_white(
    image_bgr: np.ndarray,
    text_mask_gray: np.ndarray,
    bubble_mask_gray: np.ndarray,
    dilate_text_px: int = 5,
    blur_kernel_size: int = 5,
) -> np.ndarray:
    """
    Fills the text area within a bubble with white using a robust, anti-aliased blend.

    This function adopts the highly reliable "clipping mask" strategy. It ensures
    complete text removal by dilating the text mask, then prevents any spill-over
    by clipping it against the bubble's boundary. The result is then blurred for a
    seamless blend.

    Args:
        image_bgr: The source image in BGR format.
        text_mask_gray: A mask containing only the text pixels to be removed.
        bubble_mask_gray: A mask of the entire speech bubble, used as a clipping boundary.
        dilate_text_px: Pixels to dilate the text mask to cover anti-aliased edges.
        blur_kernel_size: The size of the Gaussian blur kernel for anti-aliasing. Must be odd.

    Returns:
        A new image with the text area smoothly filled in white.
    """
    if text_mask_gray.dtype != np.uint8:
        text_mask_gray = text_mask_gray.astype(np.uint8)
    if bubble_mask_gray.dtype != np.uint8:
        bubble_mask_gray = bubble_mask_gray.astype(np.uint8)

    # Ensure masks are binary (0 or 255)
    _, text_mask = cv2.threshold(text_mask_gray, 1, 255, cv2.THRESH_BINARY)
    _, bubble_mask = cv2.threshold(bubble_mask_gray, 1, 255, cv2.THRESH_BINARY)

    # --- Robust Mask Refinement using Clipping ---

    # 1. Dilate the text mask to ensure it fully covers the text's anti-aliased edges.
    if dilate_text_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_text_mask = cv2.dilate(text_mask, kernel, iterations=dilate_text_px)
    else:
        dilated_text_mask = text_mask

    # 2. Clip the dilated mask against the bubble's boundary. This is the key step.
    #    It prevents any part of the fill from touching or exceeding the bubble's line art.
    clipped_mask = cv2.bitwise_and(dilated_text_mask, bubble_mask)

    # Fast exit if there's nothing to do
    if not np.any(clipped_mask):
        return image_bgr.copy()

    # 3. Blur the final, clipped mask to create soft edges for seamless blending.
    if blur_kernel_size > 0:
        ksize = blur_kernel_size if blur_kernel_size % 2 != 0 else blur_kernel_size + 1
        alpha_mask_final = cv2.GaussianBlur(clipped_mask, (ksize, ksize), 0)
    else:
        alpha_mask_final = clipped_mask

    # 4. Convert to a floating-point alpha mask (0.0 to 1.0) for blending.
    alpha_mask_float = alpha_mask_final.astype(np.float32) / 255.0

    # Prepare a solid white layer for blending.
    white_layer = np.full(image_bgr.shape, (255, 255, 255), dtype=image_bgr.dtype)

    # Convert the single-channel alpha mask to a 3-channel mask to blend BGR images.
    alpha_mask_3ch = cv2.cvtColor(alpha_mask_float, cv2.COLOR_GRAY2BGR)

    # Alpha blend the original image and the white layer.
    blended_bgr = image_bgr * (1 - alpha_mask_3ch) + white_layer * alpha_mask_3ch

    return np.clip(blended_bgr, 0, 255).astype(np.uint8)

