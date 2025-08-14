from __future__ import annotations
import cv2  # type: ignore
import numpy as np


def run_inpainting(image_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """
    Runs the LaMa inpainting model on an in-memory image and mask.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Pillow is required for inpainting stage") from exc

    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "simple-lama-inpainting is required. Install AI deps from backend/requirements-ai.txt"
        ) from exc

    h, w = image_bgr.shape[:2]
    if mask_gray.shape != (h, w):
        raise ValueError(f"Mask shape {mask_gray.shape} must match image shape {(h, w)}")

    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    # Binarize the mask to be safe
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    pil_mask = Image.fromarray(mask_bin, mode="L")

    # Optimization: if mask is empty, skip the expensive model call
    if int(np.sum(mask_bin)) == 0:
        pil_result = pil_image
    else:
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for inpainting stage") from exc

        # Monkey patch to force CPU execution for portability
        original_jit_load = torch.jit.load
        def _jit_load_cpu(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if "map_location" not in kwargs:
                kwargs["map_location"] = "cpu"
            return original_jit_load(path, *args, **kwargs)

        torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
        try:
            model = SimpleLama()
            pil_result = model(pil_image, pil_mask)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "SimpleLama inpainting failed; fix environment or inputs instead of falling back."
            ) from exc
        finally:
            if "torch" in locals():
                torch.jit.load = original_jit_load  # type: ignore[assignment]

    result_bgr = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
    return result_bgr