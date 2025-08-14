from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np


def run_inpainting(image_path: Path, combined_mask_path: Path) -> np.ndarray:
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

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not combined_mask_path.exists():
        raise FileNotFoundError(
            f"Combined mask not found: {combined_mask_path}. Run segmentation stage first."
        )

    pil_image = Image.open(image_path).convert("RGB")

    mask_gray = cv2.imread(str(combined_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Failed to read mask: {combined_mask_path}")

    img_w, img_h = pil_image.size
    if mask_gray.shape[:2] != (img_h, img_w):
        mask_gray = cv2.resize(mask_gray, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    pil_mask = Image.fromarray(mask_bin, mode="L")

    if int(np.sum(mask_bin)) == 0:
        pil_result = pil_image
    else:
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyTorch is required for inpainting stage") from exc

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


