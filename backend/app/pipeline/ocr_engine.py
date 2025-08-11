from __future__ import annotations

import cv2
import numpy as np

class MangaOcrEngine:
    """Thin wrapper around manga-ocr with lazy initialization."""

    def __init__(self) -> None:
        self._engine = None

    def _ensure(self) -> None:
        if self._engine is None:
            try:
                from manga_ocr import MangaOcr  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "manga-ocr is required. Install AI deps from backend/requirements-ai.txt"
                ) from exc
            self._engine = MangaOcr()

    def run(self, image: np.ndarray) -> str:
        from PIL import Image

        self._ensure()
        if image.ndim == 2:
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        text = self._engine(pil_img)
        return (text or "").strip()


