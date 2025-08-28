from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelRegistry:
    """Container for long-lived model instances used by the pipeline.

    Attributes:
        yolo_model: Preloaded ultralytics YOLO segmentation model.
        ocr_engine: Preloaded MangaOcrEngine wrapper.
        lama_model: Preloaded SimpleLama inpainting model.
    """

    yolo_model: Any | None = None
    ocr_engine: Any | None = None
    lama_model: Any | None = None

    @staticmethod
    def load(
        *,
        seg_model_path: Path,
        preload_ocr: bool = True,
    ) -> "ModelRegistry":
        """Load all heavy models once and return a registry instance.

        This function imports heavy deps lazily and initializes them. It is
        intended to be called at application startup (GPU service).
        """
        # YOLO segmentation model
        try:
            from ultralytics import YOLO  # type: ignore
            yolo_model = YOLO(str(seg_model_path))
        except Exception as exc:  # noqa: BLE001
            # Allow the caller to decide how to handle missing models
            yolo_model = None

        # OCR engine (wrapper) with preloading to instantiate manga-ocr
        try:
            from app.pipeline.ocr.engine import MangaOcrEngine

            ocr_engine = MangaOcrEngine()
            if preload_ocr:
                # Force initialization of the underlying manga-ocr model
                try:
                    ocr_engine.ensure_loaded()
                except Exception:
                    # Defer loading to first use if preload fails
                    pass
        except Exception:  # noqa: BLE001
            ocr_engine = None

        # SimpleLama inpainting model
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore

            lama_model = SimpleLama()
        except Exception:  # noqa: BLE001
            lama_model = None

        return ModelRegistry(yolo_model=yolo_model, ocr_engine=ocr_engine, lama_model=lama_model)


