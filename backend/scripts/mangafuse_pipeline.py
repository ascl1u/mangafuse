"""
MangaFuse AI Pipeline — Phase 2 Step 2.1: Bubble Segmentation (CLI wrapper)

This CLI wraps the pipeline code located under app.pipeline.* modules.
Behavior, flags, and outputs remain identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the backend package root is on sys.path when running from repo root
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import json
import os

import cv2

from app.pipeline.io import ensure_dir, read_image_bgr, save_png
from app.pipeline.masks import save_masks
from app.pipeline.segmentation import run_segmentation
from app.pipeline.visualization import make_overlay
from app.pipeline.textio import write_text_json, read_text_json, save_text_records
from app.pipeline.crops import tight_crop_from_mask
from app.pipeline.preprocess import binarize_for_ocr
from app.pipeline.ocr_engine import MangaOcrEngine
from app.pipeline.translator import GeminiTranslator
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MangaFuse AI Pipeline (Phase 2)")
    parser.add_argument("--image", type=str, required=True, help="Path to input page image")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["seg", "ocr", "translate", "inpaint", "typeset", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(Path("assets/models/model.pt")),
        help="Path to local YOLOv8 segmentation weights (.pt)",
    )
    parser.add_argument("--debug", action="store_true", help="Save additional debug outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")

    # Future flags (parsed but unused in 2.1)
    parser.add_argument("--font", type=str, default=str(Path("assets/fonts/AnimeAce.ttf")), help="Path to TTF font")
    parser.add_argument("--use-placeholder-text", action="store_true", help="Bypass OCR/translation during typeset stage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    seg_model_path = Path(args.seg_model)
    # Load environment variables from backend/.env first, then any nearer .env
    backend_env = _BACKEND_DIR / ".env"
    if backend_env.exists():
        load_dotenv(backend_env, override=False)
    load_dotenv(override=False)


    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    ensure_dir(out_dir)
    ensure_dir(out_dir / "masks")

    stage = args.stage
    if stage == "seg" or stage == "all":
        overlay_path = out_dir / "segmentation_overlay.png"
        masks_dir = out_dir / "masks"
        combined_mask_path = masks_dir / "all_mask.png"
        json_path = out_dir / "text.json"

        if not args.force and overlay_path.exists() and combined_mask_path.exists() and json_path.exists():
            print(json.dumps({"stage": "seg", "status": "skip", "reason": "artifacts_exist"}))
        else:
            image_bgr = read_image_bgr(image_path)
            result = run_segmentation(image_bgr=image_bgr, seg_model_path=seg_model_path)

            polygons = result.get("polygons", [])
            masks = result.get("masks", [])
            confidences = result.get("confidences", [])

            overlay_bgr = make_overlay(image_bgr, masks, polygons, confidences)
            save_png(overlay_path, overlay_bgr)

            h, w = image_bgr.shape[:2]
            save_masks(masks_dir, masks, combined_mask_path, image_height=h, image_width=w)

            write_text_json(json_path, polygons)

            if args.debug:
                debug_path = out_dir / "segmentation_overlay_ids.png"
                debug_img = overlay_bgr.copy()
                for idx, poly in enumerate(polygons, start=1):
                    if len(poly) == 0:
                        continue
                    x0, y0 = int(poly[0][0]), int(poly[0][1])
                    cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                save_png(debug_path, debug_img)

            print(json.dumps({"stage": "seg", "status": "ok", "num_bubbles": len(polygons)}))

        if stage == "seg":
            return

    if stage in ("ocr", "translate"):
        json_path = out_dir / "text.json"
        masks_dir = out_dir / "masks"

        if not json_path.exists():
            raise FileNotFoundError("text.json not found. Run segmentation stage first.")

        data = read_text_json(json_path)
        bubbles = data.get("bubbles", [])
        if not bubbles:
            print(json.dumps({"stage": stage, "status": "ok", "num_bubbles": 0}))
            return

        image_bgr = read_image_bgr(image_path)

        # Prepare engines
        ocr_engine = MangaOcrEngine()

        updated_ocr = 0

        # OCR pass only for explicit OCR stage (idempotent with --force rules)
        if stage == "ocr":
            for rec in bubbles:
                has_ja = isinstance(rec.get("ja_text"), str) and rec["ja_text"].strip() != ""
                if has_ja and not args.force:
                    continue
                bubble_id = int(rec.get("id"))
                polygon = rec.get("polygon") or []
                mask_path = masks_dir / f"{bubble_id}.png"
                crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_path, polygon)
                try:
                    bin_img = binarize_for_ocr(crop_bgr)
                    ja_text = ocr_engine.run(bin_img)
                except Exception:
                    ja_text = ocr_engine.run(crop_bgr)
                rec["ja_text"] = ja_text
                updated_ocr += 1

        if stage == "translate":
            api_key = os.getenv("GOOGLE_API_KEY")
            translator = GeminiTranslator(api_key=api_key or "")

            # Ensure all have ja_text
            for rec in bubbles:
                if not isinstance(rec.get("ja_text"), str) or rec["ja_text"].strip() == "":
                    bubble_id = int(rec.get("id"))
                    polygon = rec.get("polygon") or []
                    mask_path = masks_dir / f"{bubble_id}.png"
                    crop_bgr, _bbox = tight_crop_from_mask(image_bgr, mask_path, polygon)
                    try:
                        rec["ja_text"] = ocr_engine.run(binarize_for_ocr(crop_bgr))
                    except Exception:
                        rec["ja_text"] = ocr_engine.run(crop_bgr)

            # Batch translate those missing en_text unless force
            indices, texts = [], []
            for idx, rec in enumerate(bubbles):
                ja = (rec.get("ja_text") or "").strip()
                has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip() != ""
                if not ja:
                    continue
                if has_en and not args.force:
                    continue
                indices.append(idx)
                texts.append(ja)

            if texts:
                en_list = translator.translate_batch(texts)
                for i, en in zip(indices, en_list):
                    bubbles[i]["en_text"] = en

        # Persist and report
        save_text_records(json_path, bubbles)
        print(
            json.dumps(
                {
                    "stage": stage,
                    "status": "ok",
                    "num_bubbles": len(bubbles),
                    "updated_ocr": updated_ocr if stage in ("ocr", "translate") else 0,
                }
            )
        )
        return

    if stage in ("inpaint", "typeset"):
        raise NotImplementedError(
            f"Stage '{stage}' is not implemented yet. Complete Step 2.3 for inpaint/typeset."
        )


if __name__ == "__main__":
    main()

