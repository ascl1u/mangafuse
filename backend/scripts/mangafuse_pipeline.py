# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
import cv2

# --- Path Setup ---
# Ensure the backend package root is on sys.path when running from repo root
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# --- Local Imports ---
from app.pipeline.utils.io import ensure_dir, read_image_bgr, save_png
from app.pipeline.utils.textio import write_text_json, read_text_json, save_text_records

# --- Configuration & State ---

@dataclass
class PipelineConfig:
    """A centralized configuration object for the pipeline."""
    image_path: Path
    out_dir: Path
    seg_model_path: Path
    font_path: Path
    stage: str
    debug: bool
    force: bool
    use_placeholder_text: bool

    # Derived Paths
    file_prefix: str = field(init=False)
    masks_dir: Path = field(init=False)
    json_path: Path = field(init=False)
    overlay_path: Path = field(init=False)
    combined_mask_path: Path = field(init=False)
    text_mask_path: Path = field(init=False)
    cleaned_path: Path = field(init=False)
    final_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths after the main fields are set."""
        self.file_prefix = self.image_path.stem
        self.masks_dir = self.out_dir / "masks"
        self.json_path = self.out_dir / f"{self.file_prefix}_text.json"
        self.overlay_path = self.out_dir / f"{self.file_prefix}_segmentation_overlay.png"
        self.combined_mask_path = self.masks_dir / f"{self.file_prefix}_all_mask.png"
        self.text_mask_path = self.masks_dir / f"{self.file_prefix}_text_mask.png"
        self.cleaned_path = self.out_dir / f"{self.file_prefix}_cleaned.png"
        self.final_path = self.out_dir / f"{self.file_prefix}_final.png"

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaFuse AI Pipeline (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input image or a folder of images")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write artifacts")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["seg", "ocr", "translate", "inpaint", "typeset", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(Path("assets/models/model.pt")),
        help="Path to local YOLOv8 segmentation weights (.pt)",
    )
    parser.add_argument("--font", type=str, default=str(Path("assets/fonts/animeace2_reg.ttf")), help="Path to TTF font")
    parser.add_argument("--debug", action="store_true", help="Save additional debug outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument("--use-placeholder-text", action="store_true", help="Use placeholder text for typesetting")
    return parser.parse_args()


# --- Pipeline Stages ---

def run_segmentation_stage(config: PipelineConfig, image_bgr: np.ndarray) -> Dict[str, Any]:
    """Detects speech bubbles and returns their data in-memory."""
    from app.pipeline.segmentation.yolo import run_segmentation
    from app.pipeline.utils.masks import save_masks
    from app.pipeline.utils.visualization import make_overlay

    print("--- Running Stage: Segmentation ---")

    print(f"Running bubble segmentation on {config.image_path.name}...")
    result = run_segmentation(image_bgr=image_bgr, seg_model_path=config.seg_model_path)

    polygons = result.get("polygons", [])
    bubbles = [{"id": i, "polygon": p} for i, p in enumerate(polygons, 1)]
    result["bubbles"] = bubbles

    if config.debug:
        print("Debug mode: Saving segmentation artifacts to disk...")
        h, w = image_bgr.shape[:2]
        
        rasterized_masks: List[np.ndarray] = []
        for poly in polygons:
            mask = np.zeros((h, w), dtype=np.uint8)
            if poly:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            rasterized_masks.append(mask)

        save_masks(config.masks_dir, rasterized_masks, config.combined_mask_path, image_height=h, image_width=w)
        
        overlay_bgr = make_overlay(image_bgr, rasterized_masks, polygons, result.get("confidences", []))
        save_png(config.overlay_path, overlay_bgr)
        
        save_text_records(config.json_path, bubbles)


    print(f"Segmentation complete. Found {len(polygons)} bubbles.")
    return result

def run_ocr_stage(config: PipelineConfig, image_bgr: np.ndarray, seg_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Performs OCR on in-memory bubble data."""
    from app.pipeline.ocr.crops import tight_crop_from_mask
    from app.pipeline.ocr.preprocess import binarize_for_ocr
    from app.pipeline.ocr.engine import MangaOcrEngine

    print("--- Running Stage: OCR ---")

    bubbles = seg_result.get("bubbles", [])
    masks = seg_result.get("masks", [])
    if not bubbles:
        print("No bubbles found, skipping OCR.")
        return []

    ocr_engine = MangaOcrEngine()
    updated_count = 0

    print(f"Performing OCR on {len(bubbles)} bubbles...")
    for i, rec in enumerate(bubbles):
        if not (rec.get("ja_text") is None or config.force):
            continue

        mask_array = masks[i] if i < len(masks) else None
        polygon = rec.get("polygon", [])

        crop_bgr, _ = tight_crop_from_mask(image_bgr, mask_array, polygon)

        try:
            ja_text = ocr_engine.run(binarize_for_ocr(crop_bgr))
        except Exception:
            ja_text = ocr_engine.run(crop_bgr)

        rec["ja_text"] = ja_text
        updated_count += 1
        print(f"   - Bubble {rec['id']}: {ja_text[:30]}...")

    if config.debug:
        print("Debug mode: Saving updated text records to disk...")
        save_text_records(config.json_path, bubbles)

    print(f"OCR complete. Updated {updated_count} bubbles.")
    return bubbles

def run_translation_stage(config: PipelineConfig, bubbles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translates Japanese text from in-memory bubble data."""
    from app.pipeline.translate.gemini import GeminiTranslator

    print("--- Running Stage: Translation ---")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    if not bubbles:
        print("No bubbles to translate.")
        return []

    texts_to_translate = []
    indices_to_update = []

    for i, rec in enumerate(bubbles):
        ja_text = (rec.get("ja_text") or "").strip()
        if ja_text and (rec.get("en_text") is None or config.force):
            texts_to_translate.append(ja_text)
            indices_to_update.append(i)

    if not texts_to_translate:
        print("Skipping translation: All bubbles already have English text. Use --force to re-translate.")
        return bubbles

    print(f"Translating text for {len(texts_to_translate)} bubbles...")
    translator = GeminiTranslator(api_key=api_key)
    translated_texts = translator.translate_batch(texts_to_translate)

    for i, en_text in zip(indices_to_update, translated_texts):
        bubbles[i]["en_text"] = en_text
        print(f"   - Bubble {bubbles[i]['id']}: \"{bubbles[i]['ja_text']}\" -> \"{en_text}\"")

    if config.debug:
        print("Debug mode: Saving updated text records to disk...")
        save_text_records(config.json_path, bubbles)

    print(f"Translation complete. Generated {len(translated_texts)} English translations.")
    return bubbles

def run_inpaint_stage(config: PipelineConfig, image_bgr: np.ndarray, seg_result: Dict[str, Any], bubble_data: List[Dict[str, Any]]) -> np.ndarray:
    """Removes original text from the image using an in-memory workflow."""
    from app.pipeline.inpaint.lama import run_inpainting
    from app.pipeline.inpaint.text_mask import build_text_inpaint_mask

    print("--- Running Stage: Inpainting ---")

    if not config.force and config.cleaned_path.exists():
        print(f"Skipping inpainting: {config.cleaned_path.name} already exists. Use --force to override.")
        return read_image_bgr(config.cleaned_path)

    print("Building precise text mask for inpainting...")
    text_mask = build_text_inpaint_mask(
        image_bgr=image_bgr,
        instance_masks=seg_result.get("masks", []),
        bubbles=bubble_data
    )

    if config.debug:
        print(f"Debug mode: Saving text inpaint mask to {config.text_mask_path}")
        save_png(config.text_mask_path, text_mask)

    use_mask = text_mask
    if use_mask.sum() == 0:
        print("Text mask is empty, falling back to combined bubble mask for inpainting.")
        masks = seg_result.get("masks", [])
        if masks:
            combined_mask = np.maximum.reduce(masks).astype(np.uint8) * 255
            use_mask = combined_mask

    print(f"Running inpainting on {config.image_path.name}...")
    cleaned_bgr = run_inpainting(image_bgr, use_mask)

    save_png(config.cleaned_path, cleaned_bgr)

    print(f"Inpainting complete. Saved cleaned image to {config.cleaned_path}")
    return cleaned_bgr

def run_typeset_stage(config: PipelineConfig, cleaned_bgr: np.ndarray, bubble_data: List[Dict[str, Any]]):
    """Renders the translated text back into the bubbles using an in-memory image."""
    from app.pipeline.typeset.model import BubbleText
    from app.pipeline.typeset.render import render_typeset

    print("--- Running Stage: Typesetting ---")

    records = []
    if config.use_placeholder_text:
        print("Using placeholder text for typesetting.")
        records = [
            BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text="Placeholder Text")
            for rec in bubble_data
        ]
    else:
        for rec in bubble_data:
            text = (rec.get("en_text") or rec.get("ja_text") or "").strip()
            records.append(
                BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text=text)
            )

    print(f"Typesetting text for {len(records)} bubbles...")
    debug_overlay_path = config.out_dir / f"{config.file_prefix}_typeset_debug.png" if config.debug else None

    render_typeset(
        cleaned_image_bgr=cleaned_bgr,
        output_final_path=config.final_path,
        records=records,
        font_path=config.font_path,
        debug=config.debug,
        debug_overlay_path=debug_overlay_path,
    )

    print(f"Typesetting complete. Saved final image to {config.final_path}")
    if config.debug and debug_overlay_path:
        print(f"Saved debug typesetting overlay to {debug_overlay_path}")


# --- Image Processing Function ---

def process_image(config: PipelineConfig):
    """Runs the full pipeline for a single image."""
    print(f"Starting MangaFuse AI Pipeline for '{config.image_path.name}'...")
    print(f"Output directory: '{config.out_dir}'")
    print(f"Stage(s) to run: '{config.stage}'")

    image_bgr = read_image_bgr(config.image_path)
    seg_result = {}
    bubble_data = []
    cleaned_bgr = None

    if config.json_path.exists() and not config.force:
        print(f"Loading existing bubble data from {config.json_path.name}")
        raw_json_data = read_text_json(config.json_path)
        parsed_data = None
        
        try:
            parsed_data = json.loads(raw_json_data) if isinstance(raw_json_data, str) else raw_json_data
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Failed to parse JSON from {config.json_path.name}. Will re-run precursor stages.")

        if parsed_data:
            if isinstance(parsed_data, list):
                bubble_data = parsed_data
            elif isinstance(parsed_data, dict):
                for key in ["bubbles", "records", "data"]:
                    if isinstance(parsed_data.get(key), list):
                        bubble_data = parsed_data[key]
                        break
            
            if bubble_data:
                if all(isinstance(b, dict) for b in bubble_data):
                    polygons = [b.get("polygon", []) for b in bubble_data]
                    seg_result = {"polygons": polygons, "bubbles": bubble_data, "masks": []}
                else:
                    print(f"Warning: Data from {config.json_path.name} is not a list of dictionaries. Discarding.")
                    bubble_data = []
            else:
                print(f"Warning: Could not find a list of bubbles in {config.json_path.name}. Discarding loaded data.")
        
    if config.cleaned_path.exists() and not config.force:
        print(f"Loading existing cleaned image from {config.cleaned_path.name}")
        cleaned_bgr = read_image_bgr(config.cleaned_path)
    
    # --- Stage-based execution ---
    # FIX: Corrected the order of stages to match the specified pipeline flow.
    stages_to_run = ["seg", "ocr", "inpaint", "translate", "typeset"]
    
    try:
        end_index = stages_to_run.index(config.stage) if config.stage != "all" else len(stages_to_run) - 1
    except ValueError:
        end_index = len(stages_to_run) - 1

    active_stages = stages_to_run[:end_index + 1]

    if "seg" in active_stages and not seg_result:
        seg_result = run_segmentation_stage(config, image_bgr)
        bubble_data = seg_result.get("bubbles", [])

    if "ocr" in active_stages:
        if not bubble_data or any(b.get("ja_text") is None for b in bubble_data):
            if not seg_result:
                print("Segmentation results not found. Running segmentation first...")
                seg_result = run_segmentation_stage(config, image_bgr)
            bubble_data = run_ocr_stage(config, image_bgr, seg_result)

    if "inpaint" in active_stages and cleaned_bgr is None:
        if not bubble_data:
             print("Bubble data not found. Running segmentation and OCR first...")
             if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
             bubble_data = run_ocr_stage(config, image_bgr, seg_result)
        cleaned_bgr = run_inpaint_stage(config, image_bgr, seg_result, bubble_data)

    if "translate" in active_stages:
        if bubble_data and any((b.get("ja_text") or "").strip() and b.get("en_text") is None for b in bubble_data):
            if any(b.get("ja_text") is None for b in bubble_data):
                 print("OCR results are incomplete. Running OCR first...")
                 if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
                 bubble_data = run_ocr_stage(config, image_bgr, seg_result)
            bubble_data = run_translation_stage(config, bubble_data)
        elif not bubble_data and "translate" == config.stage:
            print("Bubble data not found. Run 'seg' and 'ocr' stages before 'translate'.")
    
    if "typeset" in active_stages:
        if cleaned_bgr is None:
            print("Cleaned image not found. Running inpainting stage first...")
            if not bubble_data:
                if not seg_result: seg_result = run_segmentation_stage(config, image_bgr)
                bubble_data = run_ocr_stage(config, image_bgr, seg_result)
            cleaned_bgr = run_inpaint_stage(config, image_bgr, seg_result, bubble_data)

        if not config.use_placeholder_text and not any("en_text" in b for b in bubble_data if (b.get("ja_text") or "").strip()):
            print("Translated text not found. Running translation stage first...")
            bubble_data = run_translation_stage(config, bubble_data)
            
        run_typeset_stage(config, cleaned_bgr, bubble_data)


# --- Main Orchestrator ---

def main():
    """Main function to parse arguments and run the selected pipeline stage(s)."""
    overall_start_time = time.time()
    args = parse_args()

    backend_env = _BACKEND_DIR / ".env"
    if backend_env.exists():
        load_dotenv(backend_env, override=False)
    load_dotenv(override=False)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    image_paths = []
    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        supported_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        image_paths = sorted([p for p in input_path.glob("*") if p.suffix.lower() in supported_extensions])
    else:
        raise ValueError(f"Input path is not a valid file or directory: {input_path}")
    
    if not image_paths:
        print(f"No supported images found in '{input_path}'.")
        return

    total_images = len(image_paths)
    print(f"Found {total_images} image(s) to process.")

    for i, image_path in enumerate(image_paths):
        image_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"PROCESSING IMAGE {i+1}/{total_images}: {image_path.name}")
        print(f"{'='*80}")

        config = PipelineConfig(
            image_path=image_path,
            out_dir=Path(args.out_dir),
            seg_model_path=Path(args.seg_model),
            font_path=Path(args.font),
            stage=args.stage,
            debug=args.debug,
            force=args.force,
            use_placeholder_text=args.use_placeholder_text
        )
        ensure_dir(config.out_dir)
        ensure_dir(config.masks_dir)
        
        process_image(config)

        image_end_time = time.time()
        print(f"\nFinished processing {image_path.name} in {image_end_time - image_start_time:.2f} seconds.")

    overall_end_time = time.time()
    print(f"\n{'='*80}")
    print(f"Pipeline finished processing all {total_images} image(s) in {overall_end_time - overall_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()