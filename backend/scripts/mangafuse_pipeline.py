# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import os
import cv2
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field

# --- Path Setup ---
# Ensure the backend package root is on sys.path when running from repo root
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# --- Local Imports ---
from app.pipeline.utils.io import ensure_dir, read_image_bgr, save_png
from app.pipeline.utils.masks import save_masks
from app.pipeline.segmentation.yolo import run_segmentation
from app.pipeline.utils.visualization import make_overlay
from app.pipeline.utils.textio import write_text_json, read_text_json, save_text_records
from app.pipeline.ocr.crops import tight_crop_from_mask
from app.pipeline.ocr.preprocess import binarize_for_ocr
from app.pipeline.ocr.engine import MangaOcrEngine
from app.pipeline.translate.gemini import GeminiTranslator
from app.pipeline.inpaint.lama import run_inpainting
from app.pipeline.typeset.model import BubbleText
from app.pipeline.typeset.render import render_typeset
from app.pipeline.inpaint.text_mask import build_text_inpaint_mask

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
    masks_dir: Path = field(init=False)
    json_path: Path = field(init=False)
    overlay_path: Path = field(init=False)
    combined_mask_path: Path = field(init=False)
    text_mask_path: Path = field(init=False)
    cleaned_path: Path = field(init=False)
    final_path: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths after the main fields are set."""
        self.masks_dir = self.out_dir / "masks"
        self.json_path = self.out_dir / "text.json"
        self.overlay_path = self.out_dir / "segmentation_overlay.png"
        self.combined_mask_path = self.masks_dir / "all_mask.png"
        self.text_mask_path = self.masks_dir / "text_mask.png"
        self.cleaned_path = self.out_dir / "cleaned.png"
        self.final_path = self.out_dir / "final.png"

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaFuse AI Pipeline (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input page image")
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

def run_segmentation_stage(config: PipelineConfig):
    """Detects speech bubbles and saves their masks and polygons."""
    print("--- Running Stage: Segmentation ---")
    stage_name = "seg"
    
    # Check if artifacts already exist
    if not config.force and all(p.exists() for p in [config.overlay_path, config.combined_mask_path, config.json_path]):
        print("Skipping segmentation: All artifacts already exist. Use --force to override.")
        print(json.dumps({"stage": stage_name, "status": "skip", "reason": "artifacts_exist"}))
        return

    image_bgr = read_image_bgr(config.image_path)
    h, w = image_bgr.shape[:2]
    
    print(f"Running bubble segmentation on {config.image_path.name}...")
    result = run_segmentation(image_bgr=image_bgr, seg_model_path=config.seg_model_path)
    polygons = result.get("polygons", [])
    
    # Save artifacts
    save_masks(config.masks_dir, result.get("masks", []), config.combined_mask_path, image_height=h, image_width=w)
    write_text_json(config.json_path, polygons)
    
    # Create and save visualization
    overlay_bgr = make_overlay(image_bgr, result.get("masks", []), polygons, result.get("confidences", []))
    save_png(config.overlay_path, overlay_bgr)

    if config.debug:
        debug_path = config.out_dir / "segmentation_overlay_ids.png"
        debug_img = overlay_bgr.copy()
        for idx, poly in enumerate(polygons, start=1):
            if poly:
                x0, y0 = int(poly[0][0]), int(poly[0][1])
                cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(debug_img, str(idx), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        save_png(debug_path, debug_img)
        print(f"Saved debug overlay with bubble IDs to {debug_path}")

    print(f"Segmentation complete. Found {len(polygons)} bubbles.")
    print(json.dumps({"stage": stage_name, "status": "ok", "num_bubbles": len(polygons)}))

def run_ocr_stage(config: PipelineConfig):
    """Performs OCR on detected bubbles to extract Japanese text."""
    print("--- Running Stage: OCR ---")
    stage_name = "ocr"
    if not config.json_path.exists():
        raise FileNotFoundError(f"{config.json_path} not found. Run the 'seg' stage first.")

    data = read_text_json(config.json_path)
    bubbles = data.get("bubbles", [])
    if not bubbles:
        print("No bubbles found in JSON, skipping OCR.")
        print(json.dumps({"stage": stage_name, "status": "ok", "num_bubbles": 0, "updated_ocr": 0}))
        return

    image_bgr = read_image_bgr(config.image_path)
    ocr_engine = MangaOcrEngine()
    
    updated_count = 0
    bubbles_to_process = []
    for rec in bubbles:
        has_ja = isinstance(rec.get("ja_text"), str) and rec["ja_text"].strip()
        if not has_ja or config.force:
            bubbles_to_process.append(rec)
    
    if not bubbles_to_process:
        print("Skipping OCR: All bubbles already have Japanese text. Use --force to re-run.")
        print(json.dumps({"stage": stage_name, "status": "skip", "reason": "all_bubbles_processed"}))
        return

    print(f"Performing OCR on {len(bubbles_to_process)} out of {len(bubbles)} bubbles...")
    for rec in bubbles_to_process:
        bubble_id = int(rec.get("id"))
        mask_path = config.masks_dir / f"{bubble_id}.png"
        
        # --- THIS IS THE CORRECTED LINE ---
        # Unpack the tuple to get only the image array
        crop_bgr, _ = tight_crop_from_mask(image_bgr, mask_path, rec.get("polygon", []))
        
        try:
            ja_text = ocr_engine.run(binarize_for_ocr(crop_bgr))
        except Exception:
            ja_text = ocr_engine.run(crop_bgr)
        rec["ja_text"] = ja_text
        updated_count += 1
        print(f"  - Bubble {bubble_id}: {ja_text[:30]}...")

    save_text_records(config.json_path, bubbles)
    print(f"OCR complete. Updated {updated_count} bubbles.")
    print(json.dumps({"stage": stage_name, "status": "ok", "num_bubbles": len(bubbles), "updated_ocr": updated_count}))

def run_translation_stage(config: PipelineConfig):
    """Translates Japanese text to English using the Gemini API."""
    print("--- Running Stage: Translation ---")
    stage_name = "translate"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    if not config.json_path.exists():
        raise FileNotFoundError(f"{config.json_path} not found. Run the 'seg' and 'ocr' stages first.")
    
    # Ensure all bubbles have Japanese text first
    run_ocr_stage(config)
    
    data = read_text_json(config.json_path)
    bubbles = data.get("bubbles", [])
    if not bubbles:
        print("No bubbles to translate.")
        return

    texts_to_translate = []
    indices_to_update = []
    ja_text_count = 0

    for i, rec in enumerate(bubbles):
        ja_text = (rec.get("ja_text") or "").strip()
        if ja_text:
            ja_text_count += 1
            has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip()
            if not has_en or config.force:
                texts_to_translate.append(ja_text)
                indices_to_update.append(i)

    print(f"Found {ja_text_count} bubbles with Japanese text.")
    if not texts_to_translate:
        print("Skipping translation: All bubbles already have English text. Use --force to re-translate.")
        print(json.dumps({"stage": stage_name, "status": "skip", "reason": "all_bubbles_translated"}))
        return
        
    print(f"Translating text for {len(texts_to_translate)} bubbles...")
    translator = GeminiTranslator(api_key=api_key)
    translated_texts = translator.translate_batch(texts_to_translate)

    for i, en_text in zip(indices_to_update, translated_texts):
        bubbles[i]["en_text"] = en_text
        print(f"  - Bubble {bubbles[i]['id']}: \"{bubbles[i]['ja_text']}\" -> \"{en_text}\"")

    save_text_records(config.json_path, bubbles)
    
    print(f"Translation complete. Generated {len(translated_texts)} English translations.")
    print(json.dumps({
        "stage": stage_name, 
        "status": "ok", 
        "num_bubbles_with_ja_text": ja_text_count, 
        "num_en_text_generated": len(translated_texts)
    }))

def run_inpaint_stage(config: PipelineConfig):
    """Removes original text from the image."""
    print("--- Running Stage: Inpainting ---")
    stage_name = "inpaint"
    
    if not config.force and config.cleaned_path.exists():
        print(f"Skipping inpainting: {config.cleaned_path.name} already exists. Use --force to override.")
        print(json.dumps({"stage": stage_name, "status": "skip", "reason": "artifact_exists"}))
        return

    # Build a precise text-only mask for better inpainting quality
    if config.force or not config.text_mask_path.exists():
        print("Text inpaint mask not found or --force is set, generating a new one...")
        if not config.json_path.exists():
            raise FileNotFoundError("text.json not found. Run segmentation stage first.")
        
        data = read_text_json(config.json_path)
        bubbles = data.get("bubbles", [])
        image_bgr = read_image_bgr(config.image_path)
        text_mask = build_text_inpaint_mask(image_bgr, config.masks_dir, bubbles)
        save_png(config.text_mask_path, cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR))
        print(f"Saved text inpaint mask to {config.text_mask_path}")

    # Use the text-only mask if it's not empty, otherwise fall back to the combined bubble mask
    use_mask_path = config.combined_mask_path
    text_mask_gray = cv2.imread(str(config.text_mask_path), cv2.IMREAD_GRAYSCALE)
    if text_mask_gray is not None and text_mask_gray.sum() > 0:
        use_mask_path = config.text_mask_path
        print("Using precise text mask for inpainting.")
    else:
        print("Text mask is empty, falling back to combined bubble mask for inpainting.")

    print(f"Running inpainting on {config.image_path.name} using mask {use_mask_path.name}...")
    result_bgr = run_inpainting(config.image_path, use_mask_path)
    save_png(config.cleaned_path, result_bgr)
    
    print(f"Inpainting complete. Saved cleaned image to {config.cleaned_path}")
    print(json.dumps({"stage": stage_name, "status": "ok"}))

def run_typeset_stage(config: PipelineConfig):
    """Renders the translated text back into the bubbles."""
    print("--- Running Stage: Typesetting ---")
    stage_name = "typeset"

    for p in [config.cleaned_path, config.json_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Run previous stages first.")

    data = read_text_json(config.json_path)
    bubbles = data.get("bubbles", [])
    records = []
    
    if config.use_placeholder_text:
        print("Using placeholder text for typesetting.")
        records = [
            BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text="Placeholder Text")
            for rec in bubbles
        ]
    else:
        for rec in bubbles:
            # Prefer English text, fall back to Japanese, then to empty
            text = (rec.get("en_text") or rec.get("ja_text") or "").strip()
            records.append(
                BubbleText(bubble_id=int(rec.get("id")), polygon=rec.get("polygon") or [], text=text)
            )

    print(f"Typesetting text for {len(records)} bubbles...")
    debug_overlay_path = config.out_dir / "typeset_debug.png" if config.debug else None
    
    render_typeset(
        cleaned_path=config.cleaned_path,
        output_final_path=config.final_path,
        records=records,
        font_path=config.font_path,
        margin_px=6,
        debug=config.debug,
        debug_overlay_path=debug_overlay_path,
    )

    print(f"Typesetting complete. Saved final image to {config.final_path}")
    if config.debug and debug_overlay_path:
        print(f"Saved debug typesetting overlay to {debug_overlay_path}")
        
    print(json.dumps({"stage": stage_name, "status": "ok", "num_bubbles": len(records)}))

# --- Main Orchestrator ---

def main():
    """Main function to parse arguments and run the selected pipeline stage(s)."""
    start_time = time.time()
    args = parse_args()

    # Load environment variables
    backend_env = _BACKEND_DIR / ".env"
    if backend_env.exists():
        load_dotenv(backend_env, override=False)
    load_dotenv(override=False)

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Create config and ensure output directories exist
    config = PipelineConfig(
        image_path=Path(args.image),
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
    
    print(f"Starting MangaFuse AI Pipeline for '{config.image_path.name}'...")
    print(f"Output directory: '{config.out_dir}'")
    print(f"Stage(s) to run: '{config.stage}'")
    
    # Stage-based execution
    if config.stage == "all":
        run_segmentation_stage(config)
        run_ocr_stage(config)
        run_translation_stage(config)
        run_inpaint_stage(config)
        run_typeset_stage(config)
    elif config.stage == "seg":
        run_segmentation_stage(config)
    elif config.stage == "ocr":
        run_ocr_stage(config)
    elif config.stage == "translate":
        run_translation_stage(config)
    elif config.stage == "inpaint":
        run_inpaint_stage(config)
    elif config.stage == "typeset":
        # Typesetting depends on previous stages; run them if needed.
        run_inpaint_stage(config) # Inpainting must exist before typesetting
        run_typeset_stage(config)

    end_time = time.time()
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()