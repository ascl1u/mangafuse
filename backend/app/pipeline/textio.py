from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io import ensure_dir


def write_text_json(
    json_path: Path,
    polygons: List[List[Tuple[float, float]]],
) -> None:
    ensure_dir(json_path.parent)
    records = []
    for idx, poly in enumerate(polygons, start=1):
        records.append(
            {
                "id": idx,
                "polygon": [[float(x), float(y)] for x, y in poly],
            }
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"bubbles": records}, f, ensure_ascii=False, indent=2)


def read_text_json(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"text.json not found at {json_path}. Run segmentation stage first to generate polygons."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    # Normalize structure
    if "bubbles" not in data or not isinstance(data["bubbles"], list):
        data = {"bubbles": []}
    return data


def save_text_records(json_path: Path, bubbles: List[Dict[str, Any]]) -> None:
    ensure_dir(json_path.parent)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"bubbles": bubbles}, f, ensure_ascii=False, indent=2)


