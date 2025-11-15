#!/usr/bin/env python3
"""
Converts the Kaggle dataset `labelled-corpus-political-bias-hugging-face`
from its folder-of-text-files structure into a single CSV consumable by
`prepare_kaggle_data.py`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


LABEL_MAPPING: Dict[str, str] = {
    "Left Data": "Left",
    "Right Data": "Right",
    "Center Data": "Neutral",
}


def collect_examples(root: Path, min_chars: int) -> List[Dict[str, str]]:
    """
    Walk the dataset folders, read every .txt file, and emit text/label rows.

    Args:
        root: Base directory containing the `Left/Right/Center` subfolders.
        min_chars: Minimum character count required to keep a document.

    Returns:
        List of dictionaries ready to be serialized to CSV.
    """
    rows: List[Dict[str, str]] = []
    for folder_name, label in LABEL_MAPPING.items():
        folder_path = root / folder_name
        if not folder_path.exists():
            continue
        # The dataset nests the files one level deeper with the same folder name.
        candidates = folder_path.rglob("*.txt")
        for txt_path in candidates:
            try:
                text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if len(text) < min_chars:
                continue
            rows.append({"text": text, "label": label})
    return rows


def main() -> int:
    """
    Parse CLI arguments, collect the labelled corpus rows, and write the merged CSV.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/labelled-corpus-political-bias-hugging-face",
        help="Root folder of the labelled corpus dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/labelled-corpus-political-bias-hugging-face/labelled_corpus.csv",
        help="Destination CSV that will contain text,label rows.",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=50,
        help="Ignore files shorter than this many characters.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[convert_labelled_corpus] Skip: {root} not found.")
        return 0

    rows = collect_examples(root, args.min_chars)
    if not rows:
        print(f"[convert_labelled_corpus] No rows collected under {root}.")
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[convert_labelled_corpus] Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
