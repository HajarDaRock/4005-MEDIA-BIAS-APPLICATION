#!/usr/bin/env python3
"""
Conversions for non-CSV Kaggle datasets into simple text/label CSVs used by
`prepare_kaggle_data.py`.

Currently handles:
- `labelled-corpus-political-bias-hugging-face` (folders of .txt files)
- `babe-media-bias-annotations-by-experts` neutral headlines
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd


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


def convert_labelled_corpus() -> None:
    """
    Convert the labelled corpus text folders into a CSV.
    """
    root = Path("data/labelled-corpus-political-bias-hugging-face")
    output_path = root / "labelled_corpus.csv"

    if not root.exists():
        print(f"[convert_labelled_corpus] Skip: {root} not found.")
        return

    rows = collect_examples(root, min_chars=50)
    if not rows:
        print(f"[convert_labelled_corpus] No rows collected under {root}.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[convert_labelled_corpus] Wrote {len(rows)} rows to {output_path}")


def convert_babe_neutral() -> None:
    """
    Convert BABE neutral headlines into a simple Neutral CSV.

    Reads `news_headlines_usa_neutral.csv` (if present) from the BABE dataset
    and writes `babe_neutral.csv` with text/label columns.
    """
    babe_root = Path("data/babe-media-bias-annotations-by-experts")
    if not babe_root.exists():
        print(f"[convert_babe] Skip: {babe_root} not found.")
        return

    candidates = list(babe_root.rglob("news_headlines_usa_neutral.csv"))
    if not candidates:
        print(f"[convert_babe] Skip: news_headlines_usa_neutral.csv not found under {babe_root}")
        return

    source = candidates[0]
    try:
        df = pd.read_csv(source, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[convert_babe] Failed to read {source}: {exc}")
        return

    if "title" not in df.columns:
        print(f"[convert_babe] Unexpected columns in {source}: {df.columns.tolist()}")
        return

    texts = df["title"].astype(str).str.strip()
    out_df = pd.DataFrame({"text": texts, "label": "Neutral"})
    out_df = out_df[out_df["text"].str.len() > 30].copy()

    output_path = babe_root / "babe_neutral.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[convert_babe] Wrote {len(out_df)} neutral rows to {output_path}")


def main() -> int:
    """
    Run all conversion steps for non-CSV datasets.
    """
    convert_labelled_corpus()
    convert_babe_neutral()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
