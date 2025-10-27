import argparse
import os
import glob
import re
from typing import List, Optional, Dict, Tuple

import pandas as pd


TEXT_CANDIDATES = [
    "text",
    "content",
    "article",
    "body",
    "full_text",
    "article_text",
    "content_text",
    "news",
    "document",
]

TITLE_CANDIDATES = [
    "title",
    "headline",
    "subject",
]

LABEL_CANDIDATES = [
    "label",
    "bias",
    "political_bias",
    "political_leaning",
    "leaning",
    "mbib_label",
    "class",
    "target",
    "ideology",
    "stance",
    "party",
    "source_bias",
    "bias_rating",
]


LEFT_TOKENS = {
    "left",
    "leanleft",
    "leftleaning",
    "leftofcenter",
    "centreleft",
    "centerleft",
    "leftcenterleft",
    "democrat",
    "democratic",
    "liberal",
    "progressive",
    "farleft",
}
RIGHT_TOKENS = {
    "right",
    "leanright",
    "rightleaning",
    "rightofcenter",
    "centreright",
    "centerright",
    "rightcenterright",
    "republican",
    "gop",
    "conservative",
    "farright",
}
NEUTRAL_TOKENS = {
    "neutral",
    "center",
    "centre",
    "centrist",
    "moderate",
    "balanced",
    "unbiased",
    "leastbiased",
    "leastbias",
}


def normalize_label_str(s: str) -> str:
    s = s.strip().lower()
    # collapse separators
    s = re.sub(r"[\s\-_/]+", "", s)
    return s


def map_label(val, num_map: Optional[Dict[int, str]] = None) -> Optional[str]:
    if pd.isna(val):
        return None
    # numeric mapping first if provided
    try:
        if num_map is not None:
            ival = int(val)
            lab = num_map.get(ival)
            return lab if lab in {"Left", "Right", "Neutral"} else None
    except Exception:
        pass

    s = str(val)
    ns = normalize_label_str(s)
    if ns in LEFT_TOKENS or ("left" in ns and "right" not in ns):
        return "Left"
    if ns in RIGHT_TOKENS or ("right" in ns and "left" not in ns):
        return "Right"
    if ns in NEUTRAL_TOKENS or "center" in ns or "centre" in ns:
        return "Neutral"
    # unknown
    return None


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def combine_title_text(df: pd.DataFrame, title_col: Optional[str], text_col: str) -> pd.Series:
    if title_col and title_col in df.columns:
        return (df[title_col].astype(str).str.strip() + ". " + df[text_col].astype(str).str.strip()).str.strip()
    return df[text_col].astype(str)


def parse_num_map(s: Optional[str]) -> Optional[Dict[int, str]]:
    if not s:
        return None
    mapping: Dict[int, str] = {}
    parts = [p for p in s.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            ki = int(k.strip())
            vv = v.strip().capitalize()
            if vv in {"Left", "Right", "Neutral"}:
                mapping[ki] = vv
        except Exception:
            continue
    return mapping if mapping else None


def process_file(path: str, num_map: Optional[Dict[int, str]]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[SKIP] Failed to read {path}: {e}")
        return pd.DataFrame(columns=["text", "label"])

    text_col = pick_column(df, TEXT_CANDIDATES)
    if not text_col:
        # try to build from multiple columns if available
        print(f"[WARN] No standard text column in {path}. Available: {df.columns.tolist()}")
        return pd.DataFrame(columns=["text", "label"])

    title_col = pick_column(df, TITLE_CANDIDATES)
    label_col = pick_column(df, LABEL_CANDIDATES)
    if not label_col:
        print(f"[WARN] No label column found in {path}. Available: {df.columns.tolist()}")
        return pd.DataFrame(columns=["text", "label"])

    series_text = combine_title_text(df, title_col, text_col)
    series_label = df[label_col].apply(lambda x: map_label(x, num_map=num_map))

    out = pd.DataFrame({"text": series_text, "label": series_label})
    out = out.dropna(subset=["text", "label"]).copy()
    # basic length filter
    out = out[out["text"].str.len() > 30]
    return out


def main():
    ap = argparse.ArgumentParser(description="Prepare Kaggle political bias datasets into a single CSV for training")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths or globs. Example: data/ds1/*.csv data/ds2/file.csv")
    ap.add_argument("--output", default="data/train.csv", help="Output CSV path")
    ap.add_argument("--numeric_map", default="", help='Optional numeric label mapping, e.g., "0:Left,1:Right,2:Neutral"')
    args = ap.parse_args()

    # Expand globs
    files: List[str] = []
    for pat in args.inputs:
        matched = glob.glob(pat)
        if not matched and os.path.isdir(pat):
            matched = glob.glob(os.path.join(pat, "*.csv"))
        files.extend(matched)

    if not files:
        raise SystemExit("No input CSV files found. Provide file paths or globs.")

    num_map = parse_num_map(args.numeric_map)
    if num_map:
        print(f"Using numeric label mapping: {num_map}")

    parts: List[pd.DataFrame] = []
    for f in files:
        print(f"Processing: {f}")
        df = process_file(f, num_map)
        if not df.empty:
            parts.append(df)
    if not parts:
        raise SystemExit("No usable data extracted. Check column names or provide --numeric_map.")

    all_df = pd.concat(parts, axis=0, ignore_index=True)
    # Drop duplicates
    all_df = all_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_df.to_csv(args.output, index=False)
    print(f"Wrote {len(all_df)} rows to {args.output}")
    print(all_df["label"].value_counts())


if __name__ == "__main__":
    main()

