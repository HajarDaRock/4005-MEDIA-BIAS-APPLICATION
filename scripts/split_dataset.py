from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train/val/test sets while preserving class ratios per label.

    Args:
        df: Full dataframe to split.
        label_col: Column name holding the class labels.
        train_frac: Fraction of each class assigned to the training split.
        val_frac: Fraction of each class assigned to the validation split.
        seed: RNG seed for deterministic shuffling.

    Returns:
        Tuple of (train_df, val_df, test_df) dataframes.
    """
    rng = random.Random(seed)

    train_rows = []
    val_rows = []
    test_rows = []

    for label, group in df.groupby(label_col):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)

        train_count = int(round(n * train_frac))
        val_count = int(round(n * val_frac))
        if train_count + val_count > n:
            val_count = max(0, n - train_count)
        test_count = n - train_count - val_count

        train_idx = indices[:train_count]
        val_idx = indices[train_count : train_count + val_count]
        test_idx = indices[train_count + val_count :]

        train_rows.append(group.loc[train_idx])
        if val_idx:
            val_rows.append(group.loc[val_idx])
        if test_idx:
            test_rows.append(group.loc[test_idx])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def main() -> int:
    """
    CLI entry point: loads data/train.csv, performs the stratified split,
    materializes the three CSVs, and prints a JSON summary of counts.
    """
    parser = argparse.ArgumentParser(description="Create stratified dataset splits.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path.")
    parser.add_argument("--output_dir", default="data", help="Directory for split CSVs.")
    parser.add_argument("--label_col", default="label", help="Name of the label column.")
    parser.add_argument("--train_frac", type=float, default=0.7, help="Fraction for training set.")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction for validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.train_frac <= 0 or args.val_frac <= 0:
        raise SystemExit("train_frac and val_frac must be positive.")
    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be less than 1.0.")

    df = pd.read_csv(args.input)
    label_counts = df[args.label_col].value_counts().to_dict()

    train_df, val_df, test_df = stratified_split(
        df, args.label_col, args.train_frac, args.val_frac, args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_split.csv"
    val_path = output_dir / "val_split.csv"
    test_path = output_dir / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary: Dict[str, Dict[str, int]] = {
        "total_rows": len(df),
        "label_counts": label_counts,
        "splits": {
            "train": train_df[args.label_col].value_counts().to_dict(),
            "val": val_df[args.label_col].value_counts().to_dict(),
            "test": test_df[args.label_col].value_counts().to_dict(),
        },
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote splits to: {train_path}, {val_path}, {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
