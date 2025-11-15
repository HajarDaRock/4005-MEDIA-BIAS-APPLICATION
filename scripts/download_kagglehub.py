from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

import kagglehub


DATASETS = [
    {
        "id": "surajkarakulath/labelled-corpus-political-bias-hugging-face",
        "dir": "data/labelled-corpus-political-bias-hugging-face",
    },
    {
        "id": "gandpablo/news-articles-for-political-bias-classification",
        "dir": "data/news-articles-for-political-bias-classification",
    },
    {
        "id": "timospinde/mbib-media-bias-identification-benchmark",
        "dir": "data/mbib-media-bias-identification-benchmark",
    },
    {
        "id": "newsanalysis/political-bias-in-mainstream-media",
        "dir": "data/political-bias-in-mainstream-media",
    },
    {
        "id": "tegmark/mediabias",
        "dir": "data/mediabias",
    },
]

ROOT = Path(__file__).resolve().parents[1]


def copy_tree(src: Path, dest: Path) -> None:
    """Copy an entire directory tree from src to dest."""
    for dirpath, _, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = dest if rel == "." else dest / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src_file = Path(dirpath) / filename
            dest_file = target_dir / filename
            shutil.copy2(src_file, dest_file)


def materialize_dataset(download_path: Path, destination: Path) -> None:
    """Place the downloaded payload under the desired destination folder."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if download_path.is_file() and download_path.suffix == ".zip":
        with zipfile.ZipFile(download_path) as archive:
            archive.extractall(destination)
        return

    if download_path.is_file():
        shutil.copy2(download_path, destination / download_path.name)
        return

    copy_tree(download_path, destination)


def main() -> int:
    ROOT.joinpath("data").mkdir(exist_ok=True)
    for dataset in DATASETS:
        dataset_id = dataset["id"]
        dest = ROOT / dataset["dir"]
        print(f"Downloading {dataset_id} via kagglehub ...")
        download_path = Path(kagglehub.dataset_download(dataset_id))
        if not download_path.exists():
            raise FileNotFoundError(f"kagglehub returned missing path for {dataset_id}")
        materialize_dataset(download_path, dest)
        print(f"Dataset available under {dest}")
    print("All KaggleHub downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
