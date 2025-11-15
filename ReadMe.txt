Media Bias Classifier (TextCNN)
================================

This project trains a TextCNN model to classify news articles as **Left**,
**Right**, or **Neutral**. The pipeline downloads several Kaggle datasets,
cleans and balances them, trains the model, and exposes a small API / web UI.

Quick Start (Windows)
---------------------
1) (Optional but recommended for NVIDIA GPUs)
   - Install a CUDA-enabled PyTorch build via:
     https://pytorch.org/get-started/locally/

2) Install dependencies in your virtual environment:

   `python -m pip install -r requirements.txt`

3) Run the end‑to‑end quickstart (downloads data, prepares it, trains, then
   starts the API):

   `powershell -ExecutionPolicy Bypass -File .\scripts\quickstart.ps1`  
   or  
   `.\scripts\quickstart.bat`

4) Open the web UI in a browser:

   `http://127.0.0.1:8000`

Data sources and conversion
---------------------------
Quickstart targets these Kaggle datasets and places them under `data/`:

- `surajkarakulath/labelled-corpus-political-bias-hugging-face`
- `gandpablo/news-articles-for-political-bias-classification`
- `timospinde/mbib-media-bias-identification-benchmark`
- `newsanalysis/political-bias-in-mainstream-media`
- `tegmark/mediabias`
- `timospinde/babe-media-bias-annotations-by-experts`

Most of these already contain CSV files with article text and a bias label and
can be read directly. Two datasets need conversion before they can be used:

- **Labelled corpus (folders of `.txt` files)**  
  `scripts/convert_datasets_tocsv.py` scans the `Left Data` / `Right Data` /
  `Center Data` folders, reads each article, and writes a single
  `data/labelled-corpus-political-bias-hugging-face/labelled_corpus.csv`
  containing `text,label` columns.

- **BABE media bias annotations**  
  The BABE dataset ships neutral headline data and several CSVs with segment‑
  level labels. The conversion script:

  - builds `babe_neutral.csv` from `news_headlines_usa_neutral.csv`
  - builds `babe_lrc.csv` from `final_labels_MBIC/SG1/SG2.csv` using the
    `type` column (`left`, `right`, `center`)  
  - merges both into `babe_all.csv` with canonical labels `Left`,
    `Right`, `Neutral`

Cleaning and balancing the data
-------------------------------
Once all relevant CSVs exist, `prepare_kaggle_data.py` is used to build a
single training file:

- Input CSVs are discovered from the Kaggle folders and from
  `labelled_corpus.csv` / `babe_all.csv`.
- For each file, it tries to locate:
  - a **text** column (`text`, `content`, `article`, `body`, etc.)
  - an optional **title/headline**, which is prepended to the text when present
  - a **label** column (`label`, `bias`, `leaning`, `political_bias`, etc.)
- Labels are normalised into the three canonical classes:
  - values containing tokens like `"left"`, `"liberal"`, `"progressive"` → `Left`
  - values containing `"right"`, `"conservative"`, `"gop"` → `Right`
  - values like `"center"`, `"neutral"`, `"leastbiased"` → `Neutral`
- Basic cleaning steps:
  - articles shorter than ~30 characters are dropped
  - duplicates are removed based on `text`

The combined dataframe is optionally **balanced** with
`--balance_labels` (used by quickstart):

- majority labels are **downsampled** so each of `Left`, `Right`, and `Neutral`
  has the same number of examples
- this happens after merging all datasets, so large Neutral‑only corpora cannot
  overwhelm Left/Right examples

The resulting balanced CSV is written to `data/train.csv`, and
`prepare_kaggle_data.py` prints the label distribution before and after
balancing.

Train/val/test splits
---------------------
`scripts/split_dataset.py` takes `data/train.csv` and produces three splits:

- `data/train_split.csv` – used for model training
- `data/val_split.csv` – used for validation and metrics
- `data/test_split.csv` – held‑out test set for future evaluation

The split is **stratified** by label so each split preserves the overall
Left/Right/Neutral proportions from `data/train.csv`. No oversampling is done
here; if the input is balanced, the splits are balanced as well.

The script prints a JSON summary of row counts and label counts for each split
and also writes this information to `metrics/split_summary.json` for later
inspection.

Model training and metrics
--------------------------
`train_textcnn.py` trains a TextCNN classifier using PyTorch:

- `--train_csv` / `--val_csv` / `--test_csv` control which splits are used.
  Quickstart trains on `train_split.csv`, validates on `val_split.csv`, and
  records label counts for `test_split.csv`.
- A small tokenizer (`simple_tokenize`) lowercases text and extracts word
  tokens.
- A vocabulary is built on the training texts only (respecting `--min_freq`).

During training:

- training and validation loss / accuracy are printed per epoch
- the best model (by validation loss) is saved to `models/textcnn_state.pt`
- vocabulary and model config are saved to `models/vocab.json`

After training, the script evaluates on the validation set and writes a rich
metrics report to `metrics/report_<timestamp>.json`, including:

- per‑class **precision**, **recall**, **F1** (via `sklearn`)
- overall **accuracy**
- macro / weighted averages
- **micro‑averaged** precision/recall/F1
- a **confusion matrix** over `Left`, `Right`, `Neutral`
- dataset information:
  - rows and label counts for train / val / test splits
  - paths to the CSVs used
- randomisation details:
  - training seed
  - whether validation came from an external CSV or an internal split

A CSV history of runs is appended to `metrics/history.csv`, capturing
timestamp, loss, accuracy, the aggregate metrics above, key hyperparameters,
and the training seed.

Visualisations
--------------
For each training run, several plots are written to the `metrics/` folder:

- `report_<timestamp>.png` – combined bar chart of per‑class precision,
  recall, and F1
- `report_<timestamp>_precision.png` – precision by class
- `report_<timestamp>_recall.png` – recall by class
- `report_<timestamp>_f1score.png` – F1 by class

These make it easy to quickly see how the model is performing on Left, Right,
and Neutral separately and to compare runs over time.

Notes
-----
- The application will use the GPU automatically if a CUDA‑enabled PyTorch
  build and a compatible GPU are available; otherwise it runs on CPU.
- Many auxiliary CSVs inside the Kaggle bundles (e.g., phrase count tables,
  raw annotation files without article‑level labels) are detected but skipped.
  They are not suitable as direct training examples because they do not map to
  full articles with a single Left/Right/Neutral label.
