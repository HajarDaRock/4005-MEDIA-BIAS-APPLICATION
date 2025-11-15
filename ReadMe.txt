Media Bias Classifier (TextCNN)
================================

This app uses a PyTorch TextCNN classifier for article bias (Left, Right, Neutral).

Quick Start
-----------
1) Install dependencies inside your virtual environment:
   `python -m pip install -r requirements.txt`
2) Run the automation (PowerShell Core preferred, Windows PowerShell works too):
   `pwsh -File scripts/quickstart.ps1`
   - Installs dependencies, ensures Kaggle credentials, tries the Kaggle CLI first, then KaggleHub, and falls back to the sample CSV if needed.
   - The script builds `data/train.csv`, trains TextCNN (quick settings for the sample, larger config when real data exists), and finally launches the FastAPI server.
3) Manual training alternative:
   - Sample data: `python train_textcnn.py --train_csv data/sample_train.csv --epochs 2 --batch_size 8 --max_len 200`
   - Default path: `python train_textcnn.py --epochs 6 --batch_size 64 --max_len 400`
   - Custom CSV: `python train_textcnn.py --train_csv "C:\path\to\data.csv" --text_col text --label_col label --epochs 6 --batch_size 64 --max_len 400`

Kaggle Data Options
-------------------
- Credentials live at `%USERPROFILE%\.kaggle\kaggle.json` (or `KAGGLE_CONFIG_DIR`). The quickstart will:
  - Copy an existing file if `KAGGLE_JSON_PATH` is set.
  - Generate the file from `KAGGLE_USERNAME`/`KAGGLE_KEY`.
  - Prompt interactively when neither option is provided (input hidden).
- Verification snippets:
  - `python -c "import kaggle; print('ok')"`
  - `kaggle --version` (ensures the CLI entry point is on PATH; the downloader now uses this binary directly).
  - `powershell Test-Path "$env:USERPROFILE\.kaggle\kaggle.json"`
- KaggleHub fallback (no CLI extension):
  - `kagglehub` is already listed in `requirements.txt`; the quickstart automatically uses it when the Kaggle CLI is missing.
  - If the Kaggle CLI is installed but still errors (for example due to a mismatched interpreter), the quickstart catches the failure and immediately reruns the downloads via KaggleHub.
  - Manual invocation is also available: `python scripts/download_kagglehub.py` downloads all five datasets (`surajkarakulath/...`, `gandpablo/...`, `timospinde/...`, `newsanalysis/...`, `tegmark/...`) into `data/`.
- Special handling:
  - `labelled-corpus-political-bias-hugging-face` ships as folders of `.txt` files. The quickstart runs `scripts/convert_labelled_corpus.py` automatically to flatten these into a CSV so they participate in training.

Training Data & Metrics
-----------------------
- Source datasets powering `data/train.csv`:
  1. `surajkarakulath/labelled-corpus-political-bias-hugging-face`
  2. `gandpablo/news-articles-for-political-bias-classification`
  3. `timospinde/mbib-media-bias-identification-benchmark`
  4. `newsanalysis/political-bias-in-mainstream-media`
  5. `tegmark/mediabias`
- Latest label distribution (after running `scripts/quickstart.ps1` on 2025-11-13):

  | Label | Samples |
  | --- | --- |
  | Left | 12,556 |
  | Right | 9,874 |
  | Neutral | 5,461 |
  | **Total** | **27,891** |

- Recommended 70/15/15 stratified split (generated via `python scripts/split_dataset.py --input data/train.csv --output_dir data --train_frac 0.7 --val_frac 0.15 --seed 42`):

  | Split | Left | Right | Neutral | Total |
  | --- | --- | --- | --- | --- |
  | Train | 8,789 | 6,912 | 3,823 | 19,524 |
  | Validation | 1,883 | 1,481 | 819 | 4,183 |
  | Test | 1,884 | 1,481 | 819 | 4,184 |

- Keep `--seed 42` for reproducible splits. If you train directly from `data/train.csv`, pass `--val_frac 0.15` to `train_textcnn.py` and hold back `test_split.csv` strictly for final evaluation.

   Outputs:
   - models/textcnn_state.pt
   - models/vocab.json

4) Run the API:
   uvicorn main:app --reload
   Open http://127.0.0.1:8000

Notes
-----
- If you see "Model not available. Please train using train_textcnn.py.", the artifacts above are missing.
- Uses GPU automatically if available; otherwise falls back to CPU.
- Restricted outlets include foxnews.com, nytimes.com, thehill.com, ctvnews.ca, ipsos.com (see article_utils.py to edit).

Evaluation & Visualization
--------------------------
- After training, the script evaluates on the validation split and saves:
  - JSON report: `metrics/report_YYYYMMDD_HHMMSS.json` (per-class Precision, Recall, F1)
  - PNG chart:  `metrics/report_YYYYMMDD_HHMMSS.png` (bar chart of P/R/F1 per class)
  - History CSV: `metrics/history.csv` (macro/weighted F1 and config per run)
- Install extras if missing: `pip install scikit-learn matplotlib`

Run The Model
-------------
- Start the API (the new prompt):
  uvicorn main:app --reload

- Use the web UI:
  Open http://127.0.0.1:8000 and submit a news URL.

- Or call the API from PowerShell (Windows):
  $body = @{ url = "https://example.com/news/article" } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/classify" -Method Post -Body $body -ContentType "application/json"

- Or using curl:
  curl -s -X POST "http://127.0.0.1:8000/classify" -H "Content-Type: application/json" -d '{"url":"https://example.com/news/article"}'

- Results persist to `ExtractedData.xlsx` (sheet `BiasResults`).

End-to-End One-Liners (Windows PowerShell)
-----------------------------------------
- Quick demo with sample data:
  pip install -r requirements.txt; python train_textcnn.py --train_csv data/sample_train.csv --epochs 2 --batch_size 8 --max_len 200; uvicorn main:app --reload

- Build combined dataset then train and run:
  python prepare_kaggle_data.py --inputs "data/*.csv" --output data/train.csv; python train_textcnn.py --epochs 6 --batch_size 64 --max_len 400; uvicorn main:app --reload

How It Works (URL Flow)
-----------------------
- Frontend posts JSON to `/classify` with `{ "url": "..." }`.
- Backend validates the URL and checks a small restricted list.
- `fetch_article(url)` scrapes title and content.
- On success, `classify_bias(content)` runs the TextCNN model to predict `Left|Right|Neutral`.
- Response returns `{ "bias": "..." }` to the frontend.
- The URL, title, content, and predicted bias are appended to `ExtractedData.xlsx` (sheet `BiasResults`).

About vocab.json
----------------
- Required for inference: it contains `word2id` plus model settings so the tokenizer maps words to the same IDs used in training.
- Saved in a readable format with `token_count`, `top_tokens` (most frequent tokens in training), and explanatory `notes`.
- Tokens like "lawmakers" appear because they are common in news; they aren’t labeled as bias on their own. You can control vocabulary size via `--min_freq` in training.

Optional: Data Prep from Multiple Sources
----------------------------------------
Kaggle datasets (optional):
- Install Kaggle CLI and auth, then run:
  pwsh -File scripts/download_kaggle.ps1

Create a combined CSV from multiple folders (recursive search):
   python prepare_kaggle_data.py --inputs \
     "data/labelled-corpus-political-bias-hugging-face/**/*.csv" \
     "data/news-articles-for-political-bias-classification/**/*.csv" \
     "data/mbib-media-bias-identification-benchmark/**/*.csv" \
     "data/political-bias-in-mainstream-media/**/*.csv" \
     "data/mediabias/**/*.csv" \
     --output data/train.csv

If a dataset has numeric labels, you can map them:
   python prepare_kaggle_data.py --inputs "data/with_nums/*.csv" --output data/train.csv --numeric_map "0:Left,1:Right,2:Neutral"

Notes on paths (Windows):
- Use double backslashes or quotes for paths with spaces, e.g.:
  python train_textcnn.py --train_csv "D:\\Data Sets\\bias.csv"

Train/Val Split
---------------
- By default, the trainer performs a stratified split with `--val_frac 0.1`, preserving label ratios for Left/Right/Neutral.
- You can disable stratification with `--stratify false`.
