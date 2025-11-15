Media Bias Classifier (TextCNN)
================================

Minimal guide to run the app.

Quick Start (Windows)
---------------------
1) (Optional, recommended for NVIDIA GPUs)
   - Install a CUDA-enabled PyTorch build via:
     https://pytorch.org/get-started/locally/

2) Install dependencies in your virtual environment:
   python -m pip install -r requirements.txt

3) Run the end-to-end quickstart (downloads data, trains, starts API):
    powershell -ExecutionPolicy Bypass -File .\scripts\quickstart.ps1   
    or
    .\scripts\quickstart.bat

4) Open the web UI in a browser:
   http://127.0.0.1:8000

Training Data (what quickstart uses)
------------------------------------
- Downloads several Kaggle datasets and combines them into `data/train.csv`, including:
  - `surajkarakulath/labelled-corpus-political-bias-hugging-face`
  - `gandpablo/news-articles-for-political-bias-classification`
  - `timospinde/mbib-media-bias-identification-benchmark`
  - `newsanalysis/political-bias-in-mainstream-media`
  - `tegmark/mediabias`
  - `timospinde/babe-media-bias-annotations-by-experts` (neutral headlines)
- Any raw formats are converted first (e.g., labelled corpus text files, BABE neutral headlines) and then **all** converted CSVs are merged before training.

That's it. The app will use GPU automatically if a CUDA-enabled PyTorch build and a compatible GPU are available; otherwise it runs on CPU.
