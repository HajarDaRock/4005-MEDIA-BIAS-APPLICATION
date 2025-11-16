Media Bias Classifier (TextCNN)
================================

**Goal:** classify news articles as **Left**, **Right**, or **Neutral** bias.  
This repository downloads several Kaggle datasets, cleans/normalises the text, balances the labels, trains a TextCNN model with PyTorch, and serves a FastAPI + HTML interface so you can paste an article and read its predicted leaning.

What’s Included
---------------
- `scripts/download_kaggle.ps1` / `download_kagglehub.py` – fetch the raw datasets.
- `prepare_kaggle_data.py` – merge, clean, and (optionally) balance everything into one CSV.
- `scripts/split_dataset.py` – build stratified train/val/test CSVs.
- `train_textcnn.py` – train/evaluate TextCNN, save weights + vocab, and produce metrics/plots.
- `main.py` + `static/` + `templates/` – FastAPI backend with a simple web UI.

Requirements
------------
- Windows 10/11 with Python 3.11.
- Kaggle API token saved to `%USERPROFILE%\.kaggle\kaggle.json`.
- Dependencies: `python -m pip install -r requirements.txt`.

GPU Acceleration
----------------
If you want to train/infer on GPU, install the matching build before running the project. The scripts auto-detect CUDA or DirectML (otherwise they fall back to CPU).

### NVIDIA GPUs (Command Prompt / PowerShell)
1. Activate your virtual environment in a Windows shell.
2. Install the CUDA-enabled wheels from https://pytorch.org/get-started/locally/ (example):
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. Run the project from that same shell; tensors use `cuda`.

### AMD / Intel GPUs on Windows (DirectML, no WSL)
1. Install the DirectML build (this installs the matching torch/torchvision versions):
   ```
   python -m pip install torch-directml
   ```
2. (Optional) Verify:
   ```
   python -c "import torch, torch_directml; d=torch_directml.device(); print('CUDA:', torch.cuda.is_available(), 'DirectML:', d)"
   ```
3. Run `train_textcnn.py` / `main.py` normally. The scripts pick DirectML whenever CUDA isn’t available; you can force it with `--device dml`.

Quick Start (recommended)
-------------------------
1. Clone/download the repo and create a virtual environment.
2. Put your Kaggle token in `%USERPROFILE%\.kaggle\kaggle.json`.
3. Install requirements: `python -m pip install -r requirements.txt` (plus the GPU-specific wheels if desired).
4. Launch the automation script (downloads data, prepares CSVs, trains, and starts the API):
   ```
   .\scripts\quickstart.bat
   ```
   or
   ```
   powershell -ExecutionPolicy Bypass -File .\scripts\quickstart.ps1
   ```
5. Open http://127.0.0.1:8000 to use the UI.
