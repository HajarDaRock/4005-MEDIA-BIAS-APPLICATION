NEW: TextCNN Classifier (PyTorch)
=================================

This app now uses a PyTorch TextCNN classifier instead of llama-cpp. The FastAPI interface stays the same.

Quick Start
-----------
1) Install dependencies (in your venv):
   pip install torch pandas fastapi uvicorn requests beautifulsoup4 openpyxl jinja2 aiofiles

2) Prepare your training data CSV with two columns:
   - text: the article text
   - label: one of Left, Right, Neutral (case-insensitive) or numeric 0/1/2

   Example rows:
   text,label
   "The proposed tax cuts ...",Right
   "Government expands healthcare ...",Left
   "The report states ...",Neutral

3) Train the model and save artifacts to models/:
   python train_textcnn.py --train_csv path/to/your.csv --text_col text --label_col label --epochs 6 --batch_size 64 --max_len 400

   Outputs:
   - models/textcnn_state.pt
   - models/vocab.json

4) Run the app:
   uvicorn main:app --reload
   Open http://127.0.0.1:8000

Notes
-----
- If the app returns "Model not available. Please train using train_textcnn.py.", it means the artifacts above are missing.
- GPU is used automatically if available; otherwise CPU is used.

Use Kaggle Datasets (Optional)
------------------------------
1) Download and unzip any of these datasets under `data/` (examples you shared):
   - labelled-corpus-political-bias-hugging-face
   - news-articles-for-political-bias-classification
   - source-based-news-classification
   - mbib-media-bias-identification-benchmark
   - political-bias-in-mainstream-media

2) Prepare a single training CSV from multiple sources:
   python prepare_kaggle_data.py --inputs "data/dataset1/*.csv" "data/dataset2/*.csv" --output data/train.csv

   If a dataset uses numeric labels, add a mapping (example):
   python prepare_kaggle_data.py --inputs "data/dataset_with_nums/*.csv" --output data/train.csv --numeric_map "0:Left,1:Right,2:Neutral"

3) Train using the generated `data/train.csv`:
   python train_textcnn.py --train_csv data/train.csv --text_col text --label_col label --epochs 6 --batch_size 64 --max_len 400

Legacy llama-cpp Instructions (Deprecated)
-----------------------------------------
First run:
pip install llama-cpp-python

Next run:
pip install fastapi uvicorn requests beautifulsoup4 pandas openpyxl jinja2 aiofiles

===============================================================================

To run it on your CPU, change classify_articles.py to look like:

//Code
from llama_cpp import Llama

def classify_bias(article_text):
    shortened = ' '.join(article_text.split()[:400])

    # CPU-only: No GPU layers enabled
    llm = Llama(
        model_path="models/mistral-7b-openorca.Q5_K_S.gguf",
        n_ctx=2048,
        chat_format="chatml",
        verbose=False
        # Removed n_gpu_layers to ensure CPU usage
    ) .... (Everything else stays the same)
//End code

--------
Run the FastAPI App
------------

In terminal:  uvicorn main:app --reload

Go to: http://127.0.0.1:8000
===============================================================================

TO RUN ON YOUR GPU (3060):

PREREQUISITES
-------------
A. CUDA Toolkit 12.8
- Download: https://developer.nvidia.com/cuda-downloads
- Confirm with: `nvcc --version`

B. NVIDIA GPU Driver (≥ 535)
- Check with: `nvidia-smi`

C. Visual Studio Build Tools (with C++ support)
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select “Desktop development with C++” during installation

D. CMake (3.27+ recommended)
- Download: https://cmake.org/download/
- Confirm with: `cmake --version`

==========================================================================
STEP 1: Set Up Python Environment
==========================================================================

# (Optional) Create a virtual environment
python -m venv venv
.
env\Scripts ctivate

# Install all required dependencies
pip install fastapi uvicorn requests beautifulsoup4 pandas openpyxl jinja2 aiofiles

==========================================================================
STEP 2: Build llama-cpp-python with CUDA (GGML_CUDA)
==========================================================================

# Run these in PowerShell (not Command Prompt!)
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

#Wait. This takes some time

# Run this. It will create a dist folder with the wheel file inside. 
pip wheel llama-cpp-python --wheel-dir dist

# This way if the wheel breaks, you can just run this line of code and it will create a new wheel file.
pip install dist/llama_cpp_python-<version>.whl

# Replace <version> with the actual version number of the wheel file you created. You can find it in the dist folder.

==========================================================================
STEP 3: Update classify_articles.py for GPU
==========================================================================

To run it on your GPU, change classify_articles.py to look like:

//Code
from llama_cpp import Llama

def classify_bias(article_text):
    shortened = ' '.join(article_text.split()[:400])
   
    # Reinitialize the model on each call (stateless behavior)
    llm = Llama(
    model_path="models/mistral-7b-openorca.Q5_K_S.gguf",
    n_ctx=2048,
    chat_format="chatml",
    verbose=False,
    n_gpu_layers=-1 #Push Operations to GPU
    ) ... (Everything else stays)
//End code

==========================================================================
STEP 4: Run the FastAPI App
==========================================================================

uvicorn main:app --reload

# Open in browser:
http://127.0.0.1:8000

==========================================================================
STEP ???: IF ALL ELSE FAILS
==========================================================================

Upload all the main files (HTML, .py) to CHAT GPT and ask it for help
