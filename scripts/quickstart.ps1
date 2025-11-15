<#
Quickstart (Kaggle preferred):
 - Assumes PyTorch is installed (CUDA build recommended for NVIDIA GPUs; see https://pytorch.org/get-started/locally/)
 - Tries to download and prepare Kaggle datasets into data/train.csv
 - Falls back to sample data if Kaggle not configured or no CSVs found
 - Trains and starts the API

Usage:
  powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
  or
  scripts\quickstart.bat
#>

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $here '..')
Set-Location $repo

$datasetSources = @(
  'surajkarakulath/labelled-corpus-political-bias-hugging-face',
  'gandpablo/news-articles-for-political-bias-classification',
  'timospinde/mbib-media-bias-identification-benchmark',
  'newsanalysis/political-bias-in-mainstream-media',
  'tegmark/mediabias',
  'timospinde/babe-media-bias-annotations-by-experts'
)
Write-Host 'Targeted Kaggle datasets:' -ForegroundColor Cyan
foreach ($src in $datasetSources) {
  Write-Host " - $src"
}
Write-Host ''

function Convert-SecureStringToPlain {
  param([System.Security.SecureString]$Secure)
  if (-not $Secure) { return $null }
  $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
  try 
  {
    return [Runtime.InteropServices.Marshal]::PtrToStringUni($ptr)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
  }
}

function Ensure-KaggleCredentials {
  param(
    [string]$cfgDir,
    [string]$cfgPath
  )

  if (Test-Path $cfgPath) {
    return $true
  }

  $jsonPath = $env:KAGGLE_JSON_PATH
  if ($jsonPath -and (Test-Path $jsonPath)) {
    New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
    Copy-Item -Path $jsonPath -Destination $cfgPath -Force
    Write-Host "Copied Kaggle credentials from $jsonPath to $cfgPath." -ForegroundColor Cyan
    return $true
  }

  $username = $env:KAGGLE_USERNAME
  $key = $env:KAGGLE_KEY
  if ($username -and $key) {
    New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
    $json = @{ username = $username; key = $key } | ConvertTo-Json -Compress
    Set-Content -Path $cfgPath -Value $json -Encoding ASCII
    Write-Host "Created $cfgPath from KAGGLE_USERNAME/KAGGLE_KEY." -ForegroundColor Cyan
    return $true
  }

  if (-not $env:CI) {
    Write-Host "Kaggle credentials not found. Please enter them to create $cfgPath." -ForegroundColor Yellow
    $promptUser = Read-Host 'Enter Kaggle username (leave blank to skip)'
    if ($promptUser) {
      $secureKey = Read-Host 'Enter Kaggle API key (input hidden)' -AsSecureString
      $keyPlain = Convert-SecureStringToPlain -Secure $secureKey
      if ($keyPlain) {
        New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
        $json = @{ username = $promptUser; key = $keyPlain } | ConvertTo-Json -Compress
        Set-Content -Path $cfgPath -Value $json -Encoding ASCII
        Write-Host "Created $cfgPath from interactive input." -ForegroundColor Cyan
        return $true
      } 
      else 
      {
        Write-Host 'No Kaggle key entered; skipping credential creation.' -ForegroundColor Yellow
      }
    }
  }

  return $false
}

function Test-PythonModule {
  param([string]$Name)
  & python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$Name') else 1)"
  return ($LASTEXITCODE -eq 0)
}

function Invoke-PythonQuiet {
  param(
    [string]$Description,
    [string]$Script,
    [string[]]$Arguments = @()
  )

  Write-Host $Description -ForegroundColor Cyan
  $output = & python $Script @Arguments 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Step failed (showing Python output):" -ForegroundColor Red
    $output | Out-Host
    throw "Python step failed: $Description"
  } 
  else 
  {
    Write-Host "Done." -ForegroundColor Green
  }
}

if (-not (Test-Path 'data')) { New-Item -Force -ItemType Directory -Path 'data' | Out-Null }

# 1) Install requirements
if (Test-Path 'requirements.txt') {
  Write-Host 'Installing requirements...' -ForegroundColor Cyan
  python -m pip install -r requirements.txt | Out-Null
}

# 2) Attempt Kaggle download (prefers CLI, falls back to KaggleHub)
$didKaggle = $false
# Detect kaggle modules and credentials
$hasKaggle = Test-PythonModule -Name 'kaggle'
$hasKaggleHub = Test-PythonModule -Name 'kagglehub'
$cfgDir = if ($env:KAGGLE_CONFIG_DIR) { $env:KAGGLE_CONFIG_DIR } else { Join-Path $env:USERPROFILE '.kaggle' }
$cfgPath = Join-Path $cfgDir 'kaggle.json'
$hasCreds = Ensure-KaggleCredentials -cfgDir $cfgDir -cfgPath $cfgPath
$downloaders = @(
  @{
    Name = 'Kaggle CLI'
    Enabled = ($hasKaggle -and $hasCreds)
    Action = {
      & powershell -ExecutionPolicy Bypass -File (Join-Path $here 'download_kaggle.ps1')
      if ($LASTEXITCODE -ne 0) 
      {
        throw "Kaggle CLI exited with code $LASTEXITCODE"
      }
    }
  },
  @{
    Name = 'KaggleHub'
    Enabled = ($hasKaggleHub -and $hasCreds)
    Action = {
      python (Join-Path $here 'download_kagglehub.py')
      if ($LASTEXITCODE -ne 0) {
        throw "KaggleHub script exited with code $LASTEXITCODE"
      }
    }
  }
)

foreach ($downloader in $downloaders) {
  if (-not $downloader.Enabled -or $didKaggle) { continue }
  try {
    Write-Host "$($downloader.Name) detected: downloading datasets..." -ForegroundColor Cyan
    & $downloader.Action
    $didKaggle = $true
  } 
  catch 
  {
    Write-Host "$($downloader.Name) download failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

if (-not $didKaggle) {
  Write-Host "Kaggle not ready. Module detected: $hasKaggle, credentials: $cfgPath (exists: $hasCreds)" -ForegroundColor Yellow
  if (-not $hasKaggleHub) {
    Write-Host 'Hint: install kagglehub for a CLI-free download fallback.' -ForegroundColor Yellow
  }
  Write-Host 'Set KAGGLE_JSON_PATH to an existing kaggle.json, or export KAGGLE_USERNAME/KAGGLE_KEY, or place kaggle.json under %USERPROFILE%\.kaggle.' -ForegroundColor Yellow
  Write-Host 'Skipping Kaggle download.' -ForegroundColor Yellow
}

# 2b) Convert non-CSV datasets (labelled corpus, BABE) into CSVs
$labelledCorpusRoot = 'data/labelled-corpus-political-bias-hugging-face'
if (Test-Path $labelledCorpusRoot) {
  try 
  {
    Invoke-PythonQuiet -Description 'Converting non-CSV datasets to CSV (labelled corpus, BABE)...' `
      -Script (Join-Path $here 'convert_datasets_tocsv.py')
  } 
  catch 
  {
    Write-Host "Dataset conversion failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 3) Build combined CSV from Kaggle folders if possible
if ($didKaggle) {
  try {
    $inputs = @(
      'data/labelled-corpus-political-bias-hugging-face/labelled_corpus.csv',
      'data/news-articles-for-political-bias-classification/**/*.csv',
      'data/mbib-media-bias-identification-benchmark/**/*.csv',
      'data/political-bias-in-mainstream-media/**/*.csv',
      'data/mediabias/**/*.csv',
      'data/babe-media-bias-annotations-by-experts/babe_all.csv'
    )
    $args = @('--inputs') + $inputs + @('--output','data/train.csv','--balance_labels')
    Invoke-PythonQuiet -Description 'Preparing combined training CSV from Kaggle folders...' `
      -Script 'prepare_kaggle_data.py' -Arguments $args
  } 
  catch {
    Write-Host "Data preparation failed: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 4) Ensure a training CSV exists (fallback to sample)
if (-not (Test-Path 'data/train.csv')) {
  if (Test-Path 'data/sample_train.csv') {
    Write-Host 'No data/train.csv found. Using sample dataset.' -ForegroundColor Yellow
    Copy-Item 'data/sample_train.csv' 'data/train.csv' -Force
  } 
  else {
    Write-Host 'No training CSV available. Please create data/train.csv.' -ForegroundColor Red
    exit 1
  }
}

# 4b) Display overall label distribution for transparency/metrics
if (Test-Path 'data/train.csv') {
  try {
    Write-Host 'Current data/train.csv label distribution:' -ForegroundColor Cyan
    python -c "import pandas as pd, json; df = pd.read_csv('data/train.csv'); summary = {'total_samples': len(df), 'label_counts': df['label'].value_counts().to_dict()}; print(json.dumps(summary, indent=2))"
  } 
  catch {
    Write-Host "Unable to summarize data/train.csv: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 5) Build train/val/test splits for refinement and metrics
$splitScript = Join-Path $here 'split_dataset.py'
if ((Test-Path 'data/train.csv') -and (Test-Path $splitScript)) {
  try {
    Write-Host 'Generating stratified data/train_split.csv, data/val_split.csv, data/test_split.csv...' -ForegroundColor Cyan
    python $splitScript --input data/train.csv --output_dir data --train_frac 0.7 --val_frac 0.15 --seed 42
    Write-Host 'Split breakdown (rows + label counts):' -ForegroundColor Cyan
    python -c "import pandas as pd, json, os; result = {name: ({'rows': len((df := pd.read_csv(f'data/{name}_split.csv'))), 'label_counts': df['label'].value_counts().to_dict()} if os.path.exists(f'data/{name}_split.csv') else 'missing') for name in ['train', 'val', 'test']}; print(json.dumps(result, indent=2))"
  } 
  catch {
    Write-Host "Split generation failed (continuing with data/train.csv): $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# 6) Train the model and set parameters
$maxInt = [int]::MaxValue
$trainingSeed = Get-Random -Minimum 1 -Maximum $maxInt
Write-Host "Using randomized training seed: $trainingSeed" -ForegroundColor Cyan

# Optional class weights passed through to train_textcnn.py. When non-empty,
# these weights are parsed there and used to scale the loss contribution of
# each label (Left/Right/Neutral). This lets us slightly favour performance
# on a particular class (e.g., Right) without changing the model architecture
# or the underlying train/val/test splits.
$classWeights = ""

$usingSample = $false
try {
  $sampleFirstLine = (Get-Content 'data/sample_train.csv' -TotalCount 2) -join "`n"
  $trainFirstLine  = (Get-Content 'data/train.csv' -TotalCount 2) -join "`n"
  $usingSample = ($sampleFirstLine -eq $trainFirstLine)
} catch {}

#If the kaggle dataset does not load, default to simple training model
if ($usingSample) {
  Write-Host 'Training model (sample data, quick settings)...' -ForegroundColor Cyan

  # Hyperparameters for sample data. Overrides the default model hyperparameters.
  $epochs      = 2
  $batch_size  = 8
  $max_len     = 200
  $filterSizes = "3,4,5"
  $numFilters  = 50
  $lr          = 0.001
  $dropout     = 0.5
  $embedDim    = 100
  $minFreq     = 2
} 

#If the kaggle datasets load, use train.csv
else {
  Write-Host 'Training model on Kaggle combined dataset...' -ForegroundColor Cyan
  # Hyperparameters for full Kaggle dataset. Overrides the default model hyperparameters.
  $epochs      = 12 # Epochs are the number of complete passes through the training dataset
  $batch_size  = 64 # Batch size is the number of samples processed before the model is updated
  $max_len     = 700 # Max length is the maximum number of tokens per input text (longer texts are truncated)
  $filterSizes = "3,4,5,7" # Filter sizes are the n-gram sizes for convolutional filters
  $numFilters  = 100 # Number of filters is the number of convolutional filters per filter size
  $lr          = 0.0007 # Learning rate controls how much to change the model in response to estimated error each time the model weights are updated
  $dropout     = 0.6 # Dropout is the fraction of input units to drop to prevent overfitting
  $embedDim    = 200 # Embedding dimension is the size of the word embedding vectors
  $minFreq     = 3 # Minimum frequency is the minimum number of occurrences for a word to be included in the vocabulary
  $classWeights = "Left:1.10,Right:1.20,Neutral:1.0" # Slightly up-weight articles in the loss so that mistakes on
}

Write-Host "Hyperparameters -> epochs=$epochs batch_size=$batch_size max_len=$max_len filter_sizes=$filterSizes num_filters=$numFilters lr=$lr dropout=$dropout embed_dim=$embedDim min_freq=$minFreq seed=$trainingSeed" -ForegroundColor DarkCyan

python train_textcnn.py `
  --train_csv data/train_split.csv `
  --val_csv data/val_split.csv `
  --test_csv data/test_split.csv `
  --epochs $epochs `
  --batch_size $batch_size `
  --max_len $max_len `
  --seed $trainingSeed `
  --lr $lr `
  --dropout $dropout `
  --embedding_dim $embedDim `
  --min_freq $minFreq `
  --class_weights $classWeights `
  --filter_sizes $filterSizes `
  --num_filters $numFilters `
  --use_lr_scheduler true `
  --lr_factor 0.5 `
  --lr_patience 1 `
  --early_stopping_patience 2

# 7) Start API front end
Write-Host 'Starting API at http://127.0.0.1:8000' -ForegroundColor Green
uvicorn main:app --reload
