<#
Downloads and unzips the requested Kaggle datasets into the data/ folder.

Prereqs:
- pip install kaggle (requirements.txt installs it)
- Place kaggle.json in %USERPROFILE%\.kaggle OR set $env:KAGGLE_CONFIG_DIR to a folder that contains kaggle.json
#>

param()

$ErrorActionPreference = 'Stop'

# Verify kaggle module is importable via current Python
& python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('kaggle') else 1)"
if ($LASTEXITCODE -ne 0) {
  throw 'Kaggle Python package is not available in this environment.'
}

# Verify credentials
$cfgDir = if ($env:KAGGLE_CONFIG_DIR) { $env:KAGGLE_CONFIG_DIR } else { Join-Path $env:USERPROFILE '.kaggle' }
$cfgPath = Join-Path $cfgDir 'kaggle.json'
if (-not (Test-Path $cfgPath)) {
  throw "kaggle.json not found. Checked: $cfgPath. Set KAGGLE_CONFIG_DIR or place kaggle.json in %USERPROFILE%\.kaggle"
}

New-Item -ItemType Directory -Force -Path data | Out-Null

$datasets = @(
  @{ id = 'surajkarakulath/labelled-corpus-political-bias-hugging-face';  dir = 'data/labelled-corpus-political-bias-hugging-face' },
  @{ id = 'gandpablo/news-articles-for-political-bias-classification';    dir = 'data/news-articles-for-political-bias-classification' },
  @{ id = 'timospinde/mbib-media-bias-identification-benchmark';           dir = 'data/mbib-media-bias-identification-benchmark' },
  @{ id = 'newsanalysis/political-bias-in-mainstream-media';               dir = 'data/political-bias-in-mainstream-media' },
  @{ id = 'tegmark/mediabias';                                             dir = 'data/mediabias' },
  @{ id = 'timospinde/babe-media-bias-annotations-by-experts';             dir = 'data/babe-media-bias-annotations-by-experts' }
)

foreach ($ds in $datasets) {
  New-Item -ItemType Directory -Force -Path $ds.dir | Out-Null
  Write-Host "Downloading $($ds.id) ..."
  & kaggle datasets download -d $ds.id -p $ds.dir --unzip
  if ($LASTEXITCODE -ne 0) {
    throw "Kaggle CLI failed for $($ds.id) (exit code $LASTEXITCODE)."
  }
}

Write-Host "Done. CSVs downloaded under data/ subfolders." -ForegroundColor Green
