# Set encoding to UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Change to script directory
Set-Location -Path $PSScriptRoot

# Detect and activate virtual environment
$venvDirs = @("env", ".env", "venv", ".venv")
foreach ($dir in $venvDirs) {
    if (Test-Path "$dir\Scripts\Activate.ps1") {
        & "$dir\Scripts\Activate.ps1"
        break
    }
}

# disable multi-thread download
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
# disable analytics
$env:GRADIO_ANALYTICS_ENABLED = "0"
# For HuggingFace mirror site to accelerate download for China users
$env:HF_ENDPOINT = "https://hf-mirror.com"
# HuggingFace cache directory
# $env:HF_HOME = "D:\Documents\fluxgym\cache"

$basePath = "D:\Documents\fluxgym"
$env:OUTPUTS_DIR = "$basePath\outputs"
$env:DATASETS_DIR = "$basePath\datasets"
$env:MODELS_DIR = "$basePath\models"

$env:CLIP_L_PATH = "$basePath\models\clip\clip_l.safetensors"
$env:T5XXL_PATH = "$basePath\models\clip\t5xxl_fp16.safetensors"
$env:VAE_PATH = "$basePath\models\vae\ae.sft"

# Display Python info
Write-Host "Python path: $(Get-Command python | Select-Object -ExpandProperty Source)"
python --version

# Launch application
python app.py
Read-Host "Press Enter to continue..."