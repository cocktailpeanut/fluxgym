@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:: disable multi-thread download
set HF_HUB_ENABLE_HF_TRANSFER=0
:: disable gradio analytics
set GRADIO_ANALYTICS_ENABLED=0
:: For HuggingFace mirror site to accelerate download for China users
set HF_ENDPOINT=https://hf-mirror.com
:: HuggingFace cache directory
:: set HF_HOME=D:\Documents\fluxgym\cache

:: Unified storage path configuration
set BASE_DIR=D:\Documents\fluxgym
set OUTPUTS_DIR=%BASE_DIR%\outputs
set DATASETS_DIR=%BASE_DIR%\datasets
set MODELS_DIR=%BASE_DIR%\models 

:: Virtual environment detection (newly added)
for %%D in ("env", ".env", "venv", ".venv") do (
    if exist "%%D\Scripts\activate.bat" (
        call "%%D\Scripts\activate.bat"
        goto :venv_found
    )
)
echo Virtual environment not found. Please create one first.
pause
exit /b 1
:venv_found

:: Display Python information (newly added)
where python
python --version

:: Model path declarations
set CLIP_L_PATH=%BASE_DIR%\models\clip\clip_l.safetensors
set T5XXL_PATH=%BASE_DIR%\models\clip\t5xxl_fp16.safetensors
set VAE_PATH=%BASE_DIR%\models\vae\ae.sft

:: Launch application (new execution command)
python app.py
pause
