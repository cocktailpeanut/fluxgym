@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Set download acceleration parameters (sync with PowerShell script)
set HF_HUB_ENABLE_HF_TRANSFER=1
set GRADIO_ANALYTICS_ENABLED=0
set HF_ENDPOINT=https://hf-mirror.com

REM HuggingFace cache directory
REM set HF_HOME=D:\Documents\fluxgym\cache

REM Unified storage path configuration
set BASE_DIR=D:\Documents\fluxgym
set OUTPUTS_DIR=%BASE_DIR%\outputs
set DATASETS_DIR=%BASE_DIR%\datasets
set MODELS_DIR=%BASE_DIR%\models

REM Virtual environment detection (newly added)
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

REM Display Python information (newly added)
where python
python --version

REM Model path declarations
set CLIP_L_PATH=%BASE_DIR%\models\clip\clip_l.safetensors
set T5XXL_PATH=%BASE_DIR%\models\clip\t5xxl_fp16.safetensors
set VAE_PATH=%BASE_DIR%\models\vae\ae.sft

REM Launch application (new execution command)
python app.py
pause
