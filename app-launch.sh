#!/usr/bin/env bash

cd "`dirname "$0"`" || exit 1

# Virtual environment detection order sync with PS1
venv_dirs=("env" ".env" "venv" ".venv")
for venv_dir in "${venv_dirs[@]}"; do
    if [ -d "$venv_dir" ]; then
        . "$venv_dir/bin/activate"
        break
    fi
done

echo "`which python`:  `python --version`"

# disable multi-thread download
export HF_HUB_ENABLE_HF_TRANSFER="0"
# disable analytics
export GRADIO_ANALYTICS_ENABLED="0"
# For HuggingFace mirror site to accelerate download for China users
export HF_ENDPOINT="https://hf-mirror.com"
# HuggingFace cache directory
#export HF_HOME="/path/to/your/custom/cache"

# Unified base directory configuration
base_dir="/root/autodl-tmp/fluxgym"
export OUTPUTS_DIR="$base_dir/outputs"
export DATASETS_DIR="$base_dir/datasets"
export MODELS_DIR="$base_dir/models"

# Model paths using base directory
export CLIP_L_PATH="$base_dir/models/clip/clip_l.safetensors"
export T5XXL_PATH="$base_dir/models/clip/t5xxl_fp16.safetensors"
export VAE_PATH="$base_dir/models/vae/ae.sft"

python app.py
