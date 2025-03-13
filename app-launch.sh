#!/usr/bin/env bash

cd "`dirname "$0"`" || exit 1

python_env_dir="env"
if [ -d ".env" ]; then
    python_env_dir="env"
elif [ -d ".venv" ]; then
    python_env_dir=".venv"
fi

. "$python_env_dir/bin/activate"
echo "`which python`:  `python --version`"

# multi-thread download
export HF_HUB_ENABLE_HF_TRANSFER="1"

# disable analytics
export GRADIO_ANALYTICS_ENABLED="0"

# HuggingFace cache directory
#export HF_HOME="/path/to/your/custom/cache"

# For HuggingFace mirror site to accelerate download for China users
export HF_ENDPOINT="https://hf-mirror.com"

# Set directory path by environment variables
export OUTPUTS_DIR="/root/autodl-tmp/fluxgym/outputs"
export DATASETS_DIR="/root/autodl-tmp/fluxgym/datasets"
export MODELS_DIR="/root/autodl-tmp/fluxgym/models"

# Set file path by environment variables
export CLIP_L_PATH="/root/autodl-tmp/fluxgym/models/clip/clip_l.safetensors"
export T5XXL_PATH="/root/autodl-tmp/fluxgym/models/clip/t5xxl_fp16.safetensors"
export VAE_PATH="/root/autodl-tmp/fluxgym/models/vae/ae.sft"

python app.py
