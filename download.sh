#!/bin/bash

# Create necessary directories
mkdir -p models/clip models/vae models/unet

# Function to download file
download_file() {
    local url=$1
    local destination=$2
    echo "Downloading $(basename $destination)..."
    if command -v wget > /dev/null; then
        wget -q --show-progress -O "$destination" "$url"
    elif command -v curl > /dev/null; then
        curl -L -o "$destination" "$url"
    else
        echo "Error: Neither wget nor curl is installed. Please install one of them and try again."
        exit 1
    fi
}

# Download CLIP models
download_file "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true" "models/clip/clip_l.safetensors"
download_file "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true" "models/clip/t5xxl_fp16.safetensors"

# Download VAE model
download_file "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft?download=true" "models/vae/ae.sft"

# Download UNET model
download_file "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/flux1-dev.sft?download=true" "models/unet/flux1-dev.sft"

echo "All model checkpoints have been downloaded successfully."