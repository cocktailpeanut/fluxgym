#!/usr/bin/env bash
# Point ./models at the RunPod network volume so base models (Krea bf16 ~23GB,
# T5 ~9GB, CLIP, VAE) download once and persist across invocations. Falls back
# to a local ./models dir if no volume is attached.
set -e
cd /app/fluxgym

VOLUME_MODELS="/runpod-volume/models"
if [ -d "/runpod-volume" ]; then
    mkdir -p "${VOLUME_MODELS}"
    # Replace ./models with a symlink to the volume (unless already linked).
    if [ ! -L "models" ]; then
        rm -rf models
        ln -s "${VOLUME_MODELS}" models
    fi
    echo "[entrypoint] models -> ${VOLUME_MODELS}"
else
    mkdir -p models
    echo "[entrypoint] No /runpod-volume found; using local ./models (not persistent)"
fi

exec python3 -u runpod/handler.py
