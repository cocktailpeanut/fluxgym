# Dockerfile

# Base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and pip
RUN apt-get update -y && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel so dependency resolution is more robust
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    ln -s /usr/bin/python3 /usr/bin/python || true

# Define environment variables for UID and GID
ENV PUID=1000 \
    PGID=1000

# Create a non-root user
RUN groupadd -g ${PGID} appuser && \
    useradd -m -s /bin/sh -u ${PUID} -g ${PGID} appuser

WORKDIR /app

# Get sd-scripts from kohya-ss and install their dependencies
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
    cd sd-scripts && \
    python3 -m pip install --no-cache-dir -r ./requirements.txt

# Install main FluxGym application dependencies
COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Install Torch, Torchvision, and Torchaudio for CUDA 12.2
RUN python3 -m pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html

# Clean up build-only artifacts
RUN rm -rf /app/sd-scripts /app/requirements.txt

# Ensure ownership for the non-root user
RUN chown -R appuser:appuser /app

# Run application as non-root
USER appuser

# Copy fluxgym application code (will be overridden by the bind mount at runtime)
COPY . /app/fluxgym

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app/fluxgym

# Run fluxgym Python application
CMD ["python3", "./app.py"]