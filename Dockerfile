# Base image with CUDA 12.6 and Ubuntu 24.04 (Python 3.12 default)
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

# Install pip and other dependencies
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential  # Install dependencies for building extensions

# Define environment variables for UID and GID and local timezone
ENV PUID=${PUID:-1000}
ENV PGID=${PGID:-1000}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Create a group with the specified GID
RUN groupadd -g "${PGID}" appuser
# Create a user with the specified UID and GID
RUN useradd -m -s /bin/sh -u "${PUID}" -g "${PGID}" appuser

WORKDIR /app

# Get sd-scripts from kohya-ss and install them
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
    cd sd-scripts && \
    # Remove -e . from requirements.txt to avoid issues during build
    sed -i '/-e \./d' requirements.txt && \
    # Install compatible versions beforehand for Python 3.12 compatibility
    pip install --no-cache-dir "scipy>=1.13.1" "PyWavelets>=1.6.0" "invisible-watermark>=0.2.0" "transformers>=4.42.0" && \
    pip install --no-cache-dir -r ./requirements.txt

# Install main application dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

RUN chown -R appuser:appuser /app

# delete redundant requirements.txt and sd-scripts directory within the container
RUN rm -r ./sd-scripts
RUN rm ./requirements.txt

#Run application as non-root
USER appuser

# Copy fluxgym application code
COPY . ./fluxgym

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app/fluxgym

# Run fluxgym Python application
CMD ["python3", "./app.py"]
