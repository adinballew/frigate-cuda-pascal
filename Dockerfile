# syntax=docker/dockerfile:1.6
# Custom Frigate image: CUDA 11.8 + onnxruntime-gpu 1.17.1 for GTX 1070 (Pascal SM6.1)
# Uses upstream stable as base — bump FRIGATE_VERSION to track upstream releases.

ARG FRIGATE_VERSION=stable
FROM ghcr.io/blakeblackshear/frigate:${FRIGATE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# ── Install CUDA 11.8 runtime libs ────────────────────────────────────────────
# GTX 1070 (Pascal, SM6.1) cannot use TensorRT 10 or CUDA 12 onnxruntime wheels.
# onnxruntime-gpu 1.17.1 is the last release supporting CUDA 11.x and Pascal.
# Required libs: cublas, cudart, curand, cufft, cusolver, cusparse, cudnn8
RUN apt-get update -qq && \
    wget -q -O /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libcublas-11-8 \
        cuda-cudart-11-8 \
        libcurand-11-8 \
        libcufft-11-8 \
        libcusolver-11-8 \
        libcusparse-11-8 && \
    wget -q -O /tmp/cudnn8.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/libcudnn8_8.9.7.29-1+cuda11.8_amd64.deb && \
    dpkg --force-downgrade -i /tmp/cudnn8.deb && rm /tmp/cudnn8.deb && \
    echo /usr/local/cuda-11.8/lib64 > /etc/ld.so.conf.d/cuda-11.8.conf && \
    ldconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Install onnxruntime-gpu ───────────────────────────────────────────────────
# Replace the CPU-only onnxruntime that ships in frigate:stable with the
# CUDA-enabled build pinned to 1.17.1 (last version with CUDA 11.8 support).
RUN pip3 install --break-system-packages --force-reinstall onnxruntime-gpu==1.17.1

# ── Patch detection_runners.py for Pascal CUDA Graph fallback ─────────────────
# Pascal GPUs (SM6.x) can't use CUDA Graphs with the yolov8n ONNX model due to
# Memcpy nodes. This patch catches the error and falls back to plain CUDA EP.
COPY scripts/patch_detection_runners.py /tmp/patch_detection_runners.py
RUN python3 /tmp/patch_detection_runners.py && rm /tmp/patch_detection_runners.py
