# syntax=docker/dockerfile:1.6
# Custom Frigate image: CUDA 11.8 + onnxruntime-gpu 1.17.1 for GTX 1070 (Pascal SM6.1)
# Uses upstream frigate:stable as base — bump FRIGATE_VERSION to track upstream releases.

ARG FRIGATE_VERSION=stable
FROM ghcr.io/blakeblackshear/frigate:${FRIGATE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# ── Install CUDA 11.8 runtime libs ────────────────────────────────────────────
# GTX 1070 (Pascal, SM6.1) cannot use TensorRT 10 or CUDA 12 onnxruntime wheels.
# onnxruntime-gpu 1.17.1 is the last release supporting CUDA 11.x and Pascal.
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
        libcusparse-11-8 \
        libcudnn8=8.9.7.29-1+cuda11.8 && \
    echo /usr/local/cuda-11.8/lib64 > /etc/ld.so.conf.d/cuda-11.8.conf && \
    ldconfig && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Install onnxruntime-gpu ───────────────────────────────────────────────────
# Replace the CPU-only onnxruntime that ships in frigate:stable.
# 1.17.1 is the last release with CUDA 11.x support (Pascal SM6.1 compatible).
RUN pip3 install --break-system-packages --force-reinstall onnxruntime-gpu==1.17.1

# ── Patch detection_runners.py for Pascal CUDA Graph fallback ─────────────────
# Pascal GPUs can't capture CUDA Graphs with yolov8n (4 Memcpy nodes).
# Ship a pre-patched copy (based on upstream dev) that wraps the graph capture
# in a try/except and falls back to plain CUDAExecutionProvider.
COPY patches/detection_runners.py /opt/frigate/frigate/detectors/detection_runners.py
