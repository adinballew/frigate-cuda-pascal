---
title: "Frigate NVR"
summary: "NVR with object detection on Unraid using GTX 1070 Pascal GPU — cameras and Pascal CUDA gotchas"
tags: [unraid, frigate, services, cuda, pascal]
template: "system"
ip-addresses:
  frigate: 192.168.0.47:5000
  unraid: 192.168.0.40
---

# Frigate NVR

NVR with object detection running on Unraid, using a GTX 1070 (Pascal) GPU
for CUDA inference. Custom image with CUDA 11.8 and onnxruntime-gpu 1.17.1
for Pascal (SM6.x) GPU support.

## Location

| Item | Value |
|------|-------|
| Host | [ip-address: unraid] (Unraid Tower) |
| Web UI | <http://[ip-address: frigate]/> |
| Container | `frigate` (br0 network, IP [ip-address: frigate]) |
| Config | `/mnt/user/appdata/frigate/config` |
| Media | `/mnt/user/appdata/frigate/media` |
| Template | `unraid/templates/my-frigate.xml` |

## Configuration

| Setting | Value |
|---------|-------|
| Image | `ghcr.io/adinballew/frigate-cuda-pascal:stable` |
| Detector | `device: cuda` |
| Network | br0 ([ip-address: frigate]), ports accessed directly: 5000 (Web UI), 8554 (RTSP) |
| GPU | NVIDIA GeForce GTX 1070 (Pascal architecture) |
| CUDA | 11.8 (required for Pascal GTX 10-series) |
| onnxruntime | `onnxruntime-gpu==1.17.1` (CUDAExecutionProvider) |

### Custom Image

The image includes CUDA 11.8 + onnxruntime-gpu 1.17.1 + a Pascal CUDA Graph
fallback patch + numpy<2 pin.

Repo: <https://github.com/adinballew/frigate-cuda-pascal> (GitHub Action builds
on push to main)

### Pascal CUDA Graph Fix

`patches/detection_runners.py` in the repo catches CudaGraph failure (Pascal
can't fuse 4 Memcpy nodes) and falls back to plain CUDA EP.

### Dockerfile Pin

`numpy>=1.24.2,<2` — onnxruntime 1.17.1 is compiled against numpy 1.x.
Without this pin, numpy 2.x causes `AttributeError: _ARRAY_API not found`.

## Cameras (Reolink)

| Camera | IP | Stream | Resolution | Framerate | Roles |
|--------|-----|--------|------------|-----------|-------|
| FrontDoor | `192.168.0.221` | main | 1080p | — | detect, record |
| FrontDoor | `192.168.0.221` | sub | 640x480 | 15fps | go2rtc tablet stream |
| Backyard | `192.168.0.245` | main | 4K | — | record |
| Backyard | `192.168.0.245` | sub | 640x480 | 15fps | detect, go2rtc tablet stream |

Camera credentials in the process environment (`FRIGATE_USER`, `FRIGATE_PASS`).

## Status

Verify the container is running:

```bash
ssh unraid "docker ps --filter name=frigate"
```

Check the Web UI:

```bash
curl -s http://192.168.0.45:5000/api/stats | jq .
```

Confirm CUDA detection is active:

```bash
ssh unraid "docker exec frigate python3 -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
```

Expected output includes `CUDAExecutionProvider`.

## Frigate Notes

- Do NOT use host network — HA owns ports 8554/8555/1984 (HA-bundled go2rtc).
  Frigate is on br0.
- Pascal CUDA Graph fallback may appear as a warning — expected on GTX 1070.
- GPU usage: ~500 MiB during normal detection on GTX 1070 (8 GB VRAM).

## Custom Image

The image includes CUDA 11.8 + onnxruntime-gpu 1.17.1 + a Pascal CUDA Graph
fallback patch + numpy<2 pin.

Repo: <https://github.com/adinballew/frigate-cuda-pascal> (GitHub Action builds
on push to main)

### Pascal CUDA Graph Fix

`patches/detection_runners.py` in the repo catches CudaGraph failure (Pascal
can't fuse 4 Memcpy nodes) and falls back to plain CUDA EP.

### Dockerfile Pin

`numpy>=1.24.2,<2` — onnxruntime 1.17.1 is compiled against numpy 1.x.
Without this pin, numpy 2.x causes `AttributeError: _ARRAY_API not found`.

CUDA 11.8 is required because Pascal (SM6.x) is not supported by TensorRT 10
(which ships in the upstream `stable-tensorrt` image).
