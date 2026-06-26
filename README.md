# frigate-cuda-pascal

Custom [Frigate](https://github.com/blakeblackshear/frigate) image with CUDA 11.8 + `onnxruntime-gpu 1.17.1` for **GTX 1070 and other Pascal (SM6.x) GPUs**.

The upstream `frigate:stable` image ships CPU-only `onnxruntime`. The `stable-tensorrt` image requires TensorRT 10 which dropped Pascal support. This image bridges that gap.

## What's different

| Change | Reason |
|--------|--------|
| CUDA 11.8 runtime libs (`cublas`, `cudart`, `curand`, `cufft`, `cusolver`, `cusparse`, `cudnn8+cuda11.8`) | Required by onnxruntime-gpu 1.17.1 |
| `onnxruntime-gpu==1.17.1` replaces the bundled CPU build | Enables `CUDAExecutionProvider` |
| `detection_runners.py` patch | Pascal can't use CUDA Graphs; patch catches the error and falls back to plain CUDA EP |

## Usage

```yaml
# Unraid template / docker-compose
image: ghcr.io/adinballew/frigate-cuda-pascal:stable
```

All other configuration (volumes, ports, env vars, `config.yml`) stays identical to upstream.

## Frigate config

```yaml
detectors:
  onnx:
    type: onnx
    device: cuda
```

## Updating upstream

The `FRIGATE_VERSION` build arg defaults to `stable`. To pin a specific version, trigger the workflow manually with the desired tag (e.g. `0.15.0`).

## GPU compatibility

Built and tested on **GTX 1070 (Pascal SM6.1)**, driver 580, CUDA 13.0 host, Unraid 7.x.
