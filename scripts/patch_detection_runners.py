"""Patch Frigate's detection_runners.py to handle CUDA Graph failure on Pascal GPUs.

Pascal (SM6.x) GPUs cannot capture CUDA Graphs when the ONNX model has Memcpy
nodes. Without this patch Frigate crashes the detector process. This wraps the
CUDA Graph path in a try/except and falls back to plain CUDAExecutionProvider.
"""
import sys

path = "/opt/frigate/frigate/detectors/detection_runners.py"

with open(path) as f:\n    src = f.read()\n\nOLD = '''        options[0] = {\n            **options[0],
            "enable_cuda_graph": True,
        }
        return CudaGraphRunner(
            ort.InferenceSession(
                model_path,
                providers=providers,
                provider_options=options,
            ),
            options[0]["device_id"],
        )'''

NEW = '''        try:
            options[0] = {
                **options[0],
                "enable_cuda_graph": True,
            }
            return CudaGraphRunner(
                ort.InferenceSession(
                    model_path,
                    providers=providers,
                    provider_options=options,
                ),
                options[0]["device_id"],
            )
        except Exception as cuda_graph_err:
            import logging as _l
            _l.getLogger(__name__).warning(
                f"CUDA Graph capture failed ({cuda_graph_err}); "
                "falling back to CUDA EP without graph capture"
            )
            options[0] = {k: v for k, v in options[0].items() if k != "enable_cuda_graph"}'''

if OLD not in src:
    print(
        "WARN: patch target not found in detection_runners.py — "
        "upstream may have changed. Skipping patch.",
        file=sys.stderr,
    )
    sys.exit(0)

src = src.replace(OLD, NEW)
with open(path, "w") as f:\n    f.write(src)\nprint("Patch applied to detection_runners.py")
