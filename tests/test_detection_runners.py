"""Tests for the Pascal CUDA Graph fallback in patches/detection_runners.py.

The patched module imports `frigate.*`, `onnxruntime` and `numpy`, which exist
only inside the Frigate image, so they are stubbed here. Only the behavior this
repository owns is exercised: what `get_optimized_runner` does when CUDA Graph
capture fails.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PATCH = Path(__file__).resolve().parent.parent / "patches" / "detection_runners.py"


def stub_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def harness(monkeypatch):
    """Load the patch against stubs; `fail_graph` makes CUDA Graph capture raise."""
    state = SimpleNamespace(fail_graph=False, session_options=[])

    def make_session(model_path, **kwargs):
        options = kwargs["provider_options"][0]
        state.session_options.append(options)
        if state.fail_graph and options.get("enable_cuda_graph"):
            raise RuntimeError("cudaGraphInstantiate failed")
        return SimpleNamespace(**kwargs)

    stubs = {
        "numpy": stub_module("numpy"),
        "onnxruntime": stub_module(
            "onnxruntime",
            InferenceSession=make_session,
            SessionOptions=object,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC=1),
        ),
        "frigate": stub_module("frigate"),
        "frigate.util": stub_module("frigate.util"),
        "frigate.util.model": stub_module(
            "frigate.util.model",
            get_ort_providers=lambda *a, **k: (
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                [{"device_id": 0}, {}],
            ),
        ),
        "frigate.util.rknn_converter": stub_module(
            "frigate.util.rknn_converter",
            auto_convert_model=lambda path: None,
            is_rknn_compatible=lambda path: False,
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)

    spec = importlib.util.spec_from_file_location("patched_detection_runners", PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # CudaGraphRunner internals need a GPU; keep only the construction seam.
    monkeypatch.setattr(module.CudaGraphRunner, "__init__", lambda self, session, device_id: None)
    monkeypatch.setattr(module.CudaGraphRunner, "is_model_supported", staticmethod(lambda model_type: True))
    # Upstream model-type classification is not owned here and pulls in `frigate.embeddings`.
    for classifier in ("is_cpu_complex_model", "is_concurrent_model"):
        monkeypatch.setattr(module.ONNXModelRunner, classifier, staticmethod(lambda model_type: False))

    return SimpleNamespace(module=module, state=state)


def test_graph_capture_success_returns_cuda_graph_runner(harness):
    runner = harness.module.get_optimized_runner("m.onnx", "0", "yolo-generic")

    assert isinstance(runner, harness.module.CudaGraphRunner)


def test_graph_capture_failure_falls_back_to_plain_cuda_without_graph_flag(harness):
    harness.state.fail_graph = True

    runner = harness.module.get_optimized_runner("m.onnx", "0", "yolo-generic")

    assert isinstance(runner, harness.module.ONNXModelRunner)
    # The final session is the fallback; it must not request CUDA Graph capture again.
    fallback = harness.state.session_options[-1]
    assert "enable_cuda_graph" not in fallback
    assert fallback["device_id"] == 0
