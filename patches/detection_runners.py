"""Base runner implementation for ONNX models."""

import logging
import os
import platform
import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import onnxruntime as ort

from frigate.util.model import get_ort_providers
from frigate.util.rknn_converter import auto_convert_model, is_rknn_compatible

logger = logging.getLogger(__name__)

# Process-wide lock serializing all OpenVINO compile/inference calls
_OPENVINO_LOCK = threading.Lock()


def is_arm64_platform() -> bool:
    """Check if we're running on an ARM platform."""
    machine = platform.machine().lower()
    return machine in ("aarch64", "arm64", "armv8", "armv7l")


def get_ort_session_options(
    is_complex_model: bool = False,
) -> ort.SessionOptions | None:
    if is_complex_model:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        return sess_options
    return None


try:
    import openvino as ov
except ImportError:
    ov = None


def get_openvino_available_devices() -> list[str]:
    if ov is None:
        logger.debug("OpenVINO is not available")
        return []
    try:
        core = ov.Core()
        available_devices = core.available_devices
        logger.debug(f"OpenVINO available devices: {available_devices}")
        return available_devices
    except Exception as e:\n        logger.warning(f"Failed to get OpenVINO available devices: {e}")
        return []


def is_openvino_gpu_npu_available() -> bool:
    available_devices = get_openvino_available_devices()
    acceleration_devices = ["GPU", "MYRIAD", "NPU", "GNA", "HDDL"]
    return any(
        avail_dev == accel_dev or avail_dev.startswith(accel_dev + ".")
        for avail_dev in available_devices
        for accel_dev in acceleration_devices
    )


class BaseModelRunner(ABC):
    def __init__(self, model_path: str, device: str, **kwargs):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def get_input_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_input_width(self) -> int:
        pass

    @abstractmethod
    def run(self, input: dict[str, Any]) -> Any | None:
        pass


class ONNXModelRunner(BaseModelRunner):
    @staticmethod
    def is_cpu_complex_model(model_type: str) -> bool:
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type in [
            EnrichmentModelTypeEnum.jina_v1.value,
            EnrichmentModelTypeEnum.jina_v2.value,
        ]

    @staticmethod
    def is_migraphx_complex_model(model_type: str) -> bool:
        from frigate.detectors.detector_config import ModelTypeEnum
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type in [
            EnrichmentModelTypeEnum.paddleocr.value,
            EnrichmentModelTypeEnum.jina_v2.value,
            ModelTypeEnum.rfdetr.value,
            ModelTypeEnum.dfine.value,
        ]

    @staticmethod
    def is_concurrent_model(model_type: str | None) -> bool:
        if not model_type:
            return False
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type == EnrichmentModelTypeEnum.jina_v2.value

    def __init__(self, ort: ort.InferenceSession, model_type: str | None = None):
        self.ort = ort
        self.model_type = model_type
        if self.is_concurrent_model(model_type):
            self._inference_lock = threading.Lock()
        else:
            self._inference_lock = None

    def get_input_names(self) -> list[str]:
        return [input.name for input in self.ort.get_inputs()]

    def get_input_width(self) -> int:
        return self.ort.get_inputs()[0].shape[3]

    def run(self, input: dict[str, Any]) -> Any | None:
        if self._inference_lock:
            with self._inference_lock:
                return self.ort.run(None, input)
        return self.ort.run(None, input)


class CudaGraphRunner(BaseModelRunner):
    """Encapsulates CUDA Graph capture and replay using ONNX Runtime IOBinding."""

    @staticmethod
    def is_model_supported(model_type: str) -> bool:
        from frigate.detectors.detector_config import ModelTypeEnum
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type not in [
            ModelTypeEnum.yolonas.value,
            ModelTypeEnum.dfine.value,
            EnrichmentModelTypeEnum.paddleocr.value,
            EnrichmentModelTypeEnum.jina_v1.value,
            EnrichmentModelTypeEnum.jina_v2.value,
            EnrichmentModelTypeEnum.yolov9_license_plate.value,
        ]

    def __init__(self, session: ort.InferenceSession, cuda_device_id: int):
        self._session = session
        self._cuda_device_id = cuda_device_id
        self._captured = False
        self._io_binding: ort.IOBinding | None = None
        self._input_name: str | None = None
        self._output_names: list[str] | None = None
        self._input_ortvalue: ort.OrtValue | None = None
        self._output_ortvalues: ort.OrtValue | None = None

    def get_input_names(self) -> list[str]:
        return [input.name for input in self._session.get_inputs()]

    def get_input_width(self) -> int:
        return self._session.get_inputs()[0].shape[3]

    def run(self, input: dict[str, Any]):
        input_name = list(input.keys())[0]
        tensor_input = input[input_name]
        tensor_input = np.ascontiguousarray(tensor_input)

        if not self._captured:
            self._io_binding = self._session.io_binding()
            self._input_name = input_name
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._input_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                tensor_input, "cuda", self._cuda_device_id
            )
            self._io_binding.bind_ortvalue_input(self._input_name, self._input_ortvalue)
            for name in self._output_names:
                self._io_binding.bind_output(name, "cuda", self._cuda_device_id)
            ro = ort.RunOptions()
            self._session.run_with_iobinding(self._io_binding, ro)
            self._captured = True
            return self._io_binding.copy_outputs_to_cpu()

        self._input_ortvalue.update_inplace(tensor_input)
        ro = ort.RunOptions()
        self._session.run_with_iobinding(self._io_binding, ro)
        return self._io_binding.copy_outputs_to_cpu()


class OpenVINOModelRunner(BaseModelRunner):
    @staticmethod
    def is_complex_model(model_type: str) -> bool:
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type in [
            EnrichmentModelTypeEnum.paddleocr.value,
            EnrichmentModelTypeEnum.jina_v2.value,
        ]

    @staticmethod
    def is_model_npu_supported(model_type: str) -> bool:
        from frigate.embeddings.types import EnrichmentModelTypeEnum
        return model_type not in [
            EnrichmentModelTypeEnum.paddleocr.value,
            EnrichmentModelTypeEnum.jina_v1.value,
            EnrichmentModelTypeEnum.jina_v2.value,
            EnrichmentModelTypeEnum.arcface.value,
        ]

    @staticmethod
    def is_detection_model(model_type: str) -> bool:
        from frigate.detectors.detector_config import ModelTypeEnum
        return model_type in [m.value for m in ModelTypeEnum]

    def __init__(self, model_path: str, device: str, model_type: str, **kwargs):
        self.model_path = model_path
        self.device = device
        self.model_type = model_type

        if device == "NPU" and not OpenVINOModelRunner.is_model_npu_supported(model_type):
            logger.warning(f"OpenVINO model {model_type} is not supported on NPU, using GPU instead")
            device = "GPU"

        self.complex_model = OpenVINOModelRunner.is_complex_model(model_type)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"OpenVINO model file {model_path} not found.")

        if ov is None:
            raise ImportError("OpenVINO is not available. Please install openvino package.")

        self.ov_core = ov.Core()
        self.ov_core.set_property(device, {"PERF_COUNT": "NO"})

        if device in ["GPU", "AUTO", "NPU"]:
            self.ov_core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})

        if device == "NPU" and OpenVINOModelRunner.is_detection_model(model_type):
            try:
                self.ov_core.set_property(device, {"NPU_TURBO": "YES"})
            except Exception as e:\n                logger.debug(f"NPU_TURBO not supported by driver: {e}")

        with _OPENVINO_LOCK:
            self.compiled_model = self.ov_core.compile_model(model=model_path, device_name=device)
            self.infer_request = self.compiled_model.create_infer_request()

        self.input_tensor: ov.Tensor | None = None

        if not self.complex_model:
            try:
                input_shape = self.compiled_model.inputs[0].get_shape()
                input_element_type = self.compiled_model.inputs[0].get_element_type()
                self.input_tensor = ov.Tensor(input_element_type, input_shape)
            except RuntimeError:
                pass

    def get_input_names(self) -> list[str]:
        return [input.get_any_name() for input in self.compiled_model.inputs]

    def get_input_width(self) -> int:
        first_input = self.compiled_model.inputs[0]
        try:
            partial_shape = first_input.get_partial_shape()
            if len(partial_shape) >= 4 and partial_shape[3].is_static:
                return partial_shape[3].get_length()
            return -1
        except Exception:
            try:
                input_shape = first_input.shape
                return input_shape[3] if len(input_shape) >= 4 else -1
            except Exception:
                return -1

    def run(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        with _OPENVINO_LOCK:
            from frigate.embeddings.types import EnrichmentModelTypeEnum

            if self.model_type in [EnrichmentModelTypeEnum.arcface.value]:
                self.infer_request = self.compiled_model.create_infer_request()

            if (
                len(inputs) == 1
                and len(self.compiled_model.inputs) == 1
                and self.input_tensor is not None
            ):
                input_data = list(inputs.values())[0]
                np.copyto(self.input_tensor.data, input_data)
                self.infer_request.infer(self.input_tensor)
            else:
                if self.complex_model:
                    try:
                        self.infer_request.reset_state()
                    except Exception:
                        pass

                for input_name, input_data in inputs.items():
                    input_port = None
                    input_index = None
                    for idx, port in enumerate(self.compiled_model.inputs):
                        if port.get_any_name() == input_name:
                            input_port = port
                            input_index = idx
                            break

                    if input_port is None:
                        raise ValueError(f"Input '{input_name}' not found in model")

                    input_element_type = input_port.get_element_type()
                    expected_dtype = input_element_type.to_dtype()
                    if input_data.dtype != expected_dtype:
                        logger.debug(f"Converting input '{input_name}' from {input_data.dtype} to {expected_dtype}")
                        input_data = input_data.astype(expected_dtype)

                    input_tensor = ov.Tensor(input_element_type, input_data.shape)
                    np.copyto(input_tensor.data, input_data)
                    self.infer_request.set_input_tensor(input_index, input_tensor)

                try:
                    self.infer_request.infer()
                except Exception as e:\n                    logger.error(f"Error during OpenVINO inference: {e}")
                    return []

            outputs = []
            for i in range(len(self.compiled_model.outputs)):
                outputs.append(self.infer_request.get_output_tensor(i).data)
            return outputs


class RKNNModelRunner(BaseModelRunner):
    def __init__(self, model_path: str, model_type: str = None, core_mask: int = 0):
        self.model_path = model_path
        self.model_type = model_type
        self.core_mask = core_mask
        self.rknn = None
        self._load_model()

    def _load_model(self):
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite(verbose=False)
            if self.rknn.load_rknn(self.model_path) != 0:
                logger.error(f"Failed to load RKNN model: {self.model_path}")
                raise RuntimeError("Failed to load RKNN model")
            if self.rknn.init_runtime(core_mask=self.core_mask) != 0:
                logger.error("Failed to initialize RKNN runtime")
                raise RuntimeError("Failed to initialize RKNN runtime")
            logger.info(f"Successfully loaded RKNN model: {self.model_path}")
        except ImportError:
            logger.error("RKNN Lite not available")
            raise ImportError("RKNN Lite not available")
        except Exception as e:\n            logger.error(f"Error loading RKNN model: {e}")
            raise

    def get_input_names(self) -> list[str]:
        model_name = os.path.basename(self.model_path).lower()
        if "vision" in model_name:
            return ["pixel_values"]
        elif "arcface" in model_name:
            return ["data"]
        else:
            if self.model_type and "jina-clip" in self.model_type:
                if "vision" in self.model_type:
                    return ["pixel_values"]
            return ["input"]

    def get_input_width(self) -> int:
        model_name = os.path.basename(self.model_path).lower()
        if "vision" in model_name:
            return 224
        elif "arcface" in model_name:
            return 112
        return -1

    def run(self, inputs: dict[str, Any]) -> Any:
        if not self.rknn:
            raise RuntimeError("RKNN model not loaded")
        try:
            input_names = self.get_input_names()
            rknn_inputs = []
            for name in input_names:
                if name in inputs:
                    if name == "pixel_values":
                        pixel_data = inputs[name]
                        if len(pixel_data.shape) == 4 and pixel_data.shape[1] == 3:
                            pixel_data = np.transpose(pixel_data, (0, 2, 3, 1))
                        rknn_inputs.append(pixel_data)
                    elif name == "data":
                        face_data = inputs[name]
                        if len(face_data.shape) == 4 and face_data.shape[1] == 3:
                            face_data = np.transpose(face_data, (0, 2, 3, 1))
                        face_data = (((face_data + 1.0) * 127.5).clip(0, 255).astype(np.uint8))
                        rknn_inputs.append(face_data)
                    else:
                        rknn_inputs.append(inputs[name])
            outputs = self.rknn.inference(inputs=rknn_inputs)
            return outputs
        except Exception as e:\n            logger.error(f"Error during RKNN inference: {e}")
            raise

    def __del__(self):
        if self.rknn:
            try:
                self.rknn.release()
            except Exception:
                pass


def get_optimized_runner(
    model_path: str, device: str | None, model_type: str, **kwargs
) -> BaseModelRunner:
    """Get an optimized runner for the hardware."""
    device = device or "AUTO"

    if device != "CPU" and is_rknn_compatible(model_path):
        rknn_path = auto_convert_model(model_path)
        if rknn_path:
            return RKNNModelRunner(rknn_path)

    providers, options = get_ort_providers(device == "CPU", device, **kwargs)

    if providers[0] == "CPUExecutionProvider":
        if device != "CPU" and is_openvino_gpu_npu_available():
            return OpenVINOModelRunner(model_path, device, model_type, **kwargs)

    if (
        CudaGraphRunner.is_model_supported(model_type)
        and providers[0] == "CUDAExecutionProvider"
    ):
        # Pascal GPUs (SM6.x) cannot capture CUDA Graphs due to Memcpy nodes in
        # yolov8n. Wrap the attempt and fall back to plain CUDAExecutionProvider.
        try:
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
            logger.warning(
                f"CUDA Graph capture failed ({cuda_graph_err}); "
                "falling back to CUDA EP without graph capture"
            )
            options[0] = {k: v for k, v in options[0].items() if k != "enable_cuda_graph"}

    if (
        providers
        and providers[0] == "MIGraphXExecutionProvider"
        and ONNXModelRunner.is_migraphx_complex_model(model_type)
    ):
        providers.pop(0)
        options.pop(0)

    return ONNXModelRunner(
        ort.InferenceSession(
            model_path,
            sess_options=get_ort_session_options(
                ONNXModelRunner.is_cpu_complex_model(model_type)
            ),
            providers=providers,
            provider_options=options,
        ),
        model_type=model_type,
    )
