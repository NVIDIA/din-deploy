# Third-Party Notices

DIN Deploy is distributed under the Apache License, Version 2.0. The project source includes or fetches the third-party components below. Their license terms remain applicable to those components. The listed links are the authoritative license texts or notices and are retained as part of the project distribution record.

| Component | License and notice |
|---|---|
| argparse | MIT: <https://github.com/p-ranav/argparse/blob/master/LICENSE> |
| dr_libs | Public domain or MIT-0: <https://github.com/mackron/dr_libs/blob/master/LICENSE> |
| lodepng | zlib: <https://github.com/lvandeve/lodepng/blob/master/LICENSE> |
| nlohmann/json | MIT: <https://github.com/nlohmann/json/blob/develop/LICENSE.MIT> |
| NVIDIA NVTX | Apache-2.0 with LLVM exception: <https://github.com/NVIDIA/NVTX/blob/master/LICENSE> |
| ONNX Runtime | MIT: <https://github.com/microsoft/onnxruntime/blob/main/LICENSE> |
| ONNX Runtime TensorRT RTX EP ABI | Apache-2.0: <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/blob/main/LICENSE> |
| Slang | Apache-2.0 with LLVM exception: <https://github.com/shader-slang/slang/blob/master/LICENSE> |
| Vulkan-Headers and Vulkan-Loader | Apache-2.0: <https://github.com/KhronosGroup/Vulkan-Headers/blob/main/LICENSE.txt> and <https://github.com/KhronosGroup/Vulkan-Loader/blob/main/LICENSE.txt> |
| nanobind | BSD-3-Clause: <https://github.com/wjakob/nanobind/blob/main/LICENSE> |
| Python dependencies | See the package license links and inventory below. |
| Distributed models and model artifacts | See the model license inventory below. Model terms may differ from this project's license. |

For components supplied through an SDK or binary package, the corresponding vendor license must be distributed with that SDK/package. In particular, TensorRT RTX is subject to the [NVIDIA TensorRT RTX Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html).

## Dependency and model license inventory

## C++

- `argparse` v3.2: MIT
- `dr_libs`: Public domain or MIT-0
- `lodepng`: zlib
- `nanobind` v2.9.2: BSD-3-Clause
- `nlohmann/json` v3.11.3: MIT
- NVIDIA NVTX v3.5.0 C/C++: Apache-2.0 WITH LLVM-exception
- ONNX Runtime SDK 1.27.0: MIT
- ONNX Runtime TensorRT RTX Execution Provider ABI: Apache-2.0
- Slang Apache-2.0 WITH LLVM-exception
- TensorRT RTX SDK: [NVIDIA TensorRT RTX Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html)
- Vulkan SDK: Apache-2.0 for Vulkan-Headers/Loader

## Python
- `av`: BSD-3-Clause: <https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt>
- `comfy-kitchen>=0.2.22`: Apache-2.0: <https://pypi.org/project/comfy-kitchen/>
- `sam-2`: Apache-2.0: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>

- `accelerate`: Apache-2.0
- `datasets`: Apache-2.0
- `diffusers`: Apache-2.0
- `huggingface_hub`: Apache-2.0
- `jiwer`: Apache-2.0
- `librosa>=0.10.2`: ISC
- `matplotlib`: [Matplotlib License / PSF-style](https://matplotlib.org/stable/project/license.html)
- `nanobind`: BSD-3-Clause
- `nemo_toolkit[asr]`: Apache-2.0
- `numba>=0.59`: BSD-2-Clause
- `numpy`: BSD-3-Clause
- `nvidia-modelopt`: Apache-2.0
- `onnx`: Apache-2.0
- `onnxruntime`: MIT
- `onnxruntime-ep-nv-tensorrt-rtx`: Apache-2.0 package metadata; bundled TensorRT RTX runtime is governed by the [NVIDIA TensorRT RTX Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html)
- `onnxscript`: MIT
- `pillow`: [HPND / Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- `pytest`: MIT
- `PyYAML`: MIT
- `ruff`: MIT
- `safetensors`: Apache-2.0
- `scikit-build-core>=0.10`: Apache-2.0
- `sentencepiece`: Apache-2.0
- `setuptools>=69`: MIT
- `soundfile`: BSD-3-Clause
- `torch`: BSD-3-Clause
- `torchaudio`: BSD-3-Clause
- `torchvision`: BSD-3-Clause
- `tqdm`: MIT and MPL-2.0
- `transformers`: Apache-2.0
- `wheel`: MIT

## Model/Artifact

- `black-forest-labs/FLUX.2-klein-4b`: Apache-2.0
- `black-forest-labs/FLUX.2-klein-4b-fp8`: Apache-2.0
- `black-forest-labs/FLUX.2-klein-4b-nvfp4`: Apache-2.0
- `facebook/sam2.1-hiera-tiny`: Apache-2.0: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>
- `facebook/sam2.1-hiera-small`: Apache-2.0: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>
- `facebook/sam2.1-hiera-base-plus`: Apache-2.0: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>
- `facebook/sam2.1-hiera-large`: Apache-2.0: <https://github.com/facebookresearch/sam2/blob/main/LICENSE>
- `nvidia/nemotron-3.5-asr-streaming-0.6b`: [OpenMDW-1.1](https://openmdw.ai/license/)
- `nvidia/parakeet-tdt-0.6b-v3`: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- `onnxmodelzoo/resnet18_Opset18_timm`: Apache-2.0
- `openai/whisper-tiny`: Apache-2.0
- `openai/whisper-base`: Apache-2.0
- `openai/whisper-large-v3`: Apache-2.0
- `openai/whisper-large-v3-turbo`: Apache-2.0
- `openai/whisper-medium`: Apache-2.0
- `openai/whisper-small`: Apache-2.0
