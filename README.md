# DIN Deploy — DO INFERENCE NOW

**Do inference now** — local inference samples for client PCs.

DIN Deploy is a collection of practical samples for exporting and running local models with ONNX Runtime. The samples prioritize CUDA-accelerated inference where available, while supporting CPU execution for cross-vendor use.

## Supported models

### Image generation

| Model | Hugging Face ID | Docs |
| --- | --- | --- |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4b`<br>`black-forest-labs/FLUX.2-klein-4b-fp8`<br>`black-forest-labs/FLUX.2-klein-4b-nvfp4` | [Flux](image_gen/flux2/README.md) |

### Speech recognition

| Model | Hugging Face ID | Docs |
| --- | --- | --- |
| OpenAI Whisper | `openai/whisper-tiny`<br>`openai/whisper-base`<br>`openai/whisper-small`<br>`openai/whisper-medium`<br>`openai/whisper-large-v3`<br>`openai/whisper-large-v3-turbo` | [Whisper](asr/whisper/README.md) |
| NVIDIA Parakeet TDT 0.6B v3 | `nvidia/parakeet-tdt-0.6b-v3` | [RNNT](asr/rnnt/README.md) |
| NVIDIA Nemotron 3.5 ASR Streaming 0.6B | `nvidia/nemotron-3.5-asr-streaming-0.6b` | [RNNT](asr/rnnt/README.md) |
| Qwen3 ASR | `Qwen/Qwen3-ASR-0.6B-hf`<br>`Qwen/Qwen3-ASR-1.7B-hf` | [Qwen3](asr/qwen3/README.md) |
| Qwen3 Forced Aligner | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | [Qwen3](asr/qwen3/README.md) |

### Computer Vision

| Model     | Hugging Face ID                                                                                                                         | Docs                          |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Meta Sam2 | `facebook/sam2.1-hiera-tiny`<br>`facebook/sam2.1-hiera-small`<br>`facebook/sam2.1-hiera-base-plus`<br>`facebook/sam2.1-hiera-large`<br> | [Sam2](vision/sam2/README.md) |

## Performance benchmarks

Results measured on a DGX Spark using CPU EP vs TensorRT RTX EP. Audio throughput is shown as multiples of real time (higher is faster).

| Model |         GPU | CPU |
| --- |------------:| ---: |
| `openai/whisper-large-v3-turbo` | [58.5×](asr/whisper/README.md#dgx-spark-performance) | [3.8×](asr/whisper/README.md#dgx-spark-performance) |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` |  39.01× | 3.24× |
| `nvidia/parakeet-tdt-0.6b-v3` | 206.41× | 14.44× |
| `facebook/sam2.1-hiera-base-plus` |    38.3 FPS | 0.5 FPS |

### FLUX.2-klein-4B quantization

For the fully CUDA backend-based pipeline we measured performance on a DGX Spark across the used quantization precisions.

![FLUX.2-klein-4B speedup by precision on DGX Spark](assets/flux2-speedup-dgx-spark.svg)

## Getting Started

Each model sample has two parts:

1. a Python exporter that downloads/converts a model into an ONNX artifact directory; and
2. a native CLI that runs that directory with ONNX Runtime using [TensorRT RTX execution provider](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) (EP) or CPU EP.

Use the model-specific README for the exporter command and CLI arguments.

## Requirements

- A C++20 compiler and CMake 3.24 or newer.
- Python 3.12 or newer for model exporters.
- CUDA Toolkit and the TensorRT RTX SDK for TensorRT RTX execution-provider builds.
- A supported CMake preset from `CMakePresets.json` (Windows, Linux, and ARM64 variants are available).

Create a Python environment and install the repository plus the extra required by the model you plan to export:

```bash
python -m pip install -e ".[parakeet]"  # or .[nemotron] / .[flux]
```

The base `pyproject.toml` contains common exporter dependencies; the optional extras add the model-family dependencies. Export the model to a directory, then pass that directory to the corresponding CLI with `--model-dir`.

### Build a sample

The supported presets automatically download TensorRT RTX during the first CMake configure step. CMake also obtains ONNX Runtime by default and builds the TensorRT RTX execution provider from source, so a separate native EP installation is not required.

```bash
cmake --preset <preset>
cmake --build out/build/<preset> --config Release
```

Choose a preset from `CMakePresets.json` (for example, `windows-x64`, `linux-x64`, or an ARM64 variant). Add `-D<setting>=<value>` to the configure command to override a setting.

| CMake setting | Default | Purpose                                                                                                                                                                                                                                                                     |
| --- | --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DIN_BUILD_TRT_RTX_EP` | `ON` | Builds the TensorRT RTX execution provider from source. Set to `OFF` when a compatible provider library was [downloaded](github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/) and placed in the executable directory.                                                                                                          |
| `DIN_ENABLE_NVTX` | `ON` | Enables NVTX profiling instrumentation. Set to `OFF` to build without it.                                                                                                                                                                                                   |
| `ONNXRUNTIME_VERSION` | `1.27.0` | ONNX Runtime version to download when `ONNXRUNTIME_ROOT` is not provided.                                                                                                                                                                                                   |
| `TRT_RTX_ROOT` | Not set | Optional path to an extracted TensorRT RTX SDK root. Set it to use a local SDK instead of the preset's download; the directory must contain `include/` and `lib/`. Releases are available from the [TensorRT RTX download page](https://developer.nvidia.com/tensorrt-rtx). |

Build one sample rather than the whole project by naming its target:

```bash
cmake --build out/build/<preset> --config Release --target din_asr_nemotron_cli
cmake --build out/build/<preset> --config Release --target din_asr_parakeet_tdt_cli
cmake --build out/build/<preset> --config Release --target din_asr_whisper_cli
cmake --build out/build/<preset> --config Release --target din_flux2_cli
```

The built executable and required runtime libraries are placed under `out/build/<preset>/bin/<configuration>/` for multi-config generators.

## Usage

Run the desired `din_*_cli` executable from the build directory with the model directory and arguments described in its sample README. Each CLI also documents its options through `--help`.

For an introductory native ONNX integration example, build and run `din_base_onnx`, which uses ResNet-18.

## Contribution Guidelines

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Contributions must include the Developer Certificate of Origin sign-off and appropriate tests and documentation updates.

## Security

Do not report potential security vulnerabilities in public issues. Report them through NVIDIA's [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html) or by email to `psirt@nvidia.com`.

## Community

DIN Deploy is a collection of samples. For usage questions, bug reports, and feature requests, open a [GitHub issue](https://github.com/NVIDIA/din-deploy/issues).
Community collaboration happens through [GitHub issues](https://github.com/NVIDIA/din-deploy/issues) and pull requests.

## License

This project is distributed under the [Apache License 2.0](LICENSE). Third-party software and model licenses are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
