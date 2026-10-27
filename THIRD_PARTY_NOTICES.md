# Third-Party Notices

DIN Deploy is distributed under the Apache License, Version 2.0. The project source includes or fetches the third-party components below. Their license terms remain applicable to those components. The listed links are the authoritative license texts or notices and are retained as part of the project distribution record.

| Component | License and notice |
|---|---|
| argparse | MIT: <https://github.com/p-ranav/argparse/blob/master/LICENSE> |
| miniaudio | MIT: <https://github.com/mackron/miniaudio/blob/0.11.23/LICENSE> |
| lodepng | zlib: <https://github.com/lvandeve/lodepng/blob/master/LICENSE> |
| nlohmann/json | MIT: <https://github.com/nlohmann/json/blob/develop/LICENSE.MIT> |
| PCRE2 | BSD-3-Clause WITH PCRE2-exception ([notice](#pcre2-notice)): <https://github.com/PCRE2Project/pcre2/blob/pcre2-10.46/LICENCE.md> |
| utf8proc | MIT and Unicode data license ([notices](#utf8proc-notices)): <https://github.com/JuliaStrings/utf8proc/blob/v2.11.0/LICENSE.md> |
| NVIDIA NVTX | Apache-2.0 with LLVM exception: <https://github.com/NVIDIA/NVTX/blob/master/LICENSE> |
| ONNX Runtime | MIT: <https://github.com/microsoft/onnxruntime/blob/main/LICENSE> |
| ONNX Runtime TensorRT RTX EP ABI | Apache-2.0: <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/blob/main/LICENSE> |
| Slang | Apache-2.0 with LLVM exception: <https://github.com/shader-slang/slang/blob/master/LICENSE> |
| Vulkan-Headers and Vulkan-Loader | Apache-2.0: <https://github.com/KhronosGroup/Vulkan-Headers/blob/main/LICENSE.txt> and <https://github.com/KhronosGroup/Vulkan-Loader/blob/main/LICENSE.txt> |
| nanobind | BSD-3-Clause: <https://github.com/wjakob/nanobind/blob/main/LICENSE> |
| Qwen3-ASR inference utilities | Apache-2.0, copyright 2026 The Alibaba Qwen team: <https://github.com/QwenLM/Qwen3-ASR/blob/main/LICENSE> |
| Transformers Qwen3-ASR processing | Apache-2.0, copyright Hugging Face: <https://github.com/huggingface/transformers/blob/v5.13.0/LICENSE> |
| Python dependencies | See the package license links and inventory below. |
| Distributed models and model artifacts | See the model license inventory below. Model terms may differ from this project's license. |

For components supplied through an SDK or binary package, the corresponding vendor license must be distributed with that SDK/package. In particular, TensorRT RTX is subject to the [NVIDIA TensorRT RTX Software License Agreement](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html).

## Dependency and model license inventory

## C++

- `argparse` v3.2: MIT
- `miniaudio` 0.11.23: MIT
- `lodepng`: zlib
- `nanobind` v2.9.2: BSD-3-Clause
- `nlohmann/json` v3.11.3: MIT
- `PCRE2` 10.46: BSD-3-Clause WITH PCRE2-exception
- `utf8proc` 2.11.0: MIT and Unicode data license
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

- `Qwen/Qwen3-ASR-0.6B-hf`: Apache-2.0
- `Qwen/Qwen3-ASR-1.7B-hf`: Apache-2.0
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`: Apache-2.0

## PCRE2 notice

```text
PCRE2 License
=============

| SPDX-License-Identifier: | BSD-3-Clause WITH PCRE2-exception |
|---------|-------|

PCRE2 is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

Releases 10.00 and above of PCRE2 are distributed under the terms of the "BSD"
licence, as specified below, with one exemption for certain binary
redistributions. The documentation for PCRE2, supplied in the "doc" directory,
is distributed under the same terms as the software itself. The data in the
testdata directory is not copyrighted and is in the public domain.

The basic library functions are written in C and are freestanding. Also
included in the distribution is a just-in-time compiler that can be used to
optimize pattern matching. This is an optional feature that can be omitted when
the library is built.


COPYRIGHT
---------

### The basic library functions

    Written by:       Philip Hazel
    Email local part: Philip.Hazel
    Email domain:     gmail.com

    Retired from University of Cambridge Computing Service,
    Cambridge, England.

    Copyright (c) 1997-2007 University of Cambridge
    Copyright (c) 2007-2024 Philip Hazel
    All rights reserved.

### PCRE2 Just-In-Time compilation support

    Written by:       Zoltan Herczeg
    Email local part: hzmester
    Email domain:     freemail.hu

    Copyright (c) 2010-2024 Zoltan Herczeg
    All rights reserved.

### Stack-less Just-In-Time compiler

    Written by:       Zoltan Herczeg
    Email local part: hzmester
    Email domain:     freemail.hu

    Copyright (c) 2009-2024 Zoltan Herczeg
    All rights reserved.

### All other contributions

Many other contributors have participated in the authorship of PCRE2. As PCRE2
has never required a Contributor Licensing Agreement, or other copyright
assignment agreement, all contributions have copyright retained by each
original contributor or their employer.


THE "BSD" LICENCE
-----------------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notices,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notices, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the University of Cambridge nor the names of any
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


EXEMPTION FOR BINARY LIBRARY-LIKE PACKAGES
------------------------------------------

The second condition in the BSD licence (covering binary redistributions) does
not apply all the way down a chain of software. If binary package A includes
PCRE2, it must respect the condition, but if package B is software that
includes package A, the condition is not imposed on package B unless it uses
PCRE2 independently.

End
```

## utf8proc notices

```text
## utf8proc license ##

**utf8proc** is a software package originally developed
by Jan Behrens and the rest of the Public Software Group, who
deserve nearly all of the credit for this library, that is now maintained by the Julia-language developers.  Like the original utf8proc,
whose copyright and license statements are reproduced below, all new
work on the utf8proc library is licensed under the [MIT "expat"
license](http://opensource.org/licenses/MIT):

*Copyright &copy; 2014-2021 by Steven G. Johnson, Jiahao Chen, Tony Kelman, Jonas Fonseca, and other contributors listed in the git history.*

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Original utf8proc license ##

*Copyright (c) 2009, 2013 Public Software Group e. V., Berlin, Germany*

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

## Unicode data license ##

This software contains data (`utf8proc_data.c`) derived from processing
the Unicode data files. The following license applies to that data:

**COPYRIGHT AND PERMISSION NOTICE**

*Copyright (c) 1991-2007 Unicode, Inc. All rights reserved. Distributed
under the Terms of Use in http://www.unicode.org/copyright.html.*

Permission is hereby granted, free of charge, to any person obtaining a
copy of the Unicode data files and any associated documentation (the "Data
Files") or Unicode software and any associated documentation (the
"Software") to deal in the Data Files or Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, and/or sell copies of the Data Files or Software, and
to permit persons to whom the Data Files or Software are furnished to do
so, provided that (a) the above copyright notice(s) and this permission
notice appear with all copies of the Data Files or Software, (b) both the
above copyright notice(s) and this permission notice appear in associated
documentation, and (c) there is clear notice in each modified Data File or
in the Software as well as in the documentation associated with the Data
File(s) or Software that the data or software has been modified.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS
INCLUDED IN THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR
CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THE DATA FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.

Unicode and the Unicode logo are trademarks of Unicode, Inc., and may be
registered in some jurisdictions. All other trademarks and registered
trademarks mentioned herein are the property of their respective owners.
```
