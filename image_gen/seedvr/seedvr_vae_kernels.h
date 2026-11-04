// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
namespace din::seedvr::vae
{
void norm(const half*, half*, float*, const half*, const half*, int, int, int, int64_t, int64_t, bool, cudaStream_t);
void copy(const half*, half*, int, int, int, int64_t, int64_t, cudaStream_t);
void residual(const half*, half*, int, int, int, int64_t, int64_t, cudaStream_t);
void boundary(const half*, half*, int, int, int, int64_t, bool, int, cudaStream_t);
void shuffle(const half*, half*, int, int, int, int, int64_t, int64_t, int, cudaStream_t);
void pad_time(const half*, half*, int, int64_t, int, cudaStream_t);
void to_float(const half*, float*, int64_t, cudaStream_t);
}  // namespace din::seedvr::vae
