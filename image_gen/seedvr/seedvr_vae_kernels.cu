// SPDX-License-Identifier: Apache-2.0
// Generalized from tensorrt-playground/apps/seedvr2_resblock/kernels.cu.
#include <cmath>

#include "seedvr_vae_kernels.h"
namespace din::seedvr::vae
{
__global__ void stats_kernel(const half* x, float* stats, int c, int hw, int64_t stride)
{
    int g = blockIdx.x % 32, t = blockIdx.x / 32, cg = c / 32;
    double sum = 0, sq = 0;
    for (int64_t j = threadIdx.x; j < int64_t(hw) * cg; j += blockDim.x)
    {
        double v = __half2float(x[t * stride + (j / cg) * c + g * cg + j % cg]);
        sum += v;
        sq += v * v;
    }
    __shared__ double a[256], b[256];
    a[threadIdx.x] = sum;
    b[threadIdx.x] = sq;
    __syncthreads();
    for (int s = 128; s; s /= 2)
    {
        if (threadIdx.x < s)
        {
            a[threadIdx.x] += a[threadIdx.x + s];
            b[threadIdx.x] += b[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (!threadIdx.x)
    {
        double m = a[0] / (int64_t(hw) * cg);
        stats[blockIdx.x * 2] = float(m);
        stats[blockIdx.x * 2 + 1] = float(1 / sqrt(fmax(0., b[0] / (int64_t(hw) * cg) - m * m) + 1e-6));
    }
}
__global__ void norm_kernel(const half* x, half* y, const float* s, const half* g, const half* b, int c, int64_t frame,
                            int64_t n, int64_t xs, int64_t ys, bool silu)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    {
        int ch = i % c, t = int(i / frame), k = (t * 32 + ch / (c / 32)) * 2;
        float v = (__half2float(x[t * xs + i % frame]) - s[k]) * s[k + 1];
        v = __half2float(__float2half_rn(v * __half2float(g[ch]) + __half2float(b[ch])));
        y[t * ys + i % frame] = __float2half_rn(silu ? v / (1 + expf(-v)) : v);
    }
}
void norm(const half* x, half* y, float* s, const half* g, const half* b, int t, int c, int hw, int64_t xs, int64_t ys,
          bool silu, cudaStream_t stream)
{
    stats_kernel<<<t * 32, 256, 0, stream>>>(x, s, c, hw, xs);
    norm_kernel<<<4096, 256, 0, stream>>>(x, y, s, g, b, c, int64_t(c) * hw, int64_t(t) * c * hw, xs, ys, silu);
}
__global__ void copy_kernel(const half* x, half* y, int64_t n, int64_t f, int64_t xs, int64_t ys, bool add)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    {
        auto j = (i / f) * ys + i % f;
        half v = x[(i / f) * xs + i % f];
        y[j] = add ? __hadd(y[j], v) : v;
    }
}
void copy(const half* x, half* y, int t, int c, int hw, int64_t xs, int64_t ys, cudaStream_t s)
{
    copy_kernel<<<4096, 256, 0, s>>>(x, y, int64_t(t) * c * hw, int64_t(c) * hw, xs, ys, false);
}
void residual(const half* x, half* y, int t, int c, int hw, int64_t xs, int64_t ys, cudaStream_t s)
{
    copy_kernel<<<4096, 256, 0, s>>>(x, y, int64_t(t) * c * hw, int64_t(c) * hw, xs, ys, true);
}
// mode 0 encode input, 1 encode output, 2 decode input, 3 decode output.
__global__ void boundary_kernel(const half* x, half* y, int t, int c, int hw, int64_t stride, bool input, int mode)
{
    int64_t n = int64_t(t) * c * hw;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    {
        int ch = int(i % c), f = int(i / (int64_t(c) * hw));
        int64_t spatial = (i / c) % hw;
        int64_t nchw = (int64_t(ch) * t + f) * hw + spatial;
        float v = __half2float(x[input ? nchw : f * stride + i % (int64_t(c) * hw)]);
        if (mode == 0)
            v = __half2float(__float2half_rn(v * 2)) - 1;
        else if (mode == 1)
            v *= 0.9150390625f;
        else if (mode == 2)
            v /= 0.9150390625f;
        else
            v = __half2float(__float2half_rn(v + 1)) * .5f;
        y[input ? f * stride + i % (int64_t(c) * hw) : nchw] = __float2half_rn(v);
    }
}
void boundary(const half* x, half* y, int t, int c, int hw, int64_t stride, bool input, int mode, cudaStream_t s)
{
    boundary_kernel<<<4096, 256, 0, s>>>(x, y, t, c, hw, stride, input, mode);
}
__global__ void shuffle_kernel(const half* x, half* y, int t, int c, int h, int w, int64_t xs, int64_t ys, int rt)
{
    int oh = h * 2, ow = w * 2, ot = t * rt - (rt - 1);
    int64_t n = int64_t(ot) * oh * ow * c;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    {
        int ch = i % c, xx = (i / c) % ow, yy = (i / c / ow) % oh, tt = int(i / c / ow / oh),
            src_t = tt == 0 ? 0 : tt + rt - 1;
        int src_c = (((yy % 2) * 2 + xx % 2) * rt + src_t % rt) * c + ch;
        y[int64_t(tt) * ys + (int64_t(yy) * ow + xx) * c + ch] =
            x[int64_t(src_t / rt) * xs + (int64_t(yy / 2) * w + xx / 2) * (c * 4 * rt) + src_c];
    }
}
void shuffle(const half* x, half* y, int t, int c, int h, int w, int64_t xs, int64_t ys, int rt, cudaStream_t s)
{
    shuffle_kernel<<<4096, 256, 0, s>>>(x, y, t, c, h, w, xs, ys, rt);
}
__global__ void pad_kernel(const half* x, half* y, int64_t frame, int t, int pad)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(t + pad) * frame;
         i += int64_t(gridDim.x) * blockDim.x)
        y[i] = x[int64_t(max(0, int(i / frame) - pad)) * frame + i % frame];
}
void pad_time(const half* x, half* y, int t, int64_t frame, int pad, cudaStream_t s)
{
    pad_kernel<<<4096, 256, 0, s>>>(x, y, frame, t, pad);
}
__global__ void float_kernel(const half* x, float* y, int64_t n)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        y[i] = __half2float(x[i]);
}
void to_float(const half* x, float* y, int64_t n, cudaStream_t s)
{
    float_kernel<<<4096, 256, 0, s>>>(x, y, n);
}
}  // namespace din::seedvr::vae
