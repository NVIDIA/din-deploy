// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// postprocess.slang — fused unpack + denormalize + unpatchify
//
// Input:  hidden_states [HW, C]  (transformer output after Euler step)
// Output: decoder_latent [C_out, H_out, W_out]
//
// C_out = C / (pi * pj), H_out = I * pi, W_out = J * pj

struct PushConstants {
    uint C;
    uint I;
    uint J;
    uint pi;
    uint pj;
    uint total_elements;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> params;

[[vk::binding(0)]]
StructuredBuffer<float> input_data;

[[vk::binding(1)]]
RWStructuredBuffer<float> output_data;

[[vk::binding(2)]]
StructuredBuffer<float> bn_mean;

[[vk::binding(3)]]
StructuredBuffer<float> bn_std;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= params.total_elements) return;

    uint C_out = params.C / (params.pi * params.pj);
    uint H_out = params.I * params.pi;
    uint W_out = params.J * params.pj;

    // Decompose output index: [C_out, H_out, W_out]
    uint c_out = idx / (H_out * W_out);
    uint rem   = idx % (H_out * W_out);
    uint i_out = rem / W_out;
    uint j_out = rem % W_out;

    // Reverse unpatchify → intermediate [C, I, J]
    uint i  = i_out / params.pi;
    uint p  = i_out % params.pi;
    uint j  = j_out / params.pj;
    uint q  = j_out % params.pj;
    uint in_c = c_out * params.pi * params.pj + p * params.pj + q;

    // Fused unpack + denormalize: read from input [HW, C]
    uint hw = i * params.J + j;
    output_data[idx] = input_data[hw * params.C + in_c] * bn_std[in_c] + bn_mean[in_c];
}
