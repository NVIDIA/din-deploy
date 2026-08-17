// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// euler.slang

// Define the push constants structure
struct PushConstants {
    float t_curr;
    float t_next;
    uint total_elements;
};

// Expose push constants to Vulkan explicitly
[[vk::push_constant]]
ConstantBuffer<PushConstants> params;

// Read/Write buffer for hidden states (binding 0)
[[vk::binding(0)]]
RWStructuredBuffer<float> hidden_states;

// Read-only buffer for transformer output (binding 1)
[[vk::binding(1)]]
StructuredBuffer<float> transformer_output;

// Workgroup size - 256 threads per workgroup in a 1D layout
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    // SV_DispatchThreadID maps directly to gl_GlobalInvocationID
    uint idx = dispatchThreadID.x;

    // Bounds check to prevent reading/writing past the end of the buffer
    if (idx >= params.total_elements) {
        return;
    }

    // Calculate dt based on the push constants
    float dt = params.t_next - params.t_curr;

    // Perform the Forward Euler step
    hidden_states[idx] += dt * transformer_output[idx];
}