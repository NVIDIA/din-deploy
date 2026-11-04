// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "../../../common/nvtx_helper.h"
#include <cstdio>

namespace din::sam3 {
// Host enqueue/wait ranges. They deliberately add no CUDA synchronization.
// Stop supports the existing timing scopes that end before their C++ scope.
class TraceRange {
public:
    explicit TraceRange(const char* name) {
#ifdef DIN_ENABLE_NVTX
        // Profiling must not introduce heap allocations into device submission.
        char label[256];
        std::snprintf(label, sizeof(label), "sam3.%s", name);
        nvtxRangePushA(label);
#endif
    }
    TraceRange(const TraceRange&) = delete;
    TraceRange& operator=(const TraceRange&) = delete;
    ~TraceRange() {Stop();}
    void Stop() {
#ifdef DIN_ENABLE_NVTX
        if (active_) {nvtxRangePop(); active_ = false;}
#endif
    }
private:
    [[maybe_unused]] bool active_ = true;
};
}
