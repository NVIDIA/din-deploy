// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef DIN_ENABLE_NVTX
#include <nvtx3/nvtx3.hpp>

#define DIN_NVTX_FUNC_RANGE() NVTX3_FUNC_RANGE()

namespace din::common
{
// Alias the scoped_range to our own namespace when NVTX is enabled
using nvtx_scoped_range = nvtx3::scoped_range;
}  // namespace din::common

#else

// No-op macro for function ranges when disabled
#define DIN_NVTX_FUNC_RANGE() \
    do                        \
    {                         \
    } while (0)

namespace din::common
{
// Dummy class that does nothing and gets optimized away
class nvtx_scoped_range
{
public:
    explicit nvtx_scoped_range(const char*) {}
};
}  // namespace din::common

#endif
