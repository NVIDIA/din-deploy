// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

#include <utf8proc.h>

namespace din::asr::qwen3::detail
{
inline bool ValidUtf8(const std::string& text)
{
    for (size_t i = 0; i < text.size();)
    {
        utf8proc_int32_t code;
        const auto n =
            utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + i), text.size() - i, &code);
        if (n < 0)
            return false;
        i += n;
    }
    return true;
}
}  // namespace din::asr::qwen3::detail
