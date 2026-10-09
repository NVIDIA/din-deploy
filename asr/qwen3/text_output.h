// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace din::asr::qwen3::detail
{
// A generation limit may split a byte-BPE character. Preserve complete characters
// and malformed sequences; only discard a valid prefix of an unfinished character.
inline void TrimIncompleteUtf8Suffix(std::string& text)
{
    if (text.empty())
        return;
    size_t start = text.size() - 1;
    while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80)
        --start;
    const auto lead = static_cast<unsigned char>(text[start]);
    const size_t expected = lead >= 0xC2 && lead <= 0xDF   ? 2
                            : lead >= 0xE0 && lead <= 0xEF ? 3
                            : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                           : 0;
    const auto available = text.size() - start;
    if (!expected || available >= expected)
        return;
    if (available > 1)
    {
        const auto second = static_cast<unsigned char>(text[start + 1]);
        // Reject overlong encodings, surrogates, and code points beyond U+10FFFF.
        if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second > 0x9F) || (lead == 0xF0 && second < 0x90) ||
            (lead == 0xF4 && second > 0x8F))
            return;
    }
    text.resize(start);
}
}  // namespace din::asr::qwen3::detail
