// SPDX-License-Identifier: Apache-2.0
#include "unicode_regex.h"

#include <memory>
#include <stdexcept>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace din::io
{
UnicodeRegex::UnicodeRegex(std::string_view pattern)
{
    int error;
    PCRE2_SIZE offset;
    code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &error,
                          &offset, nullptr);
    if (!code_)
        throw std::runtime_error("Invalid Unicode regex at byte " + std::to_string(offset));
}

UnicodeRegex::~UnicodeRegex()
{
    pcre2_code_free(code_);
}

std::vector<std::string> UnicodeRegex::FindAll(std::string_view text) const
{
    std::vector<std::string> matches;
    const std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> data(
        pcre2_match_data_create_from_pattern(code_, nullptr), pcre2_match_data_free);
    if (!data)
        throw std::bad_alloc();
    size_t offset = 0;
    while (offset < text.size())
    {
        const int result = pcre2_match(code_, reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset,
                                       offset ? PCRE2_NO_UTF_CHECK : 0, data.get(), nullptr);
        if (result == PCRE2_ERROR_NOMATCH)
            break;
        if (result < 0)
            throw std::invalid_argument("Invalid UTF-8 text or Unicode regex match failure: " + std::to_string(result));
        const auto* span = pcre2_get_ovector_pointer(data.get());
        if (span[1] <= offset)
            throw std::runtime_error("Unicode regex must consume input");
        matches.emplace_back(text.substr(span[0], span[1] - span[0]));
        offset = span[1];
    }
    return matches;
}
}  // namespace din::io
