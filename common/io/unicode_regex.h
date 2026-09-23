// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <string_view>
#include <vector>

struct pcre2_real_code_8;

namespace din::io
{
// Compiled once; matching uses per-call state and validates UTF-8.
class UnicodeRegex
{
public:
    explicit UnicodeRegex(std::string_view pattern);
    ~UnicodeRegex();
    UnicodeRegex(const UnicodeRegex&) = delete;
    UnicodeRegex& operator=(const UnicodeRegex&) = delete;
    std::vector<std::string> FindAll(std::string_view text) const;

private:
    pcre2_real_code_8* code_;
};
}  // namespace din::io
