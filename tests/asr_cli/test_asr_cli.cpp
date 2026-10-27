// SPDX-License-Identifier: Apache-2.0
#include "cli.h"

namespace
{
struct Config
{
    std::filesystem::path model_dir = "models";
    std::string provider = "cpu";
    std::filesystem::path ep_cache_dir = "cache";
    std::filesystem::path ep_context_dir = "context";
};
struct Result
{
    std::string text = "Hello 中";
    double audio_seconds = 1.0;
    double transcribe_seconds = 0.25;
    double encode_seconds = 0.2;
    double greedy_seconds = 0.05;
};
struct Pipeline
{
    explicit Pipeline(Config) {}
    Result TranscribeFile(const std::filesystem::path& path)
    {
        if (path == "fail.wav")
            throw std::runtime_error("inference failed");
        return {};
    }
    void Print(std::ostream& out, const Result& result, const din::asr::common::TranscriptionOptions&) const
    {
        out << result.text << '\n';
    }
};
struct Output
{
    int code;
    std::string out;
    std::string error;
};
Output run(std::vector<std::string> args)
{
    std::vector<char*> argv;
    for (auto& arg : args)
        argv.push_back(arg.data());
    std::ostringstream out, error;
    auto* previous_out = std::cout.rdbuf(out.rdbuf());
    auto* previous_error = std::cerr.rdbuf(error.rdbuf());
    const din::asr::common::CliSpec spec{.fallback_name = "test_asr",
                                         .artifact_label = "Test",
                                         .default_provider = "cpu",
                                         .timestamps_modes = "none|token",
                                         .allowed_timestamps = {"none", "token"}};
    const int code = din::asr::common::RunCli<Pipeline, Config>(static_cast<int>(argv.size()), argv.data(), spec);
    std::cout.rdbuf(previous_out);
    std::cerr.rdbuf(previous_error);
    return {code, out.str(), error.str()};
}
void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}
}  // namespace

int main()
{
    try
    {
        const auto json = run({"test_asr", "audio.wav", "--json"});
        require(json.code == 0, "JSON command failed");
        const auto document = nlohmann::json::parse(json.out);
        require(document.at("transcription") == "Hello 中", "JSON lost Unicode text");
        require(document.at("audio_seconds") == 1.0, "JSON lost audio duration");
        require(json.error.find("provider: cpu") != std::string::npos, "JSON timings must go to stderr");
        const auto plain = run({"test_asr", "audio.wav"});
        require(plain.code == 0 && plain.out.starts_with("Hello 中\n"), "Plain transcription changed");
        require(plain.out.find("provider: cpu") != std::string::npos, "Plain timings changed");
        const auto invalid = run({"test_asr", "audio.wav", "--timestamps", "invalid"});
        require(invalid.code != 0 && invalid.out.empty(), "Invalid options must fail without a transcript");
        const auto failure = run({"test_asr", "fail.wav", "--json"});
        require(failure.code != 0 && failure.out.empty(), "Inference failure must not print a JSON transcript");
        std::cout << "ASR CLI checks passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
