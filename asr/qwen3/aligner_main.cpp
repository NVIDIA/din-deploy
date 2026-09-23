// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>

#include "forced_aligner.h"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

int main(int argc, char** argv)
{
    using namespace din::asr::qwen3;
    try
    {
        ForcedAlignerConfig config;
        argparse::ArgumentParser parser("din_asr_qwen3_aligner_cli");
        parser.add_description("Align supplied text with audio, independently of any ASR model (up to 180 seconds).");
        parser.add_argument("audiofile");
        parser.add_argument("--transcript").required().help("UTF-8 transcript file");
        parser.add_argument("--lang-id").default_value(std::string{"English"}).help("Language code or name");
        parser.add_argument("--granularity").default_value(std::string{"words"}).choices("words", "characters");
        parser.add_argument("--units").flag().help("Treat each transcript line as one supplied alignment unit");
        parser.add_argument("--provider").default_value(config.provider).choices("cpu", "trt-rtx");
        parser.add_argument("--model-dir").default_value(config.model_dir.string());
        parser.add_argument("--ep-cache").default_value(config.ep_cache_dir.string());
        parser.add_argument("--ep-context-dir").default_value(config.ep_context_dir.string());
        parser.parse_args(argc, argv);
        config.provider = parser.get<std::string>("--provider");
        config.model_dir = parser.get<std::string>("--model-dir");
        config.ep_cache_dir = parser.get<std::string>("--ep-cache");
        config.ep_context_dir = parser.get<std::string>("--ep-context-dir");
        config.granularity = parser.get<std::string>("--granularity") == "characters" ? AlignmentGranularity::Characters
                                                                                      : AlignmentGranularity::Words;
        const auto transcript_path = parser.get<std::string>("--transcript");
        std::ifstream file(transcript_path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot read transcript: " + transcript_path);
        std::string transcript{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (transcript.starts_with("\xef\xbb\xbf"))
            transcript.erase(0, 3);
        const auto language = parser.get<std::string>("--lang-id");
        const auto audio = din::io::LoadAudio(parser.get<std::string>("audiofile"), 16000);
        Qwen3ForcedAligner aligner(std::move(config));
        const auto start = std::chrono::steady_clock::now();
        std::vector<WordTimestamp> timestamps;
        if (parser.get<bool>("--units"))
        {
            std::vector<std::string> units;
            std::istringstream lines(transcript);
            for (std::string line; std::getline(lines, line);)
            {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                if (!line.empty())
                    units.push_back(std::move(line));
            }
            timestamps = aligner.AlignUnits(audio, units);
        }
        else
            timestamps = aligner.Align(audio, transcript, language);
        const auto seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        nlohmann::json output{{"text", transcript},
                              {"language", language},
                              {"audio_seconds", audio.Duration()},
                              {"align_seconds", seconds},
                              {"timestamps", nlohmann::json::array()}};
        for (const auto& word : timestamps)
            output["timestamps"].push_back(
                {{"text", word.text}, {"start_time", word.start_time}, {"end_time", word.end_time}});
        std::cout << output.dump(2) << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Qwen3 aligner: " << error.what() << '\n';
        return 1;
    }
}
