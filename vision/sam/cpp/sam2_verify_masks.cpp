// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "sam2_io.h"

namespace
{

namespace fs = std::filesystem;

struct Args
{
    fs::path candidate;
    fs::path reference;
    double min_iou = 0.95;
};

Args ParseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        if (key == "--candidate" && i + 1 < argc)
        {
            args.candidate = argv[++i];
        }
        else if (key == "--reference" && i + 1 < argc)
        {
            args.reference = argv[++i];
        }
        else if (key == "--min-iou" && i + 1 < argc)
        {
            args.min_iou = std::stod(argv[++i]);
        }
        else
        {
            throw std::runtime_error("usage: din_sam2_verify_masks --candidate DIR --reference DIR [--min-iou VALUE]");
        }
    }
    if (args.candidate.empty() || args.reference.empty())
    {
        throw std::runtime_error("usage: din_sam2_verify_masks --candidate DIR --reference DIR [--min-iou VALUE]");
    }
    return args;
}

double MaskIou(const din::sam2::Image& reference, const din::sam2::Image& candidate)
{
    if (reference.width != candidate.width || reference.height != candidate.height)
    {
        throw std::runtime_error("mask size mismatch");
    }

    size_t intersection = 0;
    size_t union_count = 0;
    const size_t pixels = static_cast<size_t>(reference.width) * reference.height;
    for (size_t i = 0; i < pixels; ++i)
    {
        const bool ref = reference.rgb[i * 3] != 0;
        const bool cand = candidate.rgb[i * 3] != 0;
        intersection += ref && cand ? 1 : 0;
        union_count += ref || cand ? 1 : 0;
    }
    return union_count == 0 ? 1.0 : static_cast<double>(intersection) / static_cast<double>(union_count);
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const Args args = ParseArgs(argc, argv);
        const auto reference_masks = din::sam2::ListFrames(args.reference);
        if (reference_masks.empty())
        {
            throw std::runtime_error("no reference masks found in " + args.reference.string());
        }

        double sum_iou = 0.0;
        for (const auto& reference_path : reference_masks)
        {
            const auto candidate_path = args.candidate / reference_path.filename();
            if (!fs::is_regular_file(candidate_path))
            {
                throw std::runtime_error("missing candidate mask: " + candidate_path.string());
            }
            const auto reference = din::sam2::LoadPng(reference_path);
            const auto candidate = din::sam2::LoadPng(candidate_path);
            const double iou = MaskIou(reference, candidate);
            std::cout << reference_path.filename().string() << " iou=" << iou << '\n';
            if (iou < args.min_iou)
            {
                throw std::runtime_error("mask IoU below threshold for " + reference_path.filename().string());
            }
            sum_iou += iou;
        }

        std::cout << "mean_iou=" << (sum_iou / static_cast<double>(reference_masks.size())) << '\n';
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
