// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "seedvr_cuda_kernels.h"

namespace din::seedvr {
    inline void ValidateVaeTiling(int size, int overlap) {
        if (size < 0 || size % 16 || overlap < 0 || overlap % 16 ||
            (size != 0 && (size < 32 || overlap >= size))) {
            throw std::invalid_argument("VAE tile size must be 0 or a multiple of 16 >= 32; overlap must be "
                                        "a nonnegative multiple of 16 smaller than the enabled tile size");
        }
    }

    inline std::vector<size_t> TileStarts(size_t length, size_t tile, size_t overlap) {
        if (!tile || tile > length || (tile < length && overlap >= tile))
            throw std::invalid_argument("Invalid VAE tile geometry");
        std::vector<size_t> starts{0};
        while (starts.back() + tile < length) {
            starts.push_back(std::min(starts.back() + tile - overlap, length - tile));
        }
        return starts;
    }

    inline std::vector<SeedVrTile> MakeVaeTiles(size_t height, size_t width,
                                               size_t tile_height, size_t tile_width, size_t overlap) {
        const auto ys = TileStarts(height, tile_height, overlap);
        const auto xs = TileStarts(width, tile_width, overlap);
        std::vector<SeedVrTile> tiles;
        for (size_t y = 0; y < ys.size(); ++y) {
            for (size_t x = 0; x < xs.size(); ++x) {
                tiles.push_back({ys[y], xs[x], tile_height, tile_width,
                                 y ? ys[y - 1] + tile_height - ys[y] : 0,
                                 y + 1 < ys.size() ? ys[y] + tile_height - ys[y + 1] : 0,
                                 x ? xs[x - 1] + tile_width - xs[x] : 0,
                                 x + 1 < xs.size() ? xs[x] + tile_width - xs[x + 1] : 0});
            }
        }
        return tiles;
    }

    inline SeedVrTile ScaleVaeTile(SeedVrTile tile, size_t numerator, size_t denominator) {
        return {tile.y * numerator / denominator, tile.x * numerator / denominator,
                tile.height * numerator / denominator, tile.width * numerator / denominator,
                tile.top * numerator / denominator, tile.bottom * numerator / denominator,
                tile.left * numerator / denominator, tile.right * numerator / denominator};
    }
}
