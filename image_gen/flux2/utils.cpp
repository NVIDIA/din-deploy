// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

Flux2LatentStats LoadFlux2LatentStats(const std::filesystem::path& path, size_t channels)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Missing VAE latent statistics: " + path.string() +
            ". Re-export the VAE decoder.");
    try
    {
        const auto document = nlohmann::json::parse(file);
        const auto read = [&](const char* key, bool positive)
        {
            const auto& values = document.at(key);
            if (!values.is_array() || values.size() != channels)
                throw std::runtime_error(std::string(key) + " must contain " +
                    std::to_string(channels) + " values");
            std::vector<float> result;
            result.reserve(channels);
            for (const auto& value : values)
            {
                if (!value.is_number())
                    throw std::runtime_error(std::string(key) + " must contain only numbers");
                const float number = value.get<float>();
                if (!std::isfinite(number) || (positive && number <= 0.0f))
                    throw std::runtime_error(std::string(key) + " contains an invalid value");
                result.push_back(number);
            }
            return result;
        };
        return {read("bn_mean", false), read("bn_std", true)};
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error("Invalid VAE latent statistics in " + path.string() + ": " + error.what());
    }
}

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <limits.h>
#include <mach-o/dyld.h>
#elif __linux__
#include <limits.h>
#include <unistd.h>
#endif

OrtFileString toOrtFileString(const std::filesystem::path& path)
{
#ifdef _WIN32
    return path.wstring();
#else
    return path.string();
#endif
}

std::filesystem::path get_executable_parent_path()
{
    return get_executable_path().parent_path();
}

std::filesystem::path get_executable_path()
{
#ifdef _WIN32
    std::vector<wchar_t> pathBuf(MAX_PATH);
    DWORD length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));

    while (length == pathBuf.size())
    {
        pathBuf.resize(pathBuf.size() * 2);
        length = GetModuleFileNameW(NULL, pathBuf.data(), static_cast<DWORD>(pathBuf.size()));
    }

    if (length == 0)
    {
        std::cerr << "Error: GetModuleFileNameW failed with error " << GetLastError() << std::endl;
        return {};
    }
    return std::filesystem::path(pathBuf.data());

#elif __APPLE__
    std::vector<char> pathBuf(PATH_MAX);
    uint32_t length = pathBuf.size();
    if (_NSGetExecutablePath(pathBuf.data(), &length) != 0)
    {
        pathBuf.resize(length + 1);
        if (_NSGetExecutablePath(pathBuf.data(), &length) != 0)
        {
            std::cerr << "Error: _NSGetExecutablePath failed" << std::endl;
            return {};
        }
    }
    return std::filesystem::canonical(pathBuf.data());

#elif __linux__
    return std::filesystem::canonical(std::filesystem::read_symlink("/proc/self/exe"));
#else
    return {};
#endif
}

void flush_ort_stream(Ort::SyncStream& stream)
{
    const OrtEpApi& ort_ep_api = *Ort::GetApi().GetEpApi();
    const OrtSyncStreamImpl* stream_impl = ort_ep_api.SyncStream_GetImpl(stream);
    if (stream_impl == nullptr)
    {
        THROW_ERROR("Failed to get SyncStream implementation");
    }

    CHECK_ORT(stream_impl->Flush(const_cast<OrtSyncStreamImpl*>(stream_impl)));
}

void copy_tensors(Ort::Env& env, std::initializer_list<Ort::Value*> src, std::initializer_list<Ort::Value*> dst,
                  Ort::SyncStream& stream)
{
    if (src.size() != dst.size())
    {
        THROW_ERROR("CopyTensors source/destination count mismatch");
    }

    std::vector<const OrtValue*> src_values;
    std::vector<OrtValue*> dst_values;
    src_values.reserve(src.size());
    dst_values.reserve(dst.size());
    std::transform(src.begin(), src.end(), std::back_inserter(src_values),
                   [](const Ort::Value* value)
                   {
                       return static_cast<const OrtValue*>(*value);
                   });
    std::transform(dst.begin(), dst.end(), std::back_inserter(dst_values),
                   [](Ort::Value* value)
                   {
                       return static_cast<OrtValue*>(*value);
                   });

    CHECK_ORT(Ort::GetApi().CopyTensors(env, src_values.data(), dst_values.data(), stream, src_values.size()));
}

void copy_tensor(Ort::Env& env, Ort::Value& src, Ort::Value& dst, Ort::SyncStream& stream)
{
    CHECK_ORT(env.CopyTensor(static_cast<const OrtValue*>(src), static_cast<OrtValue*>(dst), stream));
}

OrtSyncNotificationImpl* create_sync_notification(Ort::SyncStream& stream)
{
    const OrtEpApi& ort_ep_api = *Ort::GetApi().GetEpApi();
    const OrtSyncStreamImpl* stream_impl = ort_ep_api.SyncStream_GetImpl(stream);
    if (stream_impl == nullptr)
    {
        THROW_ERROR("Failed to get SyncStream implementation");
    }

    OrtSyncNotificationImpl* notification = nullptr;
    CHECK_ORT(stream_impl->CreateNotification(const_cast<OrtSyncStreamImpl*>(stream_impl), &notification));
    return notification;
}

void release_sync_notification(OrtSyncNotificationImpl* notification)
{
    if (notification != nullptr)
    {
        notification->Release(notification);
    }
}

NotificationUniquePtr make_notification(Ort::SyncStream& stream)
{
    return NotificationUniquePtr(create_sync_notification(stream), release_sync_notification);
}

NotificationUniquePtr signal_and_wait_on_device(Ort::SyncStream& producer, Ort::SyncStream& consumer)
{
    auto done = make_notification(producer);
    CHECK_ORT(done->Activate(done.get()));
    CHECK_ORT(done->WaitOnDevice(done.get(), consumer));
    return done;
}

void signal_and_wait_on_host(Ort::SyncStream& producer)
{
    auto done = make_notification(producer);
    CHECK_ORT(done->Activate(done.get()));
    CHECK_ORT(done->WaitOnHost(done.get()));
}
