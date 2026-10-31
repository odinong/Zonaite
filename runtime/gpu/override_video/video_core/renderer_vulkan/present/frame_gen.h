// SPDX-License-Identifier: GPL-3.0-or-later
// Zonaite's frame generation, replacing Eden's (which runs Lossless Scaling's shaders from the user's copy
// of that program). Same interface, so Eden's renderer and present manager drive it unchanged: Process() takes
// each composited game frame, GenerateInto() draws the in-between frames before it is presented. The motion
// estimation and interpolation are Zonaite's own compute shaders (runtime/gpu/shaders/zn_interp_*), and the
// implementation is runtime/gpu/frame_interp.cpp.
#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <optional>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/present/frame_gen_pacer.h"
#include "video_core/renderer_vulkan/zonaite_capture.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;
struct Frame;
class InterpChain;

class FrameGen {
public:
    explicit FrameGen(MemoryAllocator& memory_allocator, Scheduler& scheduler);
    ~FrameGen();

    void Process(const Device& device, Frame* frame, VkFormat format, VkExtent2D guest_extent);

    [[nodiscard]] size_t WantedGenerations(size_t capacity);

    [[nodiscard]] size_t GeneratedFrameCount() const;

    void GenerateInto(const Device& device, Frame* destination, size_t generation);

private:
    void Schedule(Frame* frame);

    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    std::unique_ptr<InterpChain> chain;
    FrameGenPacer pacer;
    FrameGenPlan plan{};
    VkExtent2D built_extent{};
    VkFormat built_format{VK_FORMAT_UNDEFINED};
    u64 frame_count{};
    u64 last_count{};
    size_t last_generations{};
    u32 warm_streak{};
    bool generated{};

    // pacing: when each frame is shown (present manager Frame::present_at), spread evenly over the game frame
    std::chrono::steady_clock::time_point last_arrival{};  // the previous game frame reached Process
    std::chrono::steady_clock::time_point last_target{};   // when the previous game frame is shown
    std::chrono::steady_clock::time_point gen_base{};      // the generated frames of this interval: from ...
    std::chrono::steady_clock::duration gen_span{};        // ... over this span
    double interval{1.0 / 30.0};                           // smoothed seconds between game frames

    std::array<std::optional<SceneCamera>, 2> cameras;  // the last two game frames' cameras (depth path)
    u32 depth_frames{};                                  // game frames that took the depth path (log)
    u32 skipped_frames{};                                // game frames shown without in-between frames (log)
    u32 lost_no_shot{}, lost_no_camera{}, lost_no_depth{}, lost_other{};  // why no depth path (log)
    std::chrono::steady_clock::time_point previous_arrival{};
    std::array<size_t, 4> timing_generations{};          // frames generated per game frame (GPU timing)
};

} // namespace Vulkan
