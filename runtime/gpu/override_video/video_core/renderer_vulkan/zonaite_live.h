// SPDX-License-Identifier: GPL-3.0-or-later
// Settings that change while playing (runtime/gpu/config.cpp), and the picture adjustments pass
// (runtime/gpu/picture.cpp). Eden reads some renderer settings only when it starts (the resolution scale,
// texture filtering and decoding, memory limits); a change to one of those waits here until the GPU thread is
// between two frames, where RasterizerVulkan::ZonaiteApplyLive remakes what depends on it (tools/eden_patches.py).
#pragma once

#include <functional>

#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;
struct Frame;

struct ZonaiteLiveChange {
    bool rescale{};   // the resolution scale: images drawn at the old scale go back to native size
    bool textures{};  // texture decoding: ASTC textures are made again from guest memory
    bool samplers{};  // texture filtering: every sampler is made again
    bool memory{};    // memory limits (video memory use, or the resolution scale)
    bool shaders{};   // the shader cache is cleared: every pipeline is built again as it is next used
    std::function<void()> apply;  // writes the new values into Eden's settings (between the steps above)
};

// The change waiting for the GPU thread, if any (and then no longer waiting)
bool ZonaiteTakeLiveChange(ZonaiteLiveChange& change);

// Whether any picture adjustment is set (frames then need a storage view for the pass)
bool ZonaitePictureActive();

// Records the picture adjustments for a frame about to be shown (nothing when none is set)
// (format: the frame image's)
void ZonaitePicture(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler, Frame* frame,
                    VkFormat format);

} // namespace Vulkan
