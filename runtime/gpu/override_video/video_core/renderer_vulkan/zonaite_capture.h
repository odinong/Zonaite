// SPDX-License-Identifier: GPL-3.0-or-later
// Scene capture for frame interpolation (runtime/gpu/scene_capture.cpp). Eden's rasterizer calls
// ZonaiteSceneDraw for every draw (tools/eden_patches.py); it follows the game's frame (scene passes, then the
// full-size composite) and copies, in the GPU stream, what interpolation needs from the game's own frame:
// the scene's depth buffer, the camera (view and projection) the scene was drawn with, and the composite before
// the HUD is drawn over it. The renderer's composite then picks the shot of the frame it presents.
#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Tegra {
class MemoryManager;
namespace Engines {
class Maxwell3D;
}
} // namespace Tegra

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

// One attachment of the current draw, resolved by the rasterizer (image, format, size, subresource)
struct ZonaiteTarget {
    VkImage image{};
    VideoCore::Surface::PixelFormat format{};
    u32 width{}, height{};  // the image's real size (drawn at a resolution scale: larger than the guest's)
    s32 level{}, layer{};
    f32 scale{1.0f};        // the resolution scale it is drawn at
};

struct ZonaiteDrawTargets {
    ZonaiteTarget depth;
    ZonaiteTarget color;  // the first bound colour target
};

// What the rasterizer does with the draw: kSceneRecorded when commands were recorded outside the render pass (the
// caller then requests its render pass again), kSceneSkipDraw when the draw is left out (the HUD, hidden with F3)
constexpr int kSceneRecorded = 1;
constexpr int kSceneSkipDraw = 2;
int ZonaiteSceneDraw(const Tegra::Engines::Maxwell3D& maxwell3d, const Tegra::MemoryManager& gpu_memory,
                     Scheduler& scheduler, const ZonaiteDrawTargets& targets);

// The renderer's composite: the device address of the framebuffer being presented
void ZonaiteNoteComposite(u64 address);

// The camera a scene was drawn with: view (3x4, rows), projection (4x4, rows) and the viewport transform
struct SceneCamera {
    std::array<f32, 12> view{};
    std::array<f32, 16> proj{};
    std::array<f32, 3> vp_scale{}, vp_translate{};
    f32 texel_scale{1.0f};  // depth texels per viewport unit (the resolution scale the scene was drawn at)
};

struct SceneShot {
    u64 seq{};
    bool has_camera{}, has_depth{}, has_scene{};
    SceneCamera camera;
    VkImage depth{};  // D32_SFLOAT, depth_extent (the scene's depth surface)
    VkExtent2D depth_extent{};
    VkImage scene{};  // B10G11R11_UFLOAT, scene_extent: the composite before the HUD, linear colour
    VkExtent2D scene_extent{};
    u64 final_address{};  // device address of the last target drawn in the frame (the framebuffer presented)
};

// Frame interpolation hands over the device and allocator the capture images are made with
void SceneCaptureAttach(const Device& device, MemoryAllocator& allocator);

// The shot for the framebuffer of the composite in progress (ZonaiteNoteComposite), or null
const SceneShot* SceneCaptureForComposite();

} // namespace Vulkan
