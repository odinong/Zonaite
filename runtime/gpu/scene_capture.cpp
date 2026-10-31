// SPDX-License-Identifier: GPL-3.0-or-later
// Scene capture for frame interpolation (see override_video/video_core/renderer_vulkan/zonaite_capture.h).
//
// TotK's frame, as its render passes show it: the scene is drawn
// at up to 1600x900 into a G-buffer (D32_FLOAT depth plus 4-5 colour targets), lit and post-processed at that
// size, then one draw scales it into a full-size (1920x1080) B10G11R11 target, the HUD is drawn over it there,
// and a last pass writes that to the sRGB framebuffer the game presents. The vertex stage's constant buffer 8
// holds the scene camera during the G-buffer draws: the view matrix (3 rows) at +0x0 and the projection at +0x40.
//
// Per frame, in the GPU stream: at the first G-buffer draw, read the camera; just before the first draw into the
// full-size target (the scale-up), copy the scene depth; just before the second draw there (the first HUD draw),
// copy the target, which then holds the scene without the HUD. Draws after that end in the framebuffer presented,
// which ties the shot to its composite. Shots live in a ring, so a frame's shot survives until its composite even
// when the game is already drawing the next frame.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>

#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/zonaite_capture.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "zonaite/gpu.h"

namespace Vulkan {
namespace {

using VideoCore::Surface::PixelFormat;

constexpr size_t kRing = 4;
constexpr u32 kCameraBuffer = 8;  // vertex stage constant buffer with the scene camera (found by the census)

struct Slot {
    SceneShot shot;
    vk::Image depth, scene;
    bool depth_ready{}, scene_ready{};  // transitioned to GENERAL
    bool consumed{};
};

enum class Phase { Idle, Scene, ScaledUp, Tail };

struct Capture {
    std::mutex mutex;
    const Device* device{};
    MemoryAllocator* allocator{};
    std::array<Slot, kRing> ring;
    size_t current = kRing;
    u64 seq{};
    Phase phase{Phase::Idle};
    GPUVAddr scene_depth_address{};
    u32 scene_width{};
    ZonaiteTarget scene_depth;   // the scene depth attachment, as last seen bound
    GPUVAddr full_address{};     // the full-size target the scene is scaled into
    u32 full_width{}, full_height{};
    u64 composite_address{};
    bool logged{};
};

Capture& Get() {
    static Capture c;
    return c;
}

vk::Image MakeImage(MemoryAllocator& allocator, u32 width, u32 height, VkFormat format) {
    return allocator.CreateImage({
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {.width = width, .height = height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
}

bool ReadCamera(const Tegra::Engines::Maxwell3D& m, const Tegra::MemoryManager& mm, SceneCamera& cam) {
    const auto& cb = m.state.shader_stages[0].const_buffers[kCameraBuffer];
    if (!cb.enabled || cb.size < 0x80) {
        return false;
    }
    std::array<f32, 32> f{};
    mm.ReadBlockUnsafe(cb.address, f.data(), sizeof(f));
    std::copy_n(f.begin(), 12, cam.view.begin());
    std::copy_n(f.begin() + 16, 16, cam.proj.begin());
    // the view's rotation must be orthonormal and the projection a perspective one with w = -z (the shaders read
    // view distances from w)
    const auto dot3 = [&](int a, int b) {
        return cam.view[a * 4] * cam.view[b * 4] + cam.view[a * 4 + 1] * cam.view[b * 4 + 1] +
               cam.view[a * 4 + 2] * cam.view[b * 4 + 2];
    };
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (std::abs(dot3(a, b) - (a == b ? 1.0f : 0.0f)) > 2e-3f) return false;
    const auto& p = cam.proj;
    if (p[12] != 0 || p[13] != 0 || p[14] != -1 || p[15] != 0 || p[0] == 0 || p[5] == 0 || p[11] == 0) {
        return false;
    }
    const auto& vp = m.regs.viewport_transform[0];
    cam.vp_scale = {vp.scale_x, vp.scale_y, vp.scale_z};
    cam.vp_translate = {vp.translate_x, vp.translate_y, vp.translate_z};
    return true;
}

// Records a copy of one attachment into a slot image (created or resized as needed)
void CopyTarget(Capture& c, Scheduler& scheduler, const ZonaiteTarget& src, vk::Image& dst, bool& ready,
                VkExtent2D& extent, VkFormat format, VkImageAspectFlags aspect) {
    if (extent.width != src.width || extent.height != src.height || !dst) {
        if (dst) {
            scheduler.Finish();  // a slot image in use by earlier work (only when the game changes resolution)
        }
        dst = MakeImage(*c.allocator, src.width, src.height, format);
        extent = {src.width, src.height};
        ready = false;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src, image = *dst, was_ready = ready, aspect](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        const VkImageMemoryBarrier layout{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = was_ready ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{.aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0,
                              .layerCount = 1},
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, before, {},
                               layout);
        const VkImageCopy region{
            .srcSubresource{.aspectMask = aspect, .mipLevel = static_cast<u32>(src.level),
                            .baseArrayLayer = static_cast<u32>(src.layer), .layerCount = 1},
            .srcOffset = {},
            .dstSubresource{.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .dstOffset = {},
            .extent = {.width = src.width, .height = src.height, .depth = 1},
        };
        cmdbuf.CopyImage(src.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, region);
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, after);
    });
    ready = true;
}

u32 ColorTargets(const Tegra::Engines::Maxwell3D::Regs& r) {
    u32 n = 0;
    for (u32 i = 0; i < r.rt_control.count; ++i)
        n += r.rt[r.rt_control.Map(i)].format != Tegra::RenderTargetFormat::NONE;
    return n;
}

} // Anonymous namespace

void SceneCaptureAttach(const Device& device, MemoryAllocator& allocator) {
    auto& c = Get();
    std::scoped_lock lock{c.mutex};
    c.device = &device;
    c.allocator = &allocator;
}

int ZonaiteSceneDraw(const Tegra::Engines::Maxwell3D& m, const Tegra::MemoryManager& mm, Scheduler& scheduler,
                     const ZonaiteDrawTargets& t) {
    auto& c = Get();
    std::scoped_lock lock{c.mutex};
    if (!c.allocator) {
        return 0;
    }
    const auto& r = m.regs;
    const bool scene_draw = r.zeta_enable && r.zeta.format == Tegra::DepthFormat::Z32_FLOAT &&
                            r.zeta_size.width >= 640 && ColorTargets(r) >= 3 && t.depth.image &&
                            t.depth.format == PixelFormat::D32_FLOAT;
    if (scene_draw) {
        if (c.phase != Phase::Scene) {
            // a new frame's scene: start its shot
            c.current = (c.current + 1) % kRing;
            Slot& s = c.ring[c.current];
            s.shot.seq = ++c.seq;
            s.shot.has_camera = ReadCamera(m, mm, s.shot.camera);
            s.shot.has_depth = s.shot.has_scene = false;
            s.shot.final_address = 0;
            s.consumed = false;
            c.phase = Phase::Scene;
            c.scene_depth_address = r.zeta.Address();
            c.scene_width = r.zeta_size.width;
        }
        if (r.zeta.Address() == c.scene_depth_address) {
            c.scene_depth = t.depth;
        }
        return 0;
    }
    if (c.current >= kRing) {
        return 0;
    }
    Slot& s = c.ring[c.current];
    // F3 hides the HUD: the game draws it into the full-size target over the finished scene (the draws after the
    // first one there), before the frame's last pass copies that into the framebuffer
    if (zn::gpu::hud_hidden() && c.phase == Phase::Tail && t.color.image &&
        r.rt[r.rt_control.Map(0)].Address() == c.full_address) {
        return kSceneSkipDraw;
    }
    switch (c.phase) {
    case Phase::Idle:
    case Phase::Tail:
        break;
    case Phase::Scene:
        if (r.zeta_enable && r.zeta.Address() == c.scene_depth_address && t.depth.image) {
            c.scene_depth = t.depth;  // later scene passes (transparency, effects) on the same depth
            return 0;
        }
        // (widths in guest pixels: at a resolution scale, images are larger than the guest's)
        if (!r.zeta_enable && t.color.image && t.color.width / t.color.scale > c.scene_width + 0.5f &&
            c.scene_depth.image) {
            // the scale-up into the full-size target: the scene depth is final
            CopyTarget(c, scheduler, c.scene_depth, s.depth, s.depth_ready, s.shot.depth_extent,
                       VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
            s.shot.depth = *s.depth;
            s.shot.has_depth = true;
            s.shot.camera.texel_scale = c.scene_depth.scale;
            c.full_address = r.rt[r.rt_control.Map(0)].Address();
            c.full_width = t.color.width;
            c.full_height = t.color.height;
            c.phase = Phase::ScaledUp;
            return kSceneRecorded;
        }
        return 0;
    case Phase::ScaledUp: {
        c.phase = Phase::Tail;
        if (t.color.image && r.rt[r.rt_control.Map(0)].Address() == c.full_address &&
            t.color.format == PixelFormat::B10G11R11_FLOAT) {
            // the first draw over the scaled-up scene: keep the scene without the HUD
            CopyTarget(c, scheduler, t.color, s.scene, s.scene_ready, s.shot.scene_extent,
                       VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
            s.shot.scene = *s.scene;
            s.shot.has_scene = true;
            if (const auto a = mm.GpuToCpuAddress(r.rt[r.rt_control.Map(0)].Address())) {
                s.shot.final_address = *a;
            }
            return kSceneRecorded;
        }
        break;
    }
    }
    // the frame ends in a full-size target (the framebuffer); the next frame's first passes are smaller
    if (c.phase == Phase::Tail && t.color.image && t.color.width == c.full_width &&
        t.color.height == c.full_height) {
        if (const auto a = mm.GpuToCpuAddress(r.rt[r.rt_control.Map(0)].Address())) {
            s.shot.final_address = *a;
        }
    }
    return 0;
}

void ZonaiteNoteComposite(u64 address) {
    auto& c = Get();
    std::scoped_lock lock{c.mutex};
    c.composite_address = address;
}

const SceneShot* SceneCaptureForComposite() {
    auto& c = Get();
    std::scoped_lock lock{c.mutex};
    Slot* best = nullptr;
    for (auto& s : c.ring) {
        if (!s.consumed && s.shot.has_depth && s.shot.final_address == c.composite_address &&
            (!best || s.shot.seq > best->shot.seq)) {
            best = &s;
        }
    }
    if (!c.logged && c.seq > 300) {
        c.logged = true;
        LOG_WARNING(Render_Vulkan, "scene capture: composite {:x}, shot {} (camera {}, scene {})",
                    c.composite_address, best ? best->shot.seq : 0, best && best->shot.has_camera,
                    best && best->shot.has_scene);
    }
    if (!best) {
        return nullptr;
    }
    best->consumed = true;
    return &best->shot;
}

} // namespace Vulkan

namespace zn::gpu {

namespace {
std::atomic<bool> g_hud_hidden{false};
}

bool hud_hidden() {
    return g_hud_hidden.load(std::memory_order_relaxed);
}

void set_hud_hidden(bool hidden) {
    g_hud_hidden = hidden;
}

} // namespace zn::gpu
