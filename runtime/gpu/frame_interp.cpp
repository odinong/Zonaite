// SPDX-License-Identifier: GPL-3.0-or-later
// Frame interpolation: Vulkan::FrameGen (see override_video/.../frame_gen.h) on Zonaite's own shaders.
//
// Per game frame (Process), from the composited frame and, when the game drew a 3D scene, the scene capture
// (runtime/gpu/scene_capture.cpp: depth buffer, camera, the scene without the HUD):
// - the frame's scene image (HUD-free when captured) and its luma pyramid (1/2 .. 1/64 size);
// - depth path (both frames captured): one-way optical flow in both directions, then per pixel of each frame its
//   motion to the other frame, the camera's exact motion competing with the flow (zn_interp_motion.comp);
// - otherwise: bilateral optical flow through the in-between frame (zn_interp_flow.comp, the picture-only path).
// Per generated frame (GenerateInto), at time t:
// - depth path: both frames' pixels warped forward to t, nearest surface first (zn_interp_scatter.comp), composed
//   with occlusion handled by depth: each pixel's surface is picked (zn_interp_choose.comp), stray picks are
//   voted out and the HUD is laid back over the scene (zn_interp_compose.comp);
// - otherwise: each pixel picks the best motion candidate and blends the two frames (zn_interp_generate.comp).
// The game itself still runs and presents at 30 Hz; Eden's pacer decides how many frames to generate in between.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host_shaders/zn_interp_choose_comp_spv.h"
#include "video_core/host_shaders/zn_interp_compose_comp_spv.h"
#include "video_core/host_shaders/zn_interp_convert_comp_spv.h"
#include "video_core/host_shaders/zn_interp_down_comp_spv.h"
#include "video_core/host_shaders/zn_interp_flow_comp_spv.h"
#include "video_core/host_shaders/zn_interp_generate_comp_spv.h"
#include "video_core/host_shaders/zn_interp_luma_comp_spv.h"
#include "video_core/host_shaders/zn_interp_motion_comp_spv.h"
#include "video_core/host_shaders/zn_interp_scatter_comp_spv.h"
#include "video_core/renderer_vulkan/present/frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/zonaite_capture.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "zonaite/config.h"
#include "zonaite/gpu.h"

// Frames generated so far (the runtime's fps log reports the displayed rate from it)
std::atomic<unsigned long long> zn_generated_frames{0};
std::atomic<unsigned long long> zn_presented_frames{0};
// For the performance overlay (runtime/gpu/ui.cpp): GPU time per game frame (analysis) and per generated frame,
// and the share of the last 300 game frames that took the depth path, percent
std::atomic<float> zn_interp_input_ms{0}, zn_interp_gen_ms{0};
std::atomic<unsigned> zn_interp_depth_share{0};

namespace Vulkan {
namespace {

constexpr size_t kLevels = 6;      // luma pyramid: 1/2, 1/4, 1/8, 1/16, 1/32, 1/64 of the frame
constexpr size_t kMaxTargets = 8;  // present manager frames (MAX_FRAMES_IN_FLIGHT)
constexpr s32 kTopRadius = 4;      // full search at 1/64 size: +-256 pixels of a 1080p frame
constexpr f32 kLambda = 0.002f;    // smoothness: cost per texel away from the coarser estimate
constexpr u32 kSamplers = 6;       // depth-path descriptor layout: 0-5 sampled images, 6-12 storage images
constexpr u32 kStorages = 7;

// Push constants of the picture-only passes (zn_interp_common.glsl)
struct Push {
    f32 dst_w, dst_h;
    f32 t;
    s32 top;
    s32 radius;
    f32 lambda;
    s32 mode;
    s32 pad;
    std::array<f32, 4> content{0, 0, 1, 1};
};

// zn_interp_motion.comp
struct MotionPush {
    std::array<f32, 16> reproj;  // rows
    std::array<f32, 4> src_vp;
    std::array<f32, 4> src_z;
    std::array<f32, 4> src_proj;
    std::array<f32, 4> dst_sign;
    f32 out_w, out_h;
    s32 refine;
    f32 pad;
    std::array<f32, 4> content{0, 0, 1, 1};
};

// zn_interp_scatter.comp
struct ScatterPush {
    f32 out_w, out_h;
    f32 step;
    s32 pass;
    f32 sign;
    s32 clean;
    f32 nearer;
};

// zn_interp_compose.comp
struct ComposePush {
    f32 out_w, out_h;
    f32 t;
    s32 pad;
    std::array<f32, 4> content{0, 0, 1, 1};  // the game's picture in the frame
};

u32 Groups(u32 n) {
    return (n + 7) / 8;
}

struct Image {
    vk::Image image;
    vk::ImageView view;
    VkExtent2D extent{};
};

Image MakeImage(const Device& device, MemoryAllocator& allocator, VkExtent2D extent, VkFormat format,
                VkImageUsageFlags usage, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    Image img;
    img.extent = extent;
    img.image = allocator.CreateImage({
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {.width = extent.width, .height = extent.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    img.view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *img.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components{},
        .subresourceRange{.aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0,
                          .layerCount = 1},
    });
    return img;
}

void MemoryBarrier(vk::CommandBuffer cmdbuf, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                   VkAccessFlags src_access, VkAccessFlags dst_access) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
    };
    cmdbuf.PipelineBarrier(src_stage, dst_stage, 0, barrier);
}

void ComputeToCompute(vk::CommandBuffer cmdbuf) {
    MemoryBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

VkImageMemoryBarrier ImageBarrier(VkImage image, VkAccessFlags src_access, VkAccessFlags dst_access,
                                  VkImageLayout old_layout, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
}

VkImageCopy CopyRegion(VkExtent2D extent, VkImageAspectFlags aspect) {
    return VkImageCopy{
        .srcSubresource{.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .srcOffset = {},
        .dstSubresource{.aspectMask = aspect, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .dstOffset = {},
        .extent = {.width = extent.width, .height = extent.height, .depth = 1},
    };
}

using Mat4 = std::array<double, 16>;  // rows

// A camera cut or jump between two game frames: the view turned more than 20 degrees, or the camera moved more
// than 4 units (metres), in one frame; no in-between frame can show that honestly
bool CameraCut(const SceneCamera& a, const SceneCamera& b) {
    double trace = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) trace += static_cast<double>(a.view[i * 4 + j]) * b.view[i * 4 + j];
    const double angle = std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) * 57.29578;
    // camera position: -R^T t
    double pa[3] = {}, pb[3] = {};
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            pa[j] -= a.view[i * 4 + j] * a.view[i * 4 + 3];
            pb[j] -= b.view[i * 4 + j] * b.view[i * 4 + 3];
        }
    const double moved = std::sqrt((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]) +
                                   (pa[2] - pb[2]) * (pa[2] - pb[2]));
    return angle > 20.0 || moved > 4.0;
}

Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}

bool Invert(Mat4 m, Mat4& out) {
    Mat4 inv{};
    for (int i = 0; i < 4; ++i) inv[i * 5] = 1.0;
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
            if (std::abs(m[row * 4 + col]) > std::abs(m[pivot * 4 + col])) pivot = row;
        if (std::abs(m[pivot * 4 + col]) < 1e-12) return false;
        for (int j = 0; j < 4; ++j) {
            std::swap(m[col * 4 + j], m[pivot * 4 + j]);
            std::swap(inv[col * 4 + j], inv[pivot * 4 + j]);
        }
        const double d = m[col * 4 + col];
        for (int j = 0; j < 4; ++j) {
            m[col * 4 + j] /= d;
            inv[col * 4 + j] /= d;
        }
        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            const double f = m[row * 4 + col];
            for (int j = 0; j < 4; ++j) {
                m[row * 4 + j] -= f * m[col * 4 + j];
                inv[row * 4 + j] -= f * inv[col * 4 + j];
            }
        }
    }
    out = inv;
    return true;
}

// The camera's world -> clip transform (projection x view)
Mat4 ViewProjection(const SceneCamera& cam) {
    Mat4 view{}, proj{};
    for (int i = 0; i < 12; ++i) view[i] = cam.view[i];
    view[15] = 1.0;
    for (int i = 0; i < 16; ++i) proj[i] = cam.proj[i];
    return Mul(proj, view);
}

f32 Sign(f32 v) {
    return v < 0 ? -1.0f : 1.0f;
}

// zn_interp_motion's constants for the pixels of frame `src` moving to frame `dst`
bool MakeMotionPush(const SceneCamera& src, const SceneCamera& dst, VkExtent2D out, MotionPush& push) {
    Mat4 inverse;
    if (!Invert(ViewProjection(src), inverse)) {
        return false;
    }
    const Mat4 reproj = Mul(ViewProjection(dst), inverse);
    for (int i = 0; i < 16; ++i) push.reproj[i] = static_cast<f32>(reproj[i]);
    const auto& s = src.vp_scale;
    const auto& t = src.vp_translate;
    // in the depth surface's texels: the viewport is in guest pixels, the surface may be drawn at a scale
    const f32 k = src.texel_scale;
    push.src_vp = {k * (t[0] - std::abs(s[0])), k * (t[1] - std::abs(s[1])), k * 2 * std::abs(s[0]),
                   k * 2 * std::abs(s[1])};
    push.src_z = {Sign(s[0]), Sign(s[1]), s[2] != 0 ? s[2] : 1.0f, t[2]};
    push.src_proj = {src.proj[10], src.proj[11], 1.0f, 0.0f};
    push.dst_sign = {Sign(dst.vp_scale[0]), Sign(dst.vp_scale[1]), 0.0f, 0.0f};
    push.out_w = static_cast<f32>(out.width);
    push.out_h = static_cast<f32>(out.height);
    return true;
}


// What Input needs about one game frame's scene capture
struct InputShot {
    bool depth{};  // the depth path's data: depth, camera
    bool scene{};  // a HUD-free scene image
    VkImage depth_image{}, scene_image{};
    VkExtent2D depth_extent{}, scene_extent{};
    bool pair{};  // both this frame and the previous one take the depth path
    bool clean{};  // clean mode: things that move by themselves come from the nearer frame only
    std::array<f32, 4> content{0, 0, 1, 1};  // the game's picture in the frame (zn::gpu::content_rect)
    MotionPush from_cur{}, from_prev{};
};

} // Anonymous namespace

// Called by the present thread after each present (tools/eden_patches.py): logs the spacing of presented frames
// every 10 s, so frame pacing can be checked from the log.
void ZonaiteNotePresent(std::chrono::steady_clock::time_point present_at) {
    zn_presented_frames.fetch_add(1, std::memory_order_relaxed);
    using Clock = std::chrono::steady_clock;
    static double late_sum = 0, late_worst = 0;
    static u64 late_n = 0, late_count = 0;
    {
        static Clock::time_point previous{};
        const auto now = Clock::now();
        if (previous != Clock::time_point{})
            zn::ui::note_present_interval(std::chrono::duration<float, std::milli>(now - previous).count());
        previous = now;
    }
    if (present_at != Clock::time_point{}) {
        const float late = std::chrono::duration<float, std::milli>(Clock::now() - present_at).count();
        late_sum += late;
        late_worst = std::max(late_worst, double(late));
        late_n++;
        late_count += late > 1.0f;
    }
    static Clock::time_point last{}, window{};
    static u64 n = 0;
    static double sum = 0, sumsq = 0, worst = 0;
    const auto now = Clock::now();
    if (last != Clock::time_point{}) {
        const double ms = std::chrono::duration<double, std::milli>(now - last).count();
        n++;
        sum += ms;
        sumsq += ms * ms;
        worst = std::max(worst, ms);
    } else {
        window = now;
    }
    last = now;
    if (now - window >= std::chrono::seconds(10) && n) {
        const double mean = sum / n;
        LOG_WARNING(Render_Vulkan, "zonaite pacing: {} presents, interval {:.2f} ms, sd {:.2f}, worst {:.2f}; "
                    "after their time: mean {:.2f} ms, worst {:.2f}, {} over 1 ms", n, mean,
                    std::sqrt(std::max(0.0, sumsq / n - mean * mean)), worst, late_n ? late_sum / late_n : 0.0,
                    late_worst, late_count);
        late_sum = late_worst = 0;
        late_n = late_count = 0;
        n = 0;
        sum = sumsq = worst = 0;
        window = now;
    }
}

// The GPU time interpolation takes, from timestamps around each game frame's Input and each generated frame. The
// schedule needs it (the first in-between frame can't be shown before the GPU has made it), and the performance
// overlay shows it.
class InterpTiming {
public:
    static constexpr u32 kSlots = 8;                     // game frames in flight
    static constexpr u32 kStamps = 2 + 2 * kMaxTargets + 4;  // input start/end, start/end per generated frame,
                                                            // then input stages (kStageStamp..)
    static constexpr u32 kStageStamp = 2 + 2 * kMaxTargets;

    explicit InterpTiming(const Device& device) {
        const auto& dld = device.GetDispatchLoader();
        write = reinterpret_cast<PFN_vkCmdWriteTimestamp>(
            dld.vkGetDeviceProcAddr(*device.GetLogical(), "vkCmdWriteTimestamp"));
        period_ns = device.GetPhysical().GetProperties().limits.timestampPeriod;
        if (!write || period_ns <= 0) return;
        pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = kSlots * kStamps,
            .pipelineStatistics = 0,
        });
        logical = &device.GetLogical();
    }

    bool On() const {
        return logical != nullptr;
    }

    void Reset(vk::CommandBuffer cmdbuf, u64 count) {
        if (On()) cmdbuf.ResetQueryPool(*pool, Slot(count) * kStamps, kStamps);
    }

    void Stamp(vk::CommandBuffer cmdbuf, u64 count, u32 index) {
        if (On()) write(*cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool, Slot(count) * kStamps + index);
    }

    // Reads game frame `count`'s timestamps if the GPU has written them all (generated frames: `generations`)
    void Collect(u64 count, size_t generations) {
        if (!On()) return;
        std::array<u64, kStamps> t{};
        const u32 n = 2 + 2 * static_cast<u32>(std::min<size_t>(generations, kMaxTargets));
        if (logical->GetQueryResults(*pool, Slot(count) * kStamps, n, n * sizeof(u64), t.data(), sizeof(u64),
                                     VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
            return;
        }
        std::array<u64, 4> st{};
        const bool stages = logical->GetQueryResults(*pool, Slot(count) * kStamps + kStageStamp, 4, sizeof(st),
                                                     st.data(), sizeof(u64), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        const auto ms = [&](u64 a, u64 b) { return b > a ? (b - a) * period_ns * 1e-6 : 0.0; };
        input_ms += ms(t[0], t[1]);
        zn_interp_input_ms = zn_interp_input_ms + (static_cast<float>(ms(t[0], t[1])) - zn_interp_input_ms) * 0.05f;
        if (n > 2)
            zn_interp_gen_ms = zn_interp_gen_ms + (static_cast<float>(ms(t[2], t[3])) - zn_interp_gen_ms) * 0.05f;
        if (stages) {
            stage_ms[0] += ms(t[0], st[0]);   // copies, scene image, luma pyramid
            stage_ms[1] += ms(st[0], st[1]);  // optical flow
            stage_ms[2] += ms(st[1], st[2]);  // motion
            stage_ms[3] += ms(st[2], st[3]);  // splat
            stage_ms[4] += ms(st[3], t[1]);   // refine
            stage_frames++;
        }
        if (n > 2) {
            const double first = ms(t[0], t[3]);  // arrival work to the first generated frame done
            first_ms = first_ms == 0 ? first : first_ms + (first - first_ms) * 0.05;
        }
        for (u32 g = 0; g + 2 < n; g += 2) {
            gen_ms += ms(t[2 + g], t[3 + g]);
            gens++;
        }
        span_ms += ms(t[0], t[n - 1]);
        frames++;
        if (frames == 300) {
            stage_ms = {};
            stage_frames = 0;
            input_ms = gen_ms = span_ms = 0;
            frames = gens = 0;
        }
    }

    // Smoothed GPU time from a game frame's Input to its first generated frame, ms (0: not measured)
    double FirstFrameMs() const {
        return first_ms;
    }

private:
    static u32 Slot(u64 count) {
        return static_cast<u32>(count % kSlots);
    }

    const vk::Device* logical{};
    PFN_vkCmdWriteTimestamp write{};
    vk::QueryPool pool;
    double period_ns{};
    double input_ms{}, gen_ms{}, span_ms{};
    double first_ms{};
    std::array<double, 5> stage_ms{};
    u32 frames{}, gens{}, stage_frames{};
};

class InterpChain {
public:
    InterpChain(const Device& device_, MemoryAllocator& allocator_, VkExtent2D extent, VkFormat format)
        : device{device_}, allocator{allocator_}, full{extent}, timing{device_} {
        const auto& dld = device.GetLogical();
        depth_supported = device.GetMaxPushConstantsSize() >= sizeof(MotionPush);
        constexpr VkImageUsageFlags kStorage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        for (size_t k = 0; k < 2; ++k) {
            color[k] = MakeImage(device, allocator, extent, format,
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            scene[k] = MakeImage(device, allocator, extent, VK_FORMAT_R8G8B8A8_UNORM, kStorage);
        }
        VkExtent2D size{std::max(1u, (extent.width + 1) / 2), std::max(1u, (extent.height + 1) / 2)};
        for (size_t l = 0; l < kLevels; ++l) {
            for (size_t k = 0; k < 2; ++k) {
                luma[k][l] = MakeImage(device, allocator, size, VK_FORMAT_R32_SFLOAT, kStorage);
                oneway[k][l] = MakeImage(device, allocator, size, VK_FORMAT_R16G16B16A16_SFLOAT, kStorage);
            }
            flow[l] = MakeImage(device, allocator, size, VK_FORMAT_R16G16B16A16_SFLOAT, kStorage);
            size = {std::max(1u, (size.width + 1) / 2), std::max(1u, (size.height + 1) / 2)};
        }

        sampler = dld.CreateSampler({
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_NEVER,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        });

        // picture-only layout: 0-2 sampled, 3 storage
        set_layout = CreateWrappedDescriptorSetLayout(
            device,
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            VK_SHADER_STAGE_COMPUTE_BIT);
        layout = MakeLayout(set_layout, sizeof(Push));
        const auto pipeline = [&](std::span<const u32> code, vk::ShaderModule& module, vk::PipelineLayout& l) {
            module = CreateWrappedShaderModule(device, code);
            return CreateWrappedComputePipeline(device, l, *module);
        };
        luma_pipeline = pipeline(ZN_INTERP_LUMA_COMP_SPV, luma_module, layout);
        down_pipeline = pipeline(ZN_INTERP_DOWN_COMP_SPV, down_module, layout);
        flow_pipeline = pipeline(ZN_INTERP_FLOW_COMP_SPV, flow_module, layout);
        generate_pipeline = pipeline(ZN_INTERP_GENERATE_COMP_SPV, generate_module, layout);

        constexpr size_t sets_a = 2 * kLevels * 3 + 2 * kMaxTargets;
        pool = CreateWrappedDescriptorPool(device, sets_a * 4, sets_a,
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
        std::vector<VkDescriptorSetLayout> layouts(sets_a, *set_layout);
        descriptor_sets = CreateWrappedDescriptorSets(pool, layouts);
        size_t next = 0;
        for (size_t k = 0; k < 2; ++k) {
            for (size_t l = 0; l < kLevels; ++l) {
                luma_sets[k][l] = descriptor_sets[next++];
                const VkImageView src =
                    l == 0 ? (depth_supported ? *scene[k].view : *color[k].view) : *luma[k][l - 1].view;
                Write(luma_sets[k][l], {src, src, src}, *luma[k][l].view);

                flow_sets[k][l] = descriptor_sets[next++];
                const VkImageView coarse = l + 1 < kLevels ? *flow[l + 1].view : *flow[0].view;
                Write(flow_sets[k][l], {*luma[k ^ 1][l].view, *luma[k][l].view, coarse}, *flow[l].view);

                // one-way, from frame k to frame k^1
                oneway_sets[k][l] = descriptor_sets[next++];
                const VkImageView coarse1 = l + 1 < kLevels ? *oneway[k][l + 1].view : *oneway[k][0].view;
                Write(oneway_sets[k][l], {*luma[k][l].view, *luma[k ^ 1][l].view, coarse1}, *oneway[k][l].view);
            }
            for (size_t target = 0; target < kMaxTargets; ++target) {
                gen_sets[k][target] = descriptor_sets[next++];
            }
        }

        // depth path layout, when the device takes its push constants
        if (!depth_supported) {
            LOG_WARNING(Render_Vulkan, "frame interpolation: push constants too small, depth path off");
            return;
        }
        std::array<VkDescriptorType, kSamplers + kStorages> types{};
        for (u32 i = 0; i < types.size(); ++i)
            types[i] = i < kSamplers ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        depth_set_layout = CreateWrappedDescriptorSetLayout(device, types, VK_SHADER_STAGE_COMPUTE_BIT);
        depth_layout = MakeLayout(depth_set_layout, sizeof(MotionPush));
        convert_pipeline = pipeline(ZN_INTERP_CONVERT_COMP_SPV, convert_module, depth_layout);
        motion_pipeline = pipeline(ZN_INTERP_MOTION_COMP_SPV, motion_module, depth_layout);
        scatter_pipeline = pipeline(ZN_INTERP_SCATTER_COMP_SPV, scatter_module, depth_layout);
        choose_pipeline = pipeline(ZN_INTERP_CHOOSE_COMP_SPV, choose_module, depth_layout);
        compose_pipeline = pipeline(ZN_INTERP_COMPOSE_COMP_SPV, compose_module, depth_layout);
        choice = MakeImage(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT, kStorage);
        unconfirmed = MakeImage(device, allocator, {1, 1}, VK_FORMAT_R32_UINT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        for (size_t k = 0; k < 2; ++k) {
            motion[k] = MakeImage(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT, kStorage);
            camera[k] = MakeImage(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT, kStorage);
            winner[k] = MakeImage(device, allocator, extent, VK_FORMAT_R32G32B32A32_SFLOAT, kStorage);
            nearest[k] = MakeImage(device, allocator, extent, VK_FORMAT_R32_UINT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        }
        constexpr size_t sets_b = 2 * (2 + 1 + 1 + 1 + 1 + kMaxTargets);
        depth_pool = CreateWrappedDescriptorPool(device, sets_b * (kSamplers + kStorages), sets_b,
                                                 {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
        std::vector<VkDescriptorSetLayout> layouts_b(sets_b, *depth_set_layout);
        depth_sets = CreateWrappedDescriptorSets(depth_pool, layouts_b);
        next = 0;
        for (size_t k = 0; k < 2; ++k) {
            for (auto& set : convert_sets[k]) set = depth_sets[next++];
            motion_sets[k] = depth_sets[next++];
            scatter_sets[k] = depth_sets[next++];
            splat_sets[k] = depth_sets[next++];
            choose_sets[k] = depth_sets[next++];
            for (auto& set : compose_sets[k]) set = depth_sets[next++];
            WriteB(convert_sets[k][1], {{0, *color[k].view}, {6, *scene[k].view}});
            WriteB(scatter_sets[k],
                   {{6, *motion[k].view}, {7, *nearest[k].view}, {8, *winner[k].view}, {9, *camera[k].view}});
            WriteB(choose_sets[k], {{0, *scene[k ^ 1].view},
                                    {1, *scene[k].view},
                                    {6, *nearest[k].view},
                                    {7, *nearest[k ^ 1].view},
                                    {8, *winner[k].view},
                                    {9, *winner[k ^ 1].view},
                                    {11, *choice.view},
                                    {12, *unconfirmed.view}});
            // frame k's motion splatted onto the other frame
            WriteB(splat_sets[k], {{6, *motion[k].view},
                                   {7, *nearest[k ^ 1].view},
                                   {8, *winner[k ^ 1].view},
                                   {9, *camera[k].view}});
        }
    }

    // All images live in the GENERAL layout
    void Initialize(vk::CommandBuffer cmdbuf) {
        std::vector<VkImageMemoryBarrier> barriers;
        const auto add = [&](Image& img) {
            if (!img.image) return;
            barriers.push_back(ImageBarrier(*img.image, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                               VK_ACCESS_TRANSFER_WRITE_BIT,
                                            VK_IMAGE_LAYOUT_UNDEFINED));
        };
        for (size_t k = 0; k < 2; ++k) {
            add(color[k]);
            add(scene[k]);
            add(motion[k]);
            add(camera[k]);
            add(winner[k]);
            add(nearest[k]);
            for (auto& l : luma[k]) add(l);
            for (auto& l : oneway[k]) add(l);
        }
        for (auto& f : flow) add(f);
        add(choice);
        add(unconfirmed);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {},
                               barriers);
    }

    bool DepthSupported() const {
        return depth_supported;
    }

    // The images a scene capture is copied into, sized to the capture. Before replacing them the caller waits
    // for the GPU; Input brings new ones into the GENERAL layout.
    bool NeedsCaptureImages(VkExtent2D depth_extent, VkExtent2D scene_extent) const {
        return depth[0].extent.width != depth_extent.width || depth[0].extent.height != depth_extent.height ||
               (scene_extent.width && (scene_in[0].extent.width != scene_extent.width ||
                                       scene_in[0].extent.height != scene_extent.height));
    }

    void MakeCaptureImages(VkExtent2D depth_extent, VkExtent2D scene_extent) {
        if (!scene_extent.width) scene_extent = scene_in[0].extent.width ? scene_in[0].extent : full;
        for (size_t k = 0; k < 2; ++k) {
            depth[k] = MakeImage(device, allocator, depth_extent, VK_FORMAT_D32_SFLOAT,
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_ASPECT_DEPTH_BIT);
            scene_in[k] = MakeImage(device, allocator, scene_extent, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
        captures_fresh = true;
        for (size_t k = 0; k < 2; ++k) {
            WriteB(convert_sets[k][0], {{0, *scene_in[k].view}, {6, *scene[k].view}});
            WriteB(motion_sets[k], {{0, *scene[k].view},
                                    {1, *scene[k ^ 1].view},
                                    {2, *depth[k].view},
                                    {3, *oneway[k][0].view},
                                    {6, *motion[k].view},
                                    {7, *nearest[k].view},
                                    {8, *winner[k].view},
                                    {9, *camera[k].view}});
        }
    }

    // A new game frame: keep a copy and its scene image, build its luma pyramid, and estimate the motion between
    // it and the previous one
    void Input(vk::CommandBuffer cmdbuf, VkImage source, u64 count, const InputShot& shot) {
        timing.Reset(cmdbuf, count);
        timing.Stamp(cmdbuf, count, 0);
        InputPasses(cmdbuf, source, count, shot);
        timing.Stamp(cmdbuf, count, 1);
    }

    InterpTiming& Timing() {
        return timing;
    }

    void InputPasses(vk::CommandBuffer cmdbuf, VkImage source, u64 count, const InputShot& shot) {
        const size_t k = count % 2;
        if (captures_fresh) {
            captures_fresh = false;
            std::array<VkImageMemoryBarrier, 4> barriers{
                ImageBarrier(*depth[0].image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_ASPECT_DEPTH_BIT),
                ImageBarrier(*depth[1].image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_ASPECT_DEPTH_BIT),
                ImageBarrier(*scene_in[0].image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED),
                ImageBarrier(*scene_in[1].image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {},
                                   barriers);
        }
        MemoryBarrier(cmdbuf,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_GENERAL, *color[k].image, VK_IMAGE_LAYOUT_GENERAL,
                         CopyRegion(full, VK_IMAGE_ASPECT_COLOR_BIT));
        if (shot.depth) {
            cmdbuf.CopyImage(shot.depth_image, VK_IMAGE_LAYOUT_GENERAL, *depth[k].image, VK_IMAGE_LAYOUT_GENERAL,
                             CopyRegion(shot.depth_extent, VK_IMAGE_ASPECT_DEPTH_BIT));
        }
        if (shot.scene) {
            cmdbuf.CopyImage(shot.scene_image, VK_IMAGE_LAYOUT_GENERAL, *scene_in[k].image,
                             VK_IMAGE_LAYOUT_GENERAL, CopyRegion(shot.scene_extent, VK_IMAGE_ASPECT_COLOR_BIT));
        }
        MemoryBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

        // the scene image: the HUD-free capture, or the presented frame itself
        // (without the depth path, the luma pyramid reads the presented frame itself)
        if (depth_supported) {
            const bool captured = shot.scene && scene_in[k].image;
            RunB(cmdbuf, convert_pipeline, convert_sets[k][captured ? 0 : 1], full,
                 Push{.mode = captured ? 0 : 1, .content = shot.content});
            ComputeToCompute(cmdbuf);
        }
        for (size_t l = 0; l < kLevels; ++l) {
            Run(cmdbuf, l == 0 ? luma_pipeline : down_pipeline, luma_sets[k][l], luma[k][l].extent, {});
            ComputeToCompute(cmdbuf);
        }
        pair_depth[k] = false;
        clean[k] = shot.clean;
        content[k] = shot.content;
        timing.Stamp(cmdbuf, count, InterpTiming::kStageStamp);
        if (count == 0) {
            return;
        }
        if (shot.pair) {
            pair_depth[k] = true;
            for (const size_t from : {k, k ^ 1}) {
                for (size_t l = kLevels; l-- > 0;) {
                    const bool top = l + 1 == kLevels;
                    Run(cmdbuf, flow_pipeline, oneway_sets[from][l], oneway[from][l].extent,
                        {.top = top ? 1 : (l == 0 ? 2 : 0), .radius = top ? kTopRadius : 1, .lambda = kLambda,
                         .mode = 1});
                    ComputeToCompute(cmdbuf);
                }
            }
            timing.Stamp(cmdbuf, count, InterpTiming::kStageStamp + 1);
            RunB(cmdbuf, motion_pipeline, motion_sets[k], full, shot.from_cur);
            RunB(cmdbuf, motion_pipeline, motion_sets[k ^ 1], full, shot.from_prev);
            timing.Stamp(cmdbuf, count, InterpTiming::kStageStamp + 2);
            // each frame's motion, reversed, as candidates for the other frame's pixels
            ClearNearest(cmdbuf);
            for (s32 pass = 0; pass < 2; ++pass) {
                for (const size_t from : {k, k ^ 1}) {
                    RunB(cmdbuf, scatter_pipeline, splat_sets[from], full,
                         ScatterPush{.out_w = f32(full.width), .out_h = f32(full.height), .step = 1.0f,
                                     .pass = pass, .sign = -1.0f});
                }
                ComputeToCompute(cmdbuf);
            }
            timing.Stamp(cmdbuf, count, InterpTiming::kStageStamp + 3);
            MotionPush refine = shot.from_cur;
            refine.refine = 1;
            RunB(cmdbuf, motion_pipeline, motion_sets[k], full, refine);
            refine = shot.from_prev;
            refine.refine = 1;
            RunB(cmdbuf, motion_pipeline, motion_sets[k ^ 1], full, refine);
            ComputeToCompute(cmdbuf);
            return;
        }
        for (size_t l = kLevels; l-- > 0;) {
            const bool top = l + 1 == kLevels;
            Run(cmdbuf, flow_pipeline, flow_sets[k][l], flow[l].extent,
                {.t = 0.5f, .top = top ? 1 : (l == 0 ? 2 : 0), .radius = top ? kTopRadius : 1, .lambda = kLambda});
            ComputeToCompute(cmdbuf);
        }
    }

    // Generated frames are written straight into the present manager's frame (its storage view)
    void SetTarget(u32 target, VkImageView view) {
        if (target >= kMaxTargets || target_views[target] == view) {
            return;
        }
        target_views[target] = view;
        for (size_t k = 0; k < 2; ++k) {
            Write(gen_sets[k][target], {*color[k ^ 1].view, *color[k].view, *flow[0].view}, view);
            if (depth_supported) {
                WriteB(compose_sets[k][target], {{0, *scene[k ^ 1].view},
                                                 {1, *scene[k].view},
                                                 {2, *color[k ^ 1].view},
                                                 {3, *color[k].view},
                                                 {4, *motion[k].view},
                                                 {10, view},
                                                 {11, *choice.view},
                                                 {12, *unconfirmed.view}});
            }
        }
    }

    void Generate(vk::CommandBuffer cmdbuf, u64 count, u32 generation, f32 t, u32 target, VkImage image,
                  VkExtent2D extent) {
        if (target >= kMaxTargets || generation >= kMaxTargets) {
            return;
        }
        timing.Stamp(cmdbuf, count, 2 + 2 * generation);
        GeneratePasses(cmdbuf, count, t, target, image, extent);
        timing.Stamp(cmdbuf, count, 3 + 2 * generation);
    }

    void GeneratePasses(vk::CommandBuffer cmdbuf, u64 count, f32 t, u32 target, VkImage image,
                        VkExtent2D extent) {
        const size_t k = count % 2;
        const auto before = ImageBarrier(image, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, before);
        if (pair_depth[k]) {
            // forward warps of both frames to t, then the composition
            ClearNearest(cmdbuf);
            for (s32 pass = 0; pass < 2; ++pass) {
                RunB(cmdbuf, scatter_pipeline, scatter_sets[k], full,
                     ScatterPush{.out_w = f32(full.width), .out_h = f32(full.height), .step = 1.0f - t,
                                 .pass = pass, .sign = 1.0f, .clean = clean[k] ? 1 : 0,
                                 .nearer = t >= 0.5f ? 1.0f : 0.0f});
                RunB(cmdbuf, scatter_pipeline, scatter_sets[k ^ 1], full,
                     ScatterPush{.out_w = f32(full.width), .out_h = f32(full.height), .step = t, .pass = pass,
                                 .sign = 1.0f, .clean = clean[k] ? 1 : 0, .nearer = t < 0.5f ? 1.0f : 0.0f});
                ComputeToCompute(cmdbuf);
            }
            const ComposePush compose{.out_w = f32(extent.width), .out_h = f32(extent.height), .t = t,
                                      .content = content[k]};
            RunB(cmdbuf, choose_pipeline, choose_sets[k], extent, compose);
            ComputeToCompute(cmdbuf);
            RunB(cmdbuf, compose_pipeline, compose_sets[k][target], extent, compose);
        } else {
            Run(cmdbuf, generate_pipeline, gen_sets[k][target], extent, {.t = t});
        }
        const auto after = ImageBarrier(image, VK_ACCESS_SHADER_WRITE_BIT,
                                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                                        VK_IMAGE_LAYOUT_GENERAL);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               after);
    }

private:
    // the forward warps' depth images back to "nothing landed here"
    void ClearNearest(vk::CommandBuffer cmdbuf) {
        MemoryBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue none{.uint32 = {0xffffffffu, 0, 0, 0}};
        const VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0,
                                            .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
        for (size_t i = 0; i < 2; ++i)
            cmdbuf.ClearColorImage(*nearest[i].image, VK_IMAGE_LAYOUT_GENERAL, none, range);
        if (unconfirmed.image) {
            const VkClearColorValue zero{.uint32 = {0, 0, 0, 0}};
            cmdbuf.ClearColorImage(*unconfirmed.image, VK_IMAGE_LAYOUT_GENERAL, zero, range);
        }
        MemoryBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    vk::PipelineLayout MakeLayout(vk::DescriptorSetLayout& set, u32 push_size) {
        const VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size};
        const VkDescriptorSetLayout raw = *set;
        return device.GetLogical().CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &raw,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
    }

    // picture-only layout: three sampled images and the storage image
    void Write(VkDescriptorSet set, std::array<VkImageView, 3> sampled, VkImageView storage) {
        WriteSet(set, {{0, sampled[0]}, {1, sampled[1]}, {2, sampled[2]}, {3, storage}}, 3);
    }

    // depth-path layout: bindings below kSamplers are sampled, the others storage
    void WriteB(VkDescriptorSet set, std::initializer_list<std::pair<u32, VkImageView>> bindings) {
        WriteSet(set, bindings, kSamplers);
    }

    void WriteSet(VkDescriptorSet set, std::initializer_list<std::pair<u32, VkImageView>> bindings,
                  u32 first_storage) {
        std::vector<VkDescriptorImageInfo> infos;
        infos.reserve(bindings.size());
        std::vector<VkWriteDescriptorSet> writes;
        for (const auto& [binding, view] : bindings) {
            const bool storage = binding >= first_storage;
            infos.push_back({storage ? VK_NULL_HANDLE : *sampler, view, VK_IMAGE_LAYOUT_GENERAL});
            writes.push_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                          : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &infos.back(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            });
        }
        device.GetLogical().UpdateDescriptorSets(writes, {});
    }

    void Run(vk::CommandBuffer cmdbuf, const vk::Pipeline& pipeline, VkDescriptorSet set, VkExtent2D extent,
             Push push) {
        push.dst_w = static_cast<f32>(extent.width);
        push.dst_h = static_cast<f32>(extent.height);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
        cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, push);
        cmdbuf.Dispatch(Groups(extent.width), Groups(extent.height), 1);
    }

    template <typename P>
    void RunB(vk::CommandBuffer cmdbuf, const vk::Pipeline& pipeline, VkDescriptorSet set, VkExtent2D extent,
              P push) {
        if constexpr (std::is_same_v<P, Push>) {
            push.dst_w = static_cast<f32>(extent.width);
            push.dst_h = static_cast<f32>(extent.height);
        }
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *depth_layout, 0, set, {});
        cmdbuf.PushConstants(*depth_layout, VK_SHADER_STAGE_COMPUTE_BIT, push);
        cmdbuf.Dispatch(Groups(extent.width), Groups(extent.height), 1);
    }

    const Device& device;
    MemoryAllocator& allocator;
    VkExtent2D full;
    std::array<Image, 2> color;     // the presented frames
    std::array<Image, 2> scene;     // their scene images (display-encoded, like the presented frames)
    std::array<Image, 2> scene_in;  // the HUD-free captures, as captured
    std::array<Image, 2> depth;     // the scene depth buffers
    std::array<std::array<Image, kLevels>, 2> luma;
    std::array<Image, kLevels> flow;                     // bilateral, through the in-between frame
    std::array<std::array<Image, kLevels>, 2> oneway;    // one-way, from frame k to the other one
    std::array<Image, 2> motion;   // per pixel of frame k: motion to the other frame, view distance, cost
    std::array<Image, 2> camera;   // per pixel of frame k: the camera's motion, and whether it moves by itself
    std::array<Image, 2> nearest;  // forward warp of frame k: nearest view distance per pixel
    std::array<Image, 2> winner;   // ... and the motion of the surface that won
    Image choice;                  // per pixel of the generated frame: the surface it shows (zn_interp_choose)
    Image unconfirmed;             // 1x1: how many of its pixels the two frames don't confirm
    std::array<bool, 2> pair_depth{};
    std::array<bool, 2> clean{};
    std::array<std::array<f32, 4>, 2> content{};
    bool depth_supported{};
    bool captures_fresh{};
    vk::Sampler sampler;
    vk::DescriptorSetLayout set_layout, depth_set_layout;
    vk::PipelineLayout layout, depth_layout;
    vk::ShaderModule luma_module, down_module, flow_module, generate_module;
    vk::ShaderModule convert_module, motion_module, scatter_module, choose_module, compose_module;
    vk::Pipeline luma_pipeline, down_pipeline, flow_pipeline, generate_pipeline;
    vk::Pipeline convert_pipeline, motion_pipeline, scatter_pipeline, choose_pipeline, compose_pipeline;
    vk::DescriptorPool pool, depth_pool;
    vk::DescriptorSets descriptor_sets, depth_sets;
    std::array<std::array<VkDescriptorSet, kLevels>, 2> luma_sets{};
    std::array<std::array<VkDescriptorSet, kLevels>, 2> flow_sets{};
    std::array<std::array<VkDescriptorSet, kLevels>, 2> oneway_sets{};
    std::array<std::array<VkDescriptorSet, kMaxTargets>, 2> gen_sets{};
    std::array<std::array<VkDescriptorSet, 2>, 2> convert_sets{};
    std::array<VkDescriptorSet, 2> motion_sets{};
    std::array<VkDescriptorSet, 2> scatter_sets{};
    std::array<VkDescriptorSet, 2> splat_sets{};
    std::array<VkDescriptorSet, 2> choose_sets{};
    std::array<std::array<VkDescriptorSet, kMaxTargets>, 2> compose_sets{};
    std::array<VkImageView, kMaxTargets> target_views{};
    InterpTiming timing;
};

FrameGen::FrameGen(MemoryAllocator& memory_allocator_, Scheduler& scheduler_)
    : memory_allocator{memory_allocator_}, scheduler{scheduler_} {}

FrameGen::~FrameGen() = default;

void FrameGen::Process(const Device& device, Frame* frame, VkFormat format, VkExtent2D) {
    // the settings menu: depth-aware interpolation and pacing (ZN_INTERP_NO_DEPTH turns depth off)
    static zn::cfg::Config config = zn::cfg::get_copy();
    static u64 config_seen = zn::cfg::generation();
    if (const u64 g = zn::cfg::generation(); g != config_seen) {
        config_seen = g;
        config = zn::cfg::get_copy();
    }
    static const bool depth_allowed = !std::getenv("ZN_INTERP_NO_DEPTH");
    const int mode = std::clamp(config.interp_mode, 0, 2);
    const bool depth_wanted = depth_allowed && mode != 2;
    static bool attached = false;
    if (!attached && depth_allowed) {
        attached = true;
        SceneCaptureAttach(device, memory_allocator);
    }
    const SceneShot* shot = SceneCaptureForComposite();
    generated = false;
    frame->present_at = {};
    if (!Settings::values.frame_gen.GetValue() || !frame->storage_view) {
        if (chain) {
            scheduler.Finish();
            chain.reset();
        }
        warm_streak = 0;
        return;
    }
    const VkExtent2D extent{.width = frame->width, .height = frame->height};
    if (!chain || built_extent.width != extent.width || built_extent.height != extent.height ||
        built_format != format) {
        scheduler.Finish();
        chain = std::make_unique<InterpChain>(device, memory_allocator, extent, format);
        built_extent = extent;
        built_format = format;
        frame_count = 0;
        warm_streak = 0;
        cameras = {};
        // the log's counts are per 300 frames of this chain
        depth_frames = skipped_frames = 0;
        lost_no_shot = lost_no_camera = lost_no_depth = lost_other = 0;
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([c = chain.get()](vk::CommandBuffer cmdbuf) { c->Initialize(cmdbuf); });
    }
    const u64 count = frame_count++;
    if (count >= 4) chain->Timing().Collect(count - 4, timing_generations[(count - 4) % 4]);
    last_count = count;
    last_generations = plan.generations;
    // generate only from two consecutive frames that both went through Input (after a pause, a resize, ...)
    warm_streak = plan.warm ? warm_streak + 1 : 0;

    // this frame's scene capture, and whether it pairs with the previous frame's for the depth path
    InputShot in{};
    const size_t k = count % 2;
    cameras[k].reset();
    if (depth_wanted && shot && shot->has_depth && shot->has_camera && chain->DepthSupported()) {
        const VkExtent2D scene_extent = shot->has_scene ? shot->scene_extent : VkExtent2D{};
        if (chain->NeedsCaptureImages(shot->depth_extent, scene_extent)) {
            scheduler.Finish();
            chain->MakeCaptureImages(shot->depth_extent, scene_extent);
            cameras = {};
        }
        in.depth = true;
        in.depth_image = shot->depth;
        in.depth_extent = shot->depth_extent;
        in.scene = shot->has_scene && !chain->NeedsCaptureImages(shot->depth_extent, shot->scene_extent);
        in.scene_image = shot->scene;
        in.scene_extent = shot->scene_extent;
        cameras[k] = shot->camera;
    }
    if (count >= 1 && cameras[k] && cameras[k ^ 1]) {
        in.pair = MakeMotionPush(*cameras[k], *cameras[k ^ 1], extent, in.from_cur) &&
                  MakeMotionPush(*cameras[k ^ 1], *cameras[k], extent, in.from_prev);
    }
    depth_frames += in.pair;
    // why a frame had no depth path (log): no capture for this composite, no camera, or depth off
    if (!in.pair) {
        if (!shot) lost_no_shot++;
        else if (!shot->has_camera) lost_no_camera++;
        else if (!shot->has_depth) lost_no_depth++;
        else lost_other++;
    }

    // Clean mode (the default): in-between frames only where they can't smear. Not without the depth path, not
    // after a hitch (the frames are further apart than usual, so motion is larger and estimates worse), and not
    // across a camera cut or jump. The real frame then simply stays up until the next one, as without frame
    // generation.
    const auto arrival = std::chrono::steady_clock::now();
    const double gap = previous_arrival != std::chrono::steady_clock::time_point{}
                           ? std::chrono::duration<double>(arrival - previous_arrival).count()
                           : 0.0;
    previous_arrival = arrival;
    const bool hitch = gap > std::max(interval * 1.4, 1.0 / 30.0 * 1.25);
    const bool cut = in.pair && CameraCut(*cameras[k ^ 1], *cameras[k]);
    const bool clean = in.pair && !hitch && !cut;
    in.clean = mode == 0;
    in.content = zn::gpu::content_rect();
    in.from_cur.content = in.content;
    in.from_prev.content = in.content;
    generated = count >= 1 && warm_streak >= 2 && plan.generations > 0 && (mode != 0 || clean);
    skipped_frames += count >= 1 && warm_streak >= 2 && plan.generations > 0 && !generated;
    timing_generations[count % 4] = generated ? plan.generations : 0;
    if (config.pacing) Schedule(frame);
    if (count % 300 == 299) {
        LOG_WARNING(Render_Vulkan,
                    "frame interpolation: depth path for {} of the last 300 game frames, {} without in-between frames "
                    "(hitch, camera cut or no depth)",
                    depth_frames, skipped_frames);
        skipped_frames = 0;
        if (lost_no_shot + lost_no_camera + lost_no_depth + lost_other)
            LOG_WARNING(Render_Vulkan, "  no depth path: {} without a capture, {} without a camera, {} without depth, "
                        "{} other", lost_no_shot, lost_no_camera, lost_no_depth, lost_other);
        lost_no_shot = lost_no_camera = lost_no_depth = lost_other = 0;
        zn_interp_depth_share = depth_frames * 100 / 300;
        depth_frames = 0;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([c = chain.get(), source = *frame->image, count, in](vk::CommandBuffer cmdbuf) {
        c->Input(cmdbuf, source, count, in);
    });
}

// Show game frame k at P_k = (arrival + (n-1)/n of an interval + GPU margin) and its n-1 generated predecessors
// evenly between P_k-1 and P_k: the earliest even schedule, since generating needs frame k itself. The margin
// follows the GPU time interpolation takes (InterpTiming).
void FrameGen::Schedule(Frame* frame) {
    using Clock = std::chrono::steady_clock;
    // the GPU time to the first generated frame, plus 2 ms for the GPU's own queue (3 ms when not measured)
    const double gpu_ms = chain->Timing().FirstFrameMs();
    const double margin_ms = gpu_ms > 0 ? std::clamp(gpu_ms + 2.0, 3.0, 25.0) : 3.0;
    const auto kMargin = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(margin_ms));
    const auto now = Clock::now();
    if (last_arrival != Clock::time_point{}) {
        const double d = std::chrono::duration<double>(now - last_arrival).count();
        if (d > 0.004 && d < 0.2) {
            interval += (d - interval) * 0.1;
        }
    }
    last_arrival = now;
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(interval));
    if (!generated) {
        last_target = now;  // shown at once
        return;
    }
    const s64 n = static_cast<s64>(last_generations) + 1;
    Clock::time_point target = now + period * (n - 1) / n + kMargin;
    if (last_target == Clock::time_point{} || target - last_target > period * 2) {
        last_target = target - period;  // after a pause or hitch: start a fresh schedule
    }
    target = std::max(target, last_target + period / 2);
    gen_base = last_target;
    gen_span = target - last_target;
    last_target = target;
    frame->present_at = target;
}

size_t FrameGen::WantedGenerations(size_t capacity) {
    plan = pacer.Plan(capacity);
    return plan.generations;
}

size_t FrameGen::GeneratedFrameCount() const {
    return generated ? last_generations : 0;
}

void FrameGen::GenerateInto(const Device& device, Frame* destination, size_t generation) {
    chain->SetTarget(destination->index, *destination->storage_view);
    zn_generated_frames.fetch_add(1, std::memory_order_relaxed);
    if (gen_span.count()) destination->present_at = gen_base + gen_span * static_cast<s64>(generation + 1) /
                                             static_cast<s64>(last_generations + 1);
    const f32 t = static_cast<f32>(generation + 1) / static_cast<f32>(last_generations + 1);
    const VkExtent2D extent{.width = destination->width, .height = destination->height};
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([c = chain.get(), count = last_count, generation = static_cast<u32>(generation), t,
                      target = destination->index, image = *destination->image, extent](vk::CommandBuffer cmdbuf) {
        c->Generate(cmdbuf, count, generation, t, target, image, extent);
    });
}

} // namespace Vulkan
