// SPDX-License-Identifier: GPL-3.0-or-later
// Picture adjustments (the settings menu's Picture section): sharpening, gamma, contrast, brightness and
// saturation on every frame shown, the game's and the generated ones, as the last step before presentation (the
// menu drawn over the frame afterwards keeps its own colours). The frame is copied, and zn_picture.comp writes
// the adjusted picture back into it. RendererVulkan::Composite calls ZonaitePicture (tools/eden_patches.py).
#include <algorithm>
#include <array>
#include <vector>

#include "video_core/host_shaders/zn_picture_comp_spv.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/zonaite_live.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "zonaite/config.h"

namespace Vulkan {
namespace {

constexpr u32 kFrames = 8;  // present manager frames, by index (MAX_FRAMES_IN_FLIGHT)

struct PicturePush {
    std::array<f32, 4> tone{0.0f, 1.0f, 1.0f, 1.0f};  // brightness, contrast, saturation, gamma
    std::array<f32, 4> frame{};                        // sharpening, width, height, unused
};

// The menu's values, re-read when the settings change
struct Wanted {
    u64 generation = 0;
    bool active = false;
    PicturePush push;
};

const Wanted& Current() {
    static Wanted w;
    const u64 g = zn::cfg::generation();
    if (g != w.generation) {
        const zn::cfg::Config c = zn::cfg::get_copy();
        w.generation = g;
        w.active = !zn::cfg::picture_neutral(c);
        w.push.tone = {c.brightness, c.contrast, c.saturation, std::max(c.gamma, 0.1f)};
        w.push.frame[0] = c.sharpen;
    }
    return w;
}

// The same bits as a format the copy can be sampled with as display-encoded values
VkFormat Unorm(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default:
        return f;
    }
}

class Pass {
public:
    explicit Pass(const Device& device_) : device{device_} {
        const auto& dld = device.GetLogical();
        set_layout = CreateWrappedDescriptorSetLayout(
            device, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
            VK_SHADER_STAGE_COMPUTE_BIT);
        const VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0,
                                        .size = sizeof(PicturePush)};
        const VkDescriptorSetLayout raw = *set_layout;
        layout = dld.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &raw,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
        module = CreateWrappedShaderModule(device, ZN_PICTURE_COMP_SPV);
        pipeline = CreateWrappedComputePipeline(device, layout, *module);
        sampler = dld.CreateSampler({
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
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
        pool = CreateWrappedDescriptorPool(device, kFrames * 2, kFrames,
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
        std::vector<VkDescriptorSetLayout> layouts(kFrames, raw);
        sets = CreateWrappedDescriptorSets(pool, layouts);
    }

    void Record(MemoryAllocator& allocator, Scheduler& scheduler, Frame* frame, VkFormat frame_format,
                const PicturePush& wanted) {
        if (frame->index >= kFrames || !frame->storage_view) {
            return;
        }
        const VkExtent2D extent{.width = frame->width, .height = frame->height};
        const VkFormat format = Unorm(frame_format);
        if (!copy || copy_extent.width != extent.width || copy_extent.height != extent.height ||
            copy_format != format) {
            if (copy) {
                scheduler.Finish();  // the old copy may still be read (only when the window or format changes)
            }
            MakeCopy(allocator, extent, format);
        }
        // a frame's descriptor set changes only with its image, which is remade only once the frame is free
        if (written[frame->index] != *frame->storage_view) {
            Write(sets[frame->index], *frame->storage_view);
            written[frame->index] = *frame->storage_view;
        }
        PicturePush push = wanted;
        push.frame[1] = static_cast<f32>(extent.width);
        push.frame[2] = static_cast<f32>(extent.height);
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([this, image = *frame->image, copy_image = *copy, set = sets[frame->index], extent,
                          fresh = !copy_used, push](vk::CommandBuffer cmdbuf) {
            const VkImageSubresourceRange range{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0,
                                                .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1};
            const auto barrier = [&](VkImage img, VkAccessFlags from, VkAccessFlags to, VkImageLayout old_layout) {
                return VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = from,
                    .dstAccessMask = to,
                    .oldLayout = old_layout,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = img,
                    .subresourceRange = range,
                };
            };
            // the frame as drawn (render pass, interpolation, or an earlier copy out of it) into the copy
            const std::array to_copy{
                barrier(image, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL),
                barrier(copy_image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {},
                                   to_copy);
            const VkImageCopy region{
                .srcSubresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0,
                                .layerCount = 1},
                .srcOffset = {},
                .dstSubresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0,
                                .layerCount = 1},
                .dstOffset = {},
                .extent = {.width = extent.width, .height = extent.height, .depth = 1},
            };
            cmdbuf.CopyImage(image, VK_IMAGE_LAYOUT_GENERAL, copy_image, VK_IMAGE_LAYOUT_GENERAL, region);
            const std::array to_shader{
                barrier(image, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL),
                barrier(copy_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {},
                                   to_shader);
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
            cmdbuf.PushConstants(*layout, VK_SHADER_STAGE_COMPUTE_BIT, push);
            cmdbuf.Dispatch((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            const auto done = barrier(image, VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                          VK_ACCESS_SHADER_READ_BIT,
                                      VK_IMAGE_LAYOUT_GENERAL);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, done);
        });
        copy_used = true;
    }

private:
    void MakeCopy(MemoryAllocator& allocator, VkExtent2D extent, VkFormat format) {
        copy_view.reset();
        copy = allocator.CreateImage({
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
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        copy_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *copy,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components{},
            .subresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1,
                              .baseArrayLayer = 0, .layerCount = 1},
        });
        copy_extent = extent;
        copy_format = format;
        copy_used = false;
        written.fill(VK_NULL_HANDLE);  // every set names the copy
    }

    void Write(VkDescriptorSet set, VkImageView target) {
        const VkDescriptorImageInfo src{*sampler, *copy_view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorImageInfo dst{VK_NULL_HANDLE, target, VK_IMAGE_LAYOUT_GENERAL};
        const std::array writes{
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = set,
                                 .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &src,
                                 .pBufferInfo = nullptr, .pTexelBufferView = nullptr},
            VkWriteDescriptorSet{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = set,
                                 .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dst,
                                 .pBufferInfo = nullptr, .pTexelBufferView = nullptr},
        };
        device.GetLogical().UpdateDescriptorSets(writes, {});
    }

    const Device& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    vk::Sampler sampler;
    vk::DescriptorPool pool;
    vk::DescriptorSets sets;
    std::array<VkImageView, kFrames> written{};
    vk::Image copy;
    vk::ImageView copy_view;
    VkExtent2D copy_extent{};
    VkFormat copy_format{};
    bool copy_used = false;
};

Pass* g_pass = nullptr;  // never destroyed: the device may be gone at exit

} // namespace

bool ZonaitePictureActive() {
    return Current().active;
}

void ZonaitePicture(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler, Frame* frame,
                    VkFormat format) {
    const Wanted& w = Current();
    if (!w.active) {
        return;
    }
    if (!g_pass) {
        g_pass = new Pass(device);
    }
    g_pass->Record(allocator, scheduler, frame, format, w.push);
}

} // namespace Vulkan
