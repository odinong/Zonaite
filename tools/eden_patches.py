#!/usr/bin/env python3
"""Zonaite's changes to vendored Eden sources, applied by tools/vendor_eden.py after copying (and runnable alone).

Each patch is an exact text replacement in one file; a patch whose `old` text is missing (Eden changed) fails
loudly, and an already-applied patch is skipped. Keep this list short: every patch is maintenance.

usage:
  eden_patches.py            apply to third_party/eden
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDEN = os.path.join(ROOT, "third_party", "eden")

PATCHES = [
    (
        # TotK recycles command memory as soon as NVN considers a submission fetched; Eden's GPU thread can run far
        # behind (pipeline compiles), so reading the command lists later saw memory already rewritten with the
        # next frame's commands (torn streams, garbage methods, crashes). Capture the words at submission, the way
        # the hardware's GPFIFO fetch is effectively immediate, and hand them to the GPU as a prefetched list.
        "src/core/hle/service/nvdrv/devices/nvhost_gpu.cpp",
        """    gpu.PushGPUEntries(bind_id, std::move(entries));
""",
        """    // ZONAITE: capture the command lists now (see tools/eden_patches.py)
    if (channel_state->memory_manager) {
        boost::container::small_vector<Tegra::CommandHeader, 512> captured;
        for (const auto& header : entries.command_lists) {
            if (header.size == 0) {
                continue;
            }
            const size_t at = captured.size();
            captured.resize(at + header.size);
            channel_state->memory_manager->ReadBlock(header.addr, captured.data() + at,
                                                     header.size * sizeof(Tegra::CommandHeader));
        }
        if (!captured.empty()) {
            gpu.PushGPUEntries(bind_id, Tegra::CommandList{std::move(captured)});
        }
    } else {
        gpu.PushGPUEntries(bind_id, std::move(entries));
    }
""",
    ),
    (
        # Frame pacing for interpolated frames (runtime/gpu/frame_interp.cpp): each frame may carry the time it
        # should be shown, so the frames generated between two game frames are spread evenly over the interval
        # instead of going out back to back (which FIFO only spaces by one vblank: bunched at high refresh rates).
        "src/video_core/renderer_vulkan/vk_present_manager.h",
        """    bool storage_capable{};
};
""",
        """    bool storage_capable{};
    std::chrono::steady_clock::time_point present_at{};  // ZONAITE: show no earlier than this (zero: now)
};
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_present_manager.h",
        """#include <atomic>
#include <condition_variable>
""",
        """#include <atomic>
#include <chrono>
#include <condition_variable>
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_present_manager.cpp",
        """            void(std::exchange(lock, std::unique_lock{swapchain_mutex}));
            CopyToSwapchain(frame);
""",
        """            void(std::exchange(lock, std::unique_lock{swapchain_mutex}));
            // ZONAITE: frame pacing (see tools/eden_patches.py); sleep most of the way, then yield to the mark
            if (frame->present_at != std::chrono::steady_clock::time_point{}) {
                const auto coarse = frame->present_at - std::chrono::microseconds(1500);
                if (std::chrono::steady_clock::now() < coarse) {
                    std::this_thread::sleep_until(coarse);
                }
                while (std::chrono::steady_clock::now() < frame->present_at) {
                    std::this_thread::yield();
                }
            }
            CopyToSwapchain(frame);
            // runtime/gpu/frame_interp.cpp: pacing statistics, and how late frames are for the pacing margin
            extern void ZonaiteNotePresent(std::chrono::steady_clock::time_point);
            ZonaiteNotePresent(frame->present_at);
            zonaite_queued.fetch_sub(1);
""",
    ),
    (
        # A composite that waits on fences ran on whichever thread reached the last fence (the fence manager's
        # release thread), in parallel with the GPU thread, both recording into the one Vulkan scheduler: crashes
        # in the buffer/query caches, and stalls while the release thread sat in presentation. Hand it to the GPU
        # thread instead, like a composite without fences.
        "src/video_core/gpu.cpp",
        """                        free_swap_counters.push_back(current_request_counter);
                    }
                    renderer->Composite(composite_layers);
                };""",
        """                        free_swap_counters.push_back(current_request_counter);
                    }
                    // ZONAITE: on the GPU thread (see tools/eden_patches.py)
                    void(RequestSyncOperation([this, composite_layers] { renderer->Composite(composite_layers); }));
                    gpu_thread.TickGPU(is_async);
                };""",
    ),
    (
        # The fence manager's release thread runs async flushes that record into the Vulkan scheduler while
        # holding the buffer and texture cache locks, as the GPU thread's draws do. Presentation recorded into the
        # same scheduler without them (frame generation records a lot, and waits for frames in between): races,
        # crashes in the caches. RasterizerVulkan::LockCaches lets the renderer take the same locks.
        "src/video_core/renderer_vulkan/vk_rasterizer.h",
        """    void LoadDiskResources(""",
        """    // ZONAITE: the buffer and texture cache locks (see tools/eden_patches.py)
    void LockCaches() {
        std::lock(buffer_cache.mutex, texture_cache.mutex);
    }
    void UnlockCaches() {
        texture_cache.mutex.unlock();
        buffer_cache.mutex.unlock();
    }

    void LoadDiskResources(""",
    ),
    (
        "src/video_core/renderer_vulkan/renderer_vulkan.cpp",
        """    scheduler.RequestOutsideRenderPassOperationContext();
    blit_swapchain.DrawToFrame(device, rasterizer, frame, framebuffers,
                               render_window.GetFramebufferLayout(),
                               present_manager.SwapchainImageCount(),
                               swapchain.GetImageViewFormat());

#ifdef HAS_LSFG
    void(frame_gen.WantedGenerations(present_manager.MaxExtraFrames()));

    frame_gen.Process(device, frame, present_manager.SwapchainImageFormat(),
                      GuestExtent(framebuffers));

    const size_t generated_frames = frame_gen.GeneratedFrameCount();
    for (size_t generation = 0; generation < generated_frames; ++generation) {
        Frame* generated = present_manager.GetRenderFrame();
        blit_swapchain.PrepareFrame(device, generated, render_window.GetFramebufferLayout());
        frame_gen.GenerateInto(device, generated, generation);
        scheduler.Flush(*generated->render_ready);
        present_manager.Present(generated);
    }
#endif

    scheduler.Flush(*frame->render_ready);

    present_manager.Present(frame);
#ifdef HAS_LSFG
    scheduler.DispatchWork();
#endif
""",
        """    // ZONAITE: the fence thread's async flushes record into the scheduler under the cache locks, as draws do;
    // so does presentation, which records too (not while waiting for a free frame). See tools/eden_patches.py.
    rasterizer.LockCaches();
    scheduler.RequestOutsideRenderPassOperationContext();
    blit_swapchain.DrawToFrame(device, rasterizer, frame, framebuffers,
                               render_window.GetFramebufferLayout(),
                               present_manager.SwapchainImageCount(),
                               swapchain.GetImageViewFormat());

#ifdef HAS_LSFG
    void(frame_gen.WantedGenerations(present_manager.MaxExtraFrames()));

    // which framebuffer this presents, so frame interpolation takes that frame's scene capture
    ZonaiteNoteComposite(framebuffers.empty() ? 0 : framebuffers[0].address + framebuffers[0].offset);
    frame_gen.Process(device, frame, present_manager.SwapchainImageFormat(),
                      GuestExtent(framebuffers));
    ZonaitePicture(device, memory_allocator, scheduler, frame, present_manager.SwapchainImageFormat());

    const size_t generated_frames = frame_gen.GeneratedFrameCount();
    for (size_t generation = 0; generation < generated_frames; ++generation) {
        rasterizer.UnlockCaches();
        Frame* generated = present_manager.GetRenderFrame();
        rasterizer.LockCaches();
        blit_swapchain.PrepareFrame(device, generated, render_window.GetFramebufferLayout());
        frame_gen.GenerateInto(device, generated, generation);
        ZonaitePicture(device, memory_allocator, scheduler, generated,
                       present_manager.SwapchainImageFormat());
        scheduler.Flush(*generated->render_ready);
        present_manager.Present(generated);
    }
#endif

    scheduler.Flush(*frame->render_ready);

    present_manager.Present(frame);
#ifdef HAS_LSFG
    scheduler.DispatchWork();
#endif
    rasterizer.UnlockCaches();
    rasterizer.ZonaiteApplyLive();
""",
    ),
    (
        # The scene capture for frame interpolation (runtime/gpu/scene_capture.cpp): it may record copies
        # outside the render pass, after which the draw's render pass is requested again.
        "src/video_core/renderer_vulkan/vk_rasterizer.cpp",
        """    if (!pipeline->Configure(is_indexed))
        return;
""",
        """    if (!pipeline->Configure(is_indexed))
        return;
    // ZONAITE: scene capture for frame interpolation (see tools/eden_patches.py)
    {
        const auto& rts = texture_cache.GetRenderTargets();
        ZonaiteDrawTargets zt{};
        const auto resolve = [&](VideoCommon::ImageViewId id, ZonaiteTarget& out) {
            if (!id) {
                return;
            }
            const ImageView& view = texture_cache.GetImageView(id);
            // the real size: with a resolution scale, an image drawn scaled is larger than its guest size
            const bool scaled = texture_cache.IsRescaling(view);
            const auto& res = Settings::values.resolution_info;
            out = {view.ImageHandle(), view.format, scaled ? res.ScaleUp(view.size.width) : view.size.width,
                   scaled ? res.ScaleUp(view.size.height) : view.size.height, view.range.base.level,
                   view.range.base.layer, scaled ? res.up_factor : 1.0f};
        };
        resolve(rts.depth_buffer_id, zt.depth);
        for (const auto id : rts.color_buffer_ids) {
            if (id) {
                resolve(id, zt.color);
                break;
            }
        }
        const int scene = ZonaiteSceneDraw(*maxwell3d, *gpu_memory, scheduler, zt);
        if (scene & kSceneRecorded) {
            scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
        }
        if (scene & kSceneSkipDraw) {
            return;  // the HUD, hidden (F3)
        }
    }
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_rasterizer.cpp",
        """#include "video_core/renderer_vulkan/vk_rasterizer.h"
""",
        """#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
""",
    ),
    (
        # the render targets of the current draw, for the scene capture
        "src/video_core/texture_cache/texture_cache_base.h",
        """    Framebuffer* GetFramebuffer();
""",
        """    Framebuffer* GetFramebuffer();

    /// ZONAITE: the render targets of the current draw (tools/eden_patches.py)
    const RenderTargets& GetRenderTargets() const noexcept {
        return render_targets;
    }
""",
    ),
    (
        "src/video_core/renderer_vulkan/renderer_vulkan.cpp",
        """#include "video_core/renderer_vulkan/renderer_vulkan.h"
""",
        """#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
""",
    ),
    (
        # The settings menu and overlays (runtime/gpu/ui.cpp), drawn into the swapchain image after the frame is
        # copied in, before the barrier to presentation (the overlay leaves the image in the transfer layout)
        "src/video_core/renderer_vulkan/vk_present_manager.cpp",
        """    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, {},
                           {}, {}, post_barriers);
""",
        """    // ZONAITE: the settings menu and overlays (see tools/eden_patches.py)
    extern void ZonaiteOverlay(const vk::Instance&, const Device&, std::mutex&, vk::CommandBuffer, VkImage,
                               VkExtent2D, VkFormat, size_t);
    ZonaiteOverlay(instance, device, scheduler.submit_mutex, cmdbuf, image, extent, swapchain.GetImageViewFormat(),
                   image_count);

    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, {},
                           {}, {}, post_barriers);
""",
    ),
    (
        # With async presentation the present thread acquires swapchain images, and the default frame pacing
        # waited for the image's last tick with Scheduler::Wait, which Flush()es the scheduler when that tick is
        # not submitted yet: from the present thread, while the GPU thread records (crashes in the scheduler and
        # every cache). Wait for the last submitted tick instead; the copy to the swapchain waits for the frame's
        # own semaphore anyway. With frame generation, presents are paced by their present times
        # (runtime/gpu/frame_interp.cpp), and waiting here for work queued a few presents ago made them late
        # whenever the GPU queue held more than that (frame spacing 8.3 +- 2.6 ms instead of +- 0.4 ms).
        "src/video_core/renderer_vulkan/vk_swapchain.cpp",
        """    case Settings::FramePacingMode::Target_Auto:
        scheduler.Wait(resource_ticks[image_index]);
        break;""",
        """    case Settings::FramePacingMode::Target_Auto:
        // ZONAITE: never Flush from the present thread, and with frame generation (which paces presents itself)
        // don't wait for the GPU at all (see tools/eden_patches.py)
        if (!Settings::values.frame_gen.GetValue()) {
            scheduler.GetMasterSemaphore().Wait(
                std::min(resource_ticks[image_index], scheduler.CurrentTick() - 1));
        }
        break;""",
    ),
    # ---- Settings that change while playing (runtime/gpu/config.cpp, zonaite_live.h) ----
    (
        # Texture cache: the resolution scale, texture decoding and texture filtering change between two frames
        # (TextureCache::ZonaiteApplySettings), and the memory limits are worked out again on request
        "src/video_core/texture_cache/texture_cache_base.h",
        """#include <mutex>
""",
        """#include <functional>
#include <mutex>
#include <thread>
""",
    ),
    (
        "src/video_core/texture_cache/texture_cache_base.h",
        """    /// Notify the cache that a new frame has been queued
    void TickFrame();
""",
        """    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// ZONAITE: settings changed while playing (see tools/eden_patches.py)
    void ZonaiteApplySettings(bool rescale, bool textures, bool samplers, const std::function<void()>& apply);
    void ZonaiteMemoryLimits();
""",
    ),
    (
        "src/video_core/texture_cache/texture_cache.h",
        """    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                     DEFAULT_EXPECTED_MEMORY));
        critical_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                     DEFAULT_CRITICAL_MEMORY));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
""",
        """    ZonaiteMemoryLimits();  // ZONAITE: also when the settings change (see tools/eden_patches.py)
""",
    ),
    (
        "src/video_core/texture_cache/texture_cache.h",
        """template <class P>
void TextureCache<P>::RunGarbageCollector() {
""",
        """// ZONAITE: the memory the cache aims to stay under, from the device's (the constructor's code; called again
// when video memory use or the resolution scale changes while playing). See tools/eden_patches.py.
template <class P>
void TextureCache<P>::ZonaiteMemoryLimits() {
    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                     DEFAULT_EXPECTED_MEMORY));
        critical_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                     DEFAULT_CRITICAL_MEMORY));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
}

// ZONAITE: settings the menu changed while playing, applied between two frames (RasterizerVulkan::
// ZonaiteApplyLive; see tools/eden_patches.py). `apply` writes the new values into the settings once the old
// ones are no longer needed:
// - rescale: every image drawn at the old resolution scale goes back to native size (its contents with it) and
//   its scaled copy is freed; the next frames scale images up at the new scale as usual. Viewports and scissors
//   are worked out again.
// - textures: texture decoding changed. An image's format is fixed when it is made, so ASTC textures (only ever
//   uploaded from guest memory) are dropped and made again on next use.
// - samplers: texture filtering changed. Every sampler is made again in place, under the same id.
template <class P>
void TextureCache<P>::ZonaiteApplySettings(bool rescale, bool textures, bool samplers,
                                          const std::function<void()>& apply) {
    std::vector<ImageId> ids;
    for (auto [id, image] : slot_images) {
        if (id != NULL_IMAGE_ID) {
            ids.push_back(id);
        }
    }
    if (rescale) {
        for (const ImageId id : ids) {
            Image& image = slot_images[id];
            if (True(image.flags & ImageFlagBits::Rescaled)) {
                ScaleDown(image);
            }
        }
    }
    if (textures) {
        // decodes in flight first: they finish into images of the old format
        while (!async_decodes.empty()) {
            TickAsyncDecode();
            if (!async_decodes.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    runtime.Finish();
    if (rescale) {
        for (const ImageId id : ids) {
            Image& image = slot_images[id];
            if (image.HasScaled()) {
                total_used_memory -= (std::min)(total_used_memory, GetScaledImageSizeBytes(image));
                image.ZonaiteDropScaled();
            }
        }
    }
    if (apply) {
        apply();
    }
    if (rescale) {
        // the same images may be scaled before and after (2x to 0.5x): the viewports and scissors are worked out
        // again at the new factor either way
        for (const size_t c : active_channel_ids) {
            auto& flags = channel_storage[c].maxwell3d.dirty.flags;
            flags[Dirty::RescaleViewports] = true;
            flags[Dirty::RescaleScissors] = true;
        }
    }
    if (textures) {
        for (const ImageId id : ids) {
            Image& image = slot_images[id];
            if (!IsPixelFormatASTC(image.info.format) || True(image.flags & ImageFlagBits::GpuModified)) {
                continue;
            }
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, id);
            }
            if (True(image.flags & ImageFlagBits::Registered)) {
                UnregisterImage(id);
            }
            DeleteImage(id, true);
        }
    }
    if (samplers) {
        for (const size_t c : active_channel_ids) {
            for (const auto& [config, id] : channel_storage[c].samplers) {
                Sampler& sampler = slot_samplers[id];
                std::destroy_at(&sampler);
                std::construct_at(&sampler, runtime, config);
            }
        }
    }
}

template <class P>
void TextureCache<P>::RunGarbageCollector() {
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_texture_cache.h",
        """    bool ScaleDown(bool ignore = false);
""",
        """    bool ScaleDown(bool ignore = false);

    /// ZONAITE: frees the scaled copy of an image back at native size (see tools/eden_patches.py)
    void ZonaiteDropScaled();
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_texture_cache.cpp",
        """Image::~Image() {
""",
        """void Image::ZonaiteDropScaled() {
    if (ENABLE_MSAA_RESOLVE_CONSUME && runtime != nullptr && scaled_image) {
        runtime->EraseResolveShadow(*scaled_image);
    }
    scale_framebuffer.reset();
    normal_framebuffer.reset();
    scale_view.reset();
    normal_view.reset();
    scaled_image = {};
    has_scaled = false;
}

Image::~Image() {
""",
    ),
    (
        # the GPU ASTC decoder exists whatever the setting at startup, as it can change while playing
        "src/video_core/renderer_vulkan/vk_texture_cache.cpp",
        """    if (Settings::values.accelerate_astc.GetValue() == Settings::AstcDecodeMode::Gpu) {
        astc_decoder_pass.emplace(""",
        """    {  // ZONAITE: whatever the decoding setting, which can change while playing (see tools/eden_patches.py)
        astc_decoder_pass.emplace(""",
    ),
    (
        "src/video_core/buffer_cache/buffer_cache_base.h",
        """    ~BufferCache();

    void TickFrame();
""",
        """    ~BufferCache();

    void TickFrame();

    /// ZONAITE: the memory limits again (see tools/eden_patches.py)
    void ZonaiteMemoryLimits();
""",
    ),
    (
        "src/video_core/buffer_cache/buffer_cache.h",
        """    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}
""",
        """    ZonaiteMemoryLimits();  // ZONAITE: also when the settings change (see tools/eden_patches.py)
}

template <class P>
void BufferCache<P>::ZonaiteMemoryLimits() {
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}
""",
    ),
    (
        "src/video_core/vulkan_common/vulkan_device.h",
        """    u64 GetDeviceLocalMemory() const {
        return device_access_memory;
    }
""",
        """    u64 GetDeviceLocalMemory() const {
        return device_access_memory;
    }

    /// ZONAITE: the usable memory again (video memory use or the resolution scale changed while playing)
    void ZonaiteRefreshMemory() {
        CollectPhysicalMemoryInfo();
    }
""",
    ),
    (
        # The settings menu's "clear shader cache" while playing: every pipeline is dropped (and built again as
        # it is next used), and the cache files on disk start over
        "src/video_core/renderer_vulkan/vk_pipeline_cache.h",
        """    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);
""",
        """    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// ZONAITE: drops every pipeline, and with `disk` the cache files too (see tools/eden_patches.py)
    void ZonaiteClearPipelines(bool disk);
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_pipeline_cache.cpp",
        """GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
""",
        """// ZONAITE: see tools/eden_patches.py. Pipelines being built and written out finish first, and the GPU is
// done with every pipeline before they go. Without `disk` (a new resolution scale, which shaders are generated
// for), the files stay: they hold the game's shaders, which are translated again for the new scale.
void PipelineCache::ZonaiteClearPipelines(bool disk) {
    workers.WaitForRequests();
    serialization_thread.WaitForRequests();
    scheduler.Finish();
    scheduler.InvalidateState();
    current_pipeline = nullptr;
    graphics_cache.clear();
    compute_cache.clear();
    if (!disk) {
        return;
    }
    std::error_code ec;
    if (!pipeline_cache_filename.empty()) {
        std::filesystem::remove(pipeline_cache_filename, ec);
    }
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        std::filesystem::remove(vulkan_pipeline_cache_filename, ec);
        vulkan_pipeline_cache = device.GetLogical().CreatePipelineCache({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = 0,
            .pInitialData = nullptr,
        });
        last_cache_size.store(0, std::memory_order_relaxed);
        pipelines_since_flush = 0;
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_rasterizer.h",
        """    void TickFrame() override;
""",
        """    void TickFrame() override;

    /// ZONAITE: settings changed while playing, between two frames (see tools/eden_patches.py)
    void ZonaiteApplyLive();
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_rasterizer.cpp",
        """#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
""",
        """#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
#include "video_core/renderer_vulkan/zonaite_live.h"     // ZONAITE (runtime/gpu/override_video)
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_rasterizer.cpp",
        """void RasterizerVulkan::TickFrame() {
""",
        """// ZONAITE: settings the menu changed that Eden otherwise reads only when it starts (runtime/gpu/config.cpp,
// zonaite_live.h), applied by the GPU thread after a frame is presented. See tools/eden_patches.py.
void RasterizerVulkan::ZonaiteApplyLive() {
    ZonaiteLiveChange change;
    if (!ZonaiteTakeLiveChange(change)) {
        return;
    }
    // shaders are generated for one resolution scale: a new one needs them again
    if (change.shaders || change.rescale) {
        pipeline_cache.ZonaiteClearPipelines(change.shaders);
    }
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.ZonaiteApplySettings(change.rescale, change.textures, change.samplers, change.apply);
    if (change.memory) {
        const_cast<Device&>(device).ZonaiteRefreshMemory();
        texture_cache.ZonaiteMemoryLimits();
        buffer_cache.ZonaiteMemoryLimits();
    }
}

void RasterizerVulkan::TickFrame() {
""",
    ),
    (
        "src/video_core/renderer_vulkan/renderer_vulkan.cpp",
        """#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
""",
        """#include "video_core/renderer_vulkan/zonaite_capture.h"  // ZONAITE (runtime/gpu/override_video)
#include "video_core/renderer_vulkan/zonaite_live.h"     // ZONAITE (runtime/gpu/override_video)
""",
    ),
    (
        # frames need a storage view for picture adjustments too
        "src/video_core/renderer_vulkan/vk_blit_screen.cpp",
        """#include "video_core/renderer_vulkan/vk_scheduler.h"
""",
        """#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/zonaite_live.h"  // ZONAITE (runtime/gpu/override_video)
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_blit_screen.cpp",
        """    const bool storage_required = Settings::values.frame_gen.GetValue();
""",
        """    const bool storage_required = Settings::values.frame_gen.GetValue() || ZonaitePictureActive();  // ZONAITE
""",
    ),
    (
        # Frame generation changes while playing: the present manager has frames for the most it can use, and
        # keeps only as many queued for presentation as the current setting needs (fewer frames queued, less
        # delay, when it is turned down or off)
        "src/video_core/renderer_vulkan/vk_present_manager.cpp",
        """    const size_t generations = Settings::FrameGenMaxGenerations();
""",
        """    // ZONAITE: for the most frame generation can use (see tools/eden_patches.py)
    const size_t generations = Settings::MAX_FRAME_GEN_MULTIPLIER - 1;
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_present_manager.cpp",
        """    free_cv.wait(lock, [this] { return !free_queue.empty(); });
""",
        """    // ZONAITE: no more frames queued for presentation than the current setting needs (see tools/eden_patches.py)
    free_cv.wait(lock, [this] {
        const size_t wanted = std::clamp<size_t>(
            (Settings::FrameGenMaxGenerations() + 1) * (Settings::values.frame_gen_queue_target.GetValue() + 1),
            swapchain_image_count, frames.size());
        return !free_queue.empty() && (!use_present_thread || zonaite_queued.load() < wanted);
    });
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_present_manager.cpp",
        """    if (use_present_thread) {
        scheduler.Record([this, frame](vk::CommandBuffer) {
""",
        """    if (use_present_thread) {
        zonaite_queued.fetch_add(1);  // ZONAITE
        scheduler.Record([this, frame](vk::CommandBuffer) {
""",
    ),
    (
        "src/video_core/renderer_vulkan/vk_present_manager.h",
        """    std::jthread present_thread;
""",
        """    std::jthread present_thread;
    std::atomic<size_t> zonaite_queued{};  // ZONAITE: frames handed to the present thread, not yet shown
""",
    ),
]

CRLF = chr(13) + chr(10)
LF = chr(10)


def apply(eden=EDEN, log=print):
    for rel, old, new in PATCHES:
        path = os.path.join(eden, rel)
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        if CRLF in text:  # Eden's sources are CRLF on Windows checkouts
            old, new = old.replace(LF, CRLF), new.replace(LF, CRLF)
        if new in text:
            continue
        if text.count(old) != 1:
            sys.exit(f"eden_patches: {rel}: the text to patch is missing or ambiguous (Eden changed?)")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old, new, 1))
        log(f"patched {rel}")


if __name__ == "__main__":
    apply()
