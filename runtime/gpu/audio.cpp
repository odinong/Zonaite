// audren and hwopus: the audio renderer and the Opus decoder. Eden's audio_core (compiled unmodified against
// runtime/gpu/override) parses the game's update requests, mixes on its ADSP threads, feeds the output sink and
// decodes Opus; this file is the IPC front end (IAudioRendererManager, IAudioRenderer, IAudioDevice,
// IHardwareOpusDecoderManager, IHardwareOpusDecoder), Eden's core/hle/service/audio reimplemented on Zonaite's IPC.
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "audio_core/audio_core.h"
#include "audio_core/audio_render_manager.h"
#include "audio_core/common/audio_renderer_parameter.h"
#include "audio_core/opus/decoder.h"
#include "audio_core/opus/decoder_manager.h"
#include "audio_core/opus/parameters.h"
#include "audio_core/renderer/audio_device.h"
#include "audio_core/renderer/audio_renderer.h"
#include "audio_core/renderer/system.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_transfer_memory.h"

#include "zonaite/gpu.h"
#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

constexpr uint32_t kAudioOutOfSessions = 153 | 31 << 9;  // audio, desc 31
constexpr uint32_t kAudioInvalidHandle = 153 | 1536 << 9;
constexpr const char* kActiveDevice = "AudioTvOutput";  // docked

AudioCore::Renderer::Manager& renderer_manager() {
    static AudioCore::Renderer::Manager m{gpu::system()};
    return m;
}

std::span<const u8> in_span(IpcCtx& ctx, const Buffer& b) { return {ctx.m + b.addr, b.size}; }
std::span<u8> out_span(IpcCtx& ctx, const Buffer& b) { return {ctx.m + b.addr, b.size}; }

class AudioRenderer : public Service {
public:
    AudioRenderer(const AudioCore::AudioRendererParameterInternal& params, const KTransferMemory& tmem, u64 tmem_size,
                  u64 aruid, s32 session_id)
        : Service("audren:IAudioRenderer"), transfer_memory{tmem.addr, tmem.size} {
        renderer = std::make_unique<AudioCore::Renderer::Renderer>(gpu::system(), renderer_manager(), &rendered_event);
        renderer->Initialize(params, &transfer_memory, tmem_size, &Kernel::KProcess::Application(), aruid, session_id);

        on(0, "GetSampleRate", [this](IpcCtx& ctx) { ctx.push<u32>(renderer->GetSystem().GetSampleRate()); });
        on(1, "GetSampleCount", [this](IpcCtx& ctx) { ctx.push<u32>(renderer->GetSystem().GetSampleCount()); });
        on(2, "GetMixBufferCount", [this](IpcCtx& ctx) { ctx.push<u32>(renderer->GetSystem().GetMixBufferCount()); });
        on(3, "GetState", [this](IpcCtx& ctx) { ctx.push<u32>(renderer->GetSystem().IsActive() ? 0 : 1); });
        auto update = [this](IpcCtx& ctx) {
            std::lock_guard l(lock);
            const auto r = renderer->RequestUpdate(in_span(ctx, ctx.in_buffer()), out_span(ctx, ctx.out_buffer(1)),
                                                   out_span(ctx, ctx.out_buffer(0)));
            ctx.result = r.raw;
        };
        on(4, "RequestUpdate", update);
        on(10, "RequestUpdateAuto", update);
        on(5, "Start", [this](IpcCtx&) { renderer->Start(); });
        on(6, "Stop", [this](IpcCtx&) { renderer->Stop(); });
        on(7, "QuerySystemEvent",
           [this](IpcCtx& ctx) { ctx.push_copy(rendered_event.GetReadableEvent().CreateHandle()); });
        on(8, "SetRenderingTimeLimit",
           [this](IpcCtx& ctx) { renderer->GetSystem().SetRenderingTimeLimit(ctx.pop<u32>()); });
        on(9, "GetRenderingTimeLimit",
           [this](IpcCtx& ctx) { ctx.push<u32>(renderer->GetSystem().GetRenderingTimeLimit()); });
        on(12, "SetVoiceDropParameter",
           [this](IpcCtx& ctx) { renderer->GetSystem().SetVoiceDropParameter(ctx.pop<f32>()); });
        on(13, "GetVoiceDropParameter",
           [this](IpcCtx& ctx) { ctx.push<f32>(renderer->GetSystem().GetVoiceDropParameter()); });
    }

    ~AudioRenderer() override { renderer->Finalize(); }

private:
    std::mutex lock;
    Kernel::KEvent rendered_event{"IAudioRenderer:RenderedEvent"};
    Kernel::KTransferMemory transfer_memory;
    std::unique_ptr<AudioCore::Renderer::Renderer> renderer;
};

class AudioDevice : public Service {
public:
    AudioDevice(u64 aruid, u32 revision) : Service("audren:IAudioDevice"), device{gpu::system(), aruid, revision} {
        system_event.Signal();  // a device is connected from the start
        auto list = [this](IpcCtx& ctx) {
            const Buffer b = ctx.out_buffer();
            std::span<Name> names{reinterpret_cast<Name*>(ctx.m + b.addr), b.size / sizeof(Name)};
            ctx.push<u32>(device.ListAudioDeviceName(names));
        };
        auto list_output = [this](IpcCtx& ctx) {
            const Buffer b = ctx.out_buffer();
            std::span<Name> names{reinterpret_cast<Name*>(ctx.m + b.addr), b.size / sizeof(Name)};
            ctx.push<u32>(device.ListAudioOutputDeviceName(names));
        };
        auto set_volume = [this](IpcCtx& ctx) { device.SetDeviceVolumes(ctx.pop<f32>()); };
        auto get_volume = [this](IpcCtx& ctx) {
            const Buffer b = ctx.in_buffer();
            const std::string name(reinterpret_cast<const char*>(ctx.m + b.addr), strnlen(
                reinterpret_cast<const char*>(ctx.m + b.addr), b.size));
            ctx.push<f32>(device.GetDeviceVolume(name));
        };
        auto active_name = [](IpcCtx& ctx) {
            const Buffer b = ctx.out_buffer();
            if (b.size) {
                std::memset(ctx.m + b.addr, 0, b.size);
                std::memcpy(ctx.m + b.addr, kActiveDevice, std::min<size_t>(b.size - 1, std::strlen(kActiveDevice)));
            }
        };
        on(0, "ListAudioDeviceName", list);
        on(1, "SetAudioDeviceOutputVolume", set_volume);
        on(2, "GetAudioDeviceOutputVolume", get_volume);
        on(3, "GetActiveAudioDeviceName", active_name);
        on(4, "QueryAudioDeviceSystemEvent",
           [this](IpcCtx& ctx) { ctx.push_copy(system_event.GetReadableEvent().CreateHandle()); });
        on(5, "GetActiveChannelCount", [](IpcCtx& ctx) { ctx.push<u32>(2); });
        on(6, "ListAudioDeviceNameAuto", list);
        on(7, "SetAudioDeviceOutputVolumeAuto", set_volume);
        on(8, "GetAudioDeviceOutputVolumeAuto", get_volume);
        on(10, "GetActiveAudioDeviceNameAuto", active_name);
        on(11, "QueryAudioDeviceInputEvent",
           [this](IpcCtx& ctx) { ctx.push_copy(input_event.GetReadableEvent().CreateHandle()); });
        on(12, "QueryAudioDeviceOutputEvent",
           [this](IpcCtx& ctx) { ctx.push_copy(output_event.GetReadableEvent().CreateHandle()); });
        on(13, "GetActiveAudioOutputDeviceName", active_name);
        on(14, "ListAudioOutputDeviceName", list_output);
    }

private:
    using Name = AudioCore::Renderer::AudioDevice::AudioDeviceName;
    AudioCore::Renderer::AudioDevice device;
    Kernel::KEvent system_event{"IAudioDevice:SystemEvent"};
    Kernel::KEvent input_event{"IAudioDevice:InputEvent"};
    Kernel::KEvent output_event{"IAudioDevice:OutputEvent"};
};

class AudioRendererManager : public Service {
public:
    AudioRendererManager() : Service("audren:u") {
        on(0, "OpenAudioRenderer", [](IpcCtx& ctx) {
            const auto params = ctx.pop<AudioCore::AudioRendererParameterInternal>();
            const auto tmem_size = ctx.pop<u64>();
            const auto aruid = ctx.pop<u64>();
            auto tmem = ctx.copy_handles.empty()
                            ? nullptr
                            : std::dynamic_pointer_cast<KTransferMemory>(kernel_get(ctx.copy_handles[0]));
            if (!tmem) {
                ctx.result = kAudioInvalidHandle;
                return;
            }
            auto& manager = renderer_manager();
            const s32 session_id = manager.GetSessionId();
            if (session_id == -1 || manager.GetSessionCount() + 1 > AudioCore::MaxRendererSessions) {
                ctx.result = kAudioOutOfSessions;
                return;
            }
            ctx.push_object(std::make_shared<AudioRenderer>(params, *tmem, tmem_size, aruid, session_id));
        });
        on(1, "GetWorkBufferSize", [](IpcCtx& ctx) {
            const auto params = ctx.pop<AudioCore::AudioRendererParameterInternal>();
            u64 size = 0;
            ctx.result = renderer_manager().GetWorkBufferSize(params, size).raw;
            ctx.push<u64>(size);
        });
        on(2, "GetAudioDeviceService", [](IpcCtx& ctx) {
            const auto aruid = ctx.pop<u64>();
            ctx.push_object(std::make_shared<AudioDevice>(aruid, Common::MakeMagic('R', 'E', 'V', '1')));
        });
        on(4, "GetAudioDeviceServiceWithRevisionInfo", [](IpcCtx& ctx) {
            const auto revision = ctx.pop<u32>();
            const auto aruid = ctx.pop<u64>();
            ctx.push_object(std::make_shared<AudioDevice>(aruid, revision));
        });
    }
};

// ---------------------------------------------------------------- hwopus
namespace Opus = AudioCore::OpusDecoder;

Opus::OpusDecoderManager& opus_manager() {
    static Opus::OpusDecoderManager m{gpu::system()};
    return m;
}

std::shared_ptr<KTransferMemory> transfer_memory_arg(IpcCtx& ctx) {
    return ctx.copy_handles.empty() ? nullptr
                                    : std::dynamic_pointer_cast<KTransferMemory>(kernel_get(ctx.copy_handles[0]));
}

class HardwareOpusDecoder : public Service {
public:
    template <typename Params>
    HardwareOpusDecoder(const Params& params, const KTransferMemory& tmem, u64 tmem_size, uint32_t* result)
        : Service("hwopus:IHardwareOpusDecoder"), transfer_memory{tmem.addr, tmem.size},
          decoder{gpu::system(), opus_manager().GetHardwareOpus()} {
        *result = decoder.Initialize(params, &transfer_memory, tmem_size).raw;
        // Decode commands: output PCM in the first output buffer, Opus packets in the first input buffer.
        // Replies: bytes consumed, samples produced, then (the WithPerf and new forms) the time taken.
        // reset_arg: 0 = no reset, -1 = a bool argument says.
        auto decode = [this](bool multi_stream, bool perf, int reset_arg) {
            return [this, multi_stream, perf, reset_arg](IpcCtx& ctx) {
                const bool reset = reset_arg < 0 ? ctx.pop<u8>() != 0 : false;
                u32 size = 0, samples = 0;
                u64 time = 0;
                const auto in = in_span(ctx, ctx.in_buffer());
                const auto out = out_span(ctx, ctx.out_buffer());
                const auto r = multi_stream
                                   ? decoder.DecodeInterleavedForMultiStream(&size, &time, &samples, in, out, reset)
                                   : decoder.DecodeInterleaved(&size, &time, &samples, in, out, reset);
                ctx.result = r.raw;
                ctx.push<u32>(size);
                ctx.push<u32>(samples);
                if (perf) ctx.push<u64>(time);
            };
        };
        on(0, "DecodeInterleavedOld", decode(false, false, 0));
        on(2, "DecodeInterleavedForMultiStreamOld", decode(true, false, 0));
        on(4, "DecodeInterleavedWithPerfOld", decode(false, true, 0));
        on(5, "DecodeInterleavedForMultiStreamWithPerfOld", decode(true, true, 0));
        on(6, "DecodeInterleavedWithPerfAndResetOld", decode(false, true, -1));
        on(7, "DecodeInterleavedForMultiStreamWithPerfAndResetOld", decode(true, true, -1));
        on(8, "DecodeInterleaved", decode(false, true, -1));
        on(9, "DecodeInterleavedForMultiStream", decode(true, true, -1));
        auto set_context = [this](IpcCtx& ctx) { ctx.result = decoder.SetContext(in_span(ctx, ctx.in_buffer())).raw; };
        on(1, "SetContext", set_context);
        on(3, "SetContextForMultiStream", set_context);
    }

private:
    Kernel::KTransferMemory transfer_memory;
    Opus::OpusDecoder decoder;
};

// The multi-stream parameters are too large for the raw data and come in a pointer (X) buffer.
template <typename T>
T large_arg(IpcCtx& ctx) {
    T v{};
    const Buffer b = ctx.in_buffer();
    std::memcpy(&v, ctx.m + b.addr, std::min<size_t>(sizeof v, b.size));
    return v;
}

template <typename Params>
void open_opus_decoder(IpcCtx& ctx, const Params& params, u32 tmem_size) {
    auto tmem = transfer_memory_arg(ctx);
    if (!tmem) {
        ctx.result = kAudioInvalidHandle;
        return;
    }
    uint32_t result = 0;
    auto decoder = std::make_shared<HardwareOpusDecoder>(params, *tmem, tmem_size, &result);
    ctx.result = result;
    if (!result) ctx.push_object(decoder);
}

void push_size(IpcCtx& ctx, Result r, u32 size) {
    ctx.result = r.raw;
    ctx.push<u32>(size);
}

class HardwareOpusDecoderManager : public Service {
public:
    HardwareOpusDecoderManager() : Service("hwopus") {
        // Open*: parameters (raw, or a pointer buffer for multi-stream), u32 work buffer size, transfer memory
        on(0, "OpenHardwareOpusDecoder", [](IpcCtx& ctx) {
            const auto p = ctx.pop<Opus::OpusParameters>();
            const auto tmem_size = ctx.pop<u32>();
            Opus::OpusParametersEx ex{};
            ex.sample_rate = p.sample_rate;
            ex.channel_count = p.channel_count;
            ex.use_large_frame_size = false;
            open_opus_decoder(ctx, ex, tmem_size);
        });
        on(4, "OpenHardwareOpusDecoderEx", [](IpcCtx& ctx) {
            const auto p = ctx.pop<Opus::OpusParametersEx>();
            open_opus_decoder(ctx, p, ctx.pop<u32>());
        });
        on(2, "OpenOpusDecoderForMultiStream", [](IpcCtx& ctx) {
            const auto p = large_arg<Opus::OpusMultiStreamParameters>(ctx);
            Opus::OpusMultiStreamParametersEx ex{};
            ex.sample_rate = p.sample_rate;
            ex.channel_count = p.channel_count;
            ex.total_stream_count = p.total_stream_count;
            ex.stereo_stream_count = p.stereo_stream_count;
            ex.use_large_frame_size = false;
            ex.mappings = p.mappings;
            open_opus_decoder(ctx, ex, ctx.pop<u32>());
        });
        on(6, "OpenHardwareOpusDecoderForMultiStreamEx", [](IpcCtx& ctx) {
            open_opus_decoder(ctx, large_arg<Opus::OpusMultiStreamParametersEx>(ctx), ctx.pop<u32>());
        });
        on(1, "GetWorkBufferSize", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r = opus_manager().GetWorkBufferSize(ctx.pop<Opus::OpusParameters>(), size);
            push_size(ctx, r, size);
        });
        on(3, "GetWorkBufferSizeForMultiStream", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r =
                opus_manager().GetWorkBufferSizeForMultiStream(large_arg<Opus::OpusMultiStreamParameters>(ctx), size);
            push_size(ctx, r, size);
        });
        on(5, "GetWorkBufferSizeEx", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r = opus_manager().GetWorkBufferSizeEx(ctx.pop<Opus::OpusParametersEx>(), size);
            push_size(ctx, r, size);
        });
        on(7, "GetWorkBufferSizeForMultiStreamEx", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r = opus_manager().GetWorkBufferSizeForMultiStreamEx(
                large_arg<Opus::OpusMultiStreamParametersEx>(ctx), size);
            push_size(ctx, r, size);
        });
        on(8, "GetWorkBufferSizeExEx", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r = opus_manager().GetWorkBufferSizeExEx(ctx.pop<Opus::OpusParametersEx>(), size);
            push_size(ctx, r, size);
        });
        on(9, "GetWorkBufferSizeForMultiStreamExEx", [](IpcCtx& ctx) {
            u32 size = 0;
            const auto r = opus_manager().GetWorkBufferSizeForMultiStreamExEx(
                large_arg<Opus::OpusMultiStreamParametersEx>(ctx), size);
            push_size(ctx, r, size);
        });
    }
};

ServiceRegistrar reg("audren:u", [] { return std::make_shared<AudioRendererManager>(); });
ServiceRegistrar reg_opus("hwopus", [] { return std::make_shared<HardwareOpusDecoderManager>(); });

}  // namespace

}  // namespace zn
