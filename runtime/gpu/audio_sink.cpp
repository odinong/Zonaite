// Audio output for Eden's audio_core: a WASAPI sink. The device's buffer paces the audio renderer like the console's
// DSP does (48 kHz); without a usable output device a timer consumes the samples at the same rate, so the game is
// paced either way. Replaces Eden's sink_details.cpp (cubeb/SDL sink selection), which Zonaite doesn't build.
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>

#include <algorithm>
#include <span>
#include <utility>

#include "zonaite/config.h"
#include "zonaite/gpu.h"

#include "audio_core/common/common.h"
#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_details.h"
#include "audio_core/sink/sink_stream.h"

#include "zonaite/runtime.h"

namespace AudioCore::Sink {

namespace {

constexpr u32 kRate = 48000;
constexpr u32 kChannels = 2;

template <typename T>
void release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

class WasapiStream final : public SinkStream {
public:
    WasapiStream(Core::System& system_, StreamType type_, u32 system_channels_, const std::string& name_)
        : SinkStream{system_, type_} {
        system_channels = system_channels_;
        device_channels = kChannels;
        name = name_;
    }

    ~WasapiStream() override { Finalize(); }

    void Finalize() override {
        stop.store(true);
        if (thread.joinable()) thread.join();
    }

    void Start(bool resume) override {
        if (!paused) return;
        paused = false;
        if (!thread.joinable()) thread = std::thread([this] { type == StreamType::In ? run_paced() : run(); });
    }

    void Stop() override {
        if (paused) return;
        SignalPause();
        paused = true;
    }

private:
    // Output: reopened whenever the chosen device or the latency changes in the settings menu
    void run() {
        SetThreadDescription(GetCurrentThread(), L"zonaite audio out");
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        while (!stop.load()) {
            const zn::cfg::Config c = zn::cfg::get_copy();
            if (!run_device(c.audio_device, c.audio_latency_ms)) {
                CoUninitialize();
                run_paced();
                return;
            }
        }
        CoUninitialize();
    }

    // The game's samples with the menu's volume: muted, or scaled (up to 2x, clipped)
    static void apply_volume(std::span<s16> samples) {
        static zn::cfg::Config c = zn::cfg::get_copy();
        static u64 seen = zn::cfg::generation();
        if (const u64 g = zn::cfg::generation(); g != seen) {
            seen = g;
            c = zn::cfg::get_copy();
        }
        const bool silent = c.mute || (c.mute_unfocused && !zn::gpu::window_focused());
        const int gain = silent ? 0 : std::clamp(c.volume, 0, 200) * 256 / 100;  // 8.8 fixed point
        if (gain == 256) return;
        for (auto& v : samples) v = static_cast<s16>(std::clamp((v * gain) >> 8, -32768, 32767));
    }

    // Event-driven shared-mode rendering on one device (empty id: the default). Returns false if it can't be
    // opened (the caller falls back to pacing without sound), true when it should be reopened (settings changed).
    bool run_device(const std::string& id, int latency_ms) {
        IMMDeviceEnumerator* enumerator = nullptr;
        IMMDevice* device = nullptr;
        IAudioClient* client = nullptr;
        IAudioRenderClient* render = nullptr;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        WAVEFORMATEX fmt{};
        fmt.wFormatTag = WAVE_FORMAT_PCM;
        fmt.nChannels = kChannels;
        fmt.nSamplesPerSec = kRate;
        fmt.wBitsPerSample = 16;
        fmt.nBlockAlign = fmt.nChannels * fmt.wBitsPerSample / 8;
        fmt.nAvgBytesPerSec = fmt.nSamplesPerSec * fmt.nBlockAlign;
        UINT32 buffer_frames = 0;
        const bool have_enumerator = SUCCEEDED(CoCreateInstance(
            __uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&enumerator));
        if (have_enumerator && !id.empty()) {
            const std::wstring wid(id.begin(), id.end());  // endpoint ids are ASCII
            if (FAILED(enumerator->GetDevice(wid.c_str(), &device))) device = nullptr;
        }
        if (have_enumerator && !device) enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        const REFERENCE_TIME duration = std::clamp(latency_ms, 10, 200) * 10000LL;
        const bool ok =
            device && SUCCEEDED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&client)) &&
            SUCCEEDED(client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                         AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                                             AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                         duration, 0, &fmt, nullptr)) &&
            SUCCEEDED(client->SetEventHandle(event)) && SUCCEEDED(client->GetBufferSize(&buffer_frames)) &&
            SUCCEEDED(client->GetService(__uuidof(IAudioRenderClient), (void**)&render)) &&
            SUCCEEDED(client->Start());
        if (ok) {
            zn::log("audio: WASAPI output %s, %u Hz, %u channels, %u-frame buffer", id.empty() ? "(default)" : id.c_str(),
                    kRate, kChannels, buffer_frames);
            const u64 started = zn::cfg::generation();
            u64 checked = started;
            while (!stop.load()) {
                WaitForSingleObject(event, 100);
                if (const u64 g = zn::cfg::generation(); g != checked) {
                    checked = g;
                    const zn::cfg::Config c = zn::cfg::get_copy();
                    if (c.audio_device != id || c.audio_latency_ms != latency_ms) break;  // reopen
                }
                UINT32 padding = 0;
                if (FAILED(client->GetCurrentPadding(&padding))) break;
                const UINT32 frames = buffer_frames - padding;
                BYTE* data = nullptr;
                if (!frames || FAILED(render->GetBuffer(frames, &data))) continue;
                const std::span<s16> samples(reinterpret_cast<s16*>(data), (size_t)frames * kChannels);
                ProcessAudioOutAndRender(samples, frames);
                apply_volume(samples);
                render->ReleaseBuffer(frames, 0);
            }
            client->Stop();
        } else {
            zn::log("audio: no WASAPI output device, audio is discarded");
        }
        release(render);
        release(client);
        release(device);
        release(enumerator);
        CloseHandle(event);
        return ok;
    }

    // No device (or audio input): move samples at 48 kHz in 10 ms steps, discarding output / feeding silence.
    void run_paced() {
        constexpr u32 kFrames = kRate / 100;
        std::vector<s16> buffer(kFrames * kChannels);
        auto next = std::chrono::steady_clock::now();
        while (!stop.load()) {
            next += std::chrono::milliseconds(10);
            std::this_thread::sleep_until(next);
            if (type == StreamType::In) {
                std::fill(buffer.begin(), buffer.end(), s16{0});
                ProcessAudioIn(buffer, kFrames);
            } else {
                ProcessAudioOutAndRender(buffer, kFrames);
            }
        }
    }

    std::thread thread;
    std::atomic<bool> stop{false};
};

class WasapiSink final : public Sink {
public:
    WasapiSink() { device_channels = kChannels; }
    ~WasapiSink() override { CloseStreams(); }

    SinkStream* AcquireSinkStream(Core::System& system, u32 system_channels_, const std::string& name,
                                  StreamType type) override {
        system_channels = system_channels_;
        auto& s = streams.emplace_back(std::make_unique<WasapiStream>(system, type, system_channels_, name));
        s->SetDeviceVolume(device_volume);
        return s.get();
    }

    void CloseStream(SinkStream* stream) override {
        std::erase_if(streams, [stream](const SinkStreamPtr& s) { return s.get() == stream; });
    }

    void CloseStreams() override { streams.clear(); }

    f32 GetDeviceVolume() const override { return device_volume; }

    void SetDeviceVolume(f32 volume) override {
        device_volume = volume;
        for (auto& s : streams) s->SetDeviceVolume(volume);
    }

    void SetSystemVolume(f32 volume) override {
        for (auto& s : streams) s->SetSystemVolume(volume);
    }

private:
    std::vector<SinkStreamPtr> streams;
};

}  // namespace

}  // namespace AudioCore::Sink

namespace zn::audio {

// PKEY_Device_FriendlyName (functiondiscoverykeys_devpkey.h, which needs more of the Windows headers)
const PROPERTYKEY kFriendlyName{{0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}}, 14};

std::vector<std::pair<std::string, std::string>> output_devices() {
    std::vector<std::pair<std::string, std::string>> out;
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* enumerator = nullptr;
    IMMDeviceCollection* devices = nullptr;
    if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                   (void**)&enumerator)) &&
        SUCCEEDED(enumerator->EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE, &devices))) {
        UINT count = 0;
        devices->GetCount(&count);
        for (UINT i = 0; i < count; ++i) {
            IMMDevice* d = nullptr;
            if (FAILED(devices->Item(i, &d))) continue;
            LPWSTR wid = nullptr;
            IPropertyStore* props = nullptr;
            PROPVARIANT name;
            PropVariantInit(&name);
            if (SUCCEEDED(d->GetId(&wid)) && SUCCEEDED(d->OpenPropertyStore(STGM_READ, &props)) &&
                SUCCEEDED(props->GetValue(kFriendlyName, &name)) && name.vt == VT_LPWSTR) {
                auto utf8 = [](const wchar_t* w) {
                    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
                    std::string s(n > 0 ? n - 1 : 0, '\0');
                    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
                    return s;
                };
                out.emplace_back(utf8(wid), utf8(name.pwszVal));
            }
            PropVariantClear(&name);
            if (props) props->Release();
            if (wid) CoTaskMemFree(wid);
            d->Release();
        }
    }
    if (devices) devices->Release();
    if (enumerator) enumerator->Release();
    return out;
}

}  // namespace zn::audio

namespace AudioCore::Sink {

std::vector<Settings::AudioEngine> GetSinkIDs() { return {Settings::AudioEngine::Auto}; }

std::vector<std::string> GetDeviceListForSink(Settings::AudioEngine, bool) { return {"auto"}; }

std::unique_ptr<Sink> CreateSinkFromID(Settings::AudioEngine, std::string_view) {
    return std::make_unique<WasapiSink>();
}

}  // namespace AudioCore::Sink
