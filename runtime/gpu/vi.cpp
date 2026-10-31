// vi: displays, layers and vsync. Eden's nvnflinger (buffer queues, SurfaceFlinger, HardwareComposer; compiled
// unmodified against runtime/gpu/override) does the buffer handling and composition; this file holds the display
// and layer bookkeeping (Eden's vi::Container), the vsync thread that composes and paces frames (Eden's Conductor),
// and the IPC front ends the game talks to (vi:m/vi:s/vi:u, IApplicationDisplayService, IHOSBinderDriver).
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "common/windows/timer_resolution.h"
#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/nvnflinger/binder.h"
#include "core/hle/service/nvnflinger/hos_binder_driver_server.h"
#include "core/hle/service/nvnflinger/hwc_layer.h"
#include "core/hle/service/nvnflinger/parcel.h"
#include "core/hle/service/nvnflinger/surface_flinger.h"
#include "core/hle/service/vi/vi_types.h"

#include "zonaite/gpu.h"
#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

namespace VI = ::Service::VI;
namespace Nvnflinger = ::Service::Nvnflinger;

constexpr uint32_t kViNotFound = 114 | 7 << 9;
constexpr uint32_t kViOperationFailed = 114 | 1 << 9;
constexpr const char* kDisplayNames[] = {"Default", "External", "Edid", "Internal", "Null"};

struct ViLayer {
    u64 display_id = 0;
    s32 consumer_id = 0, producer_id = 0;
    u64 owner_aruid = 0;
    bool open = false;
};

// Display and layer state (Eden's vi::Container), the compositor, and the vsync thread.
struct Vi {
    std::mutex lock;
    Nvnflinger::HosBinderDriverServer server;
    std::unique_ptr<Nvnflinger::SurfaceFlinger> flinger;
    std::map<u64, ViLayer> layers;
    u64 next_layer_id = 1;
    std::map<u64, std::vector<std::unique_ptr<Kernel::KEvent>>> vsync_events;  // by display
    std::thread vsync_thread;

    Vi() {
        flinger = std::make_unique<Nvnflinger::SurfaceFlinger>(gpu::system(), server);
        for (u64 id = 0; id < std::size(kDisplayNames); id++) flinger->AddDisplay(id);
        vsync_thread = std::thread([this] { vsync_loop(); });
    }

    static std::optional<u64> display_by_name(const VI::DisplayName& name) {
        for (u64 id = 0; id < std::size(kDisplayNames); id++)
            if (std::strncmp(name.data(), kDisplayNames[id], name.size()) == 0) return id;
        return std::nullopt;
    }

    uint32_t create_layer(u64* out_layer_id, u64 display_id, u64 aruid) {
        if (display_id >= std::size(kDisplayNames)) return kViNotFound;
        ViLayer l{display_id, 0, 0, aruid, false};
        flinger->CreateBufferQueue(&l.consumer_id, &l.producer_id);
        flinger->CreateLayer(l.consumer_id);
        *out_layer_id = next_layer_id++;
        layers[*out_layer_id] = l;
        return 0;
    }

    uint32_t open_layer(s32* out_producer_id, u64 layer_id, u64 aruid) {
        auto it = layers.find(layer_id);
        if (it == layers.end()) return kViNotFound;
        if (it->second.open || it->second.owner_aruid != aruid) return kViOperationFailed;
        it->second.open = true;
        flinger->AddLayerToDisplayStack(it->second.display_id, it->second.consumer_id);
        *out_producer_id = it->second.producer_id;
        return 0;
    }

    uint32_t close_layer(u64 layer_id) {
        auto it = layers.find(layer_id);
        if (it == layers.end()) return kViNotFound;
        if (!it->second.open) return kViOperationFailed;
        flinger->RemoveLayerFromDisplayStack(it->second.display_id, it->second.consumer_id);
        it->second.open = false;
        return 0;
    }

    uint32_t destroy_layer(u64 layer_id) {
        auto it = layers.find(layer_id);
        if (it == layers.end()) return kViNotFound;
        flinger->DestroyLayer(it->second.consumer_id);
        flinger->DestroyBufferQueue(it->second.consumer_id, it->second.producer_id);
        layers.erase(it);
        return 0;
    }

    ViLayer* find_layer(u64 layer_id) {
        auto it = layers.find(layer_id);
        return it == layers.end() ? nullptr : &it->second;
    }

    Kernel::KEvent* new_vsync_event(u64 display_id) {
        auto& list = vsync_events[display_id];
        list.push_back(std::make_unique<Kernel::KEvent>("Display VSync"));
        return list.back().get();
    }

    // Compose every display once per vsync (60 Hz), then signal the displays' vsync events. The composer reports
    // the swap interval of the frame it presented (TotK: 2, i.e. 30 fps); the next vsync is that many 60 Hz
    // periods later, like the console's display.
    void vsync_loop() {
        SetThreadDescription(GetCurrentThread(), L"zonaite vsync");
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        Common::Windows::SetCurrentTimerResolutionToMaximum();
        using clock = std::chrono::steady_clock;
        constexpr auto kPeriod = std::chrono::nanoseconds(1'000'000'000 / 60);
        auto next = clock::now() + kPeriod;
        for (;;) {
            // sleep to ~1 ms before the deadline, then spin: Windows sleeps overshoot by up to a timer tick
            if (const auto ahead = next - clock::now(); ahead > std::chrono::milliseconds(1))
                std::this_thread::sleep_for(ahead - std::chrono::milliseconds(1));
            while (clock::now() < next) YieldProcessor();
            s32 swap_interval = 1;
            {
                std::lock_guard l(lock);
                for (u64 id = 0; id < std::size(kDisplayNames); id++) {
                    s32 interval = 1;
                    f32 speed_scale = 1.f;
                    const auto t0 = clock::now();
                    if (flinger->ComposeDisplay(&interval, &speed_scale, id)) swap_interval = interval;
                    if (const auto took = clock::now() - t0; took > std::chrono::milliseconds(100))
                        log("vi: composing display %llu took %lld ms", (unsigned long long)id,
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(took).count());
                    for (auto& e : vsync_events[id]) e->Signal();
                }
            }
            next += kPeriod * std::max<s32>(1, swap_interval);
            if (clock::now() > next + kPeriod * 4) next = clock::now() + kPeriod;  // fell far behind: resync
        }
    }
};

Vi& vi() {
    static Vi v;
    return v;
}

// IHOSBinderDriver: the game's Android-style binder transactions with the buffer queues.
class HosBinderDriver : public Service {
public:
    HosBinderDriver() : Service("vi:IHOSBinderDriver") {
        auto transact = [](IpcCtx& ctx) {
            const auto id = ctx.pop<s32>();
            const auto code = ctx.pop<u32>();
            const auto flags = ctx.pop<u32>();
            const Buffer in = ctx.in_buffer(), out = ctx.out_buffer();
            const auto binder = vi().server.TryGetBinder(id);
            if (!binder) return;
            binder->Transact(code, std::span<const u8>(ctx.m + in.addr, in.size), std::span<u8>(ctx.m + out.addr, out.size),
                             flags);
        };
        on(0, "TransactParcel", transact);
        on(3, "TransactParcelAuto", transact);
        on(1, "AdjustRefcount", [](IpcCtx& ctx) {
            const auto id = ctx.pop<s32>();
            const auto delta = ctx.pop<s32>();
            const auto type = ctx.pop<s32>();
            if (type != 0 && type != 1) {
                ctx.result = 0x1;  // ResultUnknown
                return;
            }
            vi().server.AdjustRefcount(id, delta, type == 1);
        });
        on(2, "GetNativeHandle", [](IpcCtx& ctx) {
            const auto id = ctx.pop<s32>();
            const auto type = ctx.pop<u32>();
            const auto binder = vi().server.TryGetBinder(id);
            Kernel::KReadableEvent* event = binder ? binder->GetNativeHandle(type) : nullptr;
            if (!event) {
                ctx.result = 0x1;
                return;
            }
            ctx.push_copy(event->CreateHandle());
        });
    }
};

class SystemDisplayService : public Service {
public:
    SystemDisplayService() : Service("vi:ISystemDisplayService") {
        on(1200, "GetZOrderCountMin", [](IpcCtx& ctx) { ctx.push<u64>(0); });
        on(1202, "GetZOrderCountMax", [](IpcCtx& ctx) { ctx.push<u64>(255); });
        on(2205, "SetLayerZ", [](IpcCtx& ctx) {
            const auto layer_id = ctx.pop<u64>();
            const auto z = ctx.pop<s64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            if (ViLayer* layer = v.find_layer(layer_id))
                if (auto hwc = v.flinger->FindLayer(layer->consumer_id)) hwc->z_index = static_cast<s32>(z);
        });
        on(2207, "SetLayerVisibility", [](IpcCtx& ctx) {
            const bool visible = ctx.pop<u8>() != 0;
            const auto layer_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            if (ViLayer* layer = v.find_layer(layer_id)) v.flinger->SetLayerVisibility(layer->consumer_id, visible);
        });
        on(3200, "GetDisplayMode", [](IpcCtx& ctx) {
            ctx.push(VI::DisplayMode{1920, 1080, 60.f, 0});
        });
    }
};

class ManagerDisplayService : public Service {
public:
    ManagerDisplayService() : Service("vi:IManagerDisplayService") {
        on(2010, "CreateManagedLayer", [](IpcCtx& ctx) {
            ctx.pop<u32>();  // flags
            const auto display_id = ctx.pop<u64>();
            const auto aruid = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            u64 layer_id = 0;
            ctx.result = v.create_layer(&layer_id, display_id, aruid);
            ctx.push<u64>(layer_id);
        });
        on(2011, "DestroyManagedLayer", [](IpcCtx& ctx) {
            const auto layer_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            v.close_layer(layer_id);
            ctx.result = v.destroy_layer(layer_id);
        });
        on(6000, "AddToLayerStack", [](IpcCtx&) {});
        on(6002, "SetLayerVisibility", [](IpcCtx& ctx) {
            const bool visible = ctx.pop<u8>() != 0;
            const auto layer_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            if (ViLayer* layer = v.find_layer(layer_id)) v.flinger->SetLayerVisibility(layer->consumer_id, visible);
        });
    }
};

// The parcel a layer's native window is described by: the producer's binder id, read by the game's nvn.
u64 write_native_window(IpcCtx& ctx, s32 producer_id) {
    ::Service::android::OutputParcel parcel;
    parcel.WriteInterface(VI::NativeWindow{producer_id});
    const auto data = parcel.Serialize();
    const Buffer out = ctx.out_buffer();
    std::memcpy(ctx.m + out.addr, data.data(), std::min<size_t>(out.size, data.size()));
    return data.size();
}

class ApplicationDisplayService : public Service {
public:
    ApplicationDisplayService() : Service("vi:IApplicationDisplayService") {
        on(100, "GetRelayService", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<HosBinderDriver>()); });
        on(101, "GetSystemDisplayService",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SystemDisplayService>()); });
        on(102, "GetManagerDisplayService",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ManagerDisplayService>()); });
        on(103, "GetIndirectDisplayTransactionService",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<HosBinderDriver>()); });
        on(1000, "ListDisplays", [](IpcCtx& ctx) {
            const Buffer out = ctx.out_buffer();
            if (out.size >= sizeof(VI::DisplayInfo)) {
                const VI::DisplayInfo info{};
                std::memcpy(ctx.m + out.addr, &info, sizeof info);
                ctx.push<u64>(1);
            } else {
                ctx.push<u64>(0);
            }
        });
        on(1010, "OpenDisplay", [](IpcCtx& ctx) {
            const auto name = ctx.pop<VI::DisplayName>();
            const auto id = Vi::display_by_name(name);
            if (!id) {
                ctx.result = kViNotFound;
                return;
            }
            vi();  // displays and the compositor come up with the first display
            ctx.push<u64>(*id);
        });
        on(1011, "OpenDefaultDisplay", [](IpcCtx& ctx) {
            vi();
            ctx.push<u64>(0);
        });
        on(1020, "CloseDisplay", [](IpcCtx&) {});
        on(1101, "SetDisplayEnabled", [](IpcCtx&) {});
        on(1102, "GetDisplayResolution", [](IpcCtx& ctx) {  // docked
            ctx.push<s64>(1920);
            ctx.push<s64>(1080);
        });
        on(2020, "OpenLayer", [](IpcCtx& ctx) {
            ctx.pop<VI::DisplayName>();
            const auto layer_id = ctx.pop<u64>();
            const auto aruid = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            s32 producer_id = 0;
            // The game opens the layer am created for it; its aruid is whatever the game was told by am.
            ViLayer* layer = v.find_layer(layer_id);
            ctx.result = v.open_layer(&producer_id, layer_id, layer ? layer->owner_aruid : aruid);
            ctx.push<u64>(ctx.result ? 0 : write_native_window(ctx, producer_id));
        });
        on(2021, "CloseLayer", [](IpcCtx& ctx) {
            const auto layer_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            ctx.result = v.close_layer(layer_id);
        });
        on(2030, "CreateStrayLayer", [](IpcCtx& ctx) {
            ctx.pop<u32>();  // flags
            const auto display_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            u64 layer_id = 0;
            s32 producer_id = 0;
            ctx.result = v.create_layer(&layer_id, display_id, 0);
            if (!ctx.result) ctx.result = v.open_layer(&producer_id, layer_id, 0);
            ctx.push<u64>(layer_id);
            ctx.push<u64>(ctx.result ? 0 : write_native_window(ctx, producer_id));
        });
        on(2031, "DestroyStrayLayer", [](IpcCtx& ctx) {
            const auto layer_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            v.close_layer(layer_id);
            ctx.result = v.destroy_layer(layer_id);
        });
        on(2101, "SetLayerScalingMode", [](IpcCtx&) {});
        on(2102, "ConvertScalingMode", [](IpcCtx& ctx) {
            using N = VI::NintendoScaleMode;
            using C = VI::ConvertedScaleMode;
            switch (static_cast<N>(ctx.pop<u32>())) {
            case N::None: ctx.push(C::None); break;
            case N::Freeze: ctx.push(C::Freeze); break;
            case N::ScaleToWindow: ctx.push(C::ScaleToWindow); break;
            case N::ScaleAndCrop: ctx.push(C::ScaleAndCrop); break;
            case N::PreserveAspectRatio: ctx.push(C::PreserveAspectRatio); break;
            default: ctx.result = kViOperationFailed; break;
            }
        });
        on(5202, "GetDisplayVsyncEvent", [](IpcCtx& ctx) {
            const auto display_id = ctx.pop<u64>();
            auto& v = vi();
            std::lock_guard l(v.lock);
            ctx.push_copy(v.new_vsync_event(display_id)->CreateHandle());
        });
    }
};

class ViRoot : public Service {
public:
    ViRoot(const char* name, uint32_t get_display_service_cmd) : Service(name) {
        on(get_display_service_cmd, "GetDisplayService",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ApplicationDisplayService>()); });
    }
};

ServiceRegistrar reg_u("vi:u", [] { return std::make_shared<ViRoot>("vi:u", 0); });
ServiceRegistrar reg_s("vi:s", [] { return std::make_shared<ViRoot>("vi:s", 1); });
ServiceRegistrar reg_m("vi:m", [] { return std::make_shared<ViRoot>("vi:m", 2); });

}  // namespace

uint64_t gpu::create_managed_layer(uint64_t aruid) {
    auto& v = vi();
    std::lock_guard l(v.lock);
    u64 layer_id = 0;
    v.create_layer(&layer_id, 0, aruid);
    return layer_id;
}

}  // namespace zn
