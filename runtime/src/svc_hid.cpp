// hid: input. The game reads controller state from a 0x40000-byte shared memory block that hid keeps up to date;
// the commands below are mostly configuration the game sets once. Controller state is written by
// runtime/gpu/input.cpp.
#include <algorithm>
#include <cstring>
#include <memory>

#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

constexpr uint64_t kSharedMemorySize = 0x40000;

struct HidState {
    std::shared_ptr<KSharedMemory> shared = create_shared_memory(kSharedMemorySize);
    std::shared_ptr<KEvent> style_set_event = std::make_shared<KEvent>();  // per npad on hardware; one is enough
    uint32_t supported_style_set = 0;
    int64_t joy_hold_type = 0;
};

HidState& hid() {
    static HidState s = [] {
        HidState h;
        input_start(h.shared->host);
        return h;
    }();
    return s;
}

// nn::hid::VibrationValue: the two frequency bands of HD rumble
struct VibrationValue {
    float amp_low, freq_low, amp_high, freq_high;
};

// A VibrationDeviceHandle names the controller and which of its two motors (byte 2: 0 left, 1 right)
void vibrate(uint32_t handle, const VibrationValue& v) {
    // (the game streams values for both motors all the time, idle ones at amplitude 0)
    static bool felt = false;
    if (!felt && (v.amp_low > 0 || v.amp_high > 0)) {
        felt = true;
        log("hid: the game's first vibration (handle %08x, %.2f / %.2f)", handle, v.amp_low, v.amp_high);
    }
    const int side = (handle >> 16) & 0xff;
    if (side <= 1) input_vibrate(side, v.amp_low, v.amp_high);
}

class AppletResource : public Service {
public:
    AppletResource() : Service("hid:IAppletResource") {
        on(0, "GetSharedMemoryHandle", [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(hid().shared)); });
    }
};

class ActiveVibrationDeviceList : public Service {
public:
    ActiveVibrationDeviceList() : Service("hid:IActiveVibrationDeviceList") {
        on(0, "ActivateVibrationDevice", [](IpcCtx&) {});
    }
};

class HidServer : public Service {
public:
    HidServer() : Service("hid") {
        on(0, "CreateAppletResource", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<AppletResource>()); });
        // activation and sensor settings: accepted, nothing to do until input is emulated
        for (uint32_t cmd : {1u, 11u, 21u, 31u, 66u, 67u, 69u, 70u, 72u, 79u, 91u, 103u, 104u, 107u, 109u, 122u, 123u, 124u,
                             125u, 128u, 129u, 130u, 133u, 300u, 301u, 302u, 1000u})
            on(cmd, "setting (accepted)", [](IpcCtx&) {});
        on(68, "IsSixAxisSensorFusionEnabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
        on(82, "IsSixAxisSensorAtRest", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
        on(100, "SetSupportedNpadStyleSet", [](IpcCtx& ctx) { hid().supported_style_set = ctx.pop<uint32_t>(); });
        on(101, "GetSupportedNpadStyleSet", [](IpcCtx& ctx) { ctx.push<uint32_t>(hid().supported_style_set); });
        on(102, "SetSupportedNpadIdType", [](IpcCtx&) {});
        on(106, "AcquireNpadStyleSetUpdateEventHandle",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(hid().style_set_event)); });
        on(108, "GetPlayerLedPattern", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
        on(120, "SetNpadJoyHoldType", [](IpcCtx& ctx) {
            ctx.pop<uint64_t>();  // aruid
            hid().joy_hold_type = ctx.pop<int64_t>();
        });
        on(121, "GetNpadJoyHoldType", [](IpcCtx& ctx) { ctx.push<int64_t>(hid().joy_hold_type); });
        on(200, "GetVibrationDeviceInfo", [](IpcCtx& ctx) {
            ctx.push<uint32_t>(1);  // device type: linear resonant actuator
            ctx.push<uint32_t>(0);  // position: none
        });
        on(201, "SendVibrationValue", [](IpcCtx& ctx) {
            const uint32_t handle = ctx.pop<uint32_t>();
            vibrate(handle, ctx.pop<VibrationValue>());
        });
        on(203, "CreateActiveVibrationDeviceList",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ActiveVibrationDeviceList>()); });
        on(206, "SendVibrationValues", [](IpcCtx& ctx) {
            // handles and values in two input buffers, in step
            const Buffer handles = ctx.in_buffer(0), values = ctx.in_buffer(1);
            const size_t n = std::min<size_t>(handles.size / sizeof(uint32_t), values.size / sizeof(VibrationValue));
            for (size_t i = 0; i < n; ++i) {
                uint32_t handle;
                VibrationValue v;
                std::memcpy(&handle, ctx.m + handles.addr + i * sizeof handle, sizeof handle);
                std::memcpy(&v, ctx.m + values.addr + i * sizeof v, sizeof v);
                vibrate(handle, v);
            }
        });
        on(211, "IsVibrationDeviceMounted", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
    }
};

ServiceRegistrar reg("hid", [] { return std::make_shared<HidServer>(); });

}  // namespace

void hid_signal_style_set_update() { kernel_signal(hid().style_set_event.get()); }

}  // namespace zn
