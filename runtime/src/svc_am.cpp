// appletOE: the application side of the applet manager. Zonaite reports a docked console, in focus, at
// 1920x1080, with English as the desired language. Command IDs per switchbrew "Applet Manager services".
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "zonaite/gpu.h"
#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

constexpr uint32_t kResultNoMessage = 0x680;       // am (128), desc 3
constexpr uint32_t kResultNoLaunchParam = 0x480;  // am, desc 2: no data in the channel (PopLaunchParameter)
constexpr uint32_t kResultOutOfBounds = 0x3EE80;    // am, desc 503 (storage access out of range)
constexpr uint32_t kMsgFocusStateChanged = 15;
constexpr uint64_t kAppletResourceUserId = 0x1234;

// "en-US" as a LanguageCode (8 bytes, NUL padded)
constexpr uint64_t kLanguageEnUs = 0x00000053552D6E65ull;

struct AmState {
    std::mutex lock;
    std::deque<uint32_t> messages{kMsgFocusStateChanged};
    std::shared_ptr<KEvent> message_event = [] {
        auto e = std::make_shared<KEvent>();
        e->flag = true;  // the initial FocusStateChanged message is pending
        return e;
    }();
    std::shared_ptr<KEvent> display_resolution_event = std::make_shared<KEvent>();
    // Signaled while the application is in the foreground and may launch applets: from the start, for us
    std::shared_ptr<KEvent> launchable_event = [] {
        auto e = std::make_shared<KEvent>();
        e->flag = true;
        return e;
    }();
    std::shared_ptr<KEvent> generic_event = std::make_shared<KEvent>();
};
AmState& am() {
    static AmState s;
    return s;
}

class CommonStateGetter : public Service {
public:
    CommonStateGetter() : Service("am:ICommonStateGetter") {
        on(0, "GetEventHandle", [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(am().message_event)); });
        on(1, "ReceiveMessage", [](IpcCtx& ctx) {
            std::lock_guard<std::mutex> l(am().lock);
            if (am().messages.empty()) { ctx.result = kResultNoMessage; return; }
            ctx.push<uint32_t>(am().messages.front());
            am().messages.pop_front();
            if (am().messages.empty()) kernel_clear(am().message_event.get());
        });
        on(5, "GetOperationMode", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });           // 1 = console (docked)
        on(6, "GetPerformanceMode", [](IpcCtx& ctx) { ctx.push<uint32_t>(1); });        // boost
        on(8, "GetBootMode", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(9, "GetCurrentFocusState", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });       // in focus
        on(50, "IsVrModeEnabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(60, "GetDefaultDisplayResolution", [](IpcCtx& ctx) {
            ctx.push<int32_t>(1920);
            ctx.push<int32_t>(1080);
        });
        on(61, "GetDefaultDisplayResolutionChangeEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(am().display_resolution_event)); });
        on(66, "SetCpuBoostMode", [](IpcCtx&) {});
        on(90, "GetApplicationIdByContentActionName", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
        on(900, "SetRequestExitToLibraryAppletAtExecuteNextProgramEnabled", [](IpcCtx&) {});
    }
};

class SelfController : public Service {
public:
    SelfController() : Service("am:ISelfController") {
        for (uint32_t cmd : {1u, 2u, 10u, 11u, 12u, 13u, 14u, 16u, 19u, 50u, 62u, 65u, 68u, 100u, 110u, 120u})
            on(cmd, "setting (accepted)", [](IpcCtx&) {});
        on(9, "GetLibraryAppletLaunchableEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(am().launchable_event)); });
        on(40, "CreateManagedDisplayLayer",
           [](IpcCtx& ctx) { ctx.push<uint64_t>(gpu::create_managed_layer(kAppletResourceUserId)); });
        on(44, "CreateManagedDisplaySeparableLayer", [](IpcCtx& ctx) {
            ctx.push<uint64_t>(gpu::create_managed_layer(kAppletResourceUserId));
            ctx.push<uint64_t>(gpu::create_managed_layer(kAppletResourceUserId));  // recording layer
        });
        on(63, "GetIdleTimeDetectionExtension", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(69, "IsAutoSleepDisabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
        on(90, "GetAccumulatedSuspendedTickValue", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
        on(91, "GetAccumulatedSuspendedTickChangedEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(am().generic_event)); });
    }
};

class WindowController : public Service {
public:
    WindowController() : Service("am:IWindowController") {
        on(1, "GetAppletResourceUserId", [](IpcCtx& ctx) { ctx.push<uint64_t>(kAppletResourceUserId); });
        on(10, "AcquireForegroundRights", [](IpcCtx&) {});
    }
};

class AudioController : public Service {
public:
    AudioController() : Service("am:IAudioController") {
        on(0, "SetExpectedMasterVolume", [](IpcCtx&) {});
        on(1, "GetMainAppletExpectedMasterVolume", [](IpcCtx& ctx) { ctx.push<float>(1.0f); });
        on(2, "GetLibraryAppletExpectedMasterVolume", [](IpcCtx& ctx) { ctx.push<float>(1.0f); });
        on(3, "ChangeMainAppletMasterVolume", [](IpcCtx&) {});
        on(4, "SetTransparentVolumeRate", [](IpcCtx&) {});
    }
};

class DisplayController : public Service {
public:
    DisplayController() : Service("am:IDisplayController") {}
};

// A block of data passed between applets (launch parameters).
class StorageAccessor : public Service {
public:
    explicit StorageAccessor(std::shared_ptr<std::vector<uint8_t>> data) : Service("am:IStorageAccessor") {
        on(0, "GetSize", [data](IpcCtx& ctx) { ctx.push<int64_t>((int64_t)data->size()); });
        on(10, "Write", [data](IpcCtx& ctx) {
            const auto offset = (uint64_t)ctx.pop<int64_t>();
            const Buffer in = ctx.in_buffer();
            if (offset > data->size() || in.size > data->size() - offset) {
                ctx.result = kResultOutOfBounds;
                return;
            }
            std::memcpy(data->data() + offset, ctx.m + in.addr, in.size);
        });
        on(11, "Read", [data](IpcCtx& ctx) {
            const auto offset = (uint64_t)ctx.pop<int64_t>();
            const Buffer out = ctx.out_buffer();
            if (offset > data->size() || out.size > data->size() - offset) {
                ctx.result = kResultOutOfBounds;
                return;
            }
            std::memcpy(ctx.m + out.addr, data->data() + offset, out.size);
        });
    }
};

class Storage : public Service {
public:
    explicit Storage(std::vector<uint8_t> bytes) : Service("am:IStorage") {
        auto data = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
        on(0, "Open", [data](IpcCtx& ctx) { ctx.push_object(std::make_shared<StorageAccessor>(data)); });
    }
};

// Launch parameter "preselected user" (kind 2): which account the game runs as. nn::account aborts without it.
std::shared_ptr<Storage> preselected_user_storage() {
    std::vector<uint8_t> d(0x88);
    const uint32_t magic = 0xC79497CA;
    std::memcpy(d.data(), &magic, 4);
    d[4] = 1;  // is_account_selected
    uint64_t uid[2];
    account_current_user(uid);
    std::memcpy(d.data() + 8, uid, 16);
    return std::make_shared<Storage>(std::move(d));
}

// ---- library applets (system UI the game launches: controller setup, keyboard, error display, ...). None of them
// is shown: an applet finishes as soon as it starts, reporting what its user would most likely have chosen.
constexpr uint32_t kAppletController = 0x0C;

class LibraryAppletAccessor : public Service {
public:
    explicit LibraryAppletAccessor(uint32_t applet_id) : Service("am:ILibraryAppletAccessor") {
        on(0, "GetAppletStateChangedEvent",
           [this](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(state_changed)); });
        on(1, "IsCompleted", [this](IpcCtx& ctx) { ctx.push<uint8_t>(completed); });
        on(10, "Start", [this, applet_id](IpcCtx&) {
            log("am: library applet %#x started; it completes at once", applet_id);
            if (applet_id == kAppletController) {
                // ControllerSupportResultInfo: player count, padding, selected npad id (player 1), result
                std::vector<uint8_t> r(0xC, 0);
                r[0] = 1;
                out_data.push_back(std::make_shared<Storage>(std::move(r)));
                kernel_signal(pop_out_data.get());
            }
            completed = true;
            kernel_signal(state_changed.get());
        });
        on(20, "RequestExit", [](IpcCtx&) {});
        on(25, "Terminate", [](IpcCtx&) {});
        on(30, "GetResult", [](IpcCtx&) {});
        on(100, "PushInData", [](IpcCtx&) {});  // the applet's arguments: nothing to read for an applet not shown
        on(101, "PopOutData", [this](IpcCtx& ctx) {
            if (out_data.empty()) {
                ctx.result = kResultNoLaunchParam;
                return;
            }
            ctx.push_object(out_data.front());
            out_data.erase(out_data.begin());
        });
        on(103, "PushInteractiveInData", [](IpcCtx&) {});
        on(104, "PopInteractiveOutData", [](IpcCtx& ctx) { ctx.result = kResultNoLaunchParam; });
        on(105, "GetPopOutDataEvent", [this](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(pop_out_data)); });
        on(106, "GetPopInteractiveOutDataEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(150, "RequestForAppletToGetForeground", [](IpcCtx&) {});
    }

private:
    std::shared_ptr<KEvent> state_changed = std::make_shared<KEvent>();
    std::shared_ptr<KEvent> pop_out_data = std::make_shared<KEvent>();
    std::vector<std::shared_ptr<Storage>> out_data;
    bool completed = false;
};

class LibraryAppletCreator : public Service {
public:
    LibraryAppletCreator() : Service("am:ILibraryAppletCreator") {
        on(0, "CreateLibraryApplet", [](IpcCtx& ctx) {
            const uint32_t applet_id = ctx.pop<uint32_t>();
            ctx.push_object(std::make_shared<LibraryAppletAccessor>(applet_id));
        });
        on(10, "CreateStorage", [](IpcCtx& ctx) {
            ctx.push_object(std::make_shared<Storage>(std::vector<uint8_t>((size_t)ctx.pop<int64_t>())));
        });
    }
};

class ApplicationFunctions : public Service {
public:
    ApplicationFunctions() : Service("am:IApplicationFunctions") {
        on(1, "PopLaunchParameter", [](IpcCtx& ctx) {
            constexpr uint32_t kPreselectedUser = 2;  // 1 is the user channel
            static std::atomic<bool> user_popped{false};
            if (ctx.pop<uint32_t>() == kPreselectedUser && !user_popped.exchange(true))
                ctx.push_object(preselected_user_storage());
            else
                ctx.result = kResultNoLaunchParam;
        });
        on(20, "EnsureSaveData", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
        on(21, "GetDesiredLanguage", [](IpcCtx& ctx) { ctx.push<uint64_t>(kLanguageEnUs); });
        on(22, "SetTerminateResult", [](IpcCtx&) {});
        on(27, "CreateCacheStorage", [](IpcCtx& ctx) {
            ctx.pop<uint16_t>();  // index (0)
            const auto size = ctx.pop<int64_t>();
            const auto journal = ctx.pop<int64_t>();
            fs_create_cache_storage(size, journal);
            ctx.push<uint32_t>(1);  // target storage: NAND
            ctx.push<uint64_t>(0);  // required size: none, it fit
        });
        on(40, "NotifyRunning", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(50, "GetPseudoDeviceId", [](IpcCtx& ctx) { ctx.push<uint64_t>(0x5A4F4E41ull); ctx.push<uint64_t>(1); });
        on(66, "InitializeGamePlayRecording", [](IpcCtx&) {});
        on(67, "SetGamePlayRecordingState", [](IpcCtx&) {});
        on(90, "EnableApplicationCrashReport", [](IpcCtx&) {});
        on(100, "InitializeApplicationCopyrightFrameBuffer", [](IpcCtx&) {});
        on(101, "SetApplicationCopyrightImage", [](IpcCtx&) {});
        on(102, "SetApplicationCopyrightVisibility", [](IpcCtx&) {});
        on(123, "GetPreviousProgramIndex", [](IpcCtx& ctx) { ctx.push<int32_t>(-1); });
        on(130, "GetGpuErrorDetectedSystemEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(140, "GetFriendInvitationStorageChannelEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(150, "GetNotificationStorageChannelEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(160, "GetHealthWarningDisappearedSystemEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
    }
};

class ApplicationProxy : public Service {
public:
    ApplicationProxy() : Service("am:IApplicationProxy") {
        on(0, "GetCommonStateGetter", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<CommonStateGetter>()); });
        on(1, "GetSelfController", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<SelfController>()); });
        on(2, "GetWindowController", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<WindowController>()); });
        on(3, "GetAudioController", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<AudioController>()); });
        on(4, "GetDisplayController", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<DisplayController>()); });
        on(11, "GetLibraryAppletCreator",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<LibraryAppletCreator>()); });
        on(20, "GetApplicationFunctions",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ApplicationFunctions>()); });
        on(1000, "GetDebugFunctions",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<Service>("am:IDebugFunctions")); });
    }
};

class AppletOE : public Service {
public:
    AppletOE() : Service("appletOE") {
        on(0, "OpenApplicationProxy", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ApplicationProxy>()); });
    }
};

ServiceRegistrar reg("appletOE", [] { return std::make_shared<AppletOE>(); });

}  // namespace
}  // namespace zn
