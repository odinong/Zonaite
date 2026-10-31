// Services the game opens but Zonaite has nothing behind: they accept requests and report "nothing there".
#include <memory>

#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

// prepo: play reports (telemetry). Dropped.
class PlayReport : public Service {
public:
    explicit PlayReport(const char* name) : Service(name) {
        for (uint32_t cmd : {10100u, 10101u, 10102u, 10103u, 10104u, 10105u, 10200u, 20100u, 20101u, 20102u, 20103u,
                             20104u, 20105u})
            on(cmd, "SaveReport / RequestImmediateTransmission (dropped)", [](IpcCtx&) {});
        on(10300, "GetTransmissionStatus", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(10400, "GetSystemSessionId", [](IpcCtx& ctx) { ctx.push<uint64_t>(0); });
    }
};

// nfp: amiibo. No reader is attached: initialization works, the device list is empty.
class NfpUser : public Service {
public:
    NfpUser() : Service("nfp:IUser") {
        on(0, "Initialize", [this](IpcCtx&) { state_ = 1; });
        on(1, "Finalize", [this](IpcCtx&) { state_ = 0; });
        on(2, "ListDevices", [](IpcCtx& ctx) { ctx.push<int32_t>(0); });
        on(19, "GetState", [this](IpcCtx& ctx) { ctx.push<uint32_t>(state_); });
        on(23, "AttachAvailabilityChangeEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
    }

private:
    uint32_t state_ = 0;  // 0 = not initialized, 1 = initialized
};

class NfpUserManager : public Service {
public:
    NfpUserManager() : Service("nfp:user") {
        on(0, "CreateUserInterface", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<NfpUser>()); });
    }
};

ServiceRegistrar reg_nfp("nfp:user", [] { return std::make_shared<NfpUserManager>(); });
ServiceRegistrar reg_prepo_u("prepo:u", [] { return std::make_shared<PlayReport>("prepo:u"); });
ServiceRegistrar reg_prepo_a("prepo:a", [] { return std::make_shared<PlayReport>("prepo:a"); });

}  // namespace

}  // namespace zn
