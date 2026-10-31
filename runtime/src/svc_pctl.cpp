// pctl: parental controls. Nothing is restricted, nothing is paired, the play timer is off.
#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

void event(IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); }

class ParentalControl : public Service {
public:
    ParentalControl() : Service("pctl:IParentalControlService") {
        for (uint32_t cmd : {1u, 1001u, 1002u, 1003u, 1004u, 1013u, 1016u, 1017u, 1064u, 1451u, 1452u, 1471u, 1472u,
                             1474u})
            on(cmd, "accepted", [](IpcCtx&) {});
        on(1006, "IsRestrictionTemporaryUnlocked", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(1018, "IsFreeCommunicationAvailable", [](IpcCtx&) {});
        on(1031, "IsRestrictionEnabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(1032, "GetSafetyLevel", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
        on(1061, "ConfirmStereoVisionRestrictionConfigurable", [](IpcCtx&) {});
        on(1065, "IsStereoVisionPermitted", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
        on(1403, "IsPairingActive", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(1432, "GetSynchronizationEvent", event);
        on(1453, "IsPlayTimerEnabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(1454, "GetPlayTimerRemainingTime", [](IpcCtx& ctx) { ctx.push<int64_t>(0); });
        on(1455, "IsRestrictedByPlayTimer", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
        on(1457, "GetPlayTimerEventToRequestSuspension", event);
        on(1458, "IsPlayTimerAlarmDisabled", [](IpcCtx& ctx) { ctx.push<uint8_t>(1); });
        on(1473, "GetUnlinkedEvent", event);
    }
};

class ParentalControlFactory : public Service {
public:
    explicit ParentalControlFactory(const char* name) : Service(name) {
        on(0, "CreateService", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ParentalControl>()); });
        on(1, "CreateServiceWithoutInitialize",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ParentalControl>()); });
    }
};

ServiceRegistrar reg("pctl", [] { return std::make_shared<ParentalControlFactory>("pctl"); });
ServiceRegistrar reg_a("pctl:a", [] { return std::make_shared<ParentalControlFactory>("pctl:a"); });
ServiceRegistrar reg_s("pctl:s", [] { return std::make_shared<ParentalControlFactory>("pctl:s"); });
ServiceRegistrar reg_r("pctl:r", [] { return std::make_shared<ParentalControlFactory>("pctl:r"); });

}  // namespace
}  // namespace zn
