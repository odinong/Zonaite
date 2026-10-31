// aoc:u: add-on content (DLC). Zonaite runs the base game with update only: no add-on content.
#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

class PurchaseEventManager : public Service {
public:
    PurchaseEventManager() : Service("aoc:IPurchaseEventManager") {
        on(0, "SetDefaultDeliveryTarget", [](IpcCtx&) {});
        on(1, "SetDeliveryTarget", [](IpcCtx&) {});
        on(2, "GetPurchasedEventReadableHandle",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(3, "PopPurchasedProductInfo", [](IpcCtx& ctx) { ctx.result = 0x1A3C;  /* no info: aoc desc 0x1E */ });
    }
};

class AddOnContentManager : public Service {
public:
    AddOnContentManager() : Service("aoc:u") {
        on(2, "CountAddOnContent", [](IpcCtx& ctx) { ctx.push<int32_t>(0); });
        on(3, "ListAddOnContent", [](IpcCtx& ctx) { ctx.push<int32_t>(0); });
        on(5, "GetAddOnContentBaseId", [](IpcCtx& ctx) { ctx.push<uint64_t>(0x0100F2C0115B7000ull); });
        on(7, "PrepareAddOnContent", [](IpcCtx&) {});
        on(8, "GetAddOnContentListChangedEvent",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(10, "GetAddOnContentListChangedEventWithProcessId",
           [](IpcCtx& ctx) { ctx.push_copy(kernel_add_handle(std::make_shared<KEvent>())); });
        on(11, "NotifyMountAddOnContent", [](IpcCtx&) {});
        on(12, "NotifyUnmountAddOnContent", [](IpcCtx&) {});
        on(50, "CheckAddOnContentMountStatus", [](IpcCtx&) {});
        on(100, "CreateEcPurchasedEventManager",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<PurchaseEventManager>()); });
        on(101, "CreatePermanentEcPurchasedEventManager",
           [](IpcCtx& ctx) { ctx.push_object(std::make_shared<PurchaseEventManager>()); });
    }
};

ServiceRegistrar reg("aoc:u", [] { return std::make_shared<AddOnContentManager>(); });

}  // namespace
}  // namespace zn
