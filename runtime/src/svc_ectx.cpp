// ectx:aw: error context writer. The SDK registers error contexts for crash reports; we accept and discard them.
#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

class ContextRegistrar : public Service {
public:
    ContextRegistrar() : Service("ectx:IContextRegistrar") {
        on(0, "Complete", [](IpcCtx& ctx) { ctx.push<uint32_t>(0); });
    }
};

class ErrorContextWriter : public Service {
public:
    ErrorContextWriter() : Service("ectx:aw") {
        on(0, "CreateContextRegistrar", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ContextRegistrar>()); });
    }
};

ServiceRegistrar reg("ectx:aw", [] { return std::make_shared<ErrorContextWriter>(); });

}  // namespace
}  // namespace zn
