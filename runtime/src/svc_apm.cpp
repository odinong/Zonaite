// apm: performance modes. Zonaite always reports docked ("Boost") mode, so the game uses its docked settings.
#include <mutex>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

constexpr uint32_t kPerformanceModeBoost = 1;  // docked
uint32_t g_config[2] = {0x00010000, 0x92220008};  // last configuration set per mode (normal, boost)
std::mutex g_lock;

class ApmSession : public Service {
public:
    ApmSession() : Service("apm:ISession") {
        on(0, "SetPerformanceConfiguration", [](IpcCtx& ctx) {
            uint32_t mode = ctx.pop<uint32_t>(), config = ctx.pop<uint32_t>();
            std::lock_guard<std::mutex> l(g_lock);
            if (mode < 2) g_config[mode] = config;
        });
        on(1, "GetPerformanceConfiguration", [](IpcCtx& ctx) {
            uint32_t mode = ctx.pop<uint32_t>();
            std::lock_guard<std::mutex> l(g_lock);
            ctx.push<uint32_t>(mode < 2 ? g_config[mode] : 0);
        });
        on(2, "SetCpuOverclockEnabled", [](IpcCtx&) {});
    }
};

class Apm : public Service {
public:
    explicit Apm(const char* name) : Service(name) {
        on(0, "OpenSession", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<ApmSession>()); });
        on(1, "GetPerformanceMode", [](IpcCtx& ctx) { ctx.push<uint32_t>(kPerformanceModeBoost); });
        on(6, "IsEnabledSystemModeCheck", [](IpcCtx& ctx) { ctx.push<uint8_t>(0); });
    }
};

ServiceRegistrar reg("apm", [] { return std::make_shared<Apm>("apm"); });
ServiceRegistrar reg_am("apm:am", [] { return std::make_shared<Apm>("apm:am"); });

}  // namespace
}  // namespace zn
