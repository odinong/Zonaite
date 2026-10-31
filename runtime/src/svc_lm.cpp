// lm: log manager. The SDK sends its log packets here; we print their text to the Zonaite log.
#include <string>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

class Logger : public Service {
public:
    Logger() : Service("lm:ILogger") {
        on(0, "Log", [](IpcCtx& ctx) {
            Buffer b = ctx.in_buffer();
            // Log packets are a header plus TLV chunks; print the printable runs, which carry the message text.
            std::string text, cur;
            for (uint64_t i = 0; i < b.size && i < 0x1000; i++) {
                char ch = (char)ctx.m[b.addr + i];
                if (ch >= 0x20 && ch < 0x7F) cur += ch;
                else {
                    if (cur.size() >= 4) text += (text.empty() ? "" : " | ") + cur;
                    cur.clear();
                }
            }
            if (cur.size() >= 4) text += (text.empty() ? "" : " | ") + cur;
            if (!text.empty()) log("guest log: %s", text.c_str());
        });
        on(1, "SetDestination", [](IpcCtx&) {});
    }
};

class LogManager : public Service {
public:
    LogManager() : Service("lm") {
        on(0, "OpenLogger", [](IpcCtx& ctx) { ctx.push_object(std::make_shared<Logger>()); });
    }
};

ServiceRegistrar reg("lm", [] { return std::make_shared<LogManager>(); });

}  // namespace
}  // namespace zn
