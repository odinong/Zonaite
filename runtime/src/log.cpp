// Logging and fatal error reports (registers + guest backtrace through the frame-pointer chain).
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "zonaite/guest.h"
#include "zonaite/runtime.h"

namespace zg {
struct FnEntry { uint64_t addr; GuestFn fn; };
extern const FnEntry kFunctions[];
extern const unsigned kFunctionCount;
}  // namespace zg

namespace zn {

using namespace zg;

static FILE* log_file() {
    static FILE* f = std::fopen("zonaite.log", "w");
    return f;
}

void log(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[zonaite] %s\n", buf);
    if (FILE* f = log_file()) { std::fprintf(f, "%s\n", buf); std::fflush(f); }
}

static uint64_t function_of(uint64_t addr) {
    const FnEntry* end = kFunctions + kFunctionCount;
    const FnEntry* it = std::upper_bound(kFunctions, end, addr, [](uint64_t a, const FnEntry& e) { return a < e.addr; });
    return it == kFunctions ? 0 : (it - 1)->addr;
}

static const char* module_of(uint64_t addr) {
    for (int i = 0; i < g_image.module_count; i++)
        if (addr >= g_image.modules[i].base && addr < g_image.modules[i].base + g_image.modules[i].size)
            return g_image.modules[i].name.c_str();
    return "?";
}

void fatal(Ctx* c, const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log("FATAL: %s", buf);
    if (c) log_guest_state(c);
    std::exit(3);
}

void log_guest_state(Ctx* c, bool registers) {
    if (registers)
        for (int i = 0; i < 31; i += 4)
            log("  x%-2d %016llx  x%-2d %016llx  x%-2d %016llx  x%-2d %016llx", i, (unsigned long long)c->x[i],
                i + 1, (unsigned long long)c->x[i + 1], i + 2, (unsigned long long)c->x[i + 2], i + 3,
                (unsigned long long)(i + 3 < 31 ? c->x[i + 3] : c->sp));
    log("  sp %016llx  lr %016llx (%s fn %#llx)", (unsigned long long)c->sp, (unsigned long long)c->x[30],
        module_of(c->x[30]), (unsigned long long)function_of(c->x[30]));
    // guest backtrace: [fp] = previous fp, [fp + 8] = return address
    uint64_t fp = c->x[29];
    for (int depth = 0; depth < 32 && fp && fp < kGuestSpace && !(fp & 7); depth++) {
        uint64_t ret = ld64(g_mem, fp + 8), prev = ld64(g_mem, fp);
        if (!ret) break;
        log("  #%-2d %016llx  %s fn %#llx", depth, (unsigned long long)ret, module_of(ret),
            (unsigned long long)function_of(ret));
        if (prev <= fp) break;
        fp = prev;
    }
}

}  // namespace zn
