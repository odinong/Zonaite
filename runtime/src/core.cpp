// Runtime core: guest address space, function lookup and indirect dispatch, traps.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "zonaite/guest.h"
#include "zonaite/gpu.h"
#include "zonaite/kernel.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace zg {
struct FnEntry { uint64_t addr; GuestFn fn; };
extern const FnEntry kFunctions[];
extern const unsigned kFunctionCount;
}  // namespace zg

namespace zn {

using namespace zg;

uint8_t* g_mem = nullptr;

// Guest memory is created by the GPU layer (Eden HostMemory: 8 GB DRAM backing mapped 1:1 into the 39-bit
// guest space), so the GPU can use the backing view while the guest view gets write-protected for tracking.
uint8_t* reserve_guest_space() {
    g_mem = gpu::create_system();
    if (!g_mem) {
        std::fprintf(stderr, "zonaite: could not create guest memory\n");
        std::exit(1);
    }
    return g_mem;
}

// The DRAM backing is committed as a whole; commit() only checks that guest memory stays inside it.
bool commit(uint64_t guest, uint64_t size, bool executable_data) {
    (void)executable_data;
    if (guest + size > kDramSize) fatal(nullptr, "guest memory at %#llx (+%#llx) is outside the 8 GB DRAM",
                                         (unsigned long long)guest, (unsigned long long)size);
    return true;
}

// ---- host functions callable through guest addresses (NVN entry points, callbacks)
static std::mutex g_thunk_lock;
static std::vector<HostFn> g_thunks;

uint64_t register_host_function(HostFn fn) {
    std::lock_guard<std::mutex> l(g_thunk_lock);
    g_thunks.push_back(fn);
    return kThunkBase + 16 * (g_thunks.size() - 1);
}

// ---- function lookup
GuestFn find_function(uint64_t addr) {
    const FnEntry* begin = kFunctions;
    const FnEntry* end = kFunctions + kFunctionCount;
    const FnEntry* it = std::lower_bound(begin, end, addr, [](const FnEntry& e, uint64_t a) { return e.addr < a; });
    if (it != end && it->addr == addr) return it->fn;
    return nullptr;
}

static void run_target(Ctx* c, uint8_t* m, uint64_t target) {
    if (target >= kThunkBase && target < kThunkBase + 16 * (uint64_t)g_thunks.size() && !(target & 15)) {
        g_thunks[(target - kThunkBase) / 16](c, m);
        return;
    }
    if (GuestFn fn = find_function(target)) {
        fn(c, m);
        return;
    }
    fatal(c, "indirect branch to %#llx, which is not a known function (from around pc %#llx)",
          (unsigned long long)target, (unsigned long long)c->pc);
}

}  // namespace zn

namespace zg {

void call_indirect(Ctx* c, uint8_t* m, uint64_t target) { zn::run_target(c, m, target); }

void dispatch_indirect(Ctx* c, uint8_t* m) { zn::run_target(c, m, c->pc); }

void jt_miss(Ctx* c, uint8_t*, uint64_t pc, uint64_t target) {
    zn::fatal(c, "jump table at %#llx went to %#llx, which the analysis did not list",
              (unsigned long long)pc, (unsigned long long)target);
}

void guest_trap(Ctx* c, uint8_t*, uint64_t pc, uint32_t word) {
    if (word == 0xE7FFDEFEu) zn::fatal(c, "guest abort trap at %#llx", (unsigned long long)pc);
    zn::fatal(c, "undefined instruction %#010x at %#llx", word, (unsigned long long)pc);
}

void guest_brk(Ctx* c, uint8_t*, uint64_t pc, uint32_t imm) {
    zn::fatal(c, "brk #%#x at %#llx", imm, (unsigned long long)pc);
}

}  // namespace zg
