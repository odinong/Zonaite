// Guest threads: contexts, stacks and TLS blocks; host-side calls into guest code.
#include <atomic>
#include <cstdio>

#include "zonaite/guest.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;

static std::atomic<uint64_t> g_stack_next{kStackBase};

// Allocate guest memory from the stack/TLS region (never freed for now).
uint64_t alloc_stack_region(uint64_t size) {
    size = (size + 0xFFFF) & ~0xFFFFull;
    uint64_t at = g_stack_next.fetch_add(size + 0x10000);  // 64 KB unmapped guard gap between allocations
    commit(at, size);
    return at;
}

Ctx* create_main_thread_context() {
    Ctx* c = new Ctx{};
    uint64_t stack = alloc_stack_region(kMainStackSize);
    c->sp = stack + kMainStackSize;
    uint64_t tls = alloc_stack_region(0x1000);
    c->tpidrro_el0 = tls;
    log("main thread: stack %#llx-%#llx, TLS %#llx", (unsigned long long)stack,
        (unsigned long long)(stack + kMainStackSize), (unsigned long long)tls);
    return c;
}

void guest_call(Ctx* c, uint8_t* m, uint64_t addr) {
    uint64_t saved_lr = c->x[30];
    c->x[30] = 0;  // generated code returns with a C return; LR is only for code that inspects it
    call_indirect(c, m, addr);
    c->x[30] = saved_lr;
}

}  // namespace zn
