// Horizon kernel objects for one process: handles, threads, events, and the synchronization SVCs
// (WaitSynchronization, mutex arbitration, process-wide condition variables, address arbiter).
// Guest threads are host threads. One kernel lock guards all state; each thread waits on its own condvar.
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "zonaite/gpu.h"
#include "zonaite/guest.h"
#include "zonaite/kernel.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;
using Clock = std::chrono::steady_clock;

static std::mutex g_k;  // the kernel lock

struct KThread : KObject {
    Ctx* ctx = nullptr;
    uint64_t id = 0;
    uint32_t handle = 0;
    int32_t priority = 44, ideal_core = 0;
    uint64_t affinity = 1;
    uint64_t entry = 0, arg = 0, stack_top = 0, tls = 0;
    bool started = false, exited = false;
    std::condition_variable cv;
    bool woken = false;          // set by whoever wakes this thread (under g_k)
    uint32_t wake_result = 0;
    int32_t wake_index = -1;
    uint64_t mutex_wait_addr = 0;  // waiting to own the mutex at this address
    KObject* wake_obj = nullptr;   // the object whose signal woke this thread from WaitSynchronization
    std::vector<KThread*> mutex_waiters;  // threads waiting for a mutex this thread owns
    bool signaled() const override { return exited; }
};

static std::vector<std::shared_ptr<KThread>> g_threads;  // every thread (under g_k)

// ---- emulated CPU cores. On the console a core runs one thread at a time, the highest-priority ready one, and
// game code relies on it: threads sharing a core never interleave between kernel calls, and a lower-priority
// thread never runs while a higher-priority one on its core is ready. Running every guest thread in parallel tore
// TotK's command buffers apart. So a guest thread owns a core while it runs guest code:
//  - it gives the core up only in blocking SVCs (sleep, waits, lock arbitration, IPC) and when it wakes a
//    higher-priority thread that could take its core (preemption);
//  - a free core goes to the highest-priority thread waiting for one (first come among equals);
//  - a thread may take any core its affinity mask allows, its ideal core first.
// ZN_CORES=0 turns this off (every guest thread runs in parallel); ZN_CORES=1 uses a single core.
struct CoreScheduler {
    std::mutex m;
    std::condition_variable cv;
    KThread* owner[3] = {};
    std::deque<KThread*> waiting;  // in arrival order
};
static CoreScheduler g_sched;
static thread_local int t_held_core = -1;

static int cores_mode() {
    static const int mode = [] {
        const char* s = std::getenv("ZN_CORES");
        return s ? std::atoi(s) : 3;
    }();
    return mode;
}

static int core_count() { return cores_mode() >= 3 ? 3 : 1; }

static bool may_run_on(const KThread* t, int core) {
    if (core_count() == 1) return true;
    return (t->affinity >> core) & 1;  // the console never runs a thread outside its affinity mask
}

// The core `t` gets now, or -1: a free core it may use, where no earlier or higher-priority waiter that may also
// use it takes precedence. Caller holds g_sched.m.
static int core_for(const KThread* t) {
    const int n = core_count();
    int order[3] = {0, 1, 2};
    if (n == 3 && t->ideal_core >= 0 && t->ideal_core < 3) std::swap(order[0], order[t->ideal_core]);
    for (int i = 0; i < n; i++) {
        const int core = order[i];
        if (g_sched.owner[core] || !may_run_on(t, core)) continue;
        bool beaten = false;
        for (const KThread* w : g_sched.waiting) {
            if (w == t) break;  // earlier waiters with equal priority go first; later ones need a better priority
            if (w->priority <= t->priority && may_run_on(w, core)) { beaten = true; break; }
        }
        if (!beaten)
            for (const KThread* w : g_sched.waiting)
                if (w != t && w->priority < t->priority && may_run_on(w, core)) { beaten = true; break; }
        if (!beaten) return core;
    }
    return -1;
}

void kernel_core_enter() {
    KThread* self = current_thread();
    if (!cores_mode() || t_held_core >= 0 || !self) return;
    std::unique_lock l(g_sched.m);
    g_sched.waiting.push_back(self);
    int core = -1;
    g_sched.cv.wait(l, [&] { return (core = core_for(self)) >= 0; });
    std::erase(g_sched.waiting, self);
    g_sched.owner[core] = self;
    t_held_core = core;
}

void kernel_core_leave() {
    if (t_held_core < 0) return;
    {
        std::lock_guard l(g_sched.m);
        g_sched.owner[t_held_core] = nullptr;
        t_held_core = -1;
    }
    g_sched.cv.notify_all();
}

// The core this thread runs on (svc GetCurrentProcessorNumber)
static int current_core(const KThread* t) { return t_held_core >= 0 ? t_held_core : (t ? t->ideal_core : 0); }

// SVCs that may block give up the core for their duration
static bool svc_blocks(uint32_t imm) {
    switch (imm) {
        case 0x0B: case 0x18: case 0x1A: case 0x1C: case 0x21: case 0x22: case 0x34: return true;
        default: return false;
    }
}

// SVCs that can wake threads: afterwards, yield the core to a woken higher-priority thread that wants it
static bool svc_wakes(uint32_t imm) {
    switch (imm) {
        case 0x11: case 0x1B: case 0x1D: case 0x35: return true;
        default: return false;
    }
}

void kernel_svc_before(uint32_t imm) {
    if (svc_blocks(imm)) kernel_core_leave();
}

void kernel_svc_after(uint32_t imm) {
    if (svc_blocks(imm)) {
        kernel_core_enter();
        return;
    }
    if (!svc_wakes(imm) || t_held_core < 0) return;
    const KThread* self = current_thread();
    bool preempted = false;
    {
        std::lock_guard l(g_sched.m);
        for (const KThread* w : g_sched.waiting)
            if (w->priority < self->priority && may_run_on(w, t_held_core)) { preempted = true; break; }
    }
    if (preempted) {
        kernel_core_leave();
        kernel_core_enter();
    }
}

static std::unordered_map<uint32_t, std::shared_ptr<KObject>> g_handles;
static uint32_t g_next_handle = 0x10001;
static uint64_t g_next_thread_id = 0x80;
static thread_local KThread* t_self = nullptr;

// ---- handles
static uint32_t add_handle(std::shared_ptr<KObject> o) {
    uint32_t h = g_next_handle++;
    g_handles[h] = std::move(o);
    return h;
}

static KObject* lookup(uint32_t h) {
    if (h == kCurrentThreadPseudoHandle) return t_self;
    auto it = g_handles.find(h);
    return it == g_handles.end() ? nullptr : it->second.get();
}

static KThread* lookup_thread(uint32_t h) { return dynamic_cast<KThread*>(lookup(h)); }

KThread* current_thread() { return t_self; }
zg::Ctx* current_guest_ctx() { return t_self ? t_self->ctx : nullptr; }
uint64_t current_thread_id() { return t_self ? t_self->id : 0; }

static void wake(KThread* t, uint32_t result, int32_t index = -1) {
    t->woken = true;
    t->wake_result = result;
    t->wake_index = index;
    t->cv.notify_one();
}

// ---- interface for services (ipc.cpp and HLE services)
uint32_t kernel_add_handle(std::shared_ptr<KObject> o) {
    std::lock_guard<std::mutex> l(g_k);
    return add_handle(std::move(o));
}

std::shared_ptr<KObject> kernel_get(uint32_t h) {
    std::lock_guard<std::mutex> l(g_k);
    auto it = g_handles.find(h);
    return it == g_handles.end() ? nullptr : it->second;
}

void kernel_signal(KEvent* e) {
    std::lock_guard<std::mutex> l(g_k);
    e->flag = true;
    for (KThread* w : e->waiters) { w->wake_obj = e; wake(w, 0); }
    e->waiters.clear();
}

void kernel_clear(KEvent* e) {
    std::lock_guard<std::mutex> l(g_k);
    e->flag = false;
}

void kernel_close_handle(uint32_t h) {
    std::lock_guard<std::mutex> l(g_k);
    g_handles.erase(h);
}

// Sleep (with g_k held via `lock`) until woken or the timeout passes. Returns the wake result or TimedOut.
static uint32_t sleep_until_woken(std::unique_lock<std::mutex>& lock, KThread* self, int64_t timeout_ns) {
    if (timeout_ns < 0) {
        self->cv.wait(lock, [&] { return self->woken; });
    } else if (!self->cv.wait_for(lock, std::chrono::nanoseconds(timeout_ns), [&] { return self->woken; })) {
        return kResultTimedOut;
    }
    return self->wake_result;
}

// ---- threads
void kernel_init_main_thread(Ctx* c) {
    std::lock_guard<std::mutex> l(g_k);
    auto t = std::make_shared<KThread>();
    t->ctx = c;
    t->id = g_next_thread_id++;
    t->handle = kMainThreadHandle;
    t->started = true;
    t->tls = c->tpidrro_el0;
    g_handles[kMainThreadHandle] = t;
    g_next_handle = kMainThreadHandle + 1;
    t_self = t.get();
    g_threads.push_back(t);
}

#if defined(_WIN32)
static DWORD WINAPI thread_main(void* p) {
    KThread* t = (KThread*)p;
    t_self = t;
    SetThreadDescription(GetCurrentThread(), L"guest thread");
    kernel_core_enter();
    guest_call(t->ctx, g_mem, t->entry);
    kernel_exit_thread();
}
#endif

void kernel_exit_thread() {
    kernel_core_leave();
    {
        std::lock_guard<std::mutex> l(g_k);
        KThread* t = t_self;
        t->exited = true;
        for (KThread* w : t->waiters) { w->wake_obj = t; wake(w, 0); }
        t->waiters.clear();
    }
#if defined(_WIN32)
    ExitThread(0);  // abandons the guest frames on this host stack; they hold no host resources
#endif
}

static uint32_t create_thread(uint64_t entry, uint64_t arg, uint64_t stack_top, int32_t prio, int32_t core,
                              uint32_t* out_handle) {
    auto t = std::make_shared<KThread>();
    t->entry = entry;
    t->arg = arg;
    t->stack_top = stack_top;
    t->priority = prio;
    t->ideal_core = core < 0 ? 0 : core;  // -2 = "use the process default"
    t->affinity = 1ull << t->ideal_core;
    t->tls = alloc_stack_region(0x1000);
    std::lock_guard<std::mutex> l(g_k);
    t->id = g_next_thread_id++;
    t->handle = add_handle(t);
    g_threads.push_back(t);
    *out_handle = t->handle;
    return 0;
}

static uint32_t start_thread(uint32_t h) {
    KThread* t;
    {
        std::lock_guard<std::mutex> l(g_k);
        t = lookup_thread(h);
        if (!t) return kResultInvalidHandle;
        t->started = true;
    }
    Ctx* c = new Ctx{};
    c->x[0] = t->arg;
    c->sp = t->stack_top & ~15ull;
    c->tpidrro_el0 = t->tls;
    t->ctx = c;
    // generated code nests guest calls on the host stack: reserve 64 MB (std::thread would give 1 MB)
    HANDLE th = CreateThread(nullptr, 64u << 20, thread_main, t, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!th) return kResultOutOfResource;
    CloseHandle(th);
    return 0;
}

// ---- events
static uint32_t create_event(uint32_t* w, uint32_t* r) {
    auto e = std::make_shared<KEvent>();
    std::lock_guard<std::mutex> l(g_k);
    *w = add_handle(e);  // writable and readable ends share one object
    *r = add_handle(e);
    return 0;
}

static uint32_t signal_event(uint32_t h) {
    std::lock_guard<std::mutex> l(g_k);
    auto* e = dynamic_cast<KEvent*>(lookup(h));
    if (!e) return kResultInvalidHandle;
    e->flag = true;
    for (KThread* w : e->waiters) { w->wake_obj = e; wake(w, 0); }
    e->waiters.clear();
    return 0;
}

static uint32_t clear_event(uint32_t h) {
    std::lock_guard<std::mutex> l(g_k);
    auto* e = dynamic_cast<KEvent*>(lookup(h));
    if (!e) return kResultInvalidHandle;
    e->flag = false;
    return 0;
}

static uint32_t wait_synchronization(uint8_t* m, uint64_t handles_ptr, int32_t count, int64_t timeout,
                                     int32_t* out_index) {
    std::unique_lock<std::mutex> lock(g_k);
    std::vector<KObject*> objs;
    for (int32_t i = 0; i < count; i++) {
        KObject* o = lookup(ld32(m, handles_ptr + 4 * i));
        if (!o) return kResultInvalidHandle;
        objs.push_back(o);
    }
    for (int32_t i = 0; i < count; i++)
        if (objs[i]->signaled()) { *out_index = i; return 0; }
    if (timeout == 0) return kResultTimedOut;
    if (timeout > 0)
        for (KObject* o : objs)
            if (auto* e = dynamic_cast<KEvent*>(o); e && e->gpu_fence) timeout = std::max(timeout, kGpuWaitLimitNs);
    KThread* self = t_self;
    self->woken = false;
    self->wake_obj = nullptr;
    for (KObject* o : objs) o->waiters.push_back(self);
    uint32_t r = sleep_until_woken(lock, self, timeout);
    for (KObject* o : objs) std::erase(o->waiters, self);
    if (r == 0) {
        // the object that signaled us, even if someone cleared it since (re-checking `signaled()` here used to
        // report a timeout that never happened)
        for (int32_t i = 0; i < count; i++)
            if (objs[i] == self->wake_obj) { *out_index = i; return 0; }
        for (int32_t i = 0; i < count; i++)
            if (objs[i]->signaled()) { *out_index = i; return 0; }
    }
    if (r == kResultTimedOut && timeout > 0) {  // the game gave up waiting: worth knowing (a GPU stall shows here)
        static std::atomic<int> reported{0};
        if (reported.fetch_add(1) < 40)
            log("WaitSynchronization timed out after %lld us (handle %#x, thread %#llx)", (long long)(timeout / 1000),
                ld32(m, handles_ptr), (unsigned long long)self->id);
    }
    return r == 0 ? kResultTimedOut : r;
}

// ---- mutex arbitration (Horizon userspace mutex: word = owner handle | kWaitMask when contended)
constexpr uint32_t kWaitMask = 0x40000000u;

static uint32_t arbitrate_lock(uint8_t* m, uint32_t owner_handle, uint64_t addr, uint32_t self_handle) {
    std::unique_lock<std::mutex> lock(g_k);
    uint32_t value = ld32(m, addr);
    if (value != (owner_handle | kWaitMask)) return 0;  // changed meanwhile: caller retries
    KThread* owner = lookup_thread(owner_handle);
    if (!owner) return kResultInvalidHandle;
    KThread* self = t_self;
    (void)self_handle;
    self->woken = false;
    self->mutex_wait_addr = addr;
    owner->mutex_waiters.push_back(self);
    uint32_t r = sleep_until_woken(lock, self, -1);  // woken by the unlock that hands us the mutex
    return r;
}

// Release the mutex at addr held by `owner`: hand it to the highest-priority waiter or clear the word.
static void release_mutex(uint8_t* m, KThread* owner, uint64_t addr) {
    KThread* next = nullptr;
    size_t remaining = 0;
    for (KThread* w : owner->mutex_waiters)
        if (w->mutex_wait_addr == addr) {
            if (!next || w->priority < next->priority) next = w;
        }
    if (!next) {
        aref<uint32_t>(m, addr).store(0, std::memory_order_seq_cst);
        return;
    }
    std::erase(owner->mutex_waiters, next);
    for (KThread* w : owner->mutex_waiters)
        if (w->mutex_wait_addr == addr) {
            remaining++;
        }
    // remaining waiters move to the new owner
    for (auto it = owner->mutex_waiters.begin(); it != owner->mutex_waiters.end();) {
        if ((*it)->mutex_wait_addr == addr) { next->mutex_waiters.push_back(*it); it = owner->mutex_waiters.erase(it); }
        else ++it;
    }
    aref<uint32_t>(m, addr).store(next->handle | (remaining ? kWaitMask : 0), std::memory_order_seq_cst);
    next->mutex_wait_addr = 0;
    wake(next, 0);
}

static uint32_t arbitrate_unlock(uint8_t* m, uint64_t addr) {
    std::lock_guard<std::mutex> l(g_k);
    release_mutex(m, t_self, addr);
    return 0;
}

// ---- process-wide condition variables
struct CvWaiter {
    KThread* thread;
    uint64_t mutex_addr;
    uint32_t tag;  // the waiting thread's handle, written into the mutex word when it acquires
};
static std::unordered_map<uint64_t, std::deque<CvWaiter>> g_cv;

// Try to give the mutex to a thread woken from a condition variable: take it if free, else queue on the owner.
static void acquire_after_signal(uint8_t* m, const CvWaiter& w) {
    auto ref = aref<uint32_t>(m, w.mutex_addr);
    uint32_t expected = 0;
    if (ref.compare_exchange_strong(expected, w.tag)) {
        wake(w.thread, 0);
        return;
    }
    // owned: set the waiter bit and wait for the owner's unlock
    uint32_t cur = ref.load();
    while (!ref.compare_exchange_weak(cur, cur | kWaitMask)) {}
    KThread* owner = lookup_thread(cur & ~kWaitMask);
    if (!owner) {  // owner vanished: treat as acquired failure
        wake(w.thread, kResultInvalidHandle);
        return;
    }
    w.thread->mutex_wait_addr = w.mutex_addr;
    owner->mutex_waiters.push_back(w.thread);
}

// Takes a thread that gave up waiting (timeout) off its mutex owner's waiter list, clearing the mutex word's
// waiter bit if nobody else waits for that mutex. Caller holds g_k.
static void leave_mutex_wait(uint8_t* m, KThread* self) {
    const uint64_t addr = self->mutex_wait_addr;
    self->mutex_wait_addr = 0;
    for (auto& t : g_threads) {
        auto& ws = t->mutex_waiters;
        if (std::find(ws.begin(), ws.end(), self) == ws.end()) continue;
        std::erase(ws, self);
        const bool others = std::any_of(ws.begin(), ws.end(), [&](KThread* w) { return w->mutex_wait_addr == addr; });
        if (!others) {
            auto ref = aref<uint32_t>(m, addr);
            uint32_t cur = ref.load();
            while ((cur & kWaitMask) && !ref.compare_exchange_weak(cur, cur & ~kWaitMask)) {}
        }
        return;
    }
}

static uint32_t wait_process_wide_key(uint8_t* m, uint64_t mutex_addr, uint64_t key, uint32_t tag, int64_t timeout) {
    std::unique_lock<std::mutex> lock(g_k);
    KThread* self = t_self;
    release_mutex(m, self, mutex_addr);
    self->woken = false;
    g_cv[key].push_back(CvWaiter{self, mutex_addr, tag});
    st32(m, key, 1);  // tell user space there are waiters
    uint32_t r = sleep_until_woken(lock, self, timeout);
    if (r == kResultTimedOut) {
        // Like the console's kernel: leave whichever queue we are in (the condvar's, or the mutex owner's if a
        // signal already moved us there) and return without the mutex. User space locks it again itself; taking
        // it here as well made that lock wait on its own handle forever.
        auto& q = g_cv[key];
        for (auto it = q.begin(); it != q.end(); ++it)
            if (it->thread == self) { q.erase(it); break; }
        if (q.empty()) st32(m, key, 0);
        if (self->mutex_wait_addr) leave_mutex_wait(m, self);
        return kResultTimedOut;
    }
    return r;
}

static uint32_t signal_process_wide_key(uint8_t* m, uint64_t key, int32_t count) {
    std::lock_guard<std::mutex> l(g_k);
    auto it = g_cv.find(key);
    if (it == g_cv.end()) return 0;
    auto& q = it->second;
    int32_t n = 0;
    while (!q.empty() && (count <= 0 || n < count)) {
        // highest priority first
        auto best = q.begin();
        for (auto j = q.begin(); j != q.end(); ++j)
            if (j->thread->priority < best->thread->priority) best = j;
        CvWaiter w = *best;
        q.erase(best);
        acquire_after_signal(m, w);
        n++;
    }
    if (q.empty()) st32(m, key, 0);
    return 0;
}

// ---- address arbiter
static std::unordered_map<uint64_t, std::deque<KThread*>> g_arb;

static uint32_t wait_for_address(uint8_t* m, uint64_t addr, uint32_t type, int32_t value, int64_t timeout) {
    std::unique_lock<std::mutex> lock(g_k);
    int32_t cur = (int32_t)ld32(m, addr);
    switch (type) {
        case 0: if (!(cur < value)) return kResultInvalidState; break;              // WaitIfLessThan
        case 1:                                                                    // DecrementAndWaitIfLessThan
            if (!(cur < value)) return kResultInvalidState;
            st32(m, addr, (uint32_t)(cur - 1));
            break;
        case 2: if (cur != value) return kResultInvalidState; break;               // WaitIfEqual
        default: return kResultInvalidEnumValue;
    }
    if (timeout == 0) return kResultTimedOut;
    KThread* self = t_self;
    self->woken = false;
    g_arb[addr].push_back(self);
    uint32_t r = sleep_until_woken(lock, self, timeout);
    if (r == kResultTimedOut) std::erase(g_arb[addr], self);
    return r;
}

static int32_t wake_arb(uint64_t addr, int32_t count) {
    // highest priority first (lowest number), arrival order among equals, as the kernel's waiter tree does
    auto& q = g_arb[addr];
    int32_t n = 0;
    while (!q.empty() && (count <= 0 || n < count)) {
        auto best = q.begin();
        for (auto it = q.begin(); it != q.end(); ++it)
            if ((*it)->priority < (*best)->priority) best = it;
        wake(*best, 0);
        q.erase(best);
        n++;
    }
    return n;
}

static uint32_t signal_to_address(uint8_t* m, uint64_t addr, uint32_t type, int32_t value, int32_t count) {
    std::lock_guard<std::mutex> l(g_k);
    if (type == 0) { wake_arb(addr, count); return 0; }                             // Signal
    int32_t cur = (int32_t)ld32(m, addr);
    if (cur != value) return kResultInvalidState;
    if (type == 1) {                                                                // SignalAndIncrementIfEqual
        st32(m, addr, (uint32_t)(cur + 1));
        wake_arb(addr, count);
        return 0;
    }
    if (type == 2) {  // SignalAndModifyByWaitingCountIfEqual
        // New value as Eden's KAddressArbiter computes it: with no waiters, value + 1; waking all of them
        // (count <= 0), value - 2; else value - 1 if the waiters after the first number fewer than count (so
        // all of them wake), value otherwise.
        const size_t waiting = g_arb[addr].size();
        int32_t nv;
        if (!waiting) nv = cur + 1;
        else if (count <= 0) nv = cur - 2;
        else nv = (int64_t)waiting - 1 < count ? cur - 1 : cur;
        st32(m, addr, (uint32_t)nv);
        wake_arb(addr, count);
        return 0;
    }
    return kResultInvalidEnumValue;
}

// ---- crash reports: a host exception nobody handled (a guest access to unmapped memory, a bug in the runtime)
// is reported like a guest fatal error, with the current guest thread's registers and backtrace.
extern "C" void zonaite_log_host_stack();

static LONG WINAPI on_unhandled_exception(EXCEPTION_POINTERS* e) {
    const auto* r = e->ExceptionRecord;
    zonaite_log_host_stack();  // the faulting host thread's stack (this filter runs on it)
    const uint64_t data = r->NumberParameters >= 2 ? (uint64_t)r->ExceptionInformation[1] : 0;
    const uint64_t guest = data - (uint64_t)g_mem;
    fatal(t_self ? t_self->ctx : nullptr,
          "host exception %#lx at %p in %s (%s %#llx; as a guest address %#llx)", r->ExceptionCode,
          r->ExceptionAddress, gpu::host_symbol(r->ExceptionAddress).c_str(),
          r->NumberParameters >= 1 && r->ExceptionInformation[0] == 1 ? "writing" : "reading",
          (unsigned long long)data, (unsigned long long)guest);
}

void install_crash_handler() { SetUnhandledExceptionFilter(on_unhandled_exception); }

// ---- shared memory
std::shared_ptr<KSharedMemory> create_shared_memory(uint64_t size) {
    static std::mutex lock;
    static uint64_t next = kSharedPoolBase;
    std::lock_guard<std::mutex> l(lock);
    size = (size + 0xFFF) & ~0xFFFull;
    if (next + size > kSharedPoolBase + kSharedPoolSize) fatal(nullptr, "shared memory pool exhausted");
    auto s = std::make_shared<KSharedMemory>();
    s->pool_offset = next;
    s->size = size;
    s->host = gpu::backing_pointer(next);
    std::memset(s->host, 0, size);
    next += size;
    return s;
}

static uint32_t map_shared_memory(uint32_t handle, uint64_t addr, uint64_t size, uint32_t perm) {
    auto s = std::dynamic_pointer_cast<KSharedMemory>(kernel_get(handle));
    if (!s) return kResultInvalidHandle;
    if ((addr | size) & 0xFFF || size != s->size || addr + size > kDramSize) return kResultInvalidSize;
    gpu::remap_guest(addr, s->pool_offset, size);
    map_region(addr, size, kStateShared, perm);
    return 0;
}

static uint32_t unmap_shared_memory(uint32_t handle, uint64_t addr, uint64_t size) {
    auto s = std::dynamic_pointer_cast<KSharedMemory>(kernel_get(handle));
    if (!s) return kResultInvalidHandle;
    if ((addr | size) & 0xFFF || size != s->size) return kResultInvalidSize;
    gpu::remap_guest(addr, addr, size);
    unmap_region(addr);
    return 0;
}

// ---- SVC entry points (called from svc.cpp); return true if handled
bool kernel_svc(Ctx* c, uint8_t* m, uint32_t imm) {
    switch (imm) {
        case 0x08: {  // CreateThread(entry, arg, stack_top, priority, core) -> handle
            uint32_t h = 0;
            log("CreateThread entry %#llx arg %#llx stack %#llx prio %d core %d", (unsigned long long)c->x[1],
                (unsigned long long)c->x[2], (unsigned long long)c->x[3], (int)c->x[4], (int)c->x[5]);
            c->x[0] = create_thread(c->x[1], c->x[2], c->x[3], (int32_t)c->x[4], (int32_t)c->x[5], &h);
            c->x[1] = h;
            return true;
        }
        case 0x09: c->x[0] = start_thread((uint32_t)c->x[0]); return true;
        case 0x0A: kernel_exit_thread(); return true;
        case 0x0B: {  // SleepThread(ns): 0, -1, -2 are yields
            int64_t ns = (int64_t)c->x[0];
            if (ns <= 0) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
            return true;
        }
        case 0x0C: {  // GetThreadPriority(handle) -> priority
            std::lock_guard<std::mutex> l(g_k);
            KThread* t = lookup_thread((uint32_t)c->x[1]);
            c->x[0] = t ? 0 : kResultInvalidHandle;
            c->x[1] = t ? (uint64_t)(uint32_t)t->priority : 0;
            return true;
        }
        case 0x0D: {  // SetThreadPriority(handle, priority)
            std::lock_guard<std::mutex> l(g_k);
            KThread* t = lookup_thread((uint32_t)c->x[0]);
            if (t) t->priority = (int32_t)c->x[1];
            c->x[0] = t ? 0 : kResultInvalidHandle;
            return true;
        }
        case 0x0E: {  // GetThreadCoreMask(handle) -> ideal core, affinity mask
            std::lock_guard<std::mutex> l(g_k);
            KThread* t = lookup_thread((uint32_t)c->x[2]);
            c->x[0] = t ? 0 : kResultInvalidHandle;
            c->x[1] = t ? (uint64_t)(uint32_t)t->ideal_core : 0;
            c->x[2] = t ? t->affinity : 0;
            return true;
        }
        case 0x0F: {  // SetThreadCoreMask(handle, ideal core, mask), with the kernel's rules for the ideal core:
            // -2 = the process default (core 0, mask = that core), -3 = keep it, -1 = don't care; a kept or
            // don't-care ideal core outside the new mask moves to the mask's highest core
            bool migrate = false;
            {
                std::lock_guard<std::mutex> l(g_k);
                KThread* t = lookup_thread((uint32_t)c->x[0]);
                int32_t core = (int32_t)c->x[1];
                uint64_t mask = c->x[2] & 7;
                uint32_t r = t ? 0 : kResultInvalidHandle;
                if (t && core == -2) {
                    core = 0;
                    mask = 1;
                } else if (t && (mask == 0 || (core >= 0 && (core > 2 || !((mask >> core) & 1))) || core < -3)) {
                    r = 0xE801;  // InvalidCombination
                }
                if (!r) {
                    if (core < 0) core = ((mask >> t->ideal_core) & 1) ? t->ideal_core : 63 - std::countl_zero(mask);
                    t->ideal_core = core;
                    t->affinity = mask;
                    migrate = t == t_self && t_held_core >= 0 && !((mask >> t_held_core) & 1);
                }
                c->x[0] = r;
            }
            if (migrate) {  // the caller now may not run where it is
                kernel_core_leave();
                kernel_core_enter();
            }
            return true;
        }
        case 0x10: c->x[0] = (uint64_t)(uint32_t)current_core(t_self); return true;  // GetCurrentProcessorNumber
        case 0x11: c->x[0] = signal_event((uint32_t)c->x[0]); return true;
        case 0x12: c->x[0] = clear_event((uint32_t)c->x[0]); return true;
        case 0x16: {  // CloseHandle
            std::lock_guard<std::mutex> l(g_k);
            g_handles.erase((uint32_t)c->x[0]);
            c->x[0] = 0;
            return true;
        }
        case 0x13:  // MapSharedMemory(handle, addr, size, perm)
            c->x[0] = map_shared_memory((uint32_t)c->x[0], c->x[1], c->x[2], (uint32_t)c->x[3]);
            return true;
        case 0x14:  // UnmapSharedMemory(handle, addr, size)
            c->x[0] = unmap_shared_memory((uint32_t)c->x[0], c->x[1], c->x[2]);
            return true;
        case 0x15: {  // CreateTransferMemory(addr, size, perm) -> handle
            auto t = std::make_shared<KTransferMemory>();
            t->addr = c->x[1];
            t->size = c->x[2];
            t->perm = (uint32_t)c->x[3];
            std::lock_guard<std::mutex> l(g_k);
            c->x[1] = add_handle(t);
            c->x[0] = 0;
            return true;
        }
        case 0x17: c->x[0] = clear_event((uint32_t)c->x[0]); return true;  // ResetSignal
        case 0x18: {  // WaitSynchronization(handles, count, timeout) -> index
            int32_t idx = -1;
            c->x[0] = wait_synchronization(m, c->x[1], (int32_t)c->x[2], (int64_t)c->x[3], &idx);
            c->x[1] = (uint64_t)(uint32_t)idx;
            return true;
        }
        case 0x1A: c->x[0] = arbitrate_lock(m, (uint32_t)c->x[0], c->x[1], (uint32_t)c->x[2]); return true;
        case 0x1B: c->x[0] = arbitrate_unlock(m, c->x[0]); return true;
        case 0x1C:
            c->x[0] = wait_process_wide_key(m, c->x[0], c->x[1], (uint32_t)c->x[2], (int64_t)c->x[3]);
            return true;
        case 0x1D: c->x[0] = signal_process_wide_key(m, c->x[0], (int32_t)c->x[1]); return true;
        case 0x25: {  // GetThreadId(handle) -> id
            std::lock_guard<std::mutex> l(g_k);
            KThread* t = lookup_thread((uint32_t)c->x[1]);
            c->x[0] = t ? 0 : kResultInvalidHandle;
            c->x[1] = t ? t->id : 0;
            return true;
        }
        case 0x34:
            c->x[0] = wait_for_address(m, c->x[0], (uint32_t)c->x[1], (int32_t)c->x[2], (int64_t)c->x[3]);
            return true;
        case 0x35:
            c->x[0] = signal_to_address(m, c->x[0], (uint32_t)c->x[1], (int32_t)c->x[2], (int32_t)c->x[3]);
            return true;
        case 0x45: {  // CreateEvent -> writable, readable
            uint32_t w = 0, r = 0;
            c->x[0] = create_event(&w, &r);
            c->x[1] = w;
            c->x[2] = r;
            return true;
        }
    }
    return false;
}

}  // namespace zn
