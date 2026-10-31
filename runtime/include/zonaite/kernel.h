// Kernel objects and synchronization SVCs (runtime/src/kernel.cpp).
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "zonaite/guest.h"

namespace zn {

// Result codes (module 1 = kernel): module | description << 9
constexpr uint32_t kResultInvalidState = 0xFA01;   // 125
constexpr uint32_t kResultInvalidHandle = 0xE401;  // 114
constexpr uint32_t kResultTimedOut = 0xEA01;       // 117
constexpr uint32_t kResultCancelled = 0xEC01;      // 118
constexpr uint32_t kResultOutOfResource = 0xCE01;  // 103

struct KThread;

// Anything a handle can refer to. `signaled` decides WaitSynchronization; it's read under the kernel lock.
struct KObject {
    virtual ~KObject() = default;
    virtual bool signaled() const { return false; }
    std::vector<KThread*> waiters;  // threads in WaitSynchronization on this object
};

struct KEvent : KObject {
    bool flag = false;
    // A GPU syncpoint event (nvhost-ctrl): the console's GPU always reaches the fence, quickly, so waits on it
    // with a timeout are given until it does (up to kGpuWaitLimitNs). Eden's GPU thread can stall for hundreds
    // of ms on pipeline compiles; a game that took the timeout as "done" recycled memory the GPU still used.
    bool gpu_fence = false;
    bool signaled() const override { return flag; }
};

// Guest memory lent to a service (svc CreateTransferMemory). Services are in-process, so they use it directly.
struct KTransferMemory : KObject {
    uint64_t addr = 0, size = 0;
    uint32_t perm = 0;
};

// Memory a service shares with the guest (hid's input state, ...). Its bytes live in Zonaite's shared-memory pool
// in DRAM; svc MapSharedMemory remaps the guest's chosen range onto them, so `host` and the guest see the same bytes.
struct KSharedMemory : KObject {
    uint64_t pool_offset = 0, size = 0;  // backing offset (a guest physical address) and size
    uint8_t* host = nullptr;             // the service's view
};
std::shared_ptr<KSharedMemory> create_shared_memory(uint64_t size);

constexpr int64_t kGpuWaitLimitNs = 20'000'000'000;  // 20 s

uint32_t kernel_add_handle(std::shared_ptr<KObject> o);
std::shared_ptr<KObject> kernel_get(uint32_t handle);
void kernel_signal(KEvent* e);
void kernel_clear(KEvent* e);
void kernel_close_handle(uint32_t handle);
constexpr uint32_t kResultNotFound = 0xF201;  // 121

KThread* current_thread();
// The calling guest thread's registers (null on host threads)
zg::Ctx* current_guest_ctx();
uint64_t current_thread_id();  // 0 on host threads
// Emulated cores (ZN_CORES): a guest thread owns one while running guest code; guest_svc calls the hooks
void kernel_core_enter();
void kernel_core_leave();
void kernel_svc_before(uint32_t imm);
void kernel_svc_after(uint32_t imm);
void install_crash_handler();
void kernel_init_main_thread(zg::Ctx* c);
[[noreturn]] void kernel_exit_thread();
bool kernel_svc(zg::Ctx* c, uint8_t* m, uint32_t imm);
uint64_t alloc_stack_region(uint64_t size);

}  // namespace zn
