// Horizon supervisor calls, system registers and cache operations.
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <random>
#include <cstdio>

#include "zonaite/guest.h"
#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;

const char* svc_name(uint32_t n) {
    static const char* names[0x80] = {};
    static bool init = false;
    if (!init) {
        names[0x01] = "SetHeapSize"; names[0x02] = "SetMemoryPermission"; names[0x03] = "SetMemoryAttribute";
        names[0x04] = "MapMemory"; names[0x05] = "UnmapMemory"; names[0x06] = "QueryMemory";
        names[0x07] = "ExitProcess"; names[0x08] = "CreateThread"; names[0x09] = "StartThread";
        names[0x0A] = "ExitThread"; names[0x0B] = "SleepThread"; names[0x0C] = "GetThreadPriority";
        names[0x0D] = "SetThreadPriority"; names[0x0E] = "GetThreadCoreMask"; names[0x0F] = "SetThreadCoreMask";
        names[0x10] = "GetCurrentProcessorNumber"; names[0x11] = "SignalEvent"; names[0x12] = "ClearEvent";
        names[0x13] = "MapSharedMemory"; names[0x14] = "UnmapSharedMemory"; names[0x15] = "CreateTransferMemory";
        names[0x16] = "CloseHandle"; names[0x17] = "ResetSignal"; names[0x18] = "WaitSynchronization";
        names[0x19] = "CancelSynchronization"; names[0x1A] = "ArbitrateLock"; names[0x1B] = "ArbitrateUnlock";
        names[0x1C] = "WaitProcessWideKeyAtomic"; names[0x1D] = "SignalProcessWideKey";
        names[0x1E] = "GetSystemTick"; names[0x1F] = "ConnectToNamedPort"; names[0x21] = "SendSyncRequest";
        names[0x22] = "SendSyncRequestWithUserBuffer"; names[0x24] = "GetProcessId"; names[0x25] = "GetThreadId";
        names[0x26] = "Break"; names[0x27] = "OutputDebugString"; names[0x29] = "GetInfo";
        names[0x2C] = "MapPhysicalMemory"; names[0x2D] = "UnmapPhysicalMemory"; names[0x32] = "SetThreadActivity";
        names[0x34] = "WaitForAddress"; names[0x35] = "SignalToAddress"; names[0x36] = "SynchronizePreemptionState";
        names[0x45] = "CreateEvent"; names[0x40] = "CreateSession"; names[0x43] = "ReplyAndReceive";
        init = true;
    }
    return n < 0x80 && names[n] ? names[n] : "?";
}

// Switch system counter: 19.2 MHz
uint64_t system_tick() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    uint64_t ns = (uint64_t)duration_cast<nanoseconds>(steady_clock::now() - t0).count();
    return ns / 10000 * 192 + ns % 10000 * 192 / 10000;  // 19.2 ticks per microsecond, without 128-bit division
}

static uint64_t random64() {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex lock;
    std::lock_guard<std::mutex> l(lock);
    return rng();
}

static uint64_t used_memory() { return heap_size() + g_image.size + physical_memory_size(); }

static uint32_t get_info(Ctx* c, uint32_t type, uint32_t handle, uint64_t sub, uint64_t* out) {
    (void)handle;
    switch (type) {
        case 0: *out = 0x7; return 0;                       // CoreMask: cores 0-2 for applications
        case 1: *out = 0xFFFFFFFFFFFFFFFFull >> 4; return 0;  // PriorityMask
        case 2: *out = kAliasRegionBase; return 0;
        case 3: *out = kAliasRegionSize; return 0;
        case 4: *out = kHeapRegionBase; return 0;
        case 5: *out = kHeapRegionSize; return 0;
        case 6: *out = kTotalMemorySize + system_resource_size(); return 0;  // TotalMemorySize
        case 7: *out = used_memory() + system_resource_size(); return 0;     // UsedMemorySize
        case 8: *out = 0; return 0;                         // DebuggerAttached
        case 11: *out = random64(); return 0;               // RandomEntropy (subtypes 0-3)
        case 12: *out = kAslrRegionBase; return 0;
        case 13: *out = kAslrRegionSize; return 0;
        case 14: *out = kStackRegionBase; return 0;
        case 15: *out = kStackRegionSize; return 0;
        case 16: *out = system_resource_size(); return 0;   // SystemResourceSizeTotal
        case 17: *out = 0; return 0;                        // SystemResourceSizeUsed (page tables: none)
        case 18: *out = 0x0100F2C0115B6000ull; return 0;    // ProgramId
        case 20: *out = 0; return 0;                        // UserExceptionContextAddress
        case 21: *out = kTotalMemorySize; return 0;         // TotalNonSystemMemorySize
        case 22: *out = used_memory(); return 0;            // UsedNonSystemMemorySize
        case 23: *out = 1; return 0;                        // IsApplication
    }
    fatal(c, "GetInfo: unhandled info type %u (subtype %#llx)", type, (unsigned long long)sub);
}

}  // namespace zn

namespace zg {

// Horizon SVC ABI: arguments in x0-x7, result code in w0, outputs in x1, x2, ...
static void guest_svc_impl(Ctx* c, uint8_t* m, uint32_t imm);

void guest_svc(Ctx* c, uint8_t* m, uint32_t imm) {
    zn::kernel_svc_before(imm);
    guest_svc_impl(c, m, imm);
    zn::kernel_svc_after(imm);
}

static void guest_svc_impl(Ctx* c, uint8_t* m, uint32_t imm) {
    using namespace zn;
    switch (imm) {
        case 0x01: {  // SetHeapSize(size) -> address
            uint64_t addr = 0;
            c->x[0] = svc_set_heap_size(c->x[1], &addr);
            c->x[1] = addr;
            return;
        }
        case 0x03:  // SetMemoryAttribute(addr, size, mask, attr): NVN marks GPU memory uncached. Host memory is
                    // coherent between the guest and the GPU, so only the arguments are checked.
            c->x[0] = (c->x[2] & ~8ull) || (c->x[3] & ~c->x[2]) ? 0xE001u /* InvalidCombination */ : 0;
            return;
        case 0x2C:  // MapPhysicalMemory(addr, size)
            c->x[0] = svc_map_physical_memory(m, c->x[0], c->x[1]);
            return;
        case 0x2D:  // UnmapPhysicalMemory(addr, size)
            c->x[0] = svc_unmap_physical_memory(c->x[0], c->x[1]);
            return;
        case 0x04:  // MapMemory(dst, src, size)
            c->x[0] = svc_map_memory(m, c->x[0], c->x[1], c->x[2]);
            return;
        case 0x05:  // UnmapMemory(dst, src, size)
            c->x[0] = svc_unmap_memory(m, c->x[0], c->x[1], c->x[2]);
            return;
        case 0x06: {  // QueryMemory(MemoryInfo*, address) -> page info
            uint32_t page_info = 0;
            c->x[0] = svc_query_memory(m, c->x[0], c->x[2], &page_info);
            c->x[1] = page_info;
            return;
        }
        case 0x1F: {  // ConnectToNamedPort(name) -> session handle
            uint32_t h = 0;
            c->x[0] = ipc_connect_to_named_port(m, c->x[1], &h);
            c->x[1] = h;
            return;
        }
        case 0x21:  // SendSyncRequest(session)
            c->x[0] = ipc_send_sync_request(c, m, (uint32_t)c->x[0]);
            return;
        case 0x1E:  // GetSystemTick
            c->x[0] = zn::system_tick();
            return;
        case 0x29: {  // GetInfo(type, handle, subtype) -> value
            uint64_t value = 0;
            c->x[0] = get_info(c, (uint32_t)c->x[1], (uint32_t)c->x[2], c->x[3], &value);
            c->x[1] = value;
            return;
        }
        default:
            if (kernel_svc(c, m, imm)) return;
            zn::fatal(c, "unimplemented svc %#x (%s) x0=%#llx x1=%#llx x2=%#llx", imm, zn::svc_name(imm),
                      (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
    }
}

// sysreg encoding: op0<<14 | op1<<11 | CRn<<7 | CRm<<3 | op2 (see recomp/emit.py)
uint64_t sysreg_read(Ctx* c, uint32_t enc) {
    switch (enc) {
        case 0xDF00: return 19200000;            // CNTFRQ_EL0
        case 0xDF01:                             // CNTPCT_EL0
        case 0xDF02: return zn::system_tick();   // CNTVCT_EL0
        case 0xD807: return 0x4;                 // DCZID_EL0: DC ZVA allowed, 64-byte blocks
        case 0xD801: return 0x8444C004;          // CTR_EL0 as on Cortex-A57
        case 0xC000: return 0x411FD071;          // MIDR_EL1 (Cortex-A57 r1p1), if the game ever asks
    }
    zn::fatal(c, "unhandled mrs, sysreg encoding %#x", enc);
}

void sysreg_write(Ctx* c, uint32_t enc, uint64_t value) {
    zn::fatal(c, "unhandled msr, sysreg encoding %#x value %#llx", enc, (unsigned long long)value);
}

void sys_op(Ctx* c, uint8_t* m, uint32_t enc, uint64_t value) {
    // op1=3 CRn=7 CRm=4 op2=1: DC ZVA (zero a 64-byte block)
    if (enc == ((1u << 14) | (3u << 11) | (7u << 7) | (4u << 3) | 1u)) {
        std::memset(m + (value & ~63ull), 0, 64);
        return;
    }
    // other DC / IC maintenance (CIVAC, CVAC, IVAU, ...): no-op on a coherent host
    if (((enc >> 7) & 15) == 7) return;
    zn::fatal(c, "unhandled sys op %#x value %#llx", enc, (unsigned long long)value);
}

}  // namespace zg
