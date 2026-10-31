// Zonaite override of Eden's KProcess (see ../../../README.md). There is one process, the game; its memory is
// Zonaite's guest memory. Page locking for device mappings is a no-op: guest memory is never unmapped under the GPU.
#pragma once

// Eden's k_process.h brings these in for nvdrv (Common::Literals, Common::Log2Ceil)
#include "common/bit_util.h"
#include "common/literals.h"
#include "core/hle/kernel/k_memory_block.h"
#include "core/hle/kernel/k_typed_address.h"
#include "core/hle/kernel/svc_types.h"
#include "core/hle/result.h"
#include "core/memory.h"  // Eden's k_process.h brings it in (audio_core uses Core::Memory::YUZU_PAGESIZE)

namespace Core::Memory {
class Memory;
}

namespace Kernel {

class KernelCore;

class KPageTableBase {
public:
    KProcessAddress GetHeapRegionStart() const { return 0; }
    Result QueryInfo(KMemoryInfo*, Svc::PageInfo*, KProcessAddress) const { return ResultSuccess; }
};

class KProcessPageTable {
public:
    Result LockForMapDeviceAddressSpace(bool* out_is_io, KProcessAddress, size_t, KMemoryPermission, bool, bool) {
        if (out_is_io) *out_is_io = false;
        return ResultSuccess;
    }
    Result UnlockForDeviceAddressSpace(KProcessAddress, size_t) { return ResultSuccess; }
    KPageTableBase& GetBasePageTable() { return base; }

private:
    KPageTableBase base;
};

class KProcess {
public:
    KProcessPageTable& GetPageTable() { return page_table; }
    void Open(KernelCore&) {}  // reference counting: the one process lives for the whole run
    void Close(KernelCore&) {}
    Core::Memory::Memory& GetMemory() const;
    u64 GetProgramId() const;
    // false: skips nvdrv's heap pre-mapping, which walks the kernel's memory blocks
    bool IsApplication() const { return false; }

    static KProcess& Application();  // the game

private:
    KProcessPageTable page_table;
};

} // namespace Kernel
