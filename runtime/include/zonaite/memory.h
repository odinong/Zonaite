// Guest address-space layout and memory SVCs.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace zn {

// Guest memory is Eden's Common::HostMemory: an 8 GB pagefile-backed "DRAM" (the GPU's physical view) and a 39-bit
// virtual view (the guest's address space, which the CPU uses and write-tracking protects). The first 8 GB of the
// virtual view map DRAM 1:1. Above that, only the alias region is used: svc MapPhysicalMemory maps pages of the
// physical pool there (nn::os's virtual address manager wants an alias region of several GB, like the console's).
constexpr uint64_t kDramSize = 8ull << 30;

// Region layout reported through svc GetInfo
constexpr uint64_t kAslrRegionBase = 0x0008000000ull, kAslrRegionSize = 0x01F8000000ull;  // up to 8 GB
constexpr uint64_t kHeapRegionBase = 0x0010000000ull, kHeapRegionSize = 0x0088000000ull;  // TotK's heap: 0x84400000
constexpr uint64_t kAliasRegionBase = 0x0200000000ull, kAliasRegionSize = 0x1000000000ull;  // 64 GB, above DRAM
// 0x0098000000 - 0x0150000000: physical pool, DRAM behind MapPhysicalMemory (never used at its own address). It
// backs the start of the alias region page for page, fixed: a page mapped, unmapped and mapped again gets the
// same DRAM page, so the GPU's mappings of it (taken when the game pins a buffer) stay valid. (Handing pages out
// from a free list let a freed page come back under another address while the GPU still wrote to it.)
constexpr uint64_t kPhysicalPoolBase = 0x0098000000ull, kPhysicalPoolSize = 0x00B8000000ull;  // 2.875 GB
constexpr uint64_t kFixedPhysicalSize = kPhysicalPoolSize;
// DRAM page behind alias region address `addr` in the fixed window, or 0 outside it
constexpr uint64_t fixed_physical(uint64_t addr) {
    return addr >= kAliasRegionBase && addr - kAliasRegionBase < kFixedPhysicalSize
               ? kPhysicalPoolBase + (addr - kAliasRegionBase)
               : 0;
}
constexpr uint64_t kStackRegionBase = 0x0150000000ull, kStackRegionSize = 0x0020000000ull;  // 512 MB
// 0x0190000000 - 0x01A0000000: Zonaite's own stacks and TLS blocks (kStackBase in runtime.h)
// 0x01C0000000 - 0x01E0000000: shared memory objects' backing (KSharedMemory); the guest sees it only through the
// ranges it maps with svc MapSharedMemory
constexpr uint64_t kSharedPoolBase = 0x01C0000000ull, kSharedPoolSize = 0x0020000000ull;
constexpr uint64_t kAddressSpaceEnd = 1ull << 39;
constexpr uint64_t kTotalMemorySize = 0xCD500000ull;  // application memory pool (3285 MB)

// MemoryState / permission values (subset)
constexpr uint32_t kStateFree = 0x00, kStateCode = 0x03, kStateHeap = 0x05, kStateNormal = 0x05, kStateShared = 0x06, kStateStack = 0x0B,
                   kStateInaccessible = 0x10;
constexpr uint32_t kPermRW = 3, kPermRX = 5;

// Result codes (module 1 = kernel)
constexpr uint32_t kResultInvalidMemoryRegion = 0xD401;  // 106
constexpr uint32_t kResultInvalidSize = 0xCA01, kResultOutOfMemory = 0xD001, kResultInvalidEnumValue = 0xF001;

void init_memory_map();
void map_region(uint64_t addr, uint64_t size, uint32_t state, uint32_t perm);
void unmap_region(uint64_t addr);
uint64_t heap_size();
uint64_t physical_memory_size();
uint64_t system_resource_size();
uint32_t svc_set_heap_size(uint64_t size, uint64_t* out_addr);
uint32_t svc_query_memory(uint8_t* m, uint64_t info_ptr, uint64_t addr, uint32_t* page_info);
uint32_t svc_map_memory(uint8_t* m, uint64_t dst, uint64_t src, uint64_t size);
uint32_t svc_unmap_memory(uint8_t* m, uint64_t dst, uint64_t src, uint64_t size);
uint32_t svc_map_physical_memory(uint8_t* m, uint64_t addr, uint64_t size);
uint32_t svc_unmap_physical_memory(uint64_t addr, uint64_t size);

}  // namespace zn
