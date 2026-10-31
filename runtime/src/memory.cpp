// Guest memory map: Horizon address-space regions, heap, and the memory SVCs (SetHeapSize, QueryMemory,
// MapMemory, UnmapMemory, SetMemoryPermission/Attribute).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "zonaite/gpu.h"
#include "zonaite/guest.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;

struct Region {
    uint64_t size;
    uint32_t state, perm, attr;
};

static std::mutex g_map_lock;
static std::map<uint64_t, Region> g_map;  // mapped guest ranges by start address
static uint64_t g_heap_size = 0;

static uint64_t g_physical_size = 0;  // memory mapped with MapPhysicalMemory

// Removes [addr, addr + size) from the map, trimming or splitting regions that overlap it. Caller holds the lock.
static void carve(uint64_t addr, uint64_t size) {
    const uint64_t end = addr + size;
    auto it = g_map.upper_bound(addr);
    if (it != g_map.begin()) --it;
    while (it != g_map.end() && it->first < end) {
        const uint64_t r_start = it->first, r_end = it->first + it->second.size;
        if (r_end <= addr) { ++it; continue; }
        const Region r = it->second;
        it = g_map.erase(it);
        if (r_start < addr) g_map[r_start] = Region{addr - r_start, r.state, r.perm, r.attr};
        if (r_end > end) {
            g_map[end] = Region{r_end - end, r.state, r.perm, r.attr};
            break;
        }
    }
}

void map_region(uint64_t addr, uint64_t size, uint32_t state, uint32_t perm) {
    std::lock_guard<std::mutex> l(g_map_lock);
    carve(addr, size);
    g_map[addr] = Region{size, state, perm, 0};
}

void unmap_region(uint64_t addr) {
    std::lock_guard<std::mutex> l(g_map_lock);
    g_map.erase(addr);
}

void init_memory_map() {
    for (int i = 0; i < g_image.module_count; i++) {
        const ModuleInfo& mi = g_image.modules[i];
        map_region(mi.base, mi.size, kStateCode, kPermRX);  // perms are informational: everything is RW on the host
    }
    // Zonaite's own DRAM ranges look taken to the game: it picks addresses for shared memory and stacks by
    // looking for free space, and one landing here aliased Zonaite's thread stacks (or the physical pool) and
    // corrupted memory at random. The physical pool (behind the alias region) and everything from Zonaite's
    // stacks up to the end of DRAM (stacks/TLS, shared memory backing, the GPU scratch page).
    map_region(kPhysicalPoolBase, kPhysicalPoolSize, kStateInaccessible, 0);
    map_region(kStackBase, kDramSize - kStackBase, kStateInaccessible, 0);
}

uint64_t heap_size() { return g_heap_size; }

uint64_t physical_memory_size() { return g_physical_size; }

// The system resource (kernel memory for the process's own page tables etc.) the game's NPDM asks for. nn::os
// sizes its virtual address manager from it and aborts if it is 0.
uint64_t system_resource_size() {
    static const uint64_t size = [] {
        for (const char* dir : {"/Exefs/main.npdm", "/exefs/main.npdm", "/ExeFS/main.npdm"}) {
            if (FILE* f = std::fopen((g_dump_dir + dir).c_str(), "rb")) {
                uint8_t meta[0x20]{};
                const size_t n = std::fread(meta, 1, sizeof meta, f);
                std::fclose(f);
                if (n == sizeof meta && std::memcmp(meta, "META", 4) == 0) {
                    uint32_t v;
                    std::memcpy(&v, meta + 0x14, 4);
                    return (uint64_t)v;
                }
            }
        }
        return (uint64_t)0;
    }();
    return size;
}

// ---- SVC implementations; return the Horizon result code
uint32_t svc_set_heap_size(uint64_t size, uint64_t* out_addr) {
    if (size & 0x1FFFFF) return kResultInvalidSize;  // 2 MB granularity
    if (size > kHeapRegionSize) return kResultOutOfMemory;
    if (size > g_heap_size) commit(kHeapRegionBase + g_heap_size, size - g_heap_size);
    // shrinking keeps the pages committed (cheap, and guest code never touches them again before regrowing)
    g_heap_size = size;
    if (size) map_region(kHeapRegionBase, size, kStateHeap, kPermRW);
    else unmap_region(kHeapRegionBase);
    *out_addr = kHeapRegionBase;
    log("SetHeapSize(%#llx) -> %#llx", (unsigned long long)size, (unsigned long long)kHeapRegionBase);
    return 0;
}

// MemoryInfo { u64 base; u64 size; u32 state; u32 attr; u32 perm; u32 ipc_refcount; u32 device_refcount; u32 pad; }
uint32_t svc_query_memory(uint8_t* m, uint64_t info_ptr, uint64_t addr, uint32_t* page_info) {
    std::lock_guard<std::mutex> l(g_map_lock);
    uint64_t base = 0, size = 0;
    uint32_t state = kStateFree, perm = 0, attr = 0;
    auto it = g_map.upper_bound(addr);
    if (it != g_map.begin()) {
        auto prev = std::prev(it);
        if (addr < prev->first + prev->second.size) {
            base = prev->first; size = prev->second.size;
            state = prev->second.state; perm = prev->second.perm; attr = prev->second.attr;
        }
    }
    if (!size) {  // free range: from the end of the previous mapping to the start of the next
        base = 0;
        if (it != g_map.begin()) { auto prev = std::prev(it); base = prev->first + prev->second.size; }
        uint64_t end = it != g_map.end() ? it->first : kAddressSpaceEnd;
        size = end - base;
    }
    st64(m, info_ptr + 0, base);
    st64(m, info_ptr + 8, size);
    st32(m, info_ptr + 16, state);
    st32(m, info_ptr + 20, attr);
    st32(m, info_ptr + 24, perm);
    st32(m, info_ptr + 28, 0);
    st32(m, info_ptr + 32, 0);
    st32(m, info_ptr + 36, 0);
    *page_info = 0;
    return 0;
}

// MapMemory(dst, src, size): Horizon aliases src into the stack region and makes src inaccessible meanwhile.
// With one flat guest space we move the contents instead; UnmapMemory moves them back.
uint32_t svc_map_memory(uint8_t* m, uint64_t dst, uint64_t src, uint64_t size) {
    commit(dst, size);
    std::memcpy(m + dst, m + src, size);
    map_region(dst, size, kStateStack, kPermRW);
    return 0;
}

uint32_t svc_unmap_memory(uint8_t* m, uint64_t dst, uint64_t src, uint64_t size) {
    std::memcpy(m + src, m + dst, size);
    unmap_region(dst);
    return 0;
}

// Pool pages behind each MapPhysicalMemory'd range of the alias region (guest address -> pool offset), page runs
static std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_phys_backing;  // addr -> (pool offset, size)

// MapPhysicalMemory(addr, size): back the unmapped parts of [addr, addr + size) in the alias region with fresh
// zeroed pool pages (nn::os's virtual address manager builds its own heaps this way).
uint32_t svc_map_physical_memory(uint8_t* m, uint64_t addr, uint64_t size) {
    if ((addr | size) & 0xFFF || !size) return kResultInvalidSize;
    if (addr < kAliasRegionBase || addr + size > kAliasRegionBase + kAliasRegionSize) return kResultInvalidMemoryRegion;
    std::lock_guard<std::mutex> l(g_map_lock);
    uint64_t cur = addr;
    const uint64_t end = addr + size;
    while (cur < end) {
        auto it = g_map.upper_bound(cur);
        if (it != g_map.begin()) {
            auto prev = std::prev(it);
            if (cur < prev->first + prev->second.size) {  // already mapped: skip it
                cur = prev->first + prev->second.size;
                continue;
            }
        }
        const uint64_t gap_end = std::min(end, it != g_map.end() ? it->first : end);
        std::vector<std::pair<uint64_t, uint64_t>> pieces;
        if (!fixed_physical(cur) || !fixed_physical(gap_end - 1))
            fatal(nullptr, "MapPhysicalMemory(%#llx, %#llx) is beyond the %llu MB of the alias region Zonaite backs",
                  (unsigned long long)cur, (unsigned long long)(gap_end - cur),
                  (unsigned long long)(kFixedPhysicalSize >> 20));
        pieces.emplace_back(fixed_physical(cur), gap_end - cur);
        uint64_t va = cur;
        for (auto [off, sz] : pieces) {
            gpu::remap_guest(va, off, sz);
            g_phys_backing[va] = {off, sz};
            va += sz;
        }
        std::memset(m + cur, 0, gap_end - cur);
        g_map[cur] = Region{gap_end - cur, kStateNormal, kPermRW, 0};
        g_physical_size += gap_end - cur;
        cur = gap_end;
    }
    return 0;
}

uint32_t svc_unmap_physical_memory(uint64_t addr, uint64_t size) {
    if ((addr | size) & 0xFFF || !size) return kResultInvalidSize;
    if (addr < kAliasRegionBase || addr + size > kAliasRegionBase + kAliasRegionSize) return kResultInvalidMemoryRegion;
    std::lock_guard<std::mutex> l(g_map_lock);
    const uint64_t end = addr + size;
    for (auto& [start, r] : g_map)  // count what is mapped before carving it out
        if (r.state == kStateNormal && start < end && start + r.size > addr)
            g_physical_size -= std::min(end, start + r.size) - std::max(addr, start);
    carve(addr, size);
    // forget the backing of [addr, end) (its DRAM pages stay with these addresses), splitting runs at the edges
    auto it = g_phys_backing.upper_bound(addr);
    if (it != g_phys_backing.begin()) --it;
    while (it != g_phys_backing.end() && it->first < end) {
        const uint64_t va = it->first;
        const auto [off, sz] = it->second;
        if (va + sz <= addr) { ++it; continue; }
        it = g_phys_backing.erase(it);
        const uint64_t lo = std::max(addr, va), hi = std::min(end, va + sz);
        if (va < lo) g_phys_backing[va] = {off, lo - va};
        if (va + sz > hi) g_phys_backing[hi] = {off + (hi - va), va + sz - hi};
        gpu::unmap_guest(lo, hi - lo);
    }
    return 0;
}

}  // namespace zn
