// Zonaite's replacement for the parts of Eden's core that video_core uses.
// Eden's headers are used as-is; this file defines the members video_core links against, backed by Zonaite's
// runtime. Compiled with Eden's include paths and flags (tools/build_runtime.py, "gpu" group).
#include <atomic>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>

extern std::atomic<unsigned long long> zn_generated_frames;  // runtime/gpu/frame_interp.cpp
static std::atomic<double> g_shown_fps{0};  // frames shown per second, interpolated ones included
extern std::atomic<unsigned long long> zn_presented_frames;

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef CreateEvent  // Core::Timing::CreateEvent below

#include "audio_core/audio_core.h"
#include "common/host_memory.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/memory.h"
#include "common/scratch_buffer.h"
#include "core/perf_stats.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"

#include "zonaite/gpu.h"
#include "zonaite/kernel.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

namespace {
constexpr size_t kDram = 8ull << 30;      // backing ("physical") size: guest addresses 0 - 8 GB
constexpr size_t kVirtual = 1ull << 39;   // the guest's 39-bit address space
constexpr size_t kPage = 4096;
constexpr size_t kChunk = 2ull << 20;  // identity views below 8 GB
}  // namespace

// ---------------------------------------------------------------- guest address -> DRAM offset
// Below 8 GB guest addresses are DRAM offsets, except ranges remapped elsewhere (shared memory); above 8 GB only
// remapped ranges exist (physical memory in the alias region). remap_guest/unmap_guest keep this table.
namespace {
std::shared_mutex g_remap_lock;
std::map<u64, std::pair<u64, u64>> g_remaps;  // guest address -> (DRAM offset, size)

std::optional<u64> to_backing(u64 vaddr) {
    std::shared_lock l(g_remap_lock);
    auto it = g_remaps.upper_bound(vaddr);
    if (it != g_remaps.begin()) {
        --it;
        if (vaddr < it->first + it->second.second) return it->second.first + (vaddr - it->first);
    }
    if (vaddr < kDram) return vaddr;
    return std::nullopt;
}

// Pages the GPU had write-protected (RasterizerMarkRegionCached), by guest page number: a remap resets the
// protection of the views Eden re-maps around it, so they are re-applied afterwards.
constexpr u64 kTrackedPages = (zn::kAliasRegionBase + zn::kAliasRegionSize) / kPage;
std::unique_ptr<std::atomic<u64>[]> g_protected{new std::atomic<u64>[kTrackedPages / 64]{}};

void note_protection(u64 start, u64 end, bool protect) {
    for (u64 page = start / kPage; page < end / kPage && page < kTrackedPages; page++) {
        const u64 bit = 1ull << (page & 63);
        if (protect) g_protected[page / 64].fetch_or(bit, std::memory_order_relaxed);
        else g_protected[page / 64].fetch_and(~bit, std::memory_order_relaxed);
    }
}

// Puts the GPU's write protection back on [start, end) after a remap re-mapped views there.
void reprotect(Common::HostMemory& buf, u64 start, u64 end) {
    end = std::min<u64>(end, kTrackedPages * kPage);
    u64 run = 0;
    bool in_run = false;
    for (u64 page = start / kPage; page <= end / kPage; page++) {
        const bool p = page < end / kPage && (g_protected[page / 64].load(std::memory_order_relaxed) >> (page & 63)) & 1;
        if (p && !in_run) { run = page; in_run = true; }
        if (!p && in_run) {
            buf.Protect(run * kPage, (page - run) * kPage, Common::MemoryPermission::Read);
            in_run = false;
        }
    }
}

// The views a remap of [guest, guest + size) can re-map: its 2 MB chunks below 8 GB, a generous window above
// (MapPhysicalMemory views are as large as the game's calls).
std::pair<u64, u64> remap_neighbourhood(u64 guest, u64 size) {
    if (guest < kDram) return {guest & ~(kChunk - 1), (guest + size + kChunk - 1) & ~(kChunk - 1)};
    constexpr u64 kWindow = 64ull << 20;
    return {guest > kWindow ? guest - kWindow : 0, guest + size + kWindow};
}

// Drops [guest, guest + size) from the table, keeping the parts of entries outside it. Caller holds the lock.
void forget_remaps(u64 guest, u64 size) {
    const u64 end = guest + size;
    auto it = g_remaps.upper_bound(guest);
    if (it != g_remaps.begin()) --it;
    while (it != g_remaps.end() && it->first < end) {
        const u64 va = it->first;
        const auto [off, sz] = it->second;
        if (va + sz <= guest) { ++it; continue; }
        it = g_remaps.erase(it);
        if (va < guest) g_remaps[va] = {off, guest - va};
        if (va + sz > end) g_remaps[end] = {off + (end - va), va + sz - end};
    }
}
}  // namespace

// ---------------------------------------------------------------- Core::DeviceMemory
namespace Core {

// Guest VA == physical address below 8 GB, mapped as 2 MB views: a remap inside DRAM (shared memory) makes Eden
// unmap and re-map the whole view around it, which resets its page protections, so views are kept small (and
// the protections are put back afterwards, see reprotect()).
DeviceMemory::DeviceMemory() : buffer{kDram, kVirtual} {
    for (u64 off = 0; off < kDram; off += kChunk)
        buffer.Map(off, off, kChunk, Common::MemoryPermission::ReadWrite, false);
}
DeviceMemory::~DeviceMemory() = default;

// ---------------------------------------------------------------- Core::System
struct System::Impl {
    Core::DeviceMemory device_memory;
    std::optional<Tegra::Host1x::Host1x> host1x;
    std::optional<Tegra::GPU> gpu;
    Timing::CoreTiming core_timing;
    PerfStats perf_stats{0x0100F2C0115B6000ull};
    Core::SpeedLimiter speed_limiter;
    std::unique_ptr<::AudioCore::AudioCore> audio_core;
    std::optional<Memory::Memory> memory;
    std::mutex stall_lock;
};

static std::optional<Tegra::GPU>* g_gpu_slot;  // where VideoCore::CreateGPU puts the GPU (zn::gpu::start)

System::System() : impl{std::make_unique<Impl>()} {
    // DeviceMemory exists now (first member of Impl); Host1x builds the GPU's device memory manager on top of it
    impl->host1x.emplace(*this);
    impl->memory.emplace(*this);
    g_gpu_slot = &impl->gpu;
}
System::~System() = default;

bool System::IsPoweredOn() const { return true; }
bool System::IsPaused() const { return false; }
bool System::IsShuttingDown() const { return false; }

// Created on first use (the first audren/audout session): it starts the ADSP threads and opens the sink.
AudioCore::AudioCore& System::AudioCore() {
    static std::once_flag once;
    std::call_once(once, [this] { impl->audio_core = std::make_unique<::AudioCore::AudioCore>(*this); });
    return *impl->audio_core;
}
void System::GatherGPUDirtyMemory(std::function<void(PAddr, size_t)>&) {}  // writes are tracked by page faults
void System::RegisterHostThread() {}
Tegra::GPU& System::GPU() { return *impl->gpu; }
Tegra::Host1x::Host1x& System::Host1x() { return *impl->host1x; }
const Tegra::Host1x::Host1x& System::Host1x() const { return *impl->host1x; }
DeviceMemory& System::DeviceMemory() { return impl->device_memory; }
const DeviceMemory& System::DeviceMemory() const { return impl->device_memory; }
Timing::CoreTiming& System::CoreTiming() { return impl->core_timing; }
const Timing::CoreTiming& System::CoreTiming() const { return impl->core_timing; }
PerfStats& System::GetPerfStats() { return impl->perf_stats; }
const PerfStats& System::GetPerfStats() const { return impl->perf_stats; }
Core::SpeedLimiter& System::SpeedLimiter() { return impl->speed_limiter; }
const Core::SpeedLimiter& System::SpeedLimiter() const { return impl->speed_limiter; }
// video_core stalls the game while it flushes or rebuilds caches; guest threads don't take this lock, the game
// already waits on the GPU's fences.
std::unique_lock<std::mutex> System::StallApplication() { return std::unique_lock{impl->stall_lock}; }
void System::UnstallApplication() {}
void System::SetNVDECActive(bool) {}
Memory::Memory& System::ApplicationMemory() { return *impl->memory; }
const Memory::Memory& System::ApplicationMemory() const { return *impl->memory; }

// ---------------------------------------------------------------- Core::Timing::CoreTiming (host clock)
namespace Timing {
// Identical to Eden's definition in core/core_timing.cpp (not compiled here); the event heap needs it complete.
struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;
    heap_t::handle_type handle{};

    friend bool operator>(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }
    friend bool operator<(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};

static const auto g_time0 = std::chrono::steady_clock::now();

// Timed events (audio_core's ADSP and device sessions): each scheduled event gets a host thread that sleeps until
// the event is due and runs its callback, instead of Eden's single event loop driven by emulated CPU time.
std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    return std::make_shared<EventType>(std::move(callback), std::move(name));
}

namespace {
std::mutex g_timer_lock;
std::map<EventType*, std::jthread> g_timers;

void run_timer(std::stop_token stop, std::shared_ptr<EventType> event, std::chrono::nanoseconds first,
               std::chrono::nanoseconds period) {
    using clock = std::chrono::steady_clock;
    auto next = clock::now() + first;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_until(next);
        if (stop.stop_requested()) break;
        const auto now = clock::now();
        const auto late = std::chrono::duration_cast<std::chrono::nanoseconds>(now - next);
        const auto again = event->callback(std::chrono::duration_cast<std::chrono::nanoseconds>(now - g_time0).count(),
                                           late);
        if (period.count() == 0 && !again) break;  // one-shot
        next += again ? *again : period;
        if (now - next > std::chrono::milliseconds(100)) next = now;  // fell far behind (debugger, hitch): resync
    }
}

void start_timer(const std::shared_ptr<EventType>& event, std::chrono::nanoseconds first,
                 std::chrono::nanoseconds period) {
    std::lock_guard l(g_timer_lock);
    g_timers.erase(event.get());  // rescheduling replaces the old schedule
    g_timers.emplace(event.get(), std::jthread(run_timer, event, first, period));
}
}  // namespace

CoreTiming::CoreTiming() = default;
CoreTiming::~CoreTiming() = default;

void CoreTiming::ScheduleEvent(std::chrono::nanoseconds ns_into_future, const std::shared_ptr<EventType>& event_type,
                               bool absolute_time) {
    const auto first = absolute_time ? ns_into_future - GetGlobalTimeNs() : ns_into_future;
    start_timer(event_type, first, std::chrono::nanoseconds::zero());
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time, std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type, bool absolute_time) {
    const auto first = absolute_time ? start_time - GetGlobalTimeNs() : start_time;
    start_timer(event_type, first, resched_time);
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type, UnscheduleEventType) {
    std::jthread t;
    {
        std::lock_guard l(g_timer_lock);
        auto it = g_timers.find(event_type.get());
        if (it == g_timers.end()) return;
        t = std::move(it->second);
        g_timers.erase(it);
    }
    if (t.get_id() == std::this_thread::get_id()) t.detach();  // unscheduled from its own callback
}
std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const noexcept {
    return std::chrono::steady_clock::now() - g_time0;
}
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(GetGlobalTimeNs());
}
u64 CoreTiming::GetGPUTicks() const {  // GM20B timestamp clock: 614.4 MHz
    const u64 ns = (u64)GetGlobalTimeNs().count();
    return ns / 625 * 384 + ns % 625 * 384 / 625;
}
}  // namespace Timing

// ---------------------------------------------------------------- Core::PerfStats
PerfStats::PerfStats(u64 title_id_) : title_id{title_id_} {}
PerfStats::~PerfStats() = default;
void PerfStats::EndGameFrame() { game_frames.fetch_add(1, std::memory_order_relaxed); }
void PerfStats::BeginSystemFrame() {}
// A frame was presented (nvdisp_disp0 flip): log the frame rate once a second
void PerfStats::EndSystemFrame() {
    static auto window_start = std::chrono::steady_clock::now();
    static auto last = window_start;
    static u32 frames = 0;
    static double worst_ms = 0;  // the longest frame in this window (stutter shows here, not in the average)
    frames++;
    const auto now = std::chrono::steady_clock::now();
    worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(now - last).count());
    last = now;
    if (now - window_start >= std::chrono::seconds(1)) {
        const double secs = std::chrono::duration<double>(now - window_start).count();
        const double fps = frames / secs;
        static u64 generated_before = 0;
        const u64 generated = ::zn_generated_frames.load(std::memory_order_relaxed);
        static u64 presented_before = 0;
        const u64 presented_now = ::zn_presented_frames.load(std::memory_order_relaxed);
        zn::log("fps: %.1f worst %.1f ms shown %.1f presented %.1f", fps, worst_ms,
                (frames + generated - generated_before) / secs, (presented_now - presented_before) / secs);
        g_shown_fps.store((frames + generated - generated_before) / secs, std::memory_order_relaxed);
        generated_before = generated;
        presented_before = presented_now;
        zn::gpu::note_fps(fps);
        frames = 0;
        worst_ms = 0;
        window_start = now;
    }
}

// The guest runs in real time: nothing to throttle.
void SpeedLimiter::DoSpeedLimiting(std::chrono::microseconds) {}

// ---------------------------------------------------------------- Core::Memory::Memory
namespace Memory {
struct Memory::Impl {};
Memory::Memory(Core::System& system_) : system{system_} {}
Memory::~Memory() = default;

// Backing (physical-view) pointer: the device memory manager derives physical addresses from it.
u8* Memory::GetPointerSilent(Common::ProcessAddress vaddr) {
    auto& buf = zn::gpu::system().DeviceMemory().buffer;
    const auto b = to_backing(GetInteger(vaddr));
    if (!b) {
        // Unmapped now, but alias pages have fixed DRAM pages: the GPU's mapping stays right when the game maps
        // the page (again) later. TotK frees tail pages of a physical-memory heap that a GPU buffer still spans.
        if (const u64 fixed = zn::fixed_physical(GetInteger(vaddr))) return buf.BackingBasePointer() + fixed;
        static std::atomic<int> reported{0};
        if (reported.fetch_add(1) < 4)
            zn::log("gpu: no memory behind guest address %#llx", (unsigned long long)GetInteger(vaddr));
        return nullptr;
    }
    return buf.BackingBasePointer() + *b;
}

// Through the guest's view, so ranges that span remapped pieces read correctly
u8* Memory::GetSpan(const VAddr src_addr, const std::size_t size) {
    if (!to_backing(src_addr) || !to_backing(src_addr + size - 1)) return nullptr;
    return zn::gpu::system().DeviceMemory().buffer.VirtualBasePointer() + src_addr;
}

bool Memory::ReadBlockUnsafe(Common::ProcessAddress src_addr, void* dest_buffer, std::size_t size) {
    return ReadBlock(src_addr, dest_buffer, size);
}

bool Memory::WriteBlockUnsafe(Common::ProcessAddress dest_addr, const void* src_buffer, std::size_t size) {
    u8* p = GetSpan(GetInteger(dest_addr), size);
    if (!p) return false;
    std::memcpy(p, src_buffer, size);  // through the guest view, so GPU write tracking sees it
    return true;
}

bool Memory::ZeroBlock(Common::ProcessAddress dest_addr, std::size_t size) {
    u8* p = GetSpan(GetInteger(dest_addr), size);
    if (!p) return false;
    std::memset(p, 0, size);
    return true;
}

void Memory::Write32(Common::ProcessAddress addr, u32 data) { WriteBlockUnsafe(addr, &data, sizeof data); }

bool Memory::ReadBlock(Common::ProcessAddress src_addr, void* dest_buffer, std::size_t size) {
    u8* p = GetSpan(GetInteger(src_addr), size);
    if (!p) return false;
    std::memcpy(dest_buffer, p, size);
    return true;
}

// The GPU caches this guest range: make the guest's (virtual) view read-only so the next CPU write faults and
// invalidates the GPU's copy (Eden's fastmem/NCE scheme). The GPU's own accesses use the backing view.
void Memory::RasterizerMarkRegionCached(Common::ProcessAddress vaddr, u64 size, bool cached) {
    const u64 a = GetInteger(vaddr);
    if (a == 0 || !to_backing(a)) return;
    const u64 start = a & ~(kPage - 1), end = (a + size + kPage - 1) & ~(kPage - 1);
    zn::gpu::system().DeviceMemory().buffer.Protect(
        start, end - start,
        cached ? Common::MemoryPermission::Read : Common::MemoryPermission::ReadWrite);
    note_protection(start, end, cached);
}
}  // namespace Memory

}  // namespace Core

// ---------------------------------------------------------------- Zonaite side
namespace zn::gpu {

static Core::System* g_system;
static std::atomic<bool> g_gpu_ready{false};

Core::System& system() { return *g_system; }
std::optional<Tegra::GPU>& gpu_slot() { return *Core::g_gpu_slot; }
void mark_gpu_ready() { g_gpu_ready.store(true, std::memory_order_release); }

static std::atomic<double> g_fps{0};
void note_fps(double fps) { g_fps.store(fps, std::memory_order_relaxed); }
double fps() { return g_fps.load(std::memory_order_relaxed); }
double shown_fps() { return g_shown_fps.load(std::memory_order_relaxed); }

// A guest write hit a page the GPU caches: invalidate the GPU's copy of the page, unprotect it, retry the write.
//
// Also: a partial unmap (UnmapPhysicalMemory of part of a mapping) briefly unmaps the whole Windows view and maps
// the surviving parts back, so another thread can fault on memory that stays mapped. to_backing waits for the
// remap in progress (it takes the remap lock); if the address is mapped afterwards, the access is retried.
static LONG CALLBACK on_fault_tracking(EXCEPTION_POINTERS* e);

static LONG CALLBACK on_fault_tracking(EXCEPTION_POINTERS* e) {
    const auto* r = e->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    auto& buf = g_system->DeviceMemory().buffer;
    u8* addr = (u8*)r->ExceptionInformation[1];
    if (!buf.IsInVirtualRange(addr)) return EXCEPTION_CONTINUE_SEARCH;
    const u64 vaddr = (u64)(addr - buf.VirtualBasePointer());
    const auto backing = to_backing(vaddr);
    if (!backing) return EXCEPTION_CONTINUE_SEARCH;
    if (r->ExceptionInformation[0] != 1) {  // a read: only the remap race explains it; retry, but not forever
        thread_local u64 last_fault = 0;
        thread_local u32 repeats = 0;
        repeats = vaddr == last_fault ? repeats + 1 : 0;
        last_fault = vaddr;
        return repeats < 100000 ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
    }
    const u64 page = vaddr & ~(kPage - 1);
    if (g_gpu_ready.load(std::memory_order_acquire)) {
        thread_local Common::ScratchBuffer<u32> scratch;
        g_system->Host1x().MemoryManager().ApplyOpOnPointer(
            buf.BackingBasePointer() + (*backing & ~(kPage - 1)), scratch,
            [](DAddr device_addr) { g_system->GPU().InvalidateRegion(device_addr, kPage); });
    }
    buf.Protect(page, kPage, Common::MemoryPermission::ReadWrite);
    note_protection(page, page + kPage, false);
    return EXCEPTION_CONTINUE_EXECUTION;
}

void remap_guest(uint64_t guest, uint64_t backing, uint64_t size) {
    auto& buf = g_system->DeviceMemory().buffer;
    std::unique_lock l(g_remap_lock);
    forget_remaps(guest, size);
    if (guest != backing) g_remaps[guest] = {backing, size};
    if (guest < kDram) buf.Unmap(guest, size, false);  // above DRAM the range is an unmapped placeholder already
    buf.Map(guest, backing, size, Common::MemoryPermission::ReadWrite, false);
    note_protection(guest, guest + size, false);  // new memory there: nothing cached of it
    const auto [lo, hi] = remap_neighbourhood(guest, size);
    reprotect(buf, lo, hi);
}

void unmap_guest(uint64_t guest, uint64_t size) {
    auto& buf = g_system->DeviceMemory().buffer;
    std::unique_lock l(g_remap_lock);
    forget_remaps(guest, size);
    buf.Unmap(guest, size, false);
    note_protection(guest, guest + size, false);
    const auto [lo, hi] = remap_neighbourhood(guest, size);
    reprotect(buf, lo, hi);
}

uint8_t* backing_pointer(uint64_t backing) { return g_system->DeviceMemory().buffer.BackingBasePointer() + backing; }

// Eden keeps its data (log, shader caches) in "user" next to the executable when that folder exists ("portable
// mode"), else in %APPDATA%\eden, which may belong to an Eden install. Zonaite always uses the portable folder.
// Its log goes to the console too: ZN_EDEN_LOG sets the filter (Eden's syntax, e.g. "*:Info Render:Debug").
static void init_eden_paths_and_log() {
    wchar_t exe[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::filesystem::path dir = std::filesystem::path(std::wstring(exe, n)).parent_path() / "user";
    std::error_code ec;
    std::filesystem::create_directories(dir / "cache", ec);  // Eden creates cache/shader but not its parent
    const char* filter = std::getenv("ZN_EDEN_LOG");
    Settings::values.log_filter.SetValue(filter ? filter : "*:Warning");
    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();
}

uint8_t* create_system() {
    init_eden_paths_and_log();
    Settings::values.memory_layout_mode.SetValue(Settings::MemoryLayout::Memory_8Gb);
    g_system = new Core::System();
    AddVectoredExceptionHandler(1, on_fault_tracking);
    return g_system->DeviceMemory().buffer.VirtualBasePointer();
}

}  // namespace zn::gpu

// For crash reports (runtime/src/kernel.cpp): host addresses and stacks, named from the exe's PDB
#include <dbghelp.h>

std::string zn::gpu::host_symbol(void* addr) {
    static std::mutex lock;
    std::lock_guard l(lock);
    static bool init = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 off = 0;
    char out[320];
    if (init && SymFromAddr(GetCurrentProcess(), (DWORD64)addr, &off, sym))
        std::snprintf(out, sizeof out, "%s+%#llx", sym->Name, (unsigned long long)off);
    else
        std::snprintf(out, sizeof out, "%p", addr);
    return out;
}

extern "C" void zonaite_log_host_stack() {
    static std::mutex lock;
    std::lock_guard l(lock);
    static bool init = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    void* frames[24];
    const USHORT n = CaptureStackBackTrace(1, 24, frames, nullptr);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    for (USHORT i = 0; i < n; i++) {
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 255;
        DWORD64 off = 0;
        if (init && SymFromAddr(GetCurrentProcess(), (DWORD64)frames[i], &off, sym))
            zn::log("  host #%u %s+%#llx", i, sym->Name, (unsigned long long)off);
        else
            zn::log("  host #%u %p", i, frames[i]);
    }
}
