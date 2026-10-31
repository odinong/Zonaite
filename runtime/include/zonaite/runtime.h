// Runtime-wide declarations shared by the host side (not included by generated code).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "zonaite/guest.h"

namespace zn {

constexpr uint64_t kGuestSpace = 1ull << 39;          // the guest's 39-bit address space
constexpr uint64_t kThunkBase = 0x7F00000000ull;     // fake guest addresses of host functions
constexpr uint64_t kStackBase = 0x0190000000ull;     // Zonaite's thread stacks and TLS blocks (below 8 GB DRAM)
constexpr uint32_t kCurrentThreadPseudoHandle = 0xFFFF8000u;   // Horizon pseudo-handles
constexpr uint32_t kCurrentProcessPseudoHandle = 0xFFFF8001u;
constexpr uint32_t kMainThreadHandle = 0x00010001u;            // a real handle, as the kernel passes to rtld
constexpr uint64_t kMainStackSize = 0x100000;                  // main.npdm: main thread stack 0x100000
constexpr uint64_t kTlsSize = 0x200;                           // Horizon per-thread TLS block (tpidrro_el0)

struct ModuleInfo {
    std::string name;
    uint64_t base = 0, size = 0, init = 0, fini = 0;
};
struct ImageInfo {
    uint64_t base = 0, size = 0, entry = 0;
    ModuleInfo modules[8];
    int module_count = 0;
};
extern ImageInfo g_image;
extern std::string g_dump_dir;   // the user's extracted dump (Exefs/ + RomFs/)
extern std::string g_build_dir;  // build/game: image.bin, image.json, romfs index
extern uint8_t* g_mem;

uint8_t* reserve_guest_space();
bool commit(uint64_t guest, uint64_t size, bool executable_data = false);

using HostFn = void (*)(zg::Ctx*, uint8_t*);
uint64_t register_host_function(HostFn fn);
zg::GuestFn find_function(uint64_t addr);

// Calls a guest function from host code with the current register state; arguments are set by the caller.
// Saves/restores x30 and gives the callee a fresh frame below the current SP.
void guest_call(zg::Ctx* c, uint8_t* m, uint64_t addr);

zg::Ctx* create_main_thread_context();

void log(const char* fmt, ...);
// The version of the game this build was recompiled from ("1.2.1", "1.0.0"; image.json)
const std::string& game_version();
// Where the game's saves go: %APPDATA%\Zonaite\save for 1.2.1, save-<version> for others (a save from one
// version is never shown to another), or ZN_SAVE_DIR
std::string save_dir();
// The Switch system counter (19.2 MHz) since Zonaite started (svc GetSystemTick, CNTVCT_EL0)
uint64_t system_tick();
// Name of an SVC number, for logs
const char* svc_name(uint32_t n);
[[noreturn]] void fatal(zg::Ctx* c, const char* fmt, ...);
// Logs a guest thread's registers (optional), lr and frame-pointer backtrace
void log_guest_state(zg::Ctx* c, bool registers = true);

}  // namespace zn
