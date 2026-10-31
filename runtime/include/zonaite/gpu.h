// Zonaite's interface to Eden's GPU (runtime/gpu). Usable from runtime code that doesn't include Eden headers.
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Core {
class System;
}
namespace Tegra {
class GPU;
}

namespace zn::gpu {

// Creates the emulated system (guest memory as Eden HostMemory, Host1x/SMMU) and returns the guest memory base
// (the 39-bit virtual view). Must be the first thing the runtime does.
uint8_t* create_system();

// Opens the window and starts the Vulkan renderer and the GPU thread. Returns false with a reason on failure.
bool start(std::string* error);

Core::System& system();
std::optional<Tegra::GPU>& gpu_slot();
void mark_gpu_ready();

// Points guest range [guest, guest + size) at DRAM backing [backing, backing + size) (shared memory, physical
// memory), or back at itself when backing == guest. Page aligned. The GPU's view follows (Memory::GetPointerSilent).
void remap_guest(uint64_t guest, uint64_t backing, uint64_t size);
// Makes a guest range above DRAM inaccessible again (UnmapPhysicalMemory).
void unmap_guest(uint64_t guest, uint64_t size);
// The host pointer to DRAM backing offset `backing` (a guest physical address).
uint8_t* backing_pointer(uint64_t backing);

// vi: creates a managed layer on the default display for an applet (am's CreateManagedDisplayLayer), returning
// its layer id; the applet then opens it with vi OpenLayer.
uint64_t create_managed_layer(uint64_t aruid);

// Name of the host function at `addr` (the exe's PDB), for crash reports
std::string host_symbol(void* addr);

// True while the game window has keyboard focus (keyboard input is only read then)
bool window_focused();

// Saves the next presented frame as a PNG (F12 does this too; ZN_SCREENSHOT_FRAMES=n,m,... at those frames).
void request_screenshot(const std::string& path);

// Presented frames per second, measured over the last second (0 before the first present).
double fps();
void note_fps(double fps);
// Frames shown per second (game frames plus interpolated ones), measured like fps()
double shown_fps();

// The window handle (for the settings menu and display modes)
void* window_handle();
// Applies the display settings (windowed size, fullscreen, monitor) on the window thread
void request_display_apply();
std::vector<std::string> monitor_names();
int refresh_rate();
// F3: the HUD (everything the game draws over the finished scene) is left out; not saved
bool hud_hidden();
void set_hud_hidden(bool hidden);
// The game's picture inside the presented frame, as fractions of it: x, y, width, height (less than the whole
// frame when the window's shape differs from the picture's and it is boxed in with black bars)
std::array<float, 4> content_rect();

}  // namespace zn::gpu

namespace zn::ui {

// The settings menu (runtime/gpu/ui.cpp)
bool menu_open();                  // the game gets no input while it is
bool capturing_input();            // the menu waits for a key or button to bind
// Called by the window thread for every message; true when the menu consumed it
bool handle_message(void* hwnd, unsigned msg, uint64_t wparam, int64_t lparam);
void toggle_menu();
// Present-interval history for the performance overlay (the present thread feeds it)
void note_present_interval(float ms);

}  // namespace zn::ui

namespace zn::input {

// The first gamepad input pressed right now (cfg::PadInput), or -1: the menu's "press a button" binding
int poll_pad_press();
// True while the mouse drives the camera (the window hides and holds the cursor)
bool mouse_captured();
// Gamepad connection, for the menu: the XInput slot in use, or -1
int pad_slot();
void test_rumble();  // half a second on both motors
// The gamepad sticks as read (raw) and after the deadzones and curve (shaped): lx ly rx ry, -1..1
void stick_preview(float raw[4], float shaped[4]);

}  // namespace zn::input

namespace zn::audio {

// Active output devices: (WASAPI endpoint id, friendly name)
std::vector<std::pair<std::string, std::string>> output_devices();

}  // namespace zn::audio
