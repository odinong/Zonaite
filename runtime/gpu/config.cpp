// Zonaite's settings (zonaite/config.h): defaults, user/zonaite.ini, and how they map onto Eden's renderer.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "common/settings.h"

#include "video_core/renderer_vulkan/zonaite_live.h"
#include "zonaite/config.h"
#include "zonaite/gpu.h"
#include "zonaite/runtime.h"

namespace zn::cfg {

namespace {

std::mutex g_mutex;
Config g_config = [] {
    Config c;
    reset_bindings(c);
    return c;
}();
std::atomic<uint64_t> g_generation{1};

const char* const kControlNames[kControlCount] = {
    "A", "B", "X", "Y", "L", "R", "ZL", "ZR", "+", "-", "Left stick click", "Right stick click",
    "D-pad up", "D-pad down", "D-pad left", "D-pad right",
    "Move forward", "Move back", "Move left", "Move right",
    "Camera up", "Camera down", "Camera left", "Camera right",
};
const char* const kControlKeys[kControlCount] = {
    "a", "b", "x", "y", "l", "r", "zl", "zr", "plus", "minus", "lstick", "rstick",
    "dup", "ddown", "dleft", "dright", "lup", "ldown", "lleft", "lright", "rup", "rdown", "rleft", "rright",
};
const char* const kPadNames[kPadCount] = {
    "A", "B", "X", "Y", "LB", "RB", "LT", "RT", "Start", "Back", "Left stick", "Right stick",
    "D-pad up", "D-pad down", "D-pad left", "D-pad right",
};

std::string exe_dir() {
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring w(path, n);
    w = w.substr(0, w.find_last_of(L"\\/"));
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), len, nullptr, nullptr);
    return s;
}

std::string ini_path() {
    return user_dir() + "\\zonaite.ini";
}

// One table drives both reading and writing the ini: name, and a pointer into the Config
struct Field {
    const char* name;
    int Config::*i = nullptr;
    bool Config::*b = nullptr;
    float Config::*f = nullptr;
    std::string Config::*s = nullptr;
};

const Field kFields[] = {
    {"video.display_mode", &Config::display_mode}, {"video.window_w", &Config::window_w},
    {"video.window_h", &Config::window_h}, {"video.monitor", &Config::monitor}, {"video.vsync", &Config::vsync},
    {"video.frame_gen", &Config::frame_gen}, {"video.target_rate", &Config::target_rate},
    {"video.interp_mode", &Config::interp_mode}, {"video.pacing", nullptr, &Config::pacing},
    {"video.aspect", &Config::aspect}, {"video.perf_overlay", &Config::perf_overlay},
    {"video.overlay_corner", &Config::overlay_corner}, {"video.show_hint", nullptr, &Config::show_hint},
    {"video.ui_scale", nullptr, nullptr, &Config::ui_scale},
    {"graphics.resolution", &Config::resolution}, {"graphics.scaling_filter", &Config::scaling_filter},
    {"graphics.fsr_sharpness", &Config::fsr_sharpness}, {"graphics.anti_aliasing", &Config::anti_aliasing},
    {"graphics.anisotropy", &Config::anisotropy}, {"graphics.async_shaders", nullptr, &Config::async_shaders},
    {"graphics.gpu_accuracy", &Config::gpu_accuracy}, {"graphics.astc_decode", &Config::astc_decode},
    {"graphics.astc_recompression", &Config::astc_recompression}, {"graphics.vram_mode", &Config::vram_mode},
    {"graphics.driver_pipeline_cache", nullptr, &Config::driver_pipeline_cache},
    {"graphics.fix_bloom", nullptr, &Config::fix_bloom}, {"graphics.gpu_clock", &Config::gpu_clock},
    {"picture.brightness", nullptr, nullptr, &Config::brightness},
    {"picture.contrast", nullptr, nullptr, &Config::contrast},
    {"picture.saturation", nullptr, nullptr, &Config::saturation},
    {"picture.gamma", nullptr, nullptr, &Config::gamma}, {"picture.sharpen", nullptr, nullptr, &Config::sharpen},
    {"audio.volume", &Config::volume}, {"audio.mute", nullptr, &Config::mute},
    {"audio.mute_unfocused", nullptr, &Config::mute_unfocused},
    {"audio.device", nullptr, nullptr, nullptr, &Config::audio_device},
    {"audio.latency_ms", &Config::audio_latency_ms},
    {"input.pad_index", &Config::pad_index},
    {"input.deadzone_left", nullptr, nullptr, &Config::deadzone_left},
    {"input.deadzone_right", nullptr, nullptr, &Config::deadzone_right},
    {"input.outer_deadzone", nullptr, nullptr, &Config::outer_deadzone},
    {"input.response_curve", nullptr, nullptr, &Config::response_curve},
    {"input.camera_speed_x", nullptr, nullptr, &Config::camera_speed_x},
    {"input.camera_speed_y", nullptr, nullptr, &Config::camera_speed_y},
    {"input.invert_x", nullptr, &Config::invert_x}, {"input.invert_y", nullptr, &Config::invert_y},
    {"input.swap_sticks", nullptr, &Config::swap_sticks}, {"input.trigger_threshold", &Config::trigger_threshold},
    {"input.mouse_camera", nullptr, &Config::mouse_camera},
    {"input.mouse_sensitivity", nullptr, nullptr, &Config::mouse_sensitivity},
    {"input.mouse_invert_y", nullptr, &Config::mouse_invert_y},
    {"input.keyboard_unfocused", nullptr, &Config::keyboard_unfocused},
    {"input.rumble", nullptr, &Config::rumble}, {"input.rumble_strength", &Config::rumble_strength},
    {"keys.menu", &Config::menu_key}, {"keys.screenshot", &Config::screenshot_key},
    {"keys.fullscreen", &Config::fullscreen_key}, {"keys.hud", &Config::hud_key},
};

// One ini line's value into the config (false: not a setting)
bool set_field(Config& c, const std::string& key, const std::string& value) {
    for (const auto& f : kFields) {
        if (key != f.name) continue;
        if (f.i) c.*f.i = std::atoi(value.c_str());
        if (f.b) c.*f.b = value == "1" || value == "true";
        if (f.f) c.*f.f = static_cast<float>(std::atof(value.c_str()));
        if (f.s) c.*f.s = value;
        return true;
    }
    if (key.rfind("bind.", 0) == 0) {
        for (int i = 0; i < kControlCount; ++i) {
            if (key.substr(5) != kControlKeys[i]) continue;
            Binding b;
            std::sscanf(value.c_str(), "%d,%d,%d", &b.keys[0], &b.keys[1], &b.pad);
            c.binds[i] = b;
            return true;
        }
    }
    return false;
}

}  // namespace

std::mutex& mutex() { return g_mutex; }
Config& edit() { return g_config; }

Config get_copy() {
    std::scoped_lock lock{g_mutex};
    return g_config;
}

uint64_t generation() { return g_generation.load(std::memory_order_acquire); }

void changed() {
    g_generation.fetch_add(1, std::memory_order_acq_rel);
    save();
}

std::string user_dir() {
    static const std::string dir = exe_dir() + "\\user";
    return dir;
}

void reset_bindings(Config& c, bool xbox_labels) {
    auto set = [&](Control ctl, int k1, int k2, int pad) { c.binds[ctl] = Binding{{k1, k2}, pad}; };
    // the pad's face buttons by position (Switch layout: the bottom button is B), or by their Xbox labels
    set(kA, 'E', 0, xbox_labels ? kPadA : kPadB);
    set(kB, 'Q', 0, xbox_labels ? kPadB : kPadA);
    set(kX, VK_SPACE, 0, xbox_labels ? kPadX : kPadY);
    set(kY, 'F', 0, xbox_labels ? kPadY : kPadX);
    set(kL, '1', 0, kPadLB);
    set(kR, '3', VK_RBUTTON, kPadRB);
    set(kZL, 'Z', 0, kPadLT);
    set(kZR, 'C', VK_LBUTTON, kPadRT);
    set(kPlus, VK_RETURN, 0, kPadStart);
    set(kMinus, VK_BACK, 0, kPadBack);
    set(kLStick, VK_LCONTROL, 0, kPadLS);
    set(kRStick, 'V', VK_MBUTTON, kPadRS);
    set(kDUp, VK_UP, 0, kPadUp);
    set(kDDown, VK_DOWN, 0, kPadDown);
    set(kDLeft, VK_LEFT, 0, kPadLeft);
    set(kDRight, VK_RIGHT, 0, kPadRight);
    set(kLUp, 'W', 0, kPadNone);
    set(kLDown, 'S', 0, kPadNone);
    set(kLLeft, 'A', 0, kPadNone);
    set(kLRight, 'D', 0, kPadNone);
    set(kRUp, 'I', 0, kPadNone);
    set(kRDown, 'K', 0, kPadNone);
    set(kRLeft, 'J', 0, kPadNone);
    set(kRRight, 'L', 0, kPadNone);
}

void load() {
    std::ifstream in(ini_path());
    if (!in) return;
    std::scoped_lock lock{g_mutex};
    Config& c = g_config;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        set_field(c, line.substr(0, eq), line.substr(eq + 1));
    }
    zn::log("settings: loaded %s", ini_path().c_str());
}

void save() {
    Config c = get_copy();
    CreateDirectoryA(user_dir().c_str(), nullptr);
    std::ofstream out(ini_path(), std::ios::trunc);
    if (!out) return;
    out << "# Zonaite settings (edit in the game with F1)\n";
    for (const auto& f : kFields) {
        out << f.name << '=';
        if (f.i) out << c.*f.i;
        if (f.b) out << (c.*f.b ? 1 : 0);
        if (f.f) out << c.*f.f;
        if (f.s) out << c.*f.s;
        out << '\n';
    }
    for (int i = 0; i < kControlCount; ++i)
        out << "bind." << kControlKeys[i] << '=' << c.binds[i].keys[0] << ',' << c.binds[i].keys[1] << ','
            << c.binds[i].pad << '\n';
}

const char* control_name(int control) {
    return control >= 0 && control < kControlCount ? kControlNames[control] : "?";
}

const char* pad_input_name(int pad) {
    return pad >= 0 && pad < kPadCount ? kPadNames[pad] : "-";
}

std::string key_name(int vk) {
    switch (vk) {
    case 0: return "-";
    case VK_LBUTTON: return "Mouse left";
    case VK_RBUTTON: return "Mouse right";
    case VK_MBUTTON: return "Mouse middle";
    case VK_XBUTTON1: return "Mouse 4";
    case VK_XBUTTON2: return "Mouse 5";
    case VK_LCONTROL: return "Left Ctrl";
    case VK_RCONTROL: return "Right Ctrl";
    case VK_LSHIFT: return "Left Shift";
    case VK_RSHIFT: return "Right Shift";
    case VK_LMENU: return "Left Alt";
    case VK_RMENU: return "Right Alt";
    }
    UINT scan = MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
    switch (vk) {  // extended keys, or GetKeyNameText names the numpad ones
    case VK_LEFT: case VK_UP: case VK_RIGHT: case VK_DOWN: case VK_PRIOR: case VK_NEXT: case VK_END:
    case VK_HOME: case VK_INSERT: case VK_DELETE: case VK_DIVIDE: case VK_NUMLOCK:
        scan |= 0x100;
    }
    char name[64] = {};
    if (scan && GetKeyNameTextA(static_cast<LONG>(scan << 16), name, sizeof name) > 0) return name;
    char buf[16];
    std::snprintf(buf, sizeof buf, "Key %d", vk);
    return buf;
}

namespace {

// Settings Eden reads only when it starts, changed in the menu: waiting for the GPU thread (ZonaiteTakeLiveChange)
std::mutex g_live_mutex;
Vulkan::ZonaiteLiveChange g_live;
bool g_live_pending = false;
Config g_live_config;

template <typename E>
E clamped(int v, int lo, int hi) {
    return static_cast<E>(std::clamp(v, lo, hi));
}

// The settings applied on the GPU thread, between two frames
void apply_deferred(const Config& c) {
    auto& v = Settings::values;
    v.resolution_setup.SetValue(clamped<Settings::ResolutionSetup>(c.resolution, 0, 12));
    Settings::UpdateRescalingInfo();
    v.max_anisotropy.SetValue(clamped<Settings::AnisotropyMode>(c.anisotropy, 0, 5));
    v.gpu_accuracy.SetValue(clamped<Settings::GpuAccuracy>(c.gpu_accuracy, 0, 1));
    Settings::UpdateGPUAccuracy();
    v.accelerate_astc.SetValue(clamped<Settings::AstcDecodeMode>(c.astc_decode, 0, 2));
    v.astc_recompression.SetValue(clamped<Settings::AstcRecompression>(c.astc_recompression, 0, 2));
    v.vram_usage_mode.SetValue(clamped<Settings::VramUsageMode>(c.vram_mode, 0, 1));
}

}  // namespace

bool picture_neutral(const Config& c) {
    return c.brightness == 0.0f && c.contrast == 1.0f && c.saturation == 1.0f && c.gamma == 1.0f &&
           c.sharpen == 0.0f;
}

int frame_multiplier(const Config& c) {
    // ZN_FRAMEGEN=N (debugging) overrides the menu
    const char* fg = std::getenv("ZN_FRAMEGEN");
    const int wanted = fg ? std::atoi(fg) : c.frame_gen != 0 ? c.frame_gen : (gpu::refresh_rate() + 15) / 30;
    return std::clamp(wanted, 1, 4);
}

void apply_renderer(const Config& c, bool startup) {
    auto& v = Settings::values;
    // read as the renderer goes: they take effect with the next frame (a new vsync mode recreates the swapchain)
    v.scaling_filter.SetValue(clamped<Settings::ScalingFilter>(c.scaling_filter, 0, 14));
    v.fsr_sharpening_slider.SetValue(std::clamp(c.fsr_sharpness, 0, 200));
    v.anti_aliasing.SetValue(clamped<Settings::AntiAliasing>(c.anti_aliasing, 0, 2));
    v.aspect_ratio.SetValue(clamped<Settings::AspectRatio>(c.aspect, 0, 4));
    v.frame_gen_target_rate.SetValue(static_cast<u32>(std::clamp(c.target_rate, 0, 240)));
    v.use_asynchronous_shaders.SetValue(c.async_shaders && std::getenv("ZN_SYNC_SHADERS") == nullptr);
    v.fix_bloom_effects.SetValue(c.fix_bloom);
    v.vsync_mode.SetValue(clamped<Settings::VSyncMode>(c.vsync, 0, 3));
    v.gpu_clock.SetValue(clamped<Settings::GpuClock>(c.gpu_clock, 0, 2));
    // Frame interpolation (runtime/gpu/frame_interp.cpp): N frames shown per game frame. The present manager
    // keeps frames for the most it can use, so this changes while playing too (tools/eden_patches.py).
    const int multiplier = frame_multiplier(c);
    if (startup || multiplier != (v.frame_gen.GetValue() ? static_cast<int>(v.frame_gen_multiplier.GetValue()) : 1))
        zn::log("display: %d Hz, showing %d frames per game frame", gpu::refresh_rate(), multiplier);
    v.frame_gen_multiplier.SetValue(static_cast<u32>(std::max(multiplier, 2)));
    v.frame_gen.SetValue(multiplier > 1);
    if (startup) {
        v.use_vulkan_driver_pipeline_cache.SetValue(c.driver_pipeline_cache);
        apply_deferred(c);
        return;
    }
    // the rest decide how textures, samplers and memory limits are made: the GPU thread remakes those between
    // two frames, with the new values
    std::scoped_lock lock{g_live_mutex};
    auto& l = g_live;
    l.rescale |= clamped<Settings::ResolutionSetup>(c.resolution, 0, 12) != v.resolution_setup.GetValue();
    l.textures |= clamped<Settings::AstcDecodeMode>(c.astc_decode, 0, 2) != v.accelerate_astc.GetValue() ||
                  clamped<Settings::AstcRecompression>(c.astc_recompression, 0, 2) !=
                      v.astc_recompression.GetValue();
    l.samplers |= clamped<Settings::AnisotropyMode>(c.anisotropy, 0, 5) != v.max_anisotropy.GetValue();
    l.memory |= l.rescale || clamped<Settings::VramUsageMode>(c.vram_mode, 0, 1) != v.vram_usage_mode.GetValue();
    const bool accuracy = clamped<Settings::GpuAccuracy>(c.gpu_accuracy, 0, 1) != v.gpu_accuracy.GetValue();
    g_live_config = c;
    g_live_pending |= l.rescale || l.textures || l.samplers || l.memory || accuracy;
}

void request_shader_cache_clear() {
    std::scoped_lock lock{g_live_mutex};
    if (!g_live_pending) g_live_config = get_copy();
    g_live.shaders = true;
    g_live_pending = true;
}

}  // namespace zn::cfg

namespace Vulkan {

bool ZonaiteTakeLiveChange(ZonaiteLiveChange& change) {
    using namespace zn::cfg;
    std::scoped_lock lock{g_live_mutex};
    if (!g_live_pending) {
        return false;
    }
    change = std::move(g_live);
    change.apply = [c = g_live_config] { apply_deferred(c); };
    g_live = {};
    g_live_pending = false;
    return true;
}

}  // namespace Vulkan
