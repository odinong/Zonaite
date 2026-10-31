// Controller input: player 1 is a Pro Controller, fed from an XInput gamepad, the keyboard and the mouse, through
// the bindings and stick settings of the settings menu (zonaite/config.h). Written into hid's shared memory (layout:
// Eden's hid_core headers) 200 times a second, as the console's hid service does; the game reads it from there.
//
// Default keyboard: WASD move, IJKL camera, arrows d-pad, E = A, Q = B, Space = X, F = Y, 1 = L, 3 = R, Z = ZL,
// C = ZR, Enter = +, Backspace = -, Left Ctrl = left stick click, V = right stick click; mouse left = ZR,
// right = R, middle = right stick click. With the mouse camera on, the mouse turns the camera (Alt frees the
// cursor).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <xinput.h>

#include "hid_core/resources/shared_memory_format.h"

#include "zonaite/config.h"
#include "zonaite/gpu.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

namespace zn {

namespace {

using namespace Service::HID;
using Core::HID::NpadButton;

constexpr s32 kStickMax = 32767;

// XInput, loaded at run time so a machine without it still runs (keyboard only)
using XInputGetStateFn = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
XInputGetStateFn load_xinput() {
    for (const wchar_t* dll : {L"xinput1_4.dll", L"xinput9_1_0.dll"})
        if (HMODULE m = LoadLibraryW(dll)) return (XInputGetStateFn)GetProcAddress(m, "XInputGetState");
    return nullptr;
}
const XInputGetStateFn g_xinput = load_xinput();
using XInputSetStateFn = DWORD(WINAPI*)(DWORD, XINPUT_VIBRATION*);
const XInputSetStateFn g_xinput_set = [] {
    for (const wchar_t* dll : {L"xinput1_4.dll", L"xinput9_1_0.dll"})
        if (HMODULE m = GetModuleHandleW(dll)) return (XInputSetStateFn)GetProcAddress(m, "XInputSetState");
    return (XInputSetStateFn) nullptr;
}();
// The game's vibration (svc_hid.cpp) for the left and right motor, 0 .. 1, and when it last sent any (ms)
std::array<std::atomic<float>, 2> g_rumble{};
std::atomic<s64> g_rumble_at{0};
std::atomic<s64> g_rumble_test_until{0};  // the menu's rumble test
std::atomic<int> g_pad_slot{-1};
std::atomic<bool> g_mouse_captured{false};
std::array<std::atomic<float>, 8> g_sticks{};  // raw lx ly rx ry, then shaped (the menu draws them)

u64 bit(NpadButton b) { return static_cast<u64>(b); }

// The Switch button each button control drives (cfg::Control order)
constexpr NpadButton kControlButtons[cfg::kButtonControls] = {
    NpadButton::A, NpadButton::B, NpadButton::X, NpadButton::Y, NpadButton::L, NpadButton::R, NpadButton::ZL,
    NpadButton::ZR, NpadButton::Plus, NpadButton::Minus, NpadButton::StickL, NpadButton::StickR,
    NpadButton::Up, NpadButton::Down, NpadButton::Left, NpadButton::Right,
};

struct PadState {
    u64 buttons = 0;
    float lx = 0, ly = 0, rx = 0, ry = 0;  // -1..1
};

// A gamepad as read: which PadInputs are down, and the raw sticks (-1..1)
struct RawPad {
    bool connected = false;
    bool down[cfg::kPadCount] = {};
    float lx = 0, ly = 0, rx = 0, ry = 0;
};

bool read_slot(int slot, int trigger_threshold, RawPad& p) {
    XINPUT_STATE x{};
    if (!g_xinput || g_xinput(slot, &x) != ERROR_SUCCESS) return false;
    const WORD b = x.Gamepad.wButtons;
    const WORD masks[cfg::kPadCount] = {
        XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_GAMEPAD_LEFT_SHOULDER,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, 0, 0, XINPUT_GAMEPAD_START, XINPUT_GAMEPAD_BACK, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT,
        XINPUT_GAMEPAD_DPAD_RIGHT,
    };
    for (int i = 0; i < cfg::kPadCount; ++i) p.down[i] = masks[i] && (b & masks[i]);
    p.down[cfg::kPadLT] = x.Gamepad.bLeftTrigger > trigger_threshold;
    p.down[cfg::kPadRT] = x.Gamepad.bRightTrigger > trigger_threshold;
    auto axis = [](SHORT v) { return std::max(-1.0f, v / 32767.0f); };
    p.lx = axis(x.Gamepad.sThumbLX);
    p.ly = axis(x.Gamepad.sThumbLY);
    p.rx = axis(x.Gamepad.sThumbRX);
    p.ry = axis(x.Gamepad.sThumbRY);
    p.connected = true;
    return true;
}

s64 now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The pad's motors from the game's vibration: off while the menu is open, and when the game has sent nothing
// for half a second (it streams vibration while any plays). Switch amplitudes are gentle next to an Xbox pad's
// motors, which barely turn below a fifth of full power, hence the square root.
void update_rumble(const cfg::Config& c, int slot, bool blocked) {
    static int applied_slot = -1;
    static WORD applied_l = 0, applied_r = 0;
    if (!g_xinput_set) return;
    const s64 now = now_ms();
    float l = 0.0f, r = 0.0f;
    if (now < g_rumble_test_until.load()) {
        l = r = 0.5f;
    } else if (c.rumble && !blocked && now - g_rumble_at.load() < 500) {
        l = g_rumble[0].load();
        r = g_rumble[1].load();
    }
    const float k = std::clamp(c.rumble_strength, 0, 200) / 100.0f;
    const auto motor = [k](float a) {
        return static_cast<WORD>(std::clamp(std::sqrt(std::max(a, 0.0f)) * k, 0.0f, 1.0f) * 65535.0f);
    };
    const WORD wl = motor(l), wr = motor(r);
    if (applied_slot >= 0 && applied_slot != slot && (applied_l || applied_r)) {
        XINPUT_VIBRATION off{};
        g_xinput_set(applied_slot, &off);  // the pad no longer in use
    }
    if (slot >= 0 && (slot != applied_slot || wl != applied_l || wr != applied_r)) {
        XINPUT_VIBRATION v{wl, wr};
        g_xinput_set(slot, &v);
    }
    applied_slot = slot;
    applied_l = slot >= 0 ? wl : 0;
    applied_r = slot >= 0 ? wr : 0;
}

// The configured gamepad, or the first connected one
bool read_pad(const cfg::Config& c, RawPad& p) {
    if (c.pad_index >= 0) {
        const bool ok = read_slot(c.pad_index, c.trigger_threshold, p);
        g_pad_slot = ok ? c.pad_index : -1;
        return ok;
    }
    for (int slot = 0; slot < 4; ++slot) {
        if (read_slot(slot, c.trigger_threshold, p)) {
            g_pad_slot = slot;
            return true;
        }
    }
    g_pad_slot = -1;
    return false;
}

// Radial deadzone, outer deadzone and response curve on one stick
void shape_stick(float& x, float& y, float deadzone, float outer, float curve) {
    const float m = std::sqrt(x * x + y * y);
    if (m <= deadzone || m <= 0.0f) {
        x = y = 0;
        return;
    }
    const float span = std::max(0.01f, outer - deadzone);
    const float shaped = std::pow(std::clamp((m - deadzone) / span, 0.0f, 1.0f), std::max(0.2f, curve));
    x = x / m * shaped;
    y = y / m * shaped;
}

void apply_pad(const cfg::Config& c, const RawPad& p, PadState& s) {
    for (int i = 0; i < cfg::kButtonControls; ++i) {
        const int pad = c.binds[i].pad;
        if (pad >= 0 && pad < cfg::kPadCount && p.down[pad]) s.buttons |= bit(kControlButtons[i]);
    }
    float lx = p.lx, ly = p.ly, rx = p.rx, ry = p.ry;
    if (c.swap_sticks) {
        std::swap(lx, rx);
        std::swap(ly, ry);
    }
    g_sticks[0] = lx;
    g_sticks[1] = ly;
    g_sticks[2] = rx;
    g_sticks[3] = ry;
    shape_stick(lx, ly, c.deadzone_left, c.outer_deadzone, c.response_curve);
    shape_stick(rx, ry, c.deadzone_right, c.outer_deadzone, c.response_curve);
    g_sticks[4] = lx;
    g_sticks[5] = ly;
    g_sticks[6] = rx;
    g_sticks[7] = ry;
    s.lx = lx;
    s.ly = ly;
    s.rx = rx;
    s.ry = ry;
}

bool key_down(int vk) { return vk && (GetAsyncKeyState(vk) & 0x8000) != 0; }

void read_keyboard(const cfg::Config& c, PadState& s) {
    if (!gpu::window_focused() && !c.keyboard_unfocused) return;
    auto down = [&](int control) {
        return key_down(c.binds[control].keys[0]) || key_down(c.binds[control].keys[1]);
    };
    for (int i = 0; i < cfg::kButtonControls; ++i)
        if (down(i)) s.buttons |= bit(kControlButtons[i]);
    auto axis = [&](int neg, int pos) { return (down(pos) ? 1.0f : 0.0f) - (down(neg) ? 1.0f : 0.0f); };
    const float lx = axis(cfg::kLLeft, cfg::kLRight), ly = axis(cfg::kLDown, cfg::kLUp);
    const float rx = axis(cfg::kRLeft, cfg::kRRight), ry = axis(cfg::kRDown, cfg::kRUp);
    if (lx != 0 || ly != 0) {
        const float m = std::sqrt(lx * lx + ly * ly);  // diagonals at full deflection, not beyond
        s.lx = lx / m;
        s.ly = ly / m;
    }
    if (rx != 0) s.rx = rx;
    if (ry != 0) s.ry = ry;
}

// Mouse camera: while the window has focus and the menu is closed (and Alt isn't held), the cursor is held at the
// window centre and its movement turns the camera (right stick)
void read_mouse_camera(const cfg::Config& c, PadState& s) {
    static float cam_x = 0, cam_y = 0;
    HWND hwnd = static_cast<HWND>(gpu::window_handle());
    const bool active = c.mouse_camera && hwnd && gpu::window_focused() && !ui::menu_open() && !key_down(VK_MENU);
    if (!active) {
        if (g_mouse_captured.exchange(false)) ClipCursor(nullptr);
        cam_x = cam_y = 0;
        return;
    }
    RECT r;
    GetClientRect(hwnd, &r);
    POINT centre{(r.right - r.left) / 2, (r.bottom - r.top) / 2};
    ClientToScreen(hwnd, &centre);
    POINT pos;
    GetCursorPos(&pos);
    if (!g_mouse_captured.exchange(true)) {
        POINT tl{0, 0}, br{r.right, r.bottom};
        ClientToScreen(hwnd, &tl);
        ClientToScreen(hwnd, &br);
        const RECT clip{tl.x, tl.y, br.x, br.y};
        ClipCursor(&clip);
        pos = centre;
    }
    SetCursorPos(centre.x, centre.y);
    // pixels per 5 ms tick -> stick deflection, smoothed over a few ticks so slow movement doesn't stutter
    const float gain = 0.06f * c.mouse_sensitivity;
    const float tx = std::clamp((pos.x - centre.x) * gain, -1.0f, 1.0f);
    const float ty = std::clamp(-(pos.y - centre.y) * gain * (c.mouse_invert_y ? -1.0f : 1.0f), -1.0f, 1.0f);
    cam_x += (tx - cam_x) * 0.5f;
    cam_y += (ty - cam_y) * 0.5f;
    if (std::abs(cam_x) > std::abs(s.rx)) s.rx = cam_x;
    if (std::abs(cam_y) > std::abs(s.ry)) s.ry = cam_y;
}

// Camera speed and inversion, on the right stick from every source
void finish_camera(const cfg::Config& c, PadState& s) {
    s.rx = std::clamp(s.rx * c.camera_speed_x * (c.invert_x ? -1.0f : 1.0f), -1.0f, 1.0f);
    s.ry = std::clamp(s.ry * c.camera_speed_y * (c.invert_y ? -1.0f : 1.0f), -1.0f, 1.0f);
}

// The stick directions as buttons, as hid reports them alongside the analog values
void stick_buttons(PadState& s) {
    constexpr float kThreshold = 0.5f;
    if (s.lx < -kThreshold) s.buttons |= bit(NpadButton::StickLLeft);
    if (s.lx > kThreshold) s.buttons |= bit(NpadButton::StickLRight);
    if (s.ly > kThreshold) s.buttons |= bit(NpadButton::StickLUp);
    if (s.ly < -kThreshold) s.buttons |= bit(NpadButton::StickLDown);
    if (s.rx < -kThreshold) s.buttons |= bit(NpadButton::StickRLeft);
    if (s.rx > kThreshold) s.buttons |= bit(NpadButton::StickRRight);
    if (s.ry > kThreshold) s.buttons |= bit(NpadButton::StickRUp);
    if (s.ry < -kThreshold) s.buttons |= bit(NpadButton::StickRDown);
}

// Player 1 as a connected, wired Pro Controller with full batteries
void connect_player1(NpadInternalState& n) {
    n.style_tag.raw = Core::HID::NpadStyleSet::Fullkey;
    n.assignment_mode = NpadJoyAssignmentMode::Dual;
    n.fullkey_color.attribute = ColorAttribute::Ok;
    n.fullkey_color.fullkey.body = {0x2D, 0x2D, 0x2D, 0xFF};
    n.fullkey_color.fullkey.button = {0xE6, 0xE6, 0xE6, 0xFF};
    n.joycon_color.attribute = ColorAttribute::NoController;
    n.device_type.raw = 0;
    n.device_type.fullkey.Assign(1);
    n.system_properties.raw = 0;
    n.system_properties.is_charging_joy_dual.Assign(1);
    n.system_properties.is_powered_joy_dual.Assign(1);
    n.system_properties.use_plus.Assign(1);
    n.system_properties.use_minus.Assign(1);
    n.system_properties.use_directional_buttons.Assign(1);
    n.battery_level_dual = Core::HID::NpadBatteryLevel::Full;
    n.battery_level_left = Core::HID::NpadBatteryLevel::Full;
    n.battery_level_right = Core::HID::NpadBatteryLevel::Full;
    n.applet_footer_type = AppletFooterUiType::SwitchProController;
}

void input_thread(SharedMemoryFormat* shm) {
    SetThreadDescription(GetCurrentThread(), L"zonaite input");
    NpadInternalState& player1 = shm->npad.npad_entry[0].internal_state;
    connect_player1(player1);
    hid_signal_style_set_update();
    s64 sampling = 0;
    auto next = std::chrono::steady_clock::now();
    cfg::Config c = cfg::get_copy();
    u64 seen = cfg::generation();
    for (;;) {
        if (const u64 gen = cfg::generation(); gen != seen) {
            seen = gen;
            c = cfg::get_copy();
        }
        PadState s;
        RawPad pad;
        const bool have_pad = read_pad(c, pad);
        // holding View + Menu (Back + Start) opens or closes the settings menu
        {
            static int held = 0;
            const bool combo = have_pad && pad.down[cfg::kPadBack] && pad.down[cfg::kPadStart];
            held = combo ? held + 1 : 0;
            if (held == 120) ui::toggle_menu();  // 0.6 s
        }
        const bool blocked = ui::menu_open();  // the menu has the controls
        update_rumble(c, have_pad ? g_pad_slot.load() : -1, blocked);
        if (have_pad && !blocked) apply_pad(c, pad, s);
        if (!have_pad)
            for (auto& v : g_sticks) v = 0.0f;
        if (!blocked) read_keyboard(c, s);
        read_mouse_camera(c, s);
        finish_camera(c, s);
        stick_buttons(s);
        auto stick = [](float v) { return static_cast<s32>(std::clamp(v, -1.0f, 1.0f) * kStickMax); };
        NPadGenericState g{};
        g.sampling_number = ++sampling;
        g.npad_buttons.raw = static_cast<NpadButton>(s.buttons);
        g.l_stick = {stick(s.lx), stick(s.ly)};
        g.r_stick = {stick(s.rx), stick(s.ry)};
        g.connection_status.is_connected.Assign(1);
        g.connection_status.is_wired.Assign(1);
        player1.fullkey_lifo.WriteNextEntry(g);
        player1.fullkey_lifo.timestamp = (s64)system_tick();
        next += std::chrono::milliseconds(5);
        std::this_thread::sleep_until(next);
    }
}

}  // namespace

namespace input {

int poll_pad_press() {
    const cfg::Config c = cfg::get_copy();
    for (int slot = 0; slot < 4; ++slot) {
        RawPad p;
        if (!read_slot(slot, c.trigger_threshold, p)) continue;
        for (int i = 0; i < cfg::kPadCount; ++i)
            if (p.down[i]) return i;
    }
    return -1;
}

bool mouse_captured() { return g_mouse_captured.load(); }

int pad_slot() { return g_pad_slot.load(); }

void test_rumble() { g_rumble_test_until = now_ms() + 600; }

void stick_preview(float raw[4], float shaped[4]) {
    for (int i = 0; i < 4; ++i) {
        raw[i] = g_sticks[i].load();
        shaped[i] = g_sticks[4 + i].load();
    }
}

}  // namespace input

void input_vibrate(int side, float amp_low, float amp_high) {
    g_rumble[side & 1] = std::max(amp_low, amp_high);
    g_rumble_at = now_ms();
}

void input_start(uint8_t* hid_shared_memory) {
    static_assert(sizeof(SharedMemoryFormat) == 0x40000);
    std::thread(input_thread, reinterpret_cast<SharedMemoryFormat*>(hid_shared_memory)).detach();
}

}  // namespace zn
