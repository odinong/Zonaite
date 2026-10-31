// The settings menu and on-screen overlays (Dear ImGui), drawn over every presented frame on the present thread,
// so the menu runs at the display's rate whatever the game does. F1 (or holding View + Menu on a gamepad) opens
// it; changes are saved to user/zonaite.ini at once (zonaite/config.h) and those that the renderer can take live
// apply at once, the rest on the next start (the menu says which).
//
// Eden's present manager calls Vulkan::ZonaiteOverlay after copying a frame into the swapchain image
// (tools/eden_patches.py): the menu is rendered into that image with a render pass that keeps it in the transfer
// layout Eden expects next. ImGui's input comes from the window thread (handle_message); one lock covers both.
//
// Layout: a full-screen options screen in the manner of console games. Tabs across the top (Q / E, LB / RB), one
// column of option rows (up / down to move, left / right to change, Enter or A on a row), and a panel beside them
// that explains the focused row. Every widget is drawn here rather than taken from ImGui's stock set, in one type
// scale and one accent colour. Fonts: Marcellus (titles) and Lato (text), SIL Open Font Licence, res/fonts. No game
// assets are used.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

// Eden's Vulkan headers first: they set up the platform (Win32) parts of vulkan.h that ImGui's backend shares
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_win32.h"

#include "zonaite/config.h"
#include "zonaite/gpu.h"
#include "zonaite/runtime.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// interpolation statistics (runtime/gpu/frame_interp.cpp)
extern std::atomic<float> zn_interp_input_ms, zn_interp_gen_ms;
extern std::atomic<unsigned> zn_interp_depth_share;

namespace zn::ui {

namespace {

using Clock = std::chrono::steady_clock;

std::mutex g_mutex;  // the ImGui context: the present thread draws, the window thread feeds input
std::atomic<bool> g_open{false};
std::atomic<bool> g_capturing{false};
bool g_imgui = false;
const Clock::time_point g_start = Clock::now();

// performance overlay: the last present intervals
std::array<float, 360> g_intervals{};
std::atomic<uint32_t> g_interval_pos{0};

// the menu's state
int g_page = 0;
bool g_page_entered = true;  // focus the first row of the page (on opening, and after a tab change)
bool g_was_open = false;
int g_capture_control = -1;  // a binding waiting for a key or button
int g_capture_slot = 0;      // 0, 1 keys, 2 gamepad, 3 hotkey (g_capture_control is then 100 + the hotkey)
Clock::time_point g_capture_since;
bool g_pad_released = false;
std::vector<std::pair<std::string, std::string>> g_audio_devices;
bool g_audio_listed = false;
bool g_confirm_reset = false;

// fonts, baked at g_font_scale; drawn at the layout scale g_s (they match unless the window was resized since)
ImFont* g_text = nullptr;   // Lato 21
ImFont* g_bold = nullptr;   // Lato Bold 15 (tabs, section labels, keycaps)
ImFont* g_small = nullptr;  // Lato 17
ImFont* g_title = nullptr;  // Marcellus 54
ImFont* g_head = nullptr;   // Marcellus 30
float g_font_scale = 1.0f;
float g_s = 1.0f;

// palette: near-black slate, three greys of warm ink, one accent (jade), one caution (amber)
constexpr ImU32 kInk = IM_COL32(236, 233, 226, 255);
constexpr ImU32 kMuted = IM_COL32(160, 166, 166, 255);
constexpr ImU32 kFaint = IM_COL32(104, 112, 114, 255);
constexpr ImU32 kLine = IM_COL32(255, 255, 255, 22);
constexpr ImU32 kAccent = IM_COL32(118, 214, 188, 255);
constexpr ImU32 kAccentSoft = IM_COL32(118, 214, 188, 40);
constexpr ImU32 kAmber = IM_COL32(232, 178, 92, 255);

float S(float v) { return v * g_s; }

ImU32 alpha(ImU32 c, float a) {
    return (c & 0x00FFFFFF) | (static_cast<ImU32>(((c >> 24) & 0xFF) * std::clamp(a, 0.0f, 1.0f)) << 24);
}

// --- drawing helpers ----------------------------------------------------------------------------------------------

float font_px(ImFont* f) { return f->FontSize / g_font_scale * g_s; }

ImVec2 text_size(ImFont* f, const char* t) {
    return f->CalcTextSizeA(font_px(f), FLT_MAX, 0.0f, t);
}

void text(ImDrawList* dl, ImFont* f, ImVec2 p, ImU32 c, const char* t) {
    dl->AddText(f, font_px(f), p, c, t);
}

// Letter-spaced capitals for labels and tabs
float spaced_width(ImFont* f, const char* t, float tracking) {
    float w = 0;
    for (const char* ch = t; *ch; ++ch) {
        const char s[2] = {*ch, 0};
        w += text_size(f, s).x + tracking;
    }
    return std::max(0.0f, w - tracking);
}

void spaced(ImDrawList* dl, ImFont* f, ImVec2 p, ImU32 c, const char* t, float tracking) {
    for (const char* ch = t; *ch; ++ch) {
        const char s[2] = {*ch, 0};
        text(dl, f, p, c, s);
        p.x += text_size(f, s).x + tracking;
    }
}

// A keycap: a rounded outline with a key's name in it; returns its width
void chevron(ImDrawList* dl, ImVec2 c, float size, bool right, ImU32 color);

// A keycap: a rounded outline with a key's name in it; returns its width. "<>" draws left and right arrows.
float keycap(ImDrawList* dl, ImVec2 p, const char* name, float h, bool lit = false) {
    const bool arrows = std::string(name) == "<>";
    const ImVec2 ts = arrows ? ImVec2{S(22), 0} : text_size(g_bold, name);
    const float w = std::max(h, ts.x + S(14));
    dl->AddRectFilled(p, {p.x + w, p.y + h}, lit ? kAccentSoft : IM_COL32(255, 255, 255, 14), S(5));
    dl->AddRect(p, {p.x + w, p.y + h}, lit ? alpha(kAccent, 0.8f) : IM_COL32(255, 255, 255, 50), S(5), 0, S(1));
    if (arrows) {
        chevron(dl, {p.x + w * 0.5f - S(5), p.y + h * 0.5f}, S(8), false, kMuted);
        chevron(dl, {p.x + w * 0.5f + S(5), p.y + h * 0.5f}, S(8), true, kMuted);
    } else {
        text(dl, g_bold, {p.x + (w - ts.x) * 0.5f, p.y + (h - ts.y) * 0.5f}, lit ? kInk : kMuted, name);
    }
    return w;
}

void chevron(ImDrawList* dl, ImVec2 c, float size, bool right, ImU32 color) {
    const float d = right ? 1.0f : -1.0f;
    const ImVec2 pts[3] = {{c.x - d * size * 0.35f, c.y - size * 0.6f}, {c.x + d * size * 0.35f, c.y},
                           {c.x - d * size * 0.35f, c.y + size * 0.6f}};
    dl->AddPolyline(pts, 3, color, 0, S(1.8f));
}

// --- the description panel ----------------------------------------------------------------------------------------

struct PaneInfo {
    std::string title, text;
    int extra = 0;  // 1, 2: show the left / right stick
    bool set = false;
};
PaneInfo g_pane_focus;  // the selected row's

void describe(const char* title, const char* text, int extra, bool selected) {
    if (selected) g_pane_focus = {title, text ? text : "", extra, true};
}

// --- rows ---------------------------------------------------------------------------------------------------------

struct Row {
    bool hovered = false, focused = false, held = false, activated = false;  // focused: the selected row
    ImVec2 min, max;
    ImVec2 vmin, vmax;  // the value area on the right
    bool hot() const { return hovered || focused; }
};

// Selection: one row of the page is always selected (the menu's own, not ImGui's navigation). Up / down move it,
// the mouse moves it to the row under it, left / right change that row, Enter / A use it. A binding row also has
// a selected box (g_sel_col).
int g_sel = 0, g_sel_col = 0;
int g_rows = 0, g_rows_last = 1;  // rows drawn this frame / last frame
bool g_moved = false;             // the selection moved by key this frame (scroll it into view)
int g_step = 0;                   // -1 / +1 for the selected row
bool g_activate = false;          // Enter / A on the selected row
bool g_clear = false;             // Delete / X on the selected binding box

// The next row's index, and whether the mouse just moved onto it (it then takes the selection)
int next_row(bool hovered) {
    const int idx = g_rows++;
    const ImVec2 d = ImGui::GetIO().MouseDelta;
    if (hovered && (d.x != 0 || d.y != 0)) g_sel = idx;
    return idx;
}

// Keeps a row that the keys just selected inside the list's visible part
void keep_visible(const ImVec2& a, const ImVec2& b) {
    if (!g_moved) return;
    const float top = ImGui::GetWindowPos().y, bottom = top + ImGui::GetWindowHeight();
    if (a.y < top + S(40)) ImGui::SetScrollY(ImGui::GetScrollY() - (top + S(40) - a.y));
    if (b.y > bottom - S(40)) ImGui::SetScrollY(ImGui::GetScrollY() + (b.y - bottom + S(40)));
}

void section(const char* label) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::Dummy({0, S(g_rows == 0 ? 2 : 18)});
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    spaced(dl, g_bold, {p.x + S(22), p.y}, kFaint, label, S(2.2f));
    const float tw = spaced_width(g_bold, label, S(2.2f));
    const float y = p.y + text_size(g_bold, "A").y * 0.55f;
    dl->AddLine({p.x + S(22) + tw + S(14), y}, {p.x + w - S(18), y}, kLine, S(1));
    ImGui::Dummy({w, text_size(g_bold, "A").y + S(8)});
}

Row row(const char* label, const char* help, int extra = 0, float value_width = 360) {
    Row r;
    ImGui::PushID(label);
    const float w = ImGui::GetContentRegionAvail().x, h = S(54);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##row", {w, h});
    r.hovered = ImGui::IsItemHovered();
    r.focused = next_row(r.hovered) == g_sel;
    r.held = ImGui::IsItemActive();
    r.activated = r.focused && g_activate;
    r.min = p;
    r.max = {p.x + w, p.y + h};
    r.vmin = {p.x + w - S(value_width), p.y};
    r.vmax = {p.x + w - S(20), p.y + h};
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (r.focused) {
        dl->AddRectFilledMultiColor(r.min, r.max, IM_COL32(118, 214, 188, 22), IM_COL32(255, 255, 255, 6),
                                    IM_COL32(255, 255, 255, 6), IM_COL32(118, 214, 188, 22));
        dl->AddRectFilled({p.x, p.y + S(12)}, {p.x + S(3), p.y + h - S(12)}, kAccent, S(2));
        keep_visible(r.min, r.max);
    } else if (r.hovered) {
        dl->AddRectFilled(r.min, r.max, IM_COL32(255, 255, 255, 8), S(4));
    }
    const ImVec2 ts = text_size(g_text, label);
    text(dl, g_text, {p.x + S(22), p.y + (h - ts.y) * 0.5f}, r.hot() ? kInk : alpha(kInk, 0.86f), label);
    describe(label, help, extra, r.focused);
    ImGui::PopID();
    return r;
}

// -1 / +1 from left / right (keyboard, d-pad, stick) on the selected row
int step_keys(const Row& r) {
    return r.focused ? g_step : 0;
}

// Enter / A on the selected row (the mouse is handled where it is clicked)
bool nav_activated(const Row& r) {
    return r.activated;
}

// A choice: < value > with a pip per option underneath
bool choice(const char* label, const char* help, int* v, const char* const* items, int n, int extra = 0) {
    Row r = row(label, help, extra);
    *v = std::clamp(*v, 0, n - 1);
    int d = step_keys(r);
    if (nav_activated(r)) d = 1;
    if (r.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const float mx = ImGui::GetIO().MousePos.x;
        d = mx < r.vmin.x + S(44) ? -1 : 1;
    }
    if (d) *v = (*v + d + n) % n;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = (r.vmin.y + r.vmax.y) * 0.5f;
    const ImU32 arrow = r.hot() ? kInk : kFaint;
    chevron(dl, {r.vmin.x + S(16), cy - S(3)}, S(12), false, arrow);
    chevron(dl, {r.vmax.x - S(16), cy - S(3)}, S(12), true, arrow);
    const ImVec2 ts = text_size(g_text, items[*v]);
    const float mid = (r.vmin.x + r.vmax.x) * 0.5f;
    text(dl, g_text, {mid - ts.x * 0.5f, cy - ts.y * 0.5f - S(4)}, r.hot() ? kInk : kMuted, items[*v]);
    if (n <= 12) {
        const float gap = S(10), y = cy + S(14);
        float x = mid - (n - 1) * gap * 0.5f;
        for (int i = 0; i < n; ++i, x += gap)
            dl->AddCircleFilled({x, y}, i == *v ? S(2.4f) : S(1.6f), i == *v ? kAccent : kFaint);
    }
    return d != 0;
}

bool toggle(const char* label, const char* help, bool* v) {
    Row r = row(label, help);
    bool flip = step_keys(r) != 0 || nav_activated(r) ||
                (r.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left));
    if (flip) *v = !*v;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = (r.vmin.y + r.vmax.y) * 0.5f;
    // On / Off, the current one lit, with a short bar under it
    const float half = (r.vmax.x - r.vmin.x) * 0.5f;
    const char* names[2] = {"Off", "On"};
    for (int i = 0; i < 2; ++i) {
        const bool lit = (i == 1) == *v;
        const float cx = r.vmin.x + half * (i + 0.5f);
        const ImVec2 ts = text_size(g_text, names[i]);
        text(dl, g_text, {cx - ts.x * 0.5f, cy - ts.y * 0.5f - S(3)},
             lit ? (i == 1 ? kAccent : kInk) : kFaint, names[i]);
        if (lit) dl->AddRectFilled({cx - S(14), cy + S(13)}, {cx + S(14), cy + S(15)}, i == 1 ? kAccent : kMuted, S(1));
    }
    return flip;
}

bool slider(const char* label, const char* help, float* v, float lo, float hi, float step, const char* fmt,
            int extra = 0) {
    Row r = row(label, help, extra);
    const float old = *v;
    if (const int d = step_keys(r)) *v += d * step;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = (r.vmin.y + r.vmax.y) * 0.5f;
    const float x0 = r.vmin.x + S(14), x1 = r.vmax.x - S(86);
    if (r.held && ImGui::GetIO().MousePos.x >= x0 - S(12) && ImGui::GetIO().MouseClickedPos[0].x >= x0 - S(12)) {
        const float t = std::clamp((ImGui::GetIO().MousePos.x - x0) / (x1 - x0), 0.0f, 1.0f);
        *v = lo + std::round(t * (hi - lo) / step) * step;
    }
    *v = std::clamp(*v, lo, hi);
    const float t = (hi > lo) ? (*v - lo) / (hi - lo) : 0.0f;
    dl->AddRectFilled({x0, cy - S(1.5f)}, {x1, cy + S(1.5f)}, IM_COL32(255, 255, 255, 30), S(2));
    dl->AddRectFilled({x0, cy - S(1.5f)}, {x0 + (x1 - x0) * t, cy + S(1.5f)}, r.hot() ? kAccent : alpha(kAccent, 0.7f),
                      S(2));
    dl->AddCircleFilled({x0 + (x1 - x0) * t, cy}, r.hot() ? S(7) : S(5.5f), r.hot() ? kInk : kMuted);
    char value[32];
    std::snprintf(value, sizeof value, fmt, *v);
    const ImVec2 ts = text_size(g_text, value);
    text(dl, g_text, {r.vmax.x - ts.x, cy - ts.y * 0.5f}, r.hot() ? kInk : kMuted, value);
    return *v != old;
}

bool slider_int(const char* label, const char* help, int* v, int lo, int hi, int step, const char* fmt) {
    float f = static_cast<float>(*v);
    char ffmt[32];
    std::snprintf(ffmt, sizeof ffmt, "%s", fmt);  // fmt uses %.0f for the int value
    const bool changed = slider(label, help, &f, static_cast<float>(lo), static_cast<float>(hi),
                                static_cast<float>(step), ffmt);
    *v = static_cast<int>(std::lround(f));
    return changed;
}

// A row that does something: its value side reads like a link
bool action(const char* label, const char* help, const char* what) {
    Row r = row(label, help);
    const bool go = nav_activated(r) || (r.hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = (r.vmin.y + r.vmax.y) * 0.5f;
    const ImVec2 ts = text_size(g_text, what);
    text(dl, g_text, {r.vmax.x - S(20) - ts.x, cy - ts.y * 0.5f}, r.hot() ? kAccent : kMuted, what);
    chevron(dl, {r.vmax.x - S(4), cy}, S(11), true, r.hot() ? kAccent : kFaint);
    return go;
}

// A row that only shows something
void info(const char* label, const char* help, const char* value) {
    Row r = row(label, help);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float cy = (r.vmin.y + r.vmax.y) * 0.5f;
    const ImVec2 ts = text_size(g_text, value);
    text(dl, g_text, {r.vmax.x - ts.x, cy - ts.y * 0.5f}, kMuted, value);
}

// --- bindings -----------------------------------------------------------------------------------------------------

void begin_capture(int control, int slot) {
    g_capture_control = control;
    g_capture_slot = slot;
    g_capture_since = Clock::now();
    g_pad_released = false;
    g_capturing = true;
}

void end_capture() {
    g_capture_control = -1;
    g_capturing = false;
}

int* hotkey_field(cfg::Config& c, int i) {
    return i == 0 ? &c.menu_key : i == 1 ? &c.screenshot_key : i == 2 ? &c.fullscreen_key : &c.hud_key;
}

// A binding row: the control's name and up to three chips (keys, gamepad). Click / Enter on a chip rebinds it,
// right-click / X clears it. `slots` lists (text, slot) per chip.
bool binding_row(const char* label, int control, std::initializer_list<std::pair<std::string, int>> slots,
                 int* cleared_slot) {
    ImGui::PushID(control);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float w = ImGui::GetContentRegionAvail().x, h = S(50);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const bool row_hovered = ImGui::IsMouseHoveringRect(p, {p.x + w, p.y + h}) && ImGui::IsWindowHovered();
    const int idx = next_row(row_hovered);
    const bool selected = idx == g_sel;
    const int count = static_cast<int>(slots.size());
    if (selected) g_sel_col = std::clamp(g_sel_col + (g_capturing ? 0 : g_step), 0, count - 1);
    dl->ChannelsSplit(2);
    dl->ChannelsSetCurrent(1);
    bool any_hot = selected, any_focus = selected;
    const float chip_w = S(150), gap = S(10);
    float x = p.x + w - S(20) - slots.size() * chip_w - (slots.size() - 1) * gap;
    int col = 0;
    for (const auto& [name, slot] : slots) {
        ImGui::SetCursorScreenPos({x, p.y + S(8)});
        ImGui::PushID(slot);
        bool pressed = ImGui::InvisibleButton("##chip", {chip_w, h - S(16)});
        const bool hovered = ImGui::IsItemHovered();
        if (hovered && (ImGui::GetIO().MouseDelta.x != 0 || ImGui::GetIO().MouseDelta.y != 0)) g_sel_col = col;
        const bool focused = selected && g_sel_col == col;
        if (focused && g_activate) pressed = true;
        const bool waiting = g_capturing && g_capture_control == control && g_capture_slot == slot;
        const ImVec2 a{x, p.y + S(8)}, b{x + chip_w, p.y + h - S(8)};
        const float pulse = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 6.0);
        dl->AddRectFilled(a, b, waiting ? alpha(kAccent, 0.10f + 0.12f * pulse)
                                        : IM_COL32(255, 255, 255, hovered || focused ? 22 : 10), S(6));
        dl->AddRect(a, b, waiting ? kAccent : (hovered || focused) ? alpha(kAccent, 0.7f) : IM_COL32(255, 255, 255, 34),
                    S(6), 0, S(1));
        const char* shown = waiting ? (slot == 2 ? "Press a button" : "Press a key") : name.c_str();
        const ImVec2 ts = text_size(g_small, shown);
        text(dl, g_small, {x + (chip_w - ts.x) * 0.5f, (a.y + b.y - ts.y) * 0.5f},
             waiting ? kInk : name == "-" ? kFaint : (hovered || focused) ? kInk : kMuted, shown);
        if (pressed && !waiting && !g_capturing) begin_capture(control, slot);
        if ((hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) || (focused && g_clear))
            *cleared_slot = slot;
        ImGui::PopID();
        x += chip_w + gap;
        ++col;
    }
    dl->ChannelsSetCurrent(0);
    if (selected) {
        dl->AddRectFilledMultiColor(p, {p.x + w, p.y + h}, IM_COL32(118, 214, 188, 22), IM_COL32(255, 255, 255, 6),
                                    IM_COL32(255, 255, 255, 6), IM_COL32(118, 214, 188, 22));
        dl->AddRectFilled({p.x, p.y + S(12)}, {p.x + S(3), p.y + h - S(12)}, kAccent, S(2));
        keep_visible(p, {p.x + w, p.y + h});
    } else if (row_hovered) {
        dl->AddRectFilled(p, {p.x + w, p.y + h}, IM_COL32(255, 255, 255, 8), S(4));
    }
    const ImVec2 ts = text_size(g_text, label);
    text(dl, g_text, {p.x + S(22), p.y + (h - ts.y) * 0.5f}, any_hot ? kInk : alpha(kInk, 0.86f), label);
    dl->ChannelsMerge();
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy({w, h});
    describe(label, "Choose a box with left / right or the mouse, press Enter or A (or click it), then press the "
                    "new key or button. Delete, X or a right-click clears a box; Esc cancels.",
             0, selected);
    ImGui::PopID();
    return any_hot;
}

// --- pages --------------------------------------------------------------------------------------------------------

struct Edit {
    cfg::Config c;
    bool changed = false;
    bool display = false;  // window / fullscreen settings changed
};

void page_display(Edit& e) {
    cfg::Config& c = e.c;
    section("SCREEN");
    static const char* const modes[] = {"Windowed", "Borderless fullscreen"};
    if (choice("Display mode", "Windowed, or a borderless window covering the whole monitor. F11 or Alt+Enter "
                               "switches at any time.", &c.display_mode, modes, 2))
        e.changed = e.display = true;
    const auto monitors = gpu::monitor_names();
    std::vector<const char*> names;
    for (const auto& n : monitors) names.push_back(n.c_str());
    if (!names.empty() && choice("Monitor", "The monitor the fullscreen window covers.", &c.monitor, names.data(),
                                 static_cast<int>(names.size())))
        e.changed = e.display = true;
    static const int sizes[][2] = {{1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {3200, 1800}, {3840, 2160}};
    static const char* const size_names[] = {"1280 x 720", "1600 x 900", "1920 x 1080", "2560 x 1440",
                                             "3200 x 1800", "3840 x 2160"};
    int size = -1;
    for (int i = 0; i < 6; ++i)
        if (c.window_w == sizes[i][0] && c.window_h == sizes[i][1]) size = i;
    int pick = size < 0 ? 2 : size;
    if (choice("Window size", "The size of the picture in windowed mode. The game draws at its own resolution "
                              "(see Graphics) and is scaled to fit.", &pick, size_names, 6)) {
        c.window_w = sizes[pick][0];
        c.window_h = sizes[pick][1];
        e.changed = e.display = true;
    }
    static const char* const aspects[] = {"16:9", "4:3", "21:9", "16:10", "Stretch"};
    if (choice("Aspect ratio", "16:9 is the game's own shape. The others stretch it, or box it in to fit.",
               &c.aspect, aspects, 5))
        e.changed = true;
    static const char* const vsync[] = {"Off", "Mailbox", "On", "Adaptive"};
    if (choice("Vertical sync", "On waits for the display (no tearing). Mailbox also avoids tearing, with less "
                                "delay. Off can tear. Adaptive tears only when a frame is late.",
               &c.vsync, vsync, 4))
        e.changed = true;

    section("MOTION");
    const int refresh = gpu::refresh_rate();
    char auto_name[48];
    std::snprintf(auto_name, sizeof auto_name, "Auto (%dx)", std::clamp((refresh + 15) / 30, 1, 4));
    const char* const fg[] = {auto_name, "Off", "2x", "3x", "4x"};
    if (choice("Frame generation",
               "Shows extra frames between each pair the game draws, for motion up to 4 times as smooth. The game "
               "still runs at its normal 30 frames a second, so it plays exactly the same. Auto picks what your "
               "display's refresh rate fits.",
               &c.frame_gen, fg, 5))
        e.changed = true;
    static const int caps[] = {0, 60, 90, 120, 144, 165, 240};
    static const char* const cap_names[] = {"No limit", "60", "90", "120", "144", "165", "240"};
    int cap = 0;
    for (int i = 0; i < 7; ++i)
        if (c.target_rate == caps[i]) cap = i;
    if (choice("Frame rate limit", "The most frames shown per second with frame generation on. Lower it to save "
                                   "power, or to match a display that can't keep up.", &cap, cap_names, 7)) {
        c.target_rate = caps[cap];
        e.changed = true;
    }
    static const char* const quality[] = {"Clean", "Always", "Pictures only"};
    if (choice("Frame quality",
               "Clean: extra frames are made only from the game's own depth and camera, and never across a slowdown "
               "or a camera cut, so nothing smears. At those moments the real frame simply stays up a little "
               "longer. Always: also makes them from the pictures alone (menus, loading, slowdowns), which is "
               "smoother there but can smear. Pictures only: never uses depth.",
               &c.interp_mode, quality, 3))
        e.changed = true;
    if (toggle("Even pacing", "Spaces the shown frames evenly. Off shows each as soon as it is ready, which can "
                              "look uneven.", &c.pacing))
        e.changed = true;

    section("ON SCREEN");
    {
        bool hidden = gpu::hud_hidden();
        const std::string help = "Leaves out everything the game draws over the world (hearts, map, prompts) for clean "
                                 "screenshots. Menus are hidden too while it is on. " +
                                 cfg::key_name(c.hud_key) + " switches it; it is off again at the next start.";
        if (toggle("Hide HUD", help.c_str(), &hidden)) gpu::set_hud_hidden(hidden);
    }
    static const char* const overlay[] = {"Off", "Frame rate", "Detailed"};
    if (choice("Performance overlay", "Frame rate, or with Detailed also frame times and how long the frame "
                                      "generation takes on the GPU.", &c.perf_overlay, overlay, 3))
        e.changed = true;
    static const char* const corners[] = {"Top left", "Top right", "Bottom left", "Bottom right"};
    if (choice("Overlay position", nullptr, &c.overlay_corner, corners, 4)) e.changed = true;
    if (slider("Menu size", "The size of this menu.", &c.ui_scale, 0.75f, 1.5f, 0.05f, "%.2fx")) e.changed = true;
    if (toggle("Startup hint", "A note on how to open this menu, for a few seconds after starting.", &c.show_hint))
        e.changed = true;
}

// Graphics presets: the image settings that cost the most, together (Custom when they match none)
struct Preset {
    const char* name;
    int resolution, scaling_filter, anti_aliasing, anisotropy;
};
constexpr Preset kPresets[] = {
    {"Performance", 2, 6, 0, 0},  // 0.75x, FSR up to the window
    {"Balanced", 3, 6, 0, 4},     // native, FSR, 8x filtering
    {"Quality", 5, 7, 0, 5},      // 1.5x, area (down to the window), 16x
    {"Ultra", 6, 7, 2, 5},        // 2x, area, SMAA, 16x
};

int matching_preset(const cfg::Config& c) {
    for (int i = 0; i < 4; ++i) {
        const Preset& p = kPresets[i];
        if (c.resolution == p.resolution && c.scaling_filter == p.scaling_filter &&
            c.anti_aliasing == p.anti_aliasing && c.anisotropy == p.anisotropy)
            return i + 1;
    }
    return 0;
}

void page_graphics(Edit& e) {
    cfg::Config& c = e.c;
    section("QUICK SETUP");
    {
        static const char* const names[] = {"Custom", "Performance", "Balanced", "Quality", "Ultra"};
        int pick = matching_preset(c);
        if (choice("Preset",
                   "Sets resolution, scaling, anti-aliasing and texture filtering together. Performance draws below "
                   "native for weaker GPUs; Ultra draws at twice native (3200 x 1800) and needs a fast one. Every "
                   "change applies at once, so you can step through them and watch.",
                   &pick, names, 5) &&
            pick > 0) {
            const Preset& p = kPresets[pick - 1];
            c.resolution = p.resolution;
            c.scaling_filter = p.scaling_filter;
            c.anti_aliasing = p.anti_aliasing;
            c.anisotropy = p.anisotropy;
            e.changed = true;
        }
    }

    section("IMAGE");
    static const char* const res[] = {"0.25x", "0.5x", "0.75x", "1x  Native", "1.25x", "1.5x", "2x",
                                      "3x", "4x", "5x", "6x", "7x", "8x"};
    if (choice("Internal resolution",
               "The resolution the game draws its world at. Native is up to 1600 x 900, as on the console. 2x is "
               "up to 3200 x 1800: much sharper, and needs a much faster GPU. Shaders are made for one resolution, "
               "so for a second or two after a change, things pop in as they are made again.",
               &c.resolution, res, 13))
        e.changed = true;
    static const char* const filters[] = {"Nearest", "Bilinear", "Bicubic", "Gaussian", "Lanczos", "ScaleForce",
                                          "AMD FSR 1", "Area", "Zero-tangent", "B-spline", "Mitchell", "Spline",
                                          "MMPX", "Snapdragon GSR", "GSR edges"};
    if (choice("Scaling filter", "How the picture is scaled to your window. Bilinear is soft and smooth; FSR keeps "
                                 "edges sharp when scaling up.", &c.scaling_filter, filters, 15))
        e.changed = true;
    if (c.scaling_filter == 6 &&
        slider_int("FSR sharpening", "How much FSR sharpens the scaled picture.", &c.fsr_sharpness, 0, 200, 5,
                   "%.0f%%"))
        e.changed = true;
    static const char* const aa[] = {"Off", "FXAA", "SMAA"};
    if (choice("Anti-aliasing", "Smooths jagged edges. FXAA is cheap and a little soft; SMAA is sharper.",
               &c.anti_aliasing, aa, 3))
        e.changed = true;
    static const char* const aniso[] = {"Auto", "Game default", "2x", "4x", "8x", "16x"};
    if (choice("Texture filtering", "Anisotropic filtering: ground and walls stay sharp at a distance. Costs "
                                    "little on a desktop GPU.", &c.anisotropy, aniso, 6))
        e.changed = true;
    if (toggle("Bloom fix", "Tones down glow that some effects overdo.", &c.fix_bloom)) e.changed = true;

    section("PICTURE");
    if (slider("Sharpening", "Brings out texture detail without haloes on edges (contrast adaptive sharpening). "
                             "Good with FSR or a resolution below native.", &c.sharpen, 0.0f, 1.0f, 0.05f, "%.2f"))
        e.changed = true;
    if (slider("Brightness", nullptr, &c.brightness, -0.25f, 0.25f, 0.01f, "%+.2f")) e.changed = true;
    if (slider("Contrast", nullptr, &c.contrast, 0.5f, 1.5f, 0.05f, "%.2f")) e.changed = true;
    if (slider("Saturation", "How vivid colours are. 0 is black and white.", &c.saturation, 0.0f, 2.0f, 0.05f,
               "%.2f"))
        e.changed = true;
    if (slider("Gamma", "Above 1 lifts dark areas (caves, night) without washing out the rest.", &c.gamma, 0.6f, 1.6f,
               0.05f, "%.2f"))
        e.changed = true;
    if (!cfg::picture_neutral(c) &&
        action("Reset picture", "Sharpening, brightness, contrast, saturation and gamma back to neutral.", "Reset")) {
        const cfg::Config d;
        c.sharpen = d.sharpen;
        c.brightness = d.brightness;
        c.contrast = d.contrast;
        c.saturation = d.saturation;
        c.gamma = d.gamma;
        e.changed = true;
    }

    section("SHADERS");
    if (toggle("Background shader builds",
               "New shaders are built while you keep playing, instead of the game pausing for them; what they draw "
               "appears a few frames late the first time. Anything seen before is ready at once from the cache.",
               &c.async_shaders))
        e.changed = true;
    std::error_code ec;
    const std::filesystem::path cache = std::filesystem::path(cfg::user_dir()) / "cache" / "shader";
    uintmax_t bytes = 0;
    if (std::filesystem::exists(cache, ec))
        for (auto& f : std::filesystem::recursive_directory_iterator(cache, ec))
            if (f.is_regular_file(ec)) bytes += f.file_size(ec);
    char what[64];
    std::snprintf(what, sizeof what, "Clear  (%.0f MB)", bytes / 1048576.0);
    if (action("Shader cache", "Every shader built so far, so they are ready at once next time. Clearing it builds "
                               "each again as it is next needed, so expect a few seconds of pop-in. Do it only if "
                               "something draws wrongly after a driver update.", what))
        cfg::request_shader_cache_clear();

    section("ADVANCED");
    static const char* const accuracy[] = {"Low", "High"};
    if (choice("GPU accuracy", "High is what the game needs to draw correctly. Low is faster and may glitch.",
               &c.gpu_accuracy, accuracy, 2))
        e.changed = true;
    static const char* const astc[] = {"CPU", "GPU", "CPU, background"};
    if (choice("Texture decoding", "Where the game's compressed textures are unpacked. GPU is fastest.",
               &c.astc_decode, astc, 3))
        e.changed = true;
    static const char* const recompress[] = {"Off", "BC1", "BC3"};
    if (choice("Texture recompression", "Saves video memory on GPUs that have little of it, at some cost in "
                                        "texture quality.", &c.astc_recompression, recompress, 3))
        e.changed = true;
    static const char* const vram[] = {"Conservative", "Aggressive"};
    if (choice("Video memory use", "Aggressive keeps more in video memory, for fewer reloads, on GPUs with plenty "
                                   "of it.", &c.vram_mode, vram, 2))
        e.changed = true;
    static const char* const clock[] = {"Normal", "Boost", "Overclock"};
    if (choice("Console GPU speed", "The GPU speed the game believes it has. The game lowers its resolution when "
                                    "it thinks the GPU is slow, so higher keeps it sharper.",
               &c.gpu_clock, clock, 3))
        e.changed = true;
}

void page_controls(Edit& e) {
    cfg::Config& c = e.c;
    section("GAMEPAD");
    const int slot = input::pad_slot();
    char status[48];
    std::snprintf(status, sizeof status, slot >= 0 ? "Controller %d connected" : "None connected", slot + 1);
    static const char* const slots[] = {"First connected", "Controller 1", "Controller 2", "Controller 3",
                                        "Controller 4"};
    int s = c.pad_index + 1;
    if (choice("Controller", status, &s, slots, 5)) {
        c.pad_index = s - 1;
        e.changed = true;
    }
    {
        int layout = c.binds[cfg::kA].pad == cfg::kPadA ? 1 : 0;
        static const char* const layouts[] = {"Switch positions", "Xbox labels"};
        if (choice("Button layout", "Switch positions: the bottom face button is B, as on a Nintendo controller. "
                                    "Xbox labels: the button marked A is A.", &layout, layouts, 2)) {
            cfg::Config d;
            cfg::reset_bindings(d, layout == 1);
            for (int i = 0; i < cfg::kControlCount; ++i) c.binds[i].pad = d.binds[i].pad;
            e.changed = true;
        }
    }
    if (slider("Left stick deadzone", "Stick movement smaller than this is ignored, which stops a worn stick from "
                                      "drifting.", &c.deadzone_left, 0.0f, 0.6f, 0.01f, "%.2f", 1))
        e.changed = true;
    if (slider("Right stick deadzone", "The same for the camera stick.", &c.deadzone_right, 0.0f, 0.6f, 0.01f, "%.2f",
               2))
        e.changed = true;
    if (slider("Outer deadzone", "How far a stick must go to count as pushed all the way.", &c.outer_deadzone, 0.6f,
               1.0f, 0.01f, "%.2f", 1))
        e.changed = true;
    if (slider("Response curve", "1 is linear. Higher gives finer control near the centre and full speed at the "
                                 "edge.", &c.response_curve, 0.5f, 3.0f, 0.05f, "%.2f", 1))
        e.changed = true;
    if (slider_int("Trigger threshold", "How far ZL and ZR must be pulled to count.", &c.trigger_threshold, 5, 250, 5,
                   "%.0f"))
        e.changed = true;
    if (toggle("Swap sticks", "Move with the right stick and turn the camera with the left.", &c.swap_sticks))
        e.changed = true;
    if (toggle("Rumble", "The game's vibration, on the controller's motors: impacts, explosions, the ground "
                         "shaking. Off while this menu is open.", &c.rumble))
        e.changed = true;
    if (c.rumble) {
        if (slider_int("Rumble strength", nullptr, &c.rumble_strength, 10, 200, 10, "%.0f%%")) e.changed = true;
        if (action("Test rumble", "Half a second on both motors, at the strength above.", "Test"))
            input::test_rumble();
    }

    section("CAMERA");
    if (slider("Horizontal speed", nullptr, &c.camera_speed_x, 0.2f, 2.5f, 0.05f, "%.2fx")) e.changed = true;
    if (slider("Vertical speed", nullptr, &c.camera_speed_y, 0.2f, 2.5f, 0.05f, "%.2fx")) e.changed = true;
    if (toggle("Invert horizontal", nullptr, &c.invert_x)) e.changed = true;
    if (toggle("Invert vertical", nullptr, &c.invert_y)) e.changed = true;

    section("KEYBOARD AND MOUSE");
    if (toggle("Mouse camera", "Turn the camera with the mouse while the game window has focus. Hold Alt to get "
                               "the cursor back.", &c.mouse_camera))
        e.changed = true;
    if (slider("Mouse sensitivity", nullptr, &c.mouse_sensitivity, 0.1f, 4.0f, 0.05f, "%.2f")) e.changed = true;
    if (toggle("Invert mouse", "Moving the mouse up looks down.", &c.mouse_invert_y)) e.changed = true;
    if (toggle("Keys in the background", "Read the keyboard even while another window has focus.",
               &c.keyboard_unfocused))
        e.changed = true;

    section("BINDINGS");
    for (int i = 0; i < cfg::kControlCount; ++i) {
        int cleared = -1;
        const auto key0 = cfg::key_name(c.binds[i].keys[0]), key1 = cfg::key_name(c.binds[i].keys[1]);
        if (i < cfg::kButtonControls)
            binding_row(cfg::control_name(i), i, {{key0, 0}, {key1, 1}, {cfg::pad_input_name(c.binds[i].pad), 2}},
                        &cleared);
        else
            binding_row(cfg::control_name(i), i, {{key0, 0}, {key1, 1}}, &cleared);
        if (cleared == 0 || cleared == 1) c.binds[i].keys[cleared] = 0;
        if (cleared == 2) c.binds[i].pad = cfg::kPadNone;
        if (cleared >= 0) e.changed = true;
    }
    if (action("Reset bindings", "Puts every key and button back to the defaults.", "Reset")) {
        cfg::reset_bindings(c);
        e.changed = true;
    }

    section("SHORTCUTS");
    static const char* const hotkeys[] = {"Open this menu", "Screenshot", "Fullscreen", "Hide HUD"};
    for (int i = 0; i < 4; ++i) {
        int cleared = -1;
        binding_row(hotkeys[i], 100 + i, {{cfg::key_name(*hotkey_field(c, i)), 3}}, &cleared);
    }
}

void page_audio(Edit& e) {
    cfg::Config& c = e.c;
    section("SOUND");
    if (slider_int("Volume", "Up to 200% for quiet speakers; above 100% loud sounds can clip.", &c.volume, 0, 200, 5,
                   "%.0f%%"))
        e.changed = true;
    if (toggle("Mute", nullptr, &c.mute)) e.changed = true;
    if (toggle("Mute in the background", "Silent while another window has focus.", &c.mute_unfocused))
        e.changed = true;
    section("OUTPUT");
    if (!g_audio_listed) {
        g_audio_devices = audio::output_devices();
        g_audio_listed = true;
    }
    int current = 0;
    std::vector<const char*> names{"System default"};
    for (size_t i = 0; i < g_audio_devices.size(); ++i) {
        names.push_back(g_audio_devices[i].second.c_str());
        if (g_audio_devices[i].first == c.audio_device) current = static_cast<int>(i) + 1;
    }
    if (choice("Output device", "Where the sound plays. System default follows Windows when you change devices "
                                "there.", &current, names.data(), static_cast<int>(names.size()))) {
        c.audio_device = current == 0 ? std::string() : g_audio_devices[current - 1].first;
        e.changed = true;
    }
    if (action("Find devices", "Look again for headphones or speakers connected since the list was made.",
               "Refresh"))
        g_audio_listed = false;
    if (slider_int("Latency", "How far ahead sound is prepared. Lower is more immediate; raise it if the sound "
                              "crackles.", &c.audio_latency_ms, 10, 200, 5, "%.0f ms"))
        e.changed = true;
}

void open_folder(const std::string& path) {
    std::filesystem::create_directories(path);
    ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void page_system(Edit& e) {
    section("FOLDERS");
    const std::string saves = save_dir();
    if (action("Saves", saves.c_str(), "Open")) open_folder(saves);
    char cwd[MAX_PATH] = ".";
    GetCurrentDirectoryA(MAX_PATH, cwd);
    if (action("Screenshots", cwd, "Open")) open_folder(cwd);
    if (action("Settings and caches", cfg::user_dir().c_str(), "Open")) open_folder(cfg::user_dir());
    section("SETTINGS");
    if (action("Reset everything", "Puts every setting on every page back to its default.",
               g_confirm_reset ? "Press again to confirm" : "Reset")) {
        if (g_confirm_reset) {
            cfg::Config d;
            cfg::reset_bindings(d);
            e.c = d;
            e.changed = e.display = true;
        }
        g_confirm_reset = !g_confirm_reset;
    }
}

void page_about(const char* gpu_name) {
    section("ZONAITE");
    const std::string version = "Tears of the Kingdom " + game_version();
    const std::string about = "The Legend of Zelda: Tears of the Kingdom, version " + game_version() +
                              ", recompiled for this PC from your own copy of the game.";
    info("Game", about.c_str(), version.c_str());
    info("Graphics", nullptr, gpu_name);
    char hz[32];
    std::snprintf(hz, sizeof hz, "%d Hz", gpu::refresh_rate());
    info("Display", nullptr, hz);
    section("LICENCES");
    info("Zonaite", "Free software under the GNU General Public License, version 3 or later. Full text: LICENSE.",
         "GPL 3.0");
    info("Eden", "Graphics, audio and input services come from the Eden emulator.", "GPL 3.0");
    info("Dear ImGui", "The toolkit this menu is drawn with.", "MIT");
    info("Marcellus, Lato", "The typefaces of this menu.", "SIL OFL 1.1");
    info("FFmpeg, libopus", "Video and audio decoding.", "LGPL, BSD");
    section("NOTICE");
    info("Not affiliated with Nintendo", "Zonaite contains no game code or assets. You need your own legally "
                                         "obtained copy of the game.", "");
}

const char* const kPages[] = {"DISPLAY", "GRAPHICS", "CONTROLS", "AUDIO", "SYSTEM", "ABOUT"};
const char* const kPageIntro[] = {
    "How the game appears on your screen, and frame generation.",
    "How sharp the picture is, and how shaders are built.",
    "Gamepad, keyboard and mouse: sticks, camera, and every binding.",
    "Volume and where the sound plays.",
    "Folders, and resetting settings.",
    "About Zonaite, and its licences.",
};

// The left stick (extra 1) or right stick (2) in the description panel: raw position, the shaped one, deadzones
void stick_picture(ImDrawList* dl, ImVec2 centre, float r, int which, const cfg::Config& c) {
    float raw[4], shaped[4];
    input::stick_preview(raw, shaped);
    const int i = which == 2 ? 2 : 0;
    const float dz = which == 2 ? c.deadzone_right : c.deadzone_left;
    dl->AddCircleFilled(centre, r, IM_COL32(255, 255, 255, 8), 64);
    dl->AddCircle(centre, r, IM_COL32(255, 255, 255, 40), 64, S(1));
    dl->AddCircle(centre, r * c.outer_deadzone, alpha(kAccent, 0.35f), 64, S(1));
    dl->AddCircleFilled(centre, r * dz, alpha(kAmber, 0.12f), 48);
    dl->AddCircle(centre, r * dz, alpha(kAmber, 0.6f), 48, S(1));
    dl->AddLine({centre.x - r, centre.y}, {centre.x + r, centre.y}, IM_COL32(255, 255, 255, 18), S(1));
    dl->AddLine({centre.x, centre.y - r}, {centre.x, centre.y + r}, IM_COL32(255, 255, 255, 18), S(1));
    dl->AddCircleFilled({centre.x + raw[i] * r, centre.y - raw[i + 1] * r}, S(4), kFaint);
    dl->AddCircleFilled({centre.x + shaped[i] * r, centre.y - shaped[i + 1] * r}, S(7), kAccent);
}

// The panel beside the list: the focused (or hovered) row, explained
void draw_pane(ImVec2 a, ImVec2 b, const cfg::Config& c) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const PaneInfo& p = g_pane_focus;
    dl->AddLine({a.x, a.y}, {a.x, b.y}, kLine, S(1));
    const float x = a.x + S(40), w = std::min(b.x - x, S(560));
    float y = a.y + S(6);
    if (!p.set) {
        ImGui::SetCursorScreenPos({x, y});
        ImGui::PushFont(g_small);
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kMuted));
        ImGui::PushTextWrapPos(x + w);
        ImGui::TextUnformatted(kPageIntro[g_page]);
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::PopFont();
        return;
    }
    dl->AddRectFilled({x, y}, {x + S(28), y + S(2)}, kAccent);
    y += S(16);
    ImGui::SetCursorScreenPos({x, y});
    ImGui::PushFont(g_head);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kInk));
    ImGui::PushTextWrapPos(x + w);
    ImGui::TextUnformatted(p.title.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    ImGui::PopFont();
    if (!p.text.empty()) {
        ImGui::Dummy({0, S(6)});
        ImGui::SetCursorScreenPos({x, ImGui::GetCursorScreenPos().y});
        ImGui::PushFont(g_small);
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kMuted));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, S(6)});
        ImGui::PushTextWrapPos(x + w);
        ImGui::TextUnformatted(p.text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }
    if (p.extra == 1 || p.extra == 2) {
        const float r = std::min(S(92), w * 0.32f);
        const ImVec2 centre{x + r + S(4), ImGui::GetCursorScreenPos().y + S(30) + r};
        stick_picture(dl, centre, r, p.extra, c);
        text(dl, g_small, {centre.x + r + S(24), centre.y - S(24)}, kMuted,
             p.extra == 2 ? "Right stick" : "Left stick");
        dl->AddCircleFilled({centre.x + r + S(30), centre.y + S(6)}, S(5), kAccent);
        text(dl, g_small, {centre.x + r + S(44), centre.y - S(5)}, kFaint, "what the game gets");
        dl->AddCircle({centre.x + r + S(30), centre.y + S(32)}, S(5), alpha(kAmber, 0.8f), 16, S(1));
        text(dl, g_small, {centre.x + r + S(44), centre.y + S(21)}, kFaint, "deadzone");
    }
}

// Tabs across the top, with the keys that switch them on either side
void draw_tabs(ImDrawList* dl, ImVec2 p, bool pad) {
    const float tracking = S(2.6f);
    float x = p.x;
    const float cap_h = S(26);
    x += keycap(dl, {x, p.y - S(4)}, pad ? "LB" : "Q", cap_h) + S(26);
    for (int i = 0; i < 6; ++i) {
        const float w = spaced_width(g_bold, kPages[i], tracking);
        const ImVec2 a{x - S(10), p.y - S(10)}, b{x + w + S(10), p.y + S(30)};
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(i);
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        if (ImGui::InvisibleButton("##tab", {b.x - a.x, b.y - a.y}) && g_page != i) {
            g_page = i;
            g_page_entered = true;
        }
        const bool hovered = ImGui::IsItemHovered();
        ImGui::PopItemFlag();
        ImGui::PopID();
        const bool active = g_page == i;
        spaced(dl, g_bold, {x, p.y}, active ? kInk : hovered ? kMuted : kFaint, kPages[i], tracking);
        if (active) dl->AddRectFilled({x, p.y + S(26)}, {x + w, p.y + S(28)}, kAccent, S(1));
        x += w + S(44);
    }
    keycap(dl, {x - S(18), p.y - S(4)}, pad ? "RB" : "E", cap_h);
}

// Key prompts along the bottom
void draw_prompts(ImDrawList* dl, ImVec2 right, bool pad) {
    struct Prompt {
        const char* key;
        const char* what;
    };
    const Prompt keys[] = {{"Esc", "Back"}, {"Enter", "Select"}, {"<>", "Change"}, {"Q  E", "Section"}};
    const Prompt buttons[] = {{"B", "Back"}, {"A", "Select"}, {"<>", "Change"}, {"LB  RB", "Section"}};
    const Prompt* list = pad ? buttons : keys;
    const float h = S(26);
    float x = right.x;
    for (int i = 0; i < 4; ++i) {
        const ImVec2 ts = text_size(g_small, list[i].what);
        x -= ts.x;
        text(dl, g_small, {x, right.y + (h - ts.y) * 0.5f}, kMuted, list[i].what);
        const bool arrows = std::string(list[i].key) == "<>";
        const float kw = std::max(h, (arrows ? S(22) : text_size(g_bold, list[i].key).x) + S(14));
        x -= S(10) + kw;
        keycap(dl, {x, right.y}, list[i].key, h);
        x -= S(34);
    }
}

void draw_menu(const char* gpu_name) {
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 size = io.DisplaySize;
    Edit e{cfg::get_copy()};
    g_s = std::clamp(size.y / 1080.0f, 0.5f, 3.0f) * std::clamp(e.c.ui_scale, 0.75f, 1.5f);
    if (!g_was_open) g_page_entered = true;
    g_was_open = true;
    g_pane_focus = {};
    const bool pad = input::pad_slot() >= 0;

    // tabs from the keyboard / gamepad shoulder buttons
    if (!g_capturing) {
        int d = 0;
        if (ImGui::IsKeyPressed(ImGuiKey_Q, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false)) d = -1;
        if (ImGui::IsKeyPressed(ImGuiKey_E, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false)) d = 1;
        if (d) {
            g_page = (g_page + d + 6) % 6;
            g_page_entered = true;
        }
    }

    // the game, dimmed: darkest behind the list, lighter toward the right
    ImDrawList* bg = ImGui::GetBackgroundDrawList();
    bg->AddRectFilled({0, 0}, size, IM_COL32(7, 11, 13, 246));
    bg->AddRectFilledMultiColor({size.x * 0.55f, 0}, size, IM_COL32(7, 11, 13, 0), IM_COL32(16, 30, 30, 40),
                                IM_COL32(16, 30, 30, 40), IM_COL32(7, 11, 13, 0));

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##zonaite", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float left = std::max(S(96), size.x * 0.07f), top = S(72);

    // title
    spaced(dl, g_bold, {left, top}, kAccent, "ZONAITE", S(4));
    text(dl, g_title, {left - S(3), top + S(18)}, kInk, "Settings");

    // tabs and the rule under them
    const float tabs_y = top + S(108);
    draw_tabs(dl, {left, tabs_y}, pad);
    dl->AddLine({left, tabs_y + S(42)}, {size.x - left, tabs_y + S(42)}, kLine, S(1));

    // the list, and the panel beside it
    const float list_top = tabs_y + S(62), list_bottom = size.y - S(108);
    const float list_w = std::min(S(900), (size.x - 2 * left) * 0.62f);
    ImGui::SetCursorScreenPos({left - S(22), list_top});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, S(2)});
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, S(4));
    ImGui::BeginChild("##list", {list_w + S(22), list_bottom - list_top}, ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNav);
    ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
    if (g_page_entered) {
        ImGui::SetScrollY(0);
        g_sel = g_sel_col = 0;
        g_page_entered = false;
    }
    {
        auto pressed = [](std::initializer_list<ImGuiKey> keys) {
            for (ImGuiKey k : keys)
                if (ImGui::IsKeyPressed(k)) return true;
            return false;
        };
        const bool free = !g_capturing;
        const bool up = free && pressed({ImGuiKey_UpArrow, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp});
        const bool down = free && pressed({ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown});
        g_moved = up || down;
        if (up) g_sel = std::max(0, g_sel - 1);
        if (down) g_sel = std::min(g_rows_last - 1, g_sel + 1);
        g_step = !free ? 0
                 : pressed({ImGuiKey_LeftArrow, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft})    ? -1
                 : pressed({ImGuiKey_RightArrow, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight}) ? 1
                                                                                                          : 0;
        g_activate = free && (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                              ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) ||
                              ImGui::IsKeyPressed(ImGuiKey_Space, false) ||
                              ImGui::IsKeyPressed(ImGuiKey_GamepadFaceDown, false));
        g_clear = free && (ImGui::IsKeyPressed(ImGuiKey_Delete, false) ||
                           ImGui::IsKeyPressed(ImGuiKey_GamepadFaceLeft, false));
        g_rows = 0;
    }
    switch (g_page) {
    case 0: page_display(e); break;
    case 1: page_graphics(e); break;
    case 2: page_controls(e); break;
    case 3: page_audio(e); break;
    case 4: page_system(e); break;
    default: page_about(gpu_name); break;
    }
    g_rows_last = std::max(1, g_rows);
    g_sel = std::min(g_sel, g_rows_last - 1);
    ImGui::Dummy({0, S(24)});
    ImGui::PopItemFlag();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    draw_pane({left + list_w + S(56), list_top}, {size.x - left, list_bottom}, e.c);

    // bottom: what is pending, and the prompts
    const float bottom = size.y - S(64);
    dl->AddLine({left, bottom - S(22)}, {size.x - left, bottom - S(22)}, kLine, S(1));
    text(dl, g_small, {left, bottom + S(3)}, kFaint, "Every change applies at once, and is saved as you make it");
    draw_prompts(dl, {size.x - left, bottom}, pad);
    ImGui::End();
    ImGui::PopStyleVar();

    // close: Escape or gamepad B, unless a binding is waiting
    if (!g_capturing && (ImGui::IsKeyPressed(ImGuiKey_Escape, false) ||
                         ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false))) {
        g_open = false;
        g_was_open = false;
        g_confirm_reset = false;
    }

    // a gamepad binding waiting for a button (once the button that started it is released)
    if (g_capturing && g_capture_slot == 2) {
        const int pressed = input::poll_pad_press();
        if (pressed < 0) g_pad_released = true;
        if (pressed >= 0 && g_pad_released) {
            e.c.binds[g_capture_control].pad = pressed;
            e.changed = true;
            end_capture();
        }
    }
    if (g_capturing && Clock::now() - g_capture_since > std::chrono::seconds(8)) end_capture();

    if (e.changed) {
        {
            std::scoped_lock lock{cfg::mutex()};
            cfg::edit() = e.c;
        }
        cfg::changed();
        cfg::apply_renderer(e.c, false);
        if (e.display) gpu::request_display_apply();
    }
}

// Frame rate (and, detailed, frame times and the GPU time of frame generation) in a corner
void draw_performance(const cfg::Config& c) {
    ImGuiIO& io = ImGui::GetIO();
    if (!g_open) g_s = std::clamp(io.DisplaySize.y / 1080.0f, 0.5f, 3.0f);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const double game = gpu::fps(), shown = gpu::shown_fps();
    char big[16], small[32];
    std::snprintf(big, sizeof big, "%.0f", shown);
    std::snprintf(small, sizeof small, shown > game + 1 ? "FPS   game %.0f" : "FPS", game);
    const bool detailed = c.perf_overlay >= 2;
    const float pad = S(14), w = detailed ? S(250) : text_size(g_head, big).x + text_size(g_bold, small).x + S(40);
    const float h = detailed ? S(150) : S(48);
    const int corner = std::clamp(c.overlay_corner, 0, 3);
    const ImVec2 a{(corner & 1) ? io.DisplaySize.x - S(20) - w : S(20),
                   (corner & 2) ? io.DisplaySize.y - S(20) - h : S(20)};
    dl->AddRectFilled(a, {a.x + w, a.y + h}, IM_COL32(6, 10, 12, 190), S(8));
    text(dl, g_head, {a.x + pad, a.y + S(7)}, kInk, big);
    spaced(dl, g_bold, {a.x + pad + text_size(g_head, big).x + S(8), a.y + S(19)}, kMuted, small, S(1.2f));
    if (!detailed) return;
    // frame times: a line over the last 360 presents, scaled to twice their mean
    float sum = 0, worst = 0;
    const uint32_t at = g_interval_pos.load();
    for (float v : g_intervals) {
        sum += v;
        worst = std::max(worst, v);
    }
    const float mean = sum / g_intervals.size();
    const float gx = a.x + pad, gy = a.y + S(52), gw = w - 2 * pad, gh = S(44);
    dl->AddLine({gx, gy + gh * 0.5f}, {gx + gw, gy + gh * 0.5f}, kLine, S(1));
    std::array<ImVec2, 360> pts;
    for (size_t i = 0; i < pts.size(); ++i) {
        const float v = g_intervals[(at + i) % g_intervals.size()];
        pts[i] = {gx + gw * i / (pts.size() - 1),
                  gy + gh - std::clamp(v / std::max(mean * 2.0f, 1.0f), 0.0f, 1.0f) * gh};
    }
    dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), alpha(kAccent, 0.9f), 0, S(1.2f));
    char line[64];
    std::snprintf(line, sizeof line, "%.2f ms   worst %.1f", mean, worst);
    text(dl, g_small, {gx, gy + gh + S(6)}, kMuted, line);
    const float in_ms = zn_interp_input_ms.load(), gen_ms = zn_interp_gen_ms.load();
    if (in_ms > 0) {
        std::snprintf(line, sizeof line, "frame gen %.1f + %.1f ms   depth %u%%", in_ms, gen_ms,
                      zn_interp_depth_share.load());
        text(dl, g_small, {gx, gy + gh + S(26)}, kFaint, line);
    }
}

void draw_hint(const cfg::Config& c) {
    if (!c.show_hint) return;
    const float t = std::chrono::duration<float>(Clock::now() - g_start).count();
    if (t > 14.0f) return;
    const float a = std::clamp(std::min(t - 2.0f, 14.0f - t), 0.0f, 1.0f);
    if (a <= 0) return;
    ImGuiIO& io = ImGui::GetIO();
    g_s = std::clamp(io.DisplaySize.y / 1080.0f, 0.5f, 3.0f);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const std::string key = cfg::key_name(c.menu_key);
    const char* label = "Settings";
    const float kw = std::max(S(30), text_size(g_bold, key.c_str()).x + S(14));
    const float w = kw + S(12) + text_size(g_small, label).x + S(36), h = S(46);
    const ImVec2 p{(io.DisplaySize.x - w) * 0.5f, io.DisplaySize.y - S(96)};
    dl->AddRectFilled(p, {p.x + w, p.y + h}, alpha(IM_COL32(6, 10, 12, 210), a), S(23));
    // keycap and label, faded together
    const ImVec2 kp{p.x + S(18), p.y + S(10)};
    dl->AddRect(kp, {kp.x + kw, kp.y + S(26)}, alpha(IM_COL32(255, 255, 255, 60), a), S(5), 0, S(1));
    const ImVec2 ks = text_size(g_bold, key.c_str());
    text(dl, g_bold, {kp.x + (kw - ks.x) * 0.5f, kp.y + (S(26) - ks.y) * 0.5f}, alpha(kInk, a), key.c_str());
    text(dl, g_small, {kp.x + kw + S(12), p.y + (h - text_size(g_small, label).y) * 0.5f}, alpha(kMuted, a), label);
}

bool wants_overlay(const cfg::Config& c) {
    return g_open || c.perf_overlay > 0 || (c.show_hint && Clock::now() - g_start < std::chrono::seconds(14));
}

void apply_theme(float scale) {
    ImGuiStyle& s = ImGui::GetStyle();
    s = ImGuiStyle();
    s.WindowPadding = {0, 0};
    s.WindowBorderSize = 0;
    s.ChildBorderSize = 0;
    s.ScrollbarRounding = 2;
    s.ScrollbarSize = 4;
    s.ItemSpacing = {0, 2};
    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = ImGui::ColorConvertU32ToFloat4(kInk);
    c[ImGuiCol_WindowBg] = c[ImGuiCol_ChildBg] = {0, 0, 0, 0};
    c[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
    c[ImGuiCol_ScrollbarGrab] = {1, 1, 1, 0.12f};
    c[ImGuiCol_ScrollbarGrabHovered] = {1, 1, 1, 0.22f};
    c[ImGuiCol_ScrollbarGrabActive] = ImGui::ColorConvertU32ToFloat4(kAccent);
    c[ImGuiCol_NavCursor] = {0, 0, 0, 0};  // the rows draw their own focus
    s.ScaleAllSizes(scale);
}

std::string fonts_dir() {
    // res/fonts next to the build folder: <root>/build/zonaite.exe -> <root>/res/fonts
    std::filesystem::path p = cfg::user_dir();
    return (p.parent_path().parent_path() / "res" / "fonts").string();
}

void load_fonts(float scale) {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    g_font_scale = scale;
    const std::string dir = fonts_dir();
    auto add = [&](const char* file, float size) -> ImFont* {
        const std::string path = dir + "\\" + file;
        ImFontConfig fc;
        fc.OversampleH = 3;
        fc.OversampleV = 2;
        if (std::filesystem::exists(path)) return io.Fonts->AddFontFromFileTTF(path.c_str(), size * scale, &fc);
        fc.SizePixels = size * scale;
        return io.Fonts->AddFontDefault(&fc);
    };
    g_text = add("Lato-Regular.ttf", 21);
    g_bold = add("Lato-Bold.ttf", 15);
    g_small = add("Lato-Regular.ttf", 17);
    g_title = add("Marcellus-Regular.ttf", 54);
    g_head = add("Marcellus-Regular.ttf", 30);
    io.FontDefault = g_text;
}

}  // namespace

bool menu_open() { return g_open.load(); }
bool capturing_input() { return g_capturing.load(); }

void toggle_menu() {
    if (g_capturing) return;
    g_open = !g_open;
}

void note_present_interval(float ms) {
    const uint32_t at = g_interval_pos.fetch_add(1) % g_intervals.size();
    g_intervals[at] = ms;
}

bool handle_message(void* hwnd_, unsigned msg, uint64_t wparam, int64_t lparam) {
    HWND hwnd = static_cast<HWND>(hwnd_);
    if (g_capturing) {
        // a binding waits for a key or mouse button (ignore the click that started it)
        int key = 0;
        if (msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN) {
            key = static_cast<int>(wparam);
            if (key == VK_CONTROL) key = (lparam & (1 << 24)) ? VK_RCONTROL : VK_LCONTROL;
            if (key == VK_SHIFT) key = MapVirtualKeyW((lparam >> 16) & 0xFF, MAPVK_VSC_TO_VK_EX);
            if (key == VK_MENU) key = (lparam & (1 << 24)) ? VK_RMENU : VK_LMENU;
        }
        const bool settle = Clock::now() - g_capture_since > std::chrono::milliseconds(250);
        if (settle && msg == WM_LBUTTONDOWN) key = VK_LBUTTON;
        if (settle && msg == WM_RBUTTONDOWN) key = VK_RBUTTON;
        if (settle && msg == WM_MBUTTONDOWN) key = VK_MBUTTON;
        if (settle && msg == WM_XBUTTONDOWN) key = GET_XBUTTON_WPARAM(wparam) == XBUTTON1 ? VK_XBUTTON1 : VK_XBUTTON2;
        if (key && g_capture_slot != 2) {
            std::scoped_lock lock{g_mutex};
            if (key != VK_ESCAPE) {
                const int value = (key == VK_BACK || key == VK_DELETE) && g_capture_slot != 3 ? 0 : key;
                {
                    std::scoped_lock cl{cfg::mutex()};
                    cfg::Config& c = cfg::edit();
                    if (g_capture_slot == 3)
                        *hotkey_field(c, g_capture_control - 100) = value;
                    else
                        c.binds[g_capture_control].keys[g_capture_slot] = value;
                }
                cfg::changed();
            }
            end_capture();
            return true;
        }
        if (key == VK_ESCAPE) {
            end_capture();
            return true;
        }
    }
    if (!g_open) return false;
    std::scoped_lock lock{g_mutex};
    if (!g_imgui) return false;
    ImGui_ImplWin32_WndProcHandler(hwnd, msg, static_cast<WPARAM>(wparam), static_cast<LPARAM>(lparam));
    switch (msg) {  // input the menu takes; everything else goes on to the window
    case WM_KEYDOWN: case WM_KEYUP: case WM_SYSKEYDOWN: case WM_SYSKEYUP: case WM_CHAR:
    case WM_LBUTTONDOWN: case WM_LBUTTONUP: case WM_RBUTTONDOWN: case WM_RBUTTONUP: case WM_MBUTTONDOWN:
    case WM_MBUTTONUP: case WM_MOUSEWHEEL: case WM_MOUSEHWHEEL: case WM_XBUTTONDOWN: case WM_XBUTTONUP:
        return true;
    }
    return false;
}

}  // namespace zn::ui

namespace Vulkan {

namespace {

struct Target {
    VkImageView view{};
    VkFramebuffer framebuffer{};
    VkExtent2D extent{};
};

struct OverlayState {
    VkDevice device{};
    VkRenderPass render_pass{};
    VkFormat format{};
    std::map<VkImage, Target> targets;
    std::vector<std::pair<Target, int>> retired;  // destroyed a few presents later (frames in flight)
    bool ready = false;
    std::string gpu_name;
    PFN_vkCreateImageView create_view{};
    PFN_vkDestroyImageView destroy_view{};
    PFN_vkCreateFramebuffer create_framebuffer{};
    PFN_vkDestroyFramebuffer destroy_framebuffer{};
    PFN_vkCreateRenderPass create_render_pass{};
    PFN_vkCmdBeginRenderPass begin_pass{};
    PFN_vkCmdEndRenderPass end_pass{};
};
OverlayState g_overlay;

PFN_vkGetInstanceProcAddr g_get_instance_proc = nullptr;
VkInstance g_instance = VK_NULL_HANDLE;

PFN_vkVoidFunction load_vk(const char* name, void*) {
    return g_get_instance_proc(g_instance, name);
}

bool init_overlay(const vk::Instance& instance, const Device& device, std::mutex& submit_mutex, VkFormat format,
                  size_t image_count) {
    HMODULE vulkan = GetModuleHandleW(L"vulkan-1.dll");
    g_get_instance_proc = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                       reinterpret_cast<void*>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")))
                                 : nullptr;
    if (!g_get_instance_proc) return false;
    g_instance = *instance;
    auto& o = g_overlay;
    o.device = *device.GetLogical();
    o.format = format;
    o.gpu_name = std::string(device.GetModelName());
    auto dev_fn = [&](const char* name) { return g_get_instance_proc(g_instance, name); };
    o.create_view = reinterpret_cast<PFN_vkCreateImageView>(dev_fn("vkCreateImageView"));
    o.destroy_view = reinterpret_cast<PFN_vkDestroyImageView>(dev_fn("vkDestroyImageView"));
    o.create_framebuffer = reinterpret_cast<PFN_vkCreateFramebuffer>(dev_fn("vkCreateFramebuffer"));
    o.destroy_framebuffer = reinterpret_cast<PFN_vkDestroyFramebuffer>(dev_fn("vkDestroyFramebuffer"));
    o.create_render_pass = reinterpret_cast<PFN_vkCreateRenderPass>(dev_fn("vkCreateRenderPass"));
    o.begin_pass = reinterpret_cast<PFN_vkCmdBeginRenderPass>(dev_fn("vkCmdBeginRenderPass"));
    o.end_pass = reinterpret_cast<PFN_vkCmdEndRenderPass>(dev_fn("vkCmdEndRenderPass"));
    auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(dev_fn("vkGetDeviceQueue"));

    // drawn over the copied frame; the image stays in the transfer layout Eden's next barrier expects
    const VkAttachmentDescription attachment{
        .flags = 0,
        .format = format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    };
    const VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &ref,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
    const std::array<VkSubpassDependency, 2> deps{{
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0},
        {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, 0},
    }};
    const VkRenderPassCreateInfo rp{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(deps.size()),
        .pDependencies = deps.data(),
    };
    if (o.create_render_pass(o.device, &rp, nullptr, &o.render_pass) != VK_SUCCESS) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    // fonts baked for the window's height and the menu size (layout follows the window if it changes later)
    const zn::cfg::Config c = zn::cfg::get_copy();
    RECT client{0, 0, 1920, 1080};
    GetClientRect(static_cast<HWND>(zn::gpu::window_handle()), &client);
    const float scale = std::clamp((client.bottom - client.top) / 1080.0f, 0.5f, 3.0f) *
                        std::clamp(c.ui_scale, 0.75f, 1.5f);
    zn::ui::g_s = scale;
    zn::ui::apply_theme(scale);
    zn::ui::load_fonts(scale);
    ImGui_ImplWin32_Init(zn::gpu::window_handle());
    if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_1, load_vk, nullptr)) return false;
    VkQueue queue = VK_NULL_HANDLE;
    get_queue(o.device, device.GetGraphicsFamily(), 0, &queue);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_1;
    info.Instance = g_instance;
    info.PhysicalDevice = device.GetPhysical();
    info.Device = o.device;
    info.QueueFamily = device.GetGraphicsFamily();
    info.Queue = queue;
    info.RenderPass = o.render_pass;
    info.MinImageCount = 2;
    info.ImageCount = static_cast<uint32_t>(std::max<size_t>(image_count, 2));
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.DescriptorPoolSize = 16;
    if (!ImGui_ImplVulkan_Init(&info)) return false;
    {
        std::scoped_lock lock{submit_mutex};  // the font upload submits to the graphics queue
        ImGui_ImplVulkan_CreateFontsTexture();
    }
    zn::log("settings menu: ready (%s)", zn::ui::fonts_dir().c_str());
    return true;
}

Target& target_for(VkImage image, VkExtent2D extent) {
    auto& o = g_overlay;
    auto it = o.targets.find(image);
    if (it != o.targets.end() && it->second.extent.width == extent.width &&
        it->second.extent.height == extent.height)
        return it->second;
    if (it != o.targets.end()) {
        o.retired.emplace_back(it->second, 8);
        o.targets.erase(it);
    }
    // a new swapchain: retire the old images' targets too
    if (!o.targets.empty() && (o.targets.begin()->second.extent.width != extent.width ||
                               o.targets.begin()->second.extent.height != extent.height)) {
        for (auto& [img, t] : o.targets) o.retired.emplace_back(t, 8);
        o.targets.clear();
    }
    Target t;
    t.extent = extent;
    const VkImageViewCreateInfo view{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = o.format,
        .components{},
        .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    o.create_view(o.device, &view, nullptr, &t.view);
    const VkFramebufferCreateInfo fb{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = o.render_pass,
        .attachmentCount = 1,
        .pAttachments = &t.view,
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };
    o.create_framebuffer(o.device, &fb, nullptr, &t.framebuffer);
    return o.targets[image] = t;
}

void retire_tick() {
    auto& o = g_overlay;
    for (auto it = o.retired.begin(); it != o.retired.end();) {
        if (--it->second > 0) {
            ++it;
            continue;
        }
        o.destroy_framebuffer(o.device, it->first.framebuffer, nullptr);
        o.destroy_view(o.device, it->first.view, nullptr);
        it = o.retired.erase(it);
    }
}

}  // namespace

// Eden's present manager, after copying a frame into swapchain image `image` (tools/eden_patches.py)
void ZonaiteOverlay(const vk::Instance& instance, const Device& device, std::mutex& submit_mutex,
                    vk::CommandBuffer cmdbuf, VkImage image, VkExtent2D extent, VkFormat format, size_t image_count) {
    using namespace zn::ui;
    static zn::cfg::Config c = zn::cfg::get_copy();
    static uint64_t seen = zn::cfg::generation();
    if (const uint64_t g = zn::cfg::generation(); g != seen) {
        seen = g;
        c = zn::cfg::get_copy();
    }
    if (!wants_overlay(c)) return;
    std::scoped_lock lock{g_mutex};
    if (!g_imgui) {
        static bool failed = false;
        if (failed) return;
        if (!init_overlay(instance, device, submit_mutex, format, image_count)) {
            zn::log("settings menu: could not start the overlay renderer");
            failed = true;
            return;
        }
        g_imgui = true;
    }
    retire_tick();
    if (format != g_overlay.format) return;  // a swapchain format change would need a new render pass
    Target& t = target_for(image, extent);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::GetIO().DisplaySize = {static_cast<float>(extent.width), static_cast<float>(extent.height)};
    ImGui::GetIO().MouseDrawCursor = false;
    ImGui::NewFrame();
    if (g_open) draw_menu(g_overlay.gpu_name.c_str());
    if (c.perf_overlay > 0) draw_performance(c);
    if (!g_open) draw_hint(c);
    ImGui::Render();

    const VkRenderPassBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = g_overlay.render_pass,
        .framebuffer = t.framebuffer,
        .renderArea = {{0, 0}, extent},
        .clearValueCount = 0,
        .pClearValues = nullptr,
    };
    g_overlay.begin_pass(*cmdbuf, &begin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmdbuf);
    g_overlay.end_pass(*cmdbuf);
}

}  // namespace Vulkan

// ImGui's checks (third_party/imgui/imconfig.h): logged, the first few of each, instead of stopping the game
void zn_imgui_assert(const char* expr, const char* file, int line) {
    static std::atomic<int> count{0};
    if (count.fetch_add(1) < 20) zn::log("settings menu: ImGui check failed: %s (%s:%d)", expr, file, line);
}
