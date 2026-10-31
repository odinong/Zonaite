// The game window (Win32) as Eden's EmuWindow, and GPU startup (VideoCore::CreateGPU with the Vulkan renderer).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <condition_variable>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/frontend/graphics_context.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "video_core/video_core.h"

#include "common/stb.h"

#include "zonaite/config.h"
#include "zonaite/gpu.h"
#include "zonaite/runtime.h"

namespace zn::gpu {

namespace {

constexpr int kWidth = 1920, kHeight = 1080;

class NullContext final : public Core::Frontend::GraphicsContext {};  // Vulkan needs no GL context

class Window final : public Core::Frontend::EmuWindow {
public:
    explicit Window(HWND hwnd) {
        window_info.type = Core::Frontend::WindowSystemType::Windows;
        window_info.render_surface = hwnd;
        window_info.render_surface_scale = 1.0f;
        RECT r;
        GetClientRect(hwnd, &r);
        Resize(r.right - r.left, r.bottom - r.top);
    }
    // The presented frames follow the window's client size (the renderer scales the game's 1080p image into it)
    void Resize(int width, int height) {
        if (width > 0 && height > 0) UpdateCurrentFramebufferLayout(width, height);
    }
    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<NullContext>();
    }
    bool IsShown() const override { return true; }
};

HWND g_hwnd;
std::unique_ptr<Window> g_window;
constexpr UINT kApplyDisplay = WM_APP + 1;  // apply the display settings (posted by the menu)

BOOL CALLBACK add_monitor(HMONITOR m, HDC, LPRECT, LPARAM data) {
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    if (GetMonitorInfoW(m, &mi)) reinterpret_cast<std::vector<RECT>*>(data)->push_back(mi.rcMonitor);
    return TRUE;
}

std::vector<RECT> monitor_rects() {
    std::vector<RECT> rects;
    EnumDisplayMonitors(nullptr, nullptr, add_monitor, reinterpret_cast<LPARAM>(&rects));
    return rects;
}

// Windowed (client area window_w x window_h, centred on its monitor) or borderless fullscreen on a monitor
void apply_display(HWND hwnd) {
    const cfg::Config c = cfg::get_copy();
    const auto monitors = monitor_rects();
    if (c.display_mode == 1) {
        const RECT m = !monitors.empty() ? monitors[std::clamp(c.monitor, 0, (int)monitors.size() - 1)]
                                         : RECT{0, 0, kWidth, kHeight};
        SetWindowLongW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(hwnd, HWND_TOP, m.left, m.top, m.right - m.left, m.bottom - m.top,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);
        return;
    }
    SetWindowLongW(hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
    const RECT& work = mi.rcWork;
    // The asked-for size, shrunk (keeping its shape) until the window with its frame fits the screen: a window
    // Windows has to cut short gets a picture with black bars at its sides
    int cw = std::max(320, c.window_w), ch = std::max(180, c.window_h);
    RECT frame{0, 0, 0, 0};
    AdjustWindowRect(&frame, WS_OVERLAPPEDWINDOW, FALSE);
    const int max_w = (work.right - work.left) - (frame.right - frame.left);
    const int max_h = (work.bottom - work.top) - (frame.bottom - frame.top);
    if (cw > max_w || ch > max_h) {
        const double s = std::min(static_cast<double>(max_w) / cw, static_cast<double>(max_h) / ch);
        cw = static_cast<int>(cw * s) & ~1;
        ch = static_cast<int>(ch * s) & ~1;
    }
    RECT r{0, 0, cw, ch};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int w = r.right - r.left, h = r.bottom - r.top;
    SetWindowPos(hwnd, nullptr, work.left + std::max(0L, (work.right - work.left - w) / 2),
                 work.top + std::max(0L, (work.bottom - work.top - h) / 2), w, h,
                 SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
}

void take_screenshot() {
    SYSTEMTIME t;
    GetLocalTime(&t);
    char name[64];
    std::snprintf(name, sizeof name, "zonaite_%04d%02d%02d_%02d%02d%02d.png", t.wYear, t.wMonth, t.wDay, t.wHour,
                  t.wMinute, t.wSecond);
    request_screenshot(name);
}

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CLOSE) {
        std::_Exit(0);  // closing the window ends the game
    }
    if (msg == kApplyDisplay) {
        apply_display(hwnd);
        return 0;
    }
    if (msg == WM_SIZE && g_window) g_window->Resize(LOWORD(lp), HIWORD(lp));
    if (msg == WM_SETCURSOR && LOWORD(lp) == HTCLIENT && input::mouse_captured()) {
        SetCursor(nullptr);  // the mouse is turning the camera
        return TRUE;
    }
    if ((msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN) && !(lp & (1 << 30)) && !ui::capturing_input()) {
        const cfg::Config c = cfg::get_copy();
        const int key = static_cast<int>(wp);
        const bool alt_enter = key == VK_RETURN && (GetKeyState(VK_MENU) & 0x8000);
        if (key == c.menu_key) {
            ui::toggle_menu();
            return 0;
        }
        if (key == c.screenshot_key) take_screenshot();
        if (key == c.hud_key) set_hud_hidden(!hud_hidden());
        if (key == c.fullscreen_key || alt_enter) {
            {
                std::scoped_lock lock{cfg::mutex()};
                cfg::edit().display_mode = cfg::edit().display_mode == 1 ? 0 : 1;
            }
            cfg::changed();
            apply_display(hwnd);
            return 0;
        }
    }
    if (ui::handle_message(hwnd, msg, wp, lp)) return 0;
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// The window lives on its own thread with its own message loop, so guest threads never pump messages.
void window_thread(std::promise<HWND>* ready) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"Zonaite";
    wc.hCursor = LoadCursorW(nullptr, (LPCWSTR)IDC_ARROW);
    RegisterClassW(&wc);
    const cfg::Config c = cfg::get_copy();
    RECT r{0, 0, std::max(320, c.window_w), std::max(180, c.window_h)};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowExW(0, L"Zonaite", L"Zonaite", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    if (hwnd) {
        apply_display(hwnd);  // windowed and centred, or fullscreen
        ShowWindow(hwnd, SW_SHOW);
    }
    ready->set_value(hwnd);
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
}

}  // namespace

bool window_focused() { return g_hwnd && GetForegroundWindow() == g_hwnd; }

void* window_handle() { return g_hwnd; }

std::array<float, 4> content_rect() {
    if (!g_window) return {0, 0, 1, 1};
    const auto& l = g_window->GetFramebufferLayout();
    if (!l.width || !l.height) return {0, 0, 1, 1};
    const float w = static_cast<float>(l.width), h = static_cast<float>(l.height);
    return {l.screen.left / w, l.screen.top / h, (l.screen.right - l.screen.left) / w,
            (l.screen.bottom - l.screen.top) / h};
}

void request_display_apply() {
    if (g_hwnd) PostMessageW(g_hwnd, kApplyDisplay, 0, 0);
}

std::vector<std::string> monitor_names() {
    std::vector<std::string> names;
    int i = 1;
    for (const RECT& r : monitor_rects())
        names.push_back("Display " + std::to_string(i++) + " (" + std::to_string(r.right - r.left) + " x " +
                        std::to_string(r.bottom - r.top) + ")");
    return names;
}

int refresh_rate() {
    DEVMODEW mode{};
    mode.dmSize = sizeof mode;
    return EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) && mode.dmDisplayFrequency > 1
               ? static_cast<int>(mode.dmDisplayFrequency)
               : 60;
}

// The renderer copies the next presented frame (B8G8R8A8, 1920x1080) into the buffer on its own thread; the
// callback swizzles it to RGBA and writes the PNG.
void request_screenshot(const std::string& path) {
    if (!gpu_slot()) return;
    auto& renderer = gpu_slot()->Renderer();
    if (renderer.IsScreenshotPending()) return;
    const auto layout = Layout::DefaultFrameLayout(kWidth, kHeight);
    auto pixels = std::make_shared<std::vector<uint8_t>>((size_t)layout.width * layout.height * 4);
    renderer.RequestScreenshot(
        pixels->data(),
        [pixels, path, layout](bool invert_y) {
            const uint32_t w = layout.width, h = layout.height;
            std::vector<uint8_t> rgba(pixels->size());
            for (uint32_t y = 0; y < h; y++) {
                const uint8_t* src = pixels->data() + (size_t)(invert_y ? h - 1 - y : y) * w * 4;
                uint8_t* dst = rgba.data() + (size_t)y * w * 4;
                for (uint32_t x = 0; x < w; x++) {
                    dst[x * 4 + 0] = src[x * 4 + 2];
                    dst[x * 4 + 1] = src[x * 4 + 1];
                    dst[x * 4 + 2] = src[x * 4 + 0];
                    dst[x * 4 + 3] = 255;
                }
            }
            // Eden builds stb without stdio: write through a callback
            std::ofstream out(path, std::ios::binary);
            const auto write = [](void* f, void* data, int size) {
                static_cast<std::ofstream*>(f)->write(static_cast<const char*>(data), size);
            };
            if (out && stbi_write_png_to_func(write, &out, (int)w, (int)h, 4, rgba.data(), (int)w * 4) && out)
                zn::log("screenshot: %s", path.c_str());
            else
                zn::log("screenshot: could not write %s", path.c_str());
        },
        layout);
}

// Every pipeline the game has used before (build/user/cache/shader/<title>/vulkan.bin, appended to as new ones
// are built) is compiled here, before the game runs, so play never waits on a shader compile it has seen.
void load_pipeline_cache() {
    constexpr u64 kTitleId = 0x0100F2C0115B6000;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> total{0};  // the callback runs on the pipeline workers
    std::stop_source stop;
    gpu_slot()->Renderer().ReadRasterizer()->LoadDiskResources(
        kTitleId, stop.get_token(), [&](VideoCore::LoadCallbackStage stage, size_t done, size_t count) {
            if (stage == VideoCore::LoadCallbackStage::Build && count) {
                total = count;
                if (done % 500 == 0) zn::log("shaders: building cached pipelines %zu/%zu", done, count);
            }
        });
    zn::log("shaders: %zu cached pipelines ready in %.1f s", total.load(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

bool start(std::string* error) {
    cfg::load();
    std::promise<HWND> ready;
    auto hwnd_future = ready.get_future();
    std::thread(window_thread, &ready).detach();
    g_hwnd = hwnd_future.get();
    if (!g_hwnd) {
        *error = "could not create the window";
        return false;
    }
    g_window = std::make_unique<Window>(g_hwnd);

    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
    Settings::values.use_asynchronous_gpu_emulation.SetValue(std::getenv("ZN_SYNC_GPU") == nullptr);
    Settings::values.use_disk_shader_cache.SetValue(true);
    // The settings menu's renderer settings (zonaite/config.h). Among them: a pipeline the cache does not have
    // yet is built on worker threads while play goes on (its draws are skipped for the few frames that takes)
    // instead of stalling the frame (async shaders; ZN_SYNC_SHADERS=1 builds them in place). GPU accuracy High
    // (the default) reads command lists through a flush of GPU-written memory, which TotK needs: it has the GPU
    // patch arguments into its own command buffers.
    const cfg::Config c = cfg::get_copy();
    cfg::apply_renderer(c, true);
    // No reactive flushing: Eden would wait for CPU reads of GPU-written memory to trigger the write-back, and
    // recompiled code reads guest memory directly, so it never sees them. Write GPU results back eagerly.
    Settings::values.use_reactive_flushing.SetValue(std::getenv("ZN_REACTIVE_FLUSH") != nullptr);
    // Frames are shown from the present thread (frame pacing for frame interpolation, which apply_renderer set
    // up above and the settings menu turns on and off while playing); ZN_SYNC_PRESENT=1 shows them from the GPU
    // thread instead (frame generation then can't be paced)
    Settings::values.async_presentation.SetValue(!std::getenv("ZN_SYNC_PRESENT"));
    if (const char* d = std::getenv("ZN_DMA"))  // debugging: 0 default, 1 unsafe (read in place), 2 safe (flush first)
        Settings::values.dma_accuracy.SetValue(static_cast<Settings::DmaAccuracy>(std::atoi(d)));
    VideoCore::CreateGPU(gpu_slot(), *g_window, system());
    if (!gpu_slot()) {
        *error = "the Vulkan renderer failed to initialise (see the log)";
        return false;
    }
    gpu_slot()->Start();
    load_pipeline_cache();
    mark_gpu_ready();
    return true;
}

}  // namespace zn::gpu
