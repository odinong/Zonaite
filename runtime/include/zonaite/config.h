// Zonaite's own settings (the settings menu, runtime/gpu/ui.cpp): video, graphics, input, audio and the rest,
// saved as user/zonaite.ini next to the executable. Readers take a copy under the lock (get_copy); the menu edits
// through edit() and saves. Settings that map onto Eden's renderer are applied by apply_renderer().
#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace zn::cfg {

// The Switch controls a binding drives (a Pro Controller)
enum Control : int {
    kA, kB, kX, kY, kL, kR, kZL, kZR, kPlus, kMinus, kLStick, kRStick,
    kDUp, kDDown, kDLeft, kDRight,
    kLUp, kLDown, kLLeft, kLRight,  // left stick directions (keyboard: full deflection)
    kRUp, kRDown, kRLeft, kRRight,  // right stick (camera)
    kControlCount
};
constexpr int kButtonControls = kLUp;  // the first kButtonControls controls are buttons a gamepad button can drive

// Gamepad (XInput) inputs a button control can be bound to
enum PadInput : int {
    kPadNone = -1,
    kPadA, kPadB, kPadX, kPadY, kPadLB, kPadRB, kPadLT, kPadRT, kPadStart, kPadBack, kPadLS, kPadRS,
    kPadUp, kPadDown, kPadLeft, kPadRight,
    kPadCount
};

struct Binding {
    std::array<int, 2> keys{};  // Win32 virtual-key codes (mouse buttons too: VK_LBUTTON ...), 0 = none
    int pad = kPadNone;
};

struct Config {
    // video
    int display_mode = 0;  // 0 windowed, 1 borderless fullscreen
    int window_w = 1920, window_h = 1080;
    int monitor = 0;
    int vsync = 2;          // Eden VSyncMode: 0 immediate, 1 mailbox, 2 FIFO (vsync), 3 FIFO relaxed
    int frame_gen = 0;      // frames shown per game frame: 0 auto (what the refresh rate fits), 1 off, 2-4
    int target_rate = 0;    // cap on frames shown per second with frame generation (0 = no cap)
    bool depth_interp = true;  // (older setting, replaced by interp_mode)
    // 0 clean: in-between frames only from the game's depth and camera, never across a hitch or a camera cut
    //   (the real frame stays up instead), so nothing is ever smeared; 1 always: also from the pictures alone where
    //   there is no depth (menus, loading) and across hitches; 2 pictures only
    int interp_mode = 0;
    bool pacing = true;     // spread interpolated frames evenly
    int aspect = 0;         // Eden AspectRatio
    int perf_overlay = 0;   // 0 off, 1 frame rate, 2 detailed
    int overlay_corner = 1; // 0 top left, 1 top right, 2 bottom left, 3 bottom right
    bool show_hint = true;  // "F1: settings" at startup
    float ui_scale = 1.0f;

    // graphics (Eden renderer)
    int resolution = 3;     // ResolutionSetup (3 = 1x)
    int scaling_filter = 1; // ScalingFilter (1 = bilinear)
    int fsr_sharpness = 25;
    int anti_aliasing = 0;  // 0 none, 1 FXAA, 2 SMAA
    int anisotropy = 0;     // AnisotropyMode (0 automatic)
    bool async_shaders = true;
    int gpu_accuracy = 1;   // 0 low, 1 high
    int astc_decode = 1;    // AstcDecodeMode (1 GPU)
    int astc_recompression = 0;
    int vram_mode = 0;      // 0 conservative, 1 aggressive
    bool driver_pipeline_cache = true;  // (read when the renderer starts; no longer in the menu)
    bool fix_bloom = false;
    int gpu_clock = 1;      // GpuClock (1 boost)

    // picture: adjustments to every frame shown (runtime/gpu/picture.cpp), in display-encoded values
    float brightness = 0.0f;  // added, -0.25 .. 0.25
    float contrast = 1.0f;
    float saturation = 1.0f;
    float gamma = 1.0f;       // >1 lifts the shadows
    float sharpen = 0.0f;     // contrast adaptive sharpening, 0 .. 1

    // audio
    int volume = 100;       // percent, up to 200
    bool mute = false;
    bool mute_unfocused = false;
    std::string audio_device;  // WASAPI endpoint ID, empty = the system default
    int audio_latency_ms = 40;

    // input
    std::array<Binding, kControlCount> binds{};
    int pad_index = -1;     // XInput slot, -1 = the first connected
    float deadzone_left = 0.24f, deadzone_right = 0.27f;  // fractions of full deflection
    float outer_deadzone = 0.98f;
    float response_curve = 1.0f;  // 1 linear, >1 finer control near the centre
    float camera_speed_x = 1.0f, camera_speed_y = 1.0f;
    bool invert_x = false, invert_y = false;
    bool swap_sticks = false;
    int trigger_threshold = 30;  // 0..255
    bool mouse_camera = false;   // mouse moves the camera while the window has focus (Alt frees the cursor)
    float mouse_sensitivity = 1.0f;
    bool mouse_invert_y = false;
    bool keyboard_unfocused = false;  // read the keyboard even when the window isn't focused
    bool rumble = true;          // the game's vibration on the gamepad's motors
    int rumble_strength = 100;   // percent, up to 200

    // other
    int menu_key = 0x70;       // VK_F1
    int screenshot_key = 0x7B; // VK_F12
    int fullscreen_key = 0x7A; // VK_F11
    int hud_key = 0x72;        // VK_F3: hide the HUD
};

std::mutex& mutex();
Config& edit();           // under mutex()
Config get_copy();
bool picture_neutral(const Config& c);  // no picture adjustment (the pass is skipped)
uint64_t generation();    // bumps on every change, so readers know when to re-read
void changed();           // after an edit: bump the generation and save
void load();
void save();
void reset_bindings(Config& c, bool xbox_labels = false);
std::string user_dir();   // <exe dir>/user

const char* control_name(int control);
const char* pad_input_name(int pad);
std::string key_name(int vk);

// Applies the renderer settings to Eden's: all at startup, and again whenever the menu changes them. Those Eden
// normally reads only when it starts (resolution, texture filtering and decoding, video memory, GPU accuracy)
// are handed to the GPU thread, which applies them between two frames (RasterizerVulkan::ZonaiteApplyLive)
void apply_renderer(const Config& c, bool startup);

// Frames shown per game frame for this config and display (1 = frame generation off)
int frame_multiplier(const Config& c);

// Empties the shader cache, on disk and in memory, while playing (GPU thread, between frames)
void request_shader_cache_clear();

}  // namespace zn::cfg
