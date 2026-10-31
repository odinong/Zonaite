// Zonaite's launcher: the first-run setup (what a dump is, how to make one, choosing its folder) and the
// version choice (the latest update, or the original release for its glitches), then it prepares the chosen
// version (tools/play.py --prepare: recompiling it from the player's own dump the first time, with progress)
// and starts it. It carries no game code, so it ships in the release package. Look: the settings menu's
// (runtime/gpu/ui.cpp): stone, jade and gold, Marcellus and Lato.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "gfx.h"
#include "imgui.h"

#include <shellapi.h>
#include <shobjidl.h>

namespace fs = std::filesystem;

// Dear ImGui's asserts (third_party/imgui/imconfig.h): never abort the launcher over one
void zn_imgui_assert(const char*, const char*, int) {}

namespace launcher {
namespace {

// --- look ---------------------------------------------------------------------------------------------------------

constexpr ImU32 kInk = IM_COL32(236, 233, 226, 255);
constexpr ImU32 kMuted = IM_COL32(160, 166, 166, 255);
constexpr ImU32 kFaint = IM_COL32(104, 112, 114, 255);
constexpr ImU32 kLine = IM_COL32(255, 255, 255, 22);
constexpr ImU32 kAccent = IM_COL32(118, 214, 188, 255);
constexpr ImU32 kAmber = IM_COL32(232, 178, 92, 255);
constexpr ImU32 kRed = IM_COL32(236, 120, 104, 255);

ImFont *g_text, *g_bold, *g_small, *g_title, *g_head;
float g_s = 1.0f;  // layout scale (DPI)

float S(float v) { return v * g_s; }

ImU32 alpha(ImU32 c, float a) {
    return (c & 0x00ffffff) | (static_cast<ImU32>(((c >> 24) & 0xff) * a) << 24);
}

void text(ImDrawList* dl, ImFont* f, ImVec2 p, ImU32 c, const char* s) {
    dl->AddText(f, f->FontSize, p, c, s);
}

// letter-spaced capitals (section names, the wordmark)
void spaced(ImDrawList* dl, ImFont* f, ImVec2 p, ImU32 c, const char* s, float tracking) {
    for (const char* q = s; *q; ++q) {
        const char ch[2] = {*q, 0};
        text(dl, f, p, c, ch);
        p.x += f->CalcTextSizeA(f->FontSize, FLT_MAX, 0, ch).x + tracking;
    }
}

// Wrapped body text at the cursor
void para(const char* s, ImU32 c = kMuted, ImFont* f = nullptr) {
    ImGui::PushFont(f ? f : g_text);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(c));
    ImGui::TextWrapped("%s", s);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void heading(const char* s) {
    ImGui::PushFont(g_head);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kInk));
    ImGui::TextUnformatted(s);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

// A button drawn in the menu's style: filled jade (primary) or outlined
bool button(const char* label, bool primary = false, float width = 0, bool enabled = true) {
    ImGui::PushFont(g_bold);
    const ImVec2 ts = ImGui::CalcTextSize(label);
    const ImVec2 size{std::max(width, ts.x + S(48)), S(46)};
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::PushID(label);
    const bool pressed = ImGui::InvisibleButton("##b", size) && enabled;
    const bool hot = ImGui::IsItemHovered() && enabled;
    ImGui::PopID();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 q{p.x + size.x, p.y + size.y};
    if (primary) {
        dl->AddRectFilled(p, q, enabled ? (hot ? kAccent : alpha(kAccent, 0.85f)) : alpha(kFaint, 0.4f), S(6));
    } else {
        dl->AddRectFilled(p, q, hot ? IM_COL32(255, 255, 255, 14) : IM_COL32(255, 255, 255, 6), S(6));
        dl->AddRect(p, q, hot ? alpha(kAccent, 0.8f) : IM_COL32(255, 255, 255, 40), S(6), 0, S(1));
    }
    const ImU32 tc = primary ? IM_COL32(8, 20, 18, 255) : (enabled ? (hot ? kInk : alpha(kInk, 0.9f)) : kFaint);
    dl->AddText(g_bold, g_bold->FontSize, {p.x + (size.x - ts.x) * 0.5f, p.y + (size.y - ts.y) * 0.5f}, tc, label);
    ImGui::PopFont();
    if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return pressed;
}

// --- the installation: where things are ----------------------------------------------------------------------------

fs::path g_root;  // the folder with tools/play.py (the package, or a repository checkout)

fs::path find_root() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    for (fs::path p = fs::path(buf).parent_path(); !p.empty(); p = p.parent_path()) {
        if (fs::exists(p / "tools" / "play.py")) return p;
        if (p == p.parent_path()) break;
    }
    return fs::path(buf).parent_path();
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The string value of "key" in a small JSON text (the launcher only reads its own simple files)
std::string json_string(const std::string& j, const std::string& key, size_t from = 0) {
    size_t p = j.find("\"" + key + "\"", from);
    if (p == std::string::npos) return {};
    p = j.find(':', p);
    p = j.find_first_not_of(" \t\r\n", p + 1);
    if (p == std::string::npos || j[p] != '"') return {};
    std::string out;
    for (++p; p < j.size() && j[p] != '"'; ++p) {
        if (j[p] == '\\' && p + 1 < j.size()) ++p;
        out += j[p];
    }
    return out;
}

bool json_true(const std::string& j, const std::string& key) {
    size_t p = j.find("\"" + key + "\"");
    return p != std::string::npos && j.compare(j.find(':', p) + 1, 5, " true") == 0;
}

struct Version {
    std::string version, label;
    bool verified = false;
    std::string dump;  // the remembered dump folder, if any
    bool built = false;
};
std::vector<Version> g_versions;
int g_selected = 0;

int version_key(const std::string& v) {
    int a = 0, b = 0, c = 0;
    std::sscanf(v.c_str(), "%d.%d.%d", &a, &b, &c);
    return a * 10000 + b * 100 + c;
}

// config/targets/*.json, the remembered dumps (user/dumps.json) and what is built
void load_versions() {
    std::string keep = g_versions.empty() ? std::string() : g_versions[g_selected].version;
    g_versions.clear();
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(g_root / "config" / "targets", ec)) {
        if (e.path().extension() != ".json") continue;
        const std::string j = read_file(e.path());
        Version v;
        v.version = json_string(j, "version");
        v.label = json_string(j, "label");
        v.verified = json_true(j, "verified");
        if (!v.version.empty()) g_versions.push_back(v);
    }
    std::sort(g_versions.begin(), g_versions.end(),
              [](const Version& a, const Version& b) { return version_key(a.version) > version_key(b.version); });
    const std::string dumps = read_file(g_root / "user" / "dumps.json");
    for (auto& v : g_versions) {
        v.dump = json_string(dumps, v.version);
        v.built = fs::exists(g_root / "build" / ("zonaite-" + v.version + ".exe"), ec);
    }
    if (keep.empty()) keep = json_string(read_file(g_root / "user" / "launcher.json"), "version");
    g_selected = 0;
    for (size_t i = 0; i < g_versions.size(); ++i)
        if (g_versions[i].version == keep) g_selected = static_cast<int>(i);
}

bool any_dump() {
    return std::any_of(g_versions.begin(), g_versions.end(), [](const Version& v) { return !v.dump.empty(); });
}

void remember_selection() {
    std::error_code ec;
    fs::create_directories(g_root / "user", ec);
    std::ofstream(g_root / "user" / "launcher.json") << "{\"version\": \"" << g_versions[g_selected].version
                                                       << "\"}\n";
}

// --- what rebuilding the game needs on this PC ---------------------------------------------------------------------

// Python runs the build scripts, LLVM compiles the game, and the Build Tools bring the Windows libraries it links
// against. Missing ones can be installed with winget (Windows' package manager), one after another.
struct Tool {
    const char* name;
    const char* why;
    const char* size;
    const wchar_t* winget;  // winget's arguments
    const char* url;        // the download page, for PCs without winget
    bool present = false;
    enum class State { Idle, Queued, Installing, Done, Failed } state = State::Idle;
};
Tool g_tools[] = {
    {"Python 3", "Runs Zonaite's build scripts.", "about 30 MB",
     L"install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements "
     L"--override \"/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1\"",
     "https://www.python.org/downloads/windows/"},
    {"LLVM", "Compiles the game for this PC.", "about 600 MB",
     L"install -e --id LLVM.LLVM --accept-source-agreements --accept-package-agreements --silent",
     "https://github.com/llvm/llvm-project/releases"},
    {"Visual Studio Build Tools", "Windows' C++ libraries, which the game is linked with.", "about 3 GB",
     L"install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements "
     L"--override \"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\"",
     "https://visualstudio.microsoft.com/visual-cpp-build-tools/"},
};
std::wstring g_python;  // the command that runs Python ("py -3" or python.exe), empty if none
std::wstring g_winget;  // winget.exe, empty if this PC has none

// PATH as a newly started program sees it: an install adds to it, but this process still has the old one
void refresh_path() {
    std::wstring path;
    const std::pair<HKEY, const wchar_t*> keys[] = {
        {HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment"},
        {HKEY_CURRENT_USER, L"Environment"}};
    for (const auto& [root, key] : keys) {
        const DWORD flags = RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND;
        DWORD size = 0;
        if (RegGetValueW(root, key, L"Path", flags, nullptr, nullptr, &size) != ERROR_SUCCESS) continue;
        std::wstring raw(size / sizeof(wchar_t) + 1, L'\0');
        if (RegGetValueW(root, key, L"Path", flags, nullptr, raw.data(), &size) != ERROR_SUCCESS) continue;
        raw.resize(wcslen(raw.c_str()));
        std::wstring expanded(32768, L'\0');
        expanded.resize(ExpandEnvironmentStringsW(raw.c_str(), expanded.data(), 32768));
        if (!expanded.empty() && expanded.back() == L'\0') expanded.pop_back();
        if (!path.empty()) path += L';';
        path += expanded;
    }
    if (!path.empty()) SetEnvironmentVariableW(L"PATH", path.c_str());
}

// Runs a program to its end and returns what it printed
std::string run_capture(std::wstring command) {
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE read = nullptr, write = nullptr;
    if (!CreatePipe(&read, &write, &sa, 0)) return {};
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{sizeof si};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = si.hStdError = write;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                   nullptr, &si, &pi);
    CloseHandle(write);
    std::string out;
    if (ok) {
        char buf[4096];
        DWORD n;
        while (ReadFile(read, buf, sizeof buf, &n, nullptr) && n) out.append(buf, n);
        WaitForSingleObject(pi.hProcess, 5000);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    CloseHandle(read);
    return out;
}

std::wstring search(const wchar_t* exe) {
    wchar_t buf[MAX_PATH];
    return SearchPathW(nullptr, exe, nullptr, MAX_PATH, buf, nullptr) ? std::wstring(buf) : std::wstring();
}

std::wstring expand(const wchar_t* s) {
    wchar_t buf[MAX_PATH * 2];
    ExpandEnvironmentStringsW(s, buf, MAX_PATH * 2);
    return buf;
}

// The C++ build tools (vswhere finds Visual Studio's) and a Windows SDK to link against
bool has_build_tools() {
    const std::wstring vswhere = expand(L"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe");
    if (!fs::exists(vswhere)) return false;
    const std::string found = run_capture(L"\"" + vswhere + L"\" -latest -products * -requires "
                                          L"Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath");
    if (found.find_first_not_of(" \r\n") == std::string::npos) return false;
    wchar_t kits[MAX_PATH] = {};
    DWORD size = sizeof kits;
    return RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots", L"KitsRoot10",
                        RRF_RT_REG_SZ, nullptr, kits, &size) == ERROR_SUCCESS &&
           fs::exists(fs::path(kits) / "Lib");
}

void find_tools() {
    refresh_path();
    g_python.clear();
    if (const auto py = search(L"py.exe"); !py.empty()) {
        g_python = L"\"" + py + L"\" -3";
    } else if (const auto python = search(L"python.exe");
               !python.empty() && python.find(L"WindowsApps") == std::wstring::npos) {  // (the Store's stub)
        g_python = L"\"" + python + L"\"";
    } else {  // a per-user install that isn't on PATH yet
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(expand(L"%LOCALAPPDATA%\\Programs\\Python"), ec))
            if (fs::exists(e.path() / "python.exe")) g_python = L"\"" + (e.path() / "python.exe").wstring() + L"\"";
    }
    g_tools[0].present = !g_python.empty();
    g_tools[1].present = !search(L"clang++.exe").empty() ||
                         fs::exists(expand(L"%ProgramFiles%\\LLVM\\bin\\clang++.exe"));
    g_tools[2].present = has_build_tools();
    g_winget = search(L"winget.exe");
}

bool tools_ready() {
    return std::all_of(std::begin(g_tools), std::end(g_tools), [](const Tool& t) { return t.present; });
}

std::wstring widen(const std::string& s) {
    std::wstring w(MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), static_cast<int>(w.size()));
    w.resize(w.size() - 1);
    return w;
}

std::string narrow(const std::wstring& w) {
    std::string s(WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), static_cast<int>(s.size()), nullptr, nullptr);
    s.resize(s.size() - 1);
    return s;
}

// --- a tool running in the background, its output kept --------------------------------------------------------------

struct Job {
    std::mutex mutex;
    std::vector<std::string> lines;
    std::atomic<bool> running{false};
    std::atomic<int> exit_code{0};
    HANDLE process = nullptr;
    std::thread reader;

    void start(const std::wstring& command) {
        if (reader.joinable()) reader.join();
        {
            std::scoped_lock lock{mutex};
            lines.clear();
        }
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE read = nullptr, write = nullptr;
        CreatePipe(&read, &write, &sa, 0);
        SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOW si{sizeof si};
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdOutput = si.hStdError = write;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        PROCESS_INFORMATION pi{};
        std::wstring cmd = command;
        running = true;
        const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE,
                                       CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS, nullptr,
                                       g_root.wstring().c_str(), &si, &pi);
        CloseHandle(write);
        if (!ok) {
            CloseHandle(read);
            std::scoped_lock lock{mutex};
            lines.push_back("Could not start: " + narrow(command));
            exit_code = -1;
            running = false;
            return;
        }
        CloseHandle(pi.hThread);
        process = pi.hProcess;
        reader = std::thread([this, read] {
            std::string partial;
            char buf[4096];
            DWORD n;
            while (ReadFile(read, buf, sizeof buf, &n, nullptr) && n) {
                partial.append(buf, n);
                for (size_t nl; (nl = partial.find_first_of("\r\n")) != std::string::npos;) {
                    std::string line = partial.substr(0, nl);
                    partial.erase(0, nl + 1);
                    if (!line.empty()) {
                        std::scoped_lock lock{mutex};
                        lines.push_back(line);
                    }
                }
            }
            if (!partial.empty()) {
                std::scoped_lock lock{mutex};
                lines.push_back(partial);
            }
            CloseHandle(read);
            WaitForSingleObject(process, INFINITE);
            DWORD code = 0;
            GetExitCodeProcess(process, &code);
            exit_code = static_cast<int>(code);
            CloseHandle(process);
            process = nullptr;
            running = false;
        });
    }

    void cancel() {
        if (process) TerminateProcess(process, 1);  // (play.py's own children end with their pipe)
    }

    std::vector<std::string> snapshot() {
        std::scoped_lock lock{mutex};
        return lines;
    }
};

std::wstring play_command(const std::wstring& args) {
    return g_python + L" \"" + (g_root / "tools" / "play.py").wstring() + L"\" " + args;
}

// --- screens -------------------------------------------------------------------------------------------------------

enum class Screen { Welcome, Guide, Library, Preparing, Tools };
Screen g_screen = Screen::Welcome;
Job g_job;
enum class JobKind { None, AddDump, Prepare, Install };
void open_tools(Screen back);  // the tools screen (below)
JobKind g_job_kind = JobKind::None;
std::string g_notice;        // the last dump check's result, or an error
ImU32 g_notice_colour = kMuted;
std::string g_preparing;     // the version being prepared
HANDLE g_game = nullptr;     // the game, while it runs (the launcher hides meanwhile)
DWORD g_game_started = 0;

std::wstring pick_folder() {
    std::wstring out;
    IFileOpenDialog* dialog = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_PPV_ARGS(&dialog)))) return out;
    DWORD options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST);
    dialog->SetTitle(L"Choose the folder with Exefs and RomFs in it");
    if (SUCCEEDED(dialog->Show(gfx_window()))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dialog->GetResult(&item))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                out = path;
                CoTaskMemFree(path);
            }
            item->Release();
        }
    }
    dialog->Release();
    return out;
}

void add_dump() {
    const std::wstring folder = pick_folder();
    if (folder.empty()) return;
    g_notice = "Checking " + narrow(folder) + " ...";
    g_notice_colour = kMuted;
    g_job_kind = JobKind::AddDump;
    g_job.start(play_command(L"--add-dump \"" + folder + L"\""));
}

// The dump check's JSON (tools/play.py --add-dump), as a sentence
void finish_add_dump() {
    const auto lines = g_job.snapshot();
    const std::string j = lines.empty() ? std::string() : lines.back();
    const std::string version = json_string(j, "version");
    std::vector<std::string> problems;
    if (size_t p = j.find("\"problems\""); p != std::string::npos) {
        for (size_t q = j.find('[', p) + 1; q < j.size() && j[q] != ']';) {
            const size_t a = j.find('"', q);
            if (a == std::string::npos || a > j.find(']', q)) break;
            size_t b = a + 1;
            std::string s;
            for (; b < j.size() && j[b] != '"'; ++b) {
                if (j[b] == '\\' && b + 1 < j.size()) ++b;
                s += j[b];
            }
            problems.push_back(s);
            q = b + 1;
        }
    }
    if (j.find('{') == std::string::npos) {
        g_notice = "The check could not run: " + (lines.empty() ? std::string("no output") : lines.back());
        g_notice_colour = kRed;
    } else if (problems.empty()) {
        g_notice = "Found Tears of the Kingdom " + version + ". It's ready to play.";
        g_notice_colour = kAccent;
        load_versions();
        for (size_t i = 0; i < g_versions.size(); ++i)
            if (g_versions[i].version == version) g_selected = static_cast<int>(i);
        remember_selection();
        g_screen = Screen::Library;
    } else {
        g_notice = "That folder can't be used: " + problems.front();
        g_notice_colour = kRed;
    }
}

void start_prepare() {
    remember_selection();
    g_preparing = g_versions[g_selected].version;
    g_job_kind = JobKind::Prepare;
    g_screen = Screen::Preparing;
    g_job.start(play_command(L"--prepare --version " + widen(g_preparing)));
}

void start_game() {
    const Version& v = g_versions[g_selected];
    const fs::path exe = g_root / "build" / ("zonaite-" + v.version + ".exe");
    const fs::path game = g_root / "build" / ("game-" + v.version);
    std::wstring cmd = L"\"" + exe.wstring() + L"\" \"" + game.wstring() + L"\" \"" + widen(v.dump) + L"\"";
    STARTUPINFOW si{sizeof si};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, g_root.wstring().c_str(), &si,
                        &pi)) {
        g_notice = "The game could not start (" + exe.string() + ").";
        g_notice_colour = kRed;
        g_screen = Screen::Library;
        return;
    }
    CloseHandle(pi.hThread);
    g_game = pi.hProcess;
    g_game_started = GetTickCount();
    ShowWindow(gfx_window(), SW_HIDE);
}

// The window's frame: background, the wordmark, and the content area's left edge and width
struct Frame {
    ImVec2 size;
    float left, top, width;
};

Frame begin_screen() {
    const ImVec2 size = ImGui::GetIO().DisplaySize;
    ImDrawList* bg = ImGui::GetBackgroundDrawList();
    bg->AddRectFilled({0, 0}, size, IM_COL32(7, 11, 13, 255));
    bg->AddRectFilledMultiColor({size.x * 0.45f, 0}, size, IM_COL32(7, 11, 13, 0), IM_COL32(18, 36, 34, 120),
                                IM_COL32(18, 36, 34, 120), IM_COL32(7, 11, 13, 0));
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(size);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    ImGui::Begin("##launcher", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::PopStyleVar();
    Frame f{size, std::max(S(64), size.x * 0.07f), S(56), 0};
    f.width = std::min(size.x - 2 * f.left, S(1100));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    spaced(dl, g_bold, {f.left, f.top}, kAccent, "ZONAITE", S(4));
    return f;
}

void end_screen() {
    ImGui::End();
}

void screen_welcome() {
    const Frame f = begin_screen();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    text(dl, g_title, {f.left - S(3), f.top + S(22)}, kInk, "Welcome");
    ImGui::SetCursorScreenPos({f.left, f.top + S(110)});
    ImGui::BeginGroup();
    const float col = std::min(f.width * 0.58f, S(680));
    ImGui::PushTextWrapPos(f.left + col);
    para("Zonaite plays The Legend of Zelda: Tears of the Kingdom natively on your PC. It doesn't come with the "
         "game: it is rebuilt here, on this PC, from your own copy.", kInk);
    ImGui::Dummy({0, S(14)});
    para("To start, Zonaite needs a dump: the game's files (two folders, Exefs and RomFs) extracted from a copy you "
         "own. If you already have one, choose its folder. If not, the guide shows how to make one in a few "
         "minutes.");
    ImGui::Dummy({0, S(30)});
    if (button("Choose my dump folder", true)) add_dump();
    ImGui::SameLine(0, S(14));
    if (button("How do I make a dump?")) g_screen = Screen::Guide;
    ImGui::Dummy({0, S(22)});
    if (!g_notice.empty()) para(g_notice.c_str(), g_notice_colour);
    if (g_job.running && g_job_kind == JobKind::AddDump) para("Checking the dump...", kMuted, g_small);
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();

    // the three steps, on the right
    {
        const float x = f.left + col + S(64), w = f.left + f.width - x;
        if (w > S(240)) {
            float y = f.top + S(110);
            dl->AddLine({x - S(32), y}, {x - S(32), y + S(300)}, kLine, S(1));
            spaced(dl, g_bold, {x, y}, kFaint, "HOW IT GOES", S(2.5f));
            const char* steps[][2] = {{"Make a dump", "Your copy of the game, extracted with Ryujinx."},
                                      {"Choose its folder", "Zonaite checks it and recognises the version."},
                                      {"Play", "The first start rebuilds the game for this PC, once."}};
            y += S(40);
            for (int i = 0; i < 3; ++i) {
                char n[4];
                std::snprintf(n, sizeof n, "%d", i + 1);
                text(dl, g_head, {x, y - S(4)}, alpha(kAccent, 0.8f), n);
                text(dl, g_bold, {x + S(36), y}, kInk, steps[i][0]);
                text(dl, g_small, {x + S(36), y + S(24)}, kMuted, steps[i][1]);
                y += S(80);
            }
        }
    }

    // what is needed on this PC, at the bottom
    ImGui::SetCursorScreenPos({f.left, f.size.y - S(110)});
    ImGui::PushTextWrapPos(f.left + f.width);
    if (!tools_ready()) {
        para("This PC is missing some free tools Zonaite needs to build the game. Zonaite can install them.", kAmber,
             g_small);
        ImGui::Dummy({0, S(6)});
        if (button("Set them up")) open_tools(Screen::Welcome);
    } else {
        para("Your dump stays on this PC. Nothing from the game is ever uploaded or shared.", kFaint, g_small);
    }
    ImGui::PopTextWrapPos();
    end_screen();
}

void step(int n, const char* title, const char* body) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    dl->AddCircle({p.x + S(16), p.y + S(16)}, S(15), alpha(kAccent, 0.7f), 32, S(1.5f));
    char num[4];
    std::snprintf(num, sizeof num, "%d", n);
    const ImVec2 ns = g_bold->CalcTextSizeA(g_bold->FontSize, FLT_MAX, 0, num);
    text(dl, g_bold, {p.x + S(16) - ns.x * 0.5f, p.y + S(16) - ns.y * 0.5f}, kAccent, num);
    ImGui::SetCursorScreenPos({p.x + S(52), p.y + S(2)});
    ImGui::BeginGroup();
    para(title, kInk, g_bold);
    ImGui::Dummy({0, S(2)});
    para(body);
    ImGui::EndGroup();
    ImGui::SetCursorScreenPos({p.x, ImGui::GetCursorScreenPos().y + S(16)});
}

void screen_guide() {
    const Frame f = begin_screen();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    text(dl, g_title, {f.left - S(3), f.top + S(22)}, kInk, "Making a dump");
    ImGui::SetCursorScreenPos({f.left, f.top + S(104)});
    ImGui::BeginChild("##guide", {f.width, f.size.y - f.top - S(104) - S(110)}, 0, ImGuiWindowFlags_NoBackground);
    ImGui::PushTextWrapPos(ImGui::GetCursorScreenPos().x + std::min(f.width, S(820)) - S(20));
    para("A dump is your copy of the game, taken out of its package. You need the game dumped from your own "
         "Switch (for example with DBI or NXDumpTool), and your console's keys set up in Ryujinx. Zonaite never "
         "asks for or uses keys itself.");
    ImGui::Dummy({0, S(20)});
    step(1, "Open the game in Ryujinx",
         "Add the folder with your game file (NSP or XCI) under Options > Settings > Game Directories, so Tears "
         "of the Kingdom shows in the list.");
    step(2, "Choose the version",
         "Right-click the game > Manage Title Updates. For the latest version (1.2.1), add and select the 1.2.1 "
         "update. For the original release (1.0.0, with its glitches), select \"No Update\". Each version needs "
         "its own dump; you can add both.");
    step(3, "Extract ExeFS",
         "Right-click the game > Extract Data > ExeFS, and choose a new empty folder, for example "
         "\"TotK 1.2.1\" or \"TotK 1.0.0\".");
    step(4, "Extract RomFS",
         "Right-click the game > Extract Data > RomFS, into the same folder. It is about 16 GB and takes a few "
         "minutes.");
    step(5, "Choose the folder here",
         "The folder should now hold Exefs and RomFs. Choose it below: Zonaite checks it and recognises the "
         "version on its own.");
    ImGui::Dummy({0, S(4)});
    para("Other emulators built on Ryujinx work the same way. If the check says the RomFS and ExeFS don't match, "
         "one was extracted with a different update selected: extract both again with the same choice.", kFaint,
         g_small);
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::SetCursorScreenPos({f.left, f.size.y - S(96)});
    if (button("Choose my dump folder", true)) add_dump();
    ImGui::SameLine(0, S(14));
    if (button("Back")) g_screen = any_dump() ? Screen::Library : Screen::Welcome;
    if (!g_notice.empty() || g_job.running) {
        ImGui::SameLine(0, S(24));
        ImGui::BeginGroup();
        ImGui::Dummy({0, S(10)});
        ImGui::PushTextWrapPos(f.left + f.width);
        para(g_job.running ? "Checking the dump..." : g_notice.c_str(), g_job.running ? kMuted : g_notice_colour,
             g_small);
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }
    end_screen();
}

const char* version_blurb(const std::string& v) {
    if (v == "1.0.0")
        return "The game as it launched in May 2023, before any update. The glitches later updates removed, "
               "like the item duplication tricks, still work here.";
    return "The latest update: every fix and improvement Nintendo made. The best way to play the game normally.";
}

void screen_library() {
    const Frame f = begin_screen();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    text(dl, g_title, {f.left - S(3), f.top + S(22)}, kInk, "Tears of the Kingdom");
    const float top = f.top + S(120);
    spaced(dl, g_bold, {f.left, top}, kFaint, "VERSION", S(2.5f));

    // the versions, one card each
    const float card_w = std::min(S(420), f.width * 0.42f), card_h = S(96);
    float y = top + S(30);
    for (size_t i = 0; i < g_versions.size(); ++i) {
        const Version& v = g_versions[i];
        const ImVec2 p{f.left, y}, q{f.left + card_w, y + card_h};
        ImGui::SetCursorScreenPos(p);
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::InvisibleButton("##card", {card_w, card_h})) {
            g_selected = static_cast<int>(i);
            g_notice.clear();
        }
        const bool hot = ImGui::IsItemHovered();
        ImGui::PopID();
        const bool sel = static_cast<int>(i) == g_selected;
        dl->AddRectFilled(p, q, sel ? IM_COL32(118, 214, 188, 22) : hot ? IM_COL32(255, 255, 255, 10)
                                                                         : IM_COL32(255, 255, 255, 5),
                          S(8));
        dl->AddRect(p, q, sel ? alpha(kAccent, 0.75f) : kLine, S(8), 0, S(1));
        text(dl, g_head, {p.x + S(22), p.y + S(14)}, sel ? kInk : alpha(kInk, 0.85f), v.version.c_str());
        const float vw = g_head->CalcTextSizeA(g_head->FontSize, FLT_MAX, 0, v.version.c_str()).x;
        text(dl, g_text, {p.x + S(22) + vw + S(14), p.y + S(22)}, kMuted, v.label.c_str());
        const char* status = v.dump.empty() ? "No dump yet" : v.built ? "Ready" : "First start rebuilds the game";
        text(dl, g_small, {p.x + S(22), p.y + S(60)}, v.dump.empty() ? kAmber : v.built ? kAccent : kMuted, status);
        y += card_h + S(14);
    }

    // the selected version: what it is, its dump, and Play
    if (!g_versions.empty()) {
        const Version& v = g_versions[g_selected];
        const float x = f.left + card_w + S(48), w = f.left + f.width - x;
        ImGui::SetCursorScreenPos({x, top + S(30)});
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(x + w);
        heading((v.version + "  " + v.label).c_str());
        ImGui::Dummy({0, S(8)});
        para(version_blurb(v.version));
        ImGui::Dummy({0, S(18)});
        if (v.dump.empty()) {
            para("No dump of this version yet. Make one with \"No Update\" or the matching update selected (see "
                 "the guide), then add it.",
                 kAmber, g_small);
            ImGui::Dummy({0, S(16)});
            if (button("Add a dump", true)) add_dump();
            ImGui::SameLine(0, S(14));
            if (button("How do I make one?")) g_screen = Screen::Guide;
        } else {
            para(("Dump: " + v.dump).c_str(), kFaint, g_small);
            if (!v.verified)
                para("Support for this version is new: tell whoever shared Zonaite with you if anything is off.",
                     kAmber, g_small);
            if (!v.built)
                para("The first start rebuilds this version from your dump, which takes about 10 to 20 minutes. "
                     "After that it starts at once.",
                     kMuted, g_small);
            ImGui::Dummy({0, S(16)});
            if (button("Play", true, S(200), !g_job.running)) {
                if (tools_ready() || v.built)
                    start_prepare();
                else
                    open_tools(Screen::Library);  // the tools to build it first
            }
            ImGui::SameLine(0, S(14));
            if (button("Use another dump", false, 0, !g_job.running)) add_dump();
            if (!tools_ready() && !v.built)
                para("Building the game needs a few free tools this PC doesn't have yet. Play installs them first.",
                     kAmber, g_small);
        }
        ImGui::Dummy({0, S(14)});
        if (!g_notice.empty()) para(g_notice.c_str(), g_notice_colour, g_small);
        if (g_job.running && g_job_kind == JobKind::AddDump) para("Checking the dump...", kMuted, g_small);
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }

    // footer: folders and the guide
    ImGui::SetCursorScreenPos({f.left, f.size.y - S(84)});
    if (button("Dump guide")) g_screen = Screen::Guide;
    ImGui::SameLine(0, S(12));
    if (button("Saves")) {
        wchar_t* appdata = nullptr;
        size_t n = 0;
        _wdupenv_s(&appdata, &n, L"APPDATA");
        const fs::path saves = fs::path(appdata ? appdata : L".") / L"Zonaite";
        free(appdata);
        std::error_code ec;
        fs::create_directories(saves, ec);
        ShellExecuteW(nullptr, L"open", saves.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
    end_screen();
}

// "compiled 120/1031 (...)" lines of tools/build_game.py: how far the rebuild is
float build_progress(const std::vector<std::string>& lines, std::string* stage) {
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        int done = 0, total = 0;
        if (std::sscanf(it->c_str(), "compiled %d/%d", &done, &total) == 2 && total) {
            *stage = "Compiling the game for this PC";
            return 0.1f + 0.85f * done / total;
        }
        if (it->rfind("Linking", 0) == 0) {
            *stage = "Linking";
            return 0.97f;
        }
        if (it->rfind("analysis:", 0) == 0 || it->rfind("generated", 0) == 0 || it->rfind("RomFS index", 0) == 0) {
            *stage = "Translating the game's code";
            return 0.08f;
        }
    }
    *stage = "Getting ready";
    return 0.02f;
}

void screen_preparing() {
    const Frame f = begin_screen();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    text(dl, g_title, {f.left - S(3), f.top + S(22)}, kInk, ("Preparing " + g_preparing).c_str());
    const auto lines = g_job.snapshot();
    std::string stage;
    const float progress = g_job.running ? build_progress(lines, &stage) : 1.0f;
    const float y = f.top + S(130);
    text(dl, g_text, {f.left, y}, kMuted, g_job.running ? stage.c_str() : "Done");
    const float bar_w = std::min(f.width, S(760));
    dl->AddRectFilled({f.left, y + S(40)}, {f.left + bar_w, y + S(46)}, IM_COL32(255, 255, 255, 18), S(3));
    dl->AddRectFilled({f.left, y + S(40)}, {f.left + bar_w * progress, y + S(46)}, kAccent, S(3));
    ImGui::SetCursorScreenPos({f.left, y + S(70)});
    ImGui::PushTextWrapPos(f.left + bar_w);
    para("The first start of a version rebuilds the whole game for this PC from your dump. It takes a while and "
         "uses most of the CPU (at low priority, so the PC stays usable). Later starts skip this.",
         kFaint, g_small);
    ImGui::PopTextWrapPos();
    // the last lines of output
    ImGui::SetCursorScreenPos({f.left, y + S(140)});
    ImGui::BeginChild("##log", {bar_w, f.size.y - y - S(140) - S(110)}, 0, ImGuiWindowFlags_NoBackground);
    ImGui::PushFont(g_small);
    const size_t from = lines.size() > 200 ? lines.size() - 200 : 0;
    for (size_t i = from; i < lines.size(); ++i) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kFaint));
        ImGui::TextUnformatted(lines[i].c_str());
        ImGui::PopStyleColor();
    }
    if (g_job.running) ImGui::SetScrollHereY(1.0f);
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::SetCursorScreenPos({f.left, f.size.y - S(84)});
    if (g_job.running) {
        if (button("Cancel")) g_job.cancel();
    } else if (button("Back")) {
        g_screen = Screen::Library;
    }
    end_screen();
}

// --- the tools screen: what's missing, and installing it ------------------------------------------------------------

Screen g_tools_return = Screen::Welcome;  // where Back and Continue go

void start_next_install() {
    for (auto& t : g_tools) {
        if (t.state != Tool::State::Queued) continue;
        t.state = Tool::State::Installing;
        g_job_kind = JobKind::Install;
        g_job.start(L"\"" + g_winget + L"\" " + t.winget);
        return;
    }
}

void install_missing() {
    for (auto& t : g_tools)
        if (!t.present) t.state = Tool::State::Queued;
    start_next_install();
}

// An install ended: whether the tool is there now decides, not winget's exit code (some need a restart to show)
void finish_install() {
    find_tools();
    for (auto& t : g_tools)
        if (t.state == Tool::State::Installing) t.state = t.present ? Tool::State::Done : Tool::State::Failed;
    start_next_install();
}

void open_tools(Screen back) {
    g_tools_return = back;
    find_tools();
    g_screen = Screen::Tools;
}

void screen_tools() {
    const Frame f = begin_screen();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    text(dl, g_title, {f.left - S(3), f.top + S(22)}, kInk, "A few tools first");
    const float col = std::min(f.width, S(860));
    ImGui::SetCursorScreenPos({f.left, f.top + S(110)});
    ImGui::BeginGroup();
    ImGui::PushTextWrapPos(f.left + col);
    para("Zonaite builds the game on your PC the first time you play, and that takes three free tools. Zonaite can "
         "install the ones this PC is missing. Windows will ask for permission for some of them.");
    ImGui::Dummy({0, S(20)});
    const bool installing = g_job.running && g_job_kind == JobKind::Install;
    for (auto& t : g_tools) {
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float h = S(70);
        dl->AddRectFilled(p, {p.x + col, p.y + h}, IM_COL32(255, 255, 255, 6), S(8));
        text(dl, g_bold, {p.x + S(20), p.y + S(14)}, kInk, t.name);
        std::string detail = std::string(t.why) + "  " + t.size;
        text(dl, g_small, {p.x + S(20), p.y + S(38)}, kMuted, detail.c_str());
        const char* status = "Missing";
        ImU32 colour = kAmber;
        if (t.present) {
            status = "Installed";
            colour = kAccent;
        } else if (t.state == Tool::State::Queued) {
            status = "Waiting";
            colour = kMuted;
        } else if (t.state == Tool::State::Installing) {
            status = "Installing...";
            colour = kInk;
        } else if (t.state == Tool::State::Failed) {
            status = "Not installed yet";
            colour = kRed;
        }
        const ImVec2 ss = g_bold->CalcTextSizeA(g_bold->FontSize, FLT_MAX, 0, status);
        float right = p.x + col - S(20);
        if (!t.present && g_winget.empty()) {  // no winget: the download page instead
            ImGui::SetCursorScreenPos({right - S(130), p.y + S(12)});
            ImGui::PushID(t.name);
            if (button("Download", false, S(130)))
                ShellExecuteW(nullptr, L"open", widen(t.url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::PopID();
            right -= S(150);
        }
        text(dl, g_bold, {right - ss.x, p.y + (h - ss.y) * 0.5f}, colour, status);
        ImGui::SetCursorScreenPos({p.x, p.y + h + S(10)});
    }
    ImGui::Dummy({0, S(12)});
    if (tools_ready()) {
        if (button("Continue", true, S(200))) g_screen = g_tools_return;
    } else if (!g_winget.empty()) {
        if (button(installing ? "Installing..." : "Install what's missing", true, 0, !g_job.running)) install_missing();
    }
    ImGui::SameLine(0, S(14));
    if (button("Check again", false, 0, !g_job.running)) find_tools();
    ImGui::SameLine(0, S(14));
    if (button("Back", false, 0, !installing)) g_screen = g_tools_return;
    ImGui::Dummy({0, S(14)});
    if (g_winget.empty())
        para("This PC doesn't have winget, so the tools have to be installed by hand: download each one, install it, "
             "then press Check again. For Python, tick \"Add python.exe to PATH\"; for the Build Tools, choose "
             "\"Desktop development with C++\".",
             kMuted, g_small);
    else
        para("Installing accepts each tool's licence. The Build Tools are the biggest download and can take 10 "
             "minutes or more. If a tool still shows as missing afterwards, restart the PC and open Zonaite again.",
             kFaint, g_small);
    if (installing) {  // the installer's last words
        const auto lines = g_job.snapshot();
        for (size_t i = lines.size() > 3 ? lines.size() - 3 : 0; i < lines.size(); ++i) {
            std::string l = lines[i];
            l.erase(0, l.find_first_not_of(" -\\|/"));
            if (!l.empty()) para(l.c_str(), kFaint, g_small);
        }
    }
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    end_screen();
}

void load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    const float scale = g_s;
    ImFontConfig fc;
    fc.OversampleH = 2;
    const auto add = [&](const char* file, float size) {
        const fs::path path = g_root / "res" / "fonts" / file;
        if (fs::exists(path)) return io.Fonts->AddFontFromFileTTF(path.string().c_str(), size * scale, &fc);
        ImFontConfig d;
        d.SizePixels = size * scale;
        return io.Fonts->AddFontDefault(&d);
    };
    g_text = add("Lato-Regular.ttf", 19);
    g_bold = add("Lato-Bold.ttf", 15);
    g_small = add("Lato-Regular.ttf", 16);
    g_title = add("Marcellus-Regular.ttf", 52);
    g_head = add("Marcellus-Regular.ttf", 30);
    io.FontDefault = g_text;
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScrollbarSize = S(6);
    style.Colors[ImGuiCol_ScrollbarBg] = {0, 0, 0, 0};
    style.Colors[ImGuiCol_ScrollbarGrab] = {1, 1, 1, 0.12f};
}

}  // namespace
}  // namespace launcher

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    using namespace launcher;
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    g_root = find_root();
    std::string error;
    if (!gfx_open(L"Zonaite", 1180, 720, &error)) {
        MessageBoxW(nullptr, widen(error).c_str(), L"Zonaite", MB_ICONERROR);
        return 1;
    }
    g_s = static_cast<float>(GetDpiForWindow(gfx_window())) / 96.0f;
    load_fonts();
    find_tools();
    load_versions();
    g_screen = any_dump() ? Screen::Library : Screen::Welcome;
    while (gfx_begin()) {
        // a job finished: what it means
        static bool was_running = false;
        if (was_running && !g_job.running) {
            if (g_job_kind == JobKind::AddDump) {
                finish_add_dump();
            } else if (g_job_kind == JobKind::Install) {
                finish_install();  // (it may start the next install)
            } else if (g_job_kind == JobKind::Prepare) {
                load_versions();
                if (g_job.exit_code == 0) {
                    g_screen = Screen::Library;
                    start_game();
                } else {
                    g_notice = "Preparing the game failed: the log above says why.";
                    g_notice_colour = kRed;
                }
            }
            if (!g_job.running) g_job_kind = JobKind::None;
        }
        was_running = g_job.running;
        // the game ended: back to the launcher (with a note if it stopped right away)
        if (g_game && WaitForSingleObject(g_game, 0) == WAIT_OBJECT_0) {
            DWORD code = 0;
            GetExitCodeProcess(g_game, &code);
            CloseHandle(g_game);
            g_game = nullptr;
            if (code != 0 && GetTickCount() - g_game_started < 30000) {
                g_notice = "The game stopped right after starting. zonaite.log in the Zonaite folder says why.";
                g_notice_colour = kRed;
            }
            ShowWindow(gfx_window(), SW_SHOW);
            SetForegroundWindow(gfx_window());
        }
        if (g_game) Sleep(100);  // hidden while the game runs
        switch (g_screen) {
        case Screen::Welcome: screen_welcome(); break;
        case Screen::Guide: screen_guide(); break;
        case Screen::Library: screen_library(); break;
        case Screen::Preparing: screen_preparing(); break;
        case Screen::Tools: screen_tools(); break;
        }
        gfx_end();
    }
    g_job.cancel();
    if (g_job.reader.joinable()) g_job.reader.join();
    gfx_close();
    return 0;
}
