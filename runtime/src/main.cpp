// Zonaite runtime entry: load the guest image, set up the main thread and start the game the way rtld does.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "zonaite/guest.h"
#include "zonaite/gpu.h"
#include "zonaite/kernel.h"
#include "zonaite/memory.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;

ImageInfo g_image;

static std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "zonaite: cannot open %s\n", path.c_str()); std::exit(1); }
    std::string d;
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) d.append(buf, n);
    std::fclose(f);
    return d;
}

// Minimal reader for the few fields of image.json the runtime needs (written by tools/build_game.py).
static uint64_t json_u64(const std::string& j, const std::string& key, size_t from = 0) {
    size_t p = j.find("\"" + key + "\":", from);
    if (p == std::string::npos) { std::fprintf(stderr, "zonaite: image.json lacks %s\n", key.c_str()); std::exit(1); }
    return std::strtoull(j.c_str() + p + key.size() + 3, nullptr, 10);
}

static std::string g_game_version = "1.2.1";  // builds from before image.json named it were 1.2.1

const std::string& game_version() { return g_game_version; }

static void load_image(const std::string& dir) {
    std::string j = read_file(dir + "/image.json");
    if (size_t v = j.find("\"version\": \""); v != std::string::npos) {
        v += 12;
        g_game_version = j.substr(v, j.find('"', v) - v);
    }
    g_image.base = json_u64(j, "base");
    g_image.size = json_u64(j, "size");
    g_image.entry = json_u64(j, "entry");
    size_t p = 0;
    while ((p = j.find("\"name\":", p)) != std::string::npos && g_image.module_count < 8) {
        ModuleInfo& mi = g_image.modules[g_image.module_count++];
        size_t open = j.find('"', p + 7) + 1;  // skip `"name":` and any space before the value's quote
        size_t q = j.find('"', open);
        mi.name = j.substr(open, q - open);
        mi.base = json_u64(j, "base", p);
        mi.size = json_u64(j, "size", p);
        size_t ip = j.find("\"init\":", p);
        mi.init = std::strtoull(j.c_str() + ip + 7, nullptr, 10);
        size_t fp = j.find("\"fini\":", p);
        mi.fini = std::strtoull(j.c_str() + fp + 7, nullptr, 10);
        p = q;
    }
    std::string img = read_file(dir + "/image.bin");
    if (img.size() != g_image.size) { std::fprintf(stderr, "zonaite: image.bin size mismatch\n"); std::exit(1); }
    commit(g_image.base, g_image.size);
    std::memcpy(g_mem + g_image.base, img.data(), img.size());
    log("loaded %zu MB image at %#llx, %d modules, nn::init::Start at %#llx", img.size() >> 20,
        (unsigned long long)g_image.base, g_image.module_count, (unsigned long long)g_image.entry);
}

// rtld's three callbacks to nn::init::Start
static void rtld_set_flag(Ctx*, uint8_t*) {}
static void rtld_run_inits(Ctx* c, uint8_t* m) {
    for (int i = 0; i < g_image.module_count; i++)
        if (g_image.modules[i].init) {
            log("init %s (%#llx)", g_image.modules[i].name.c_str(), (unsigned long long)g_image.modules[i].init);
            guest_call(c, m, g_image.modules[i].init);
        }
}
static void rtld_run_finis(Ctx* c, uint8_t* m) {
    for (int i = 0; i < g_image.module_count; i++)
        if (g_image.modules[i].fini) guest_call(c, m, g_image.modules[i].fini);
}

}  // namespace zn

int main(int argc, char** argv) {
    using namespace zn;
    std::string dir = argc > 1 ? argv[1] : "build/game";
    g_build_dir = dir;
    g_dump_dir = argc > 2 ? argv[2] : "";
    install_crash_handler();
    reserve_guest_space();
    load_image(dir);
    init_memory_map();
    Ctx* c = create_main_thread_context();
    kernel_init_main_thread(c);
    std::string gpu_error;
    if (!gpu::start(&gpu_error)) fatal(nullptr, "GPU: %s", gpu_error.c_str());
    log("GPU: Vulkan renderer started");
    uint64_t flag = register_host_function(rtld_set_flag);
    uint64_t inits = register_host_function(rtld_run_inits);
    uint64_t finis = register_host_function(rtld_run_finis);
    c->x[0] = kMainThreadHandle;
    c->x[1] = g_image.modules[0].base;  // main module base (rtld passes the address right after itself)
    c->x[2] = flag;
    c->x[3] = inits;
    c->x[4] = finis;
    kernel_core_enter();
    log("calling nn::init::Start");
    guest_call(c, g_mem, g_image.entry);
    log("nn::init::Start returned");
    return 0;
}
