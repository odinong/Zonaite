// Service registration: each service file declares `ServiceRegistrar reg("name", factory);`.
#pragma once
#include <functional>
#include <memory>
#include <string>

#include "zonaite/ipc.h"

namespace zn {

void register_service(const std::string& name, std::function<std::shared_ptr<Service>()> make);

struct ServiceRegistrar {
    ServiceRegistrar(const std::string& name, std::function<std::shared_ptr<Service>()> make) {
        register_service(name, std::move(make));
    }
};

// fs: creates the game's cache storage (am CreateCacheStorage)
void fs_create_cache_storage(int64_t size, int64_t journal_size);

// hid: input.cpp writes controller state into hid's shared memory; it starts with the memory (svc_hid.cpp) and
// signals the style-set event when it connects a controller.
void input_start(uint8_t* hid_shared_memory);
// The game's vibration for one side of the controller (0 left, 1 right): amplitudes of its two bands, 0 .. 1
void input_vibrate(int side, float amp_low, float amp_high);
void hid_signal_style_set_update();

// acc: the uid of the one local user (svc_acc.cpp); am hands it to the game as the preselected user.
void account_current_user(uint64_t uid[2]);

}  // namespace zn
