// Registry of HLE services handed out by sm:. Each service lives in its own file and registers here.
#include <functional>
#include <map>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"

namespace zn {

static std::map<std::string, std::function<std::shared_ptr<Service>()>>& registry() {
    static std::map<std::string, std::function<std::shared_ptr<Service>()>> r;
    return r;
}

void register_service(const std::string& name, std::function<std::shared_ptr<Service>()> make) {
    registry()[name] = std::move(make);
}

std::shared_ptr<Service> open_service(const std::string& name) {
    auto it = registry().find(name);
    return it == registry().end() ? nullptr : it->second();
}

}  // namespace zn
