// Zonaite override of Eden's service manager (see ../../../../README.md). Eden code looks other services up here
// to reach their internals (SurfaceFlinger asks for nvdrv's device table); each service class it asks for provides
// a static ZonaiteGetService(name) in its own override header.
#pragma once
#include <memory>
#include <string>

namespace Service::SM {

class ServiceManager {
public:
    template <typename T>
    std::shared_ptr<T> GetService(const std::string& name, bool = false) const {
        return T::ZonaiteGetService(name);
    }
};

} // namespace Service::SM
