// Zonaite override of Eden's NVDRV IPC interface (see ../../../../README.md). Zonaite's front end is
// runtime/gpu/nvdrv.cpp; this class only gives Eden code (SurfaceFlinger) the shared device table.
#pragma once
#include <memory>
#include <string>

#include "core/hle/service/nvdrv/nvdrv.h"  // Eden's version includes it; SurfaceFlinger relies on that

namespace Service::Nvidia {

class NVDRV {
public:
    static std::shared_ptr<NVDRV> ZonaiteGetService(const std::string& name);
    std::shared_ptr<Module> GetModule() const;
};

} // namespace Service::Nvidia
