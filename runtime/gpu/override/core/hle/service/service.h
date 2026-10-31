// Zonaite override of Eden's service framework header (see ../../../README.md). nvdrv's headers include it but
// only nvdrv_interface (not compiled; Zonaite has its own IPC front end) uses the framework.
#pragma once
#include <cstddef>
#include <mutex>
#include "common/bit_util.h"  // Eden's service.h brings this in (nvhost_as_gpu uses Common::Log2Ceil)
#include "common/common_types.h"
