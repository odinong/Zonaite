// The VulkanMemoryAllocator implementation unit (Eden gets it from its VMA package; we compile it here).
// Same configuration as video_core/vulkan_common/vma.h: Vulkan functions are loaded at run time.
#include "video_core/vulkan_common/vma.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
