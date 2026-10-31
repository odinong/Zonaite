// Zonaite override of Eden's KEvent (see ../../../README.md): an event backed by a Zonaite kernel event, so the
// guest can wait on it through a handle (runtime/gpu/kernel_shim.cpp).
#pragma once

#include <memory>
#include <string>

#include "core/hle/result.h"

namespace Kernel {

class KernelCore;
class KEvent;

// The waitable side of a KEvent (what services hand out as copy handles).
class KReadableEvent {
public:
    explicit KReadableEvent(KEvent& parent_) : parent{parent_} {}
    KEvent& GetParent() { return parent; }
    u32 CreateHandle();

private:
    KEvent& parent;
};

class KEvent {
public:
    explicit KEvent(std::string name);
    ~KEvent();

    Result Signal(KernelCore&) { return Signal(); }
    Result Clear(KernelCore&) { return Clear(); }
    Result Signal();
    Result Clear();

    // A new guest handle to this event (for QueryEvent and the like).
    u32 CreateHandle();
    // A GPU syncpoint event: guest waits on it with a timeout last until the GPU gets there (zn::KEvent::gpu_fence)
    void MarkGpuFence();

    const std::string& GetName() const { return name; }
    KReadableEvent& GetReadableEvent() { return readable; }

private:
    std::string name;
    KReadableEvent readable{*this};
    std::shared_ptr<void> impl;  // zn::KEvent
};

} // namespace Kernel
