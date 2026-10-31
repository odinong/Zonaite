// Zonaite override of Eden's KTransferMemory (see ../../../README.md): guest memory lent to a service. Services
// run in-process, so they use the guest range directly.
#pragma once
#include "core/hle/kernel/k_process.h"

namespace Kernel {

class KTransferMemory {
public:
    KTransferMemory(KProcessAddress address_, size_t size_) : address{address_}, size{size_} {}
    KProcessAddress GetSourceAddress() const { return address; }
    size_t GetSize() const { return size; }
    KProcess* GetOwner() const { return &KProcess::Application(); }

private:
    KProcessAddress address;
    size_t size;
};

} // namespace Kernel
