// nvdrv: the GPU driver service. Eden's nvdrv devices (core/hle/service/nvdrv/devices, compiled unmodified against
// the header overrides in runtime/gpu/override) do the work; this file is the glue: the kernel pieces they use
// (events, the process), the device table (Eden's Module, declared in nvdrv.h) and the IPC front end the game talks
// to (Eden's NVDRV interface, reimplemented on Zonaite's IPC).
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/core.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/nvdrv/core/container.h"
#include "core/hle/service/nvdrv/devices/nvdevice.h"
#include "core/hle/service/nvdrv/devices/nvdisp_disp0.h"
#include "core/hle/service/nvdrv/devices/nvhost_as_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_ctrl.h"
#include "core/hle/service/nvdrv/devices/nvhost_ctrl_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_gpu.h"
#include "core/hle/service/nvdrv/devices/nvhost_nvdec.h"
#include "core/hle/service/nvdrv/devices/nvhost_nvjpg.h"
#include "core/hle/service/nvdrv/devices/nvhost_vic.h"
#include "core/hle/service/nvdrv/devices/nvmap.h"
#include "core/hle/service/nvdrv/nvdrv.h"
#include "core/hle/service/nvdrv/nvdrv_interface.h"
#include "core/hle/service/sm/sm.h"
#include "core/memory.h"

#include "zonaite/gpu.h"
#include "zonaite/ipc.h"
#include "zonaite/kernel.h"
#include "zonaite/runtime.h"
#include "zonaite/services.h"

// ---------------------------------------------------------------- kernel pieces (header overrides)
namespace Kernel {

class KernelCore {};  // nothing in nvdrv reads it; Signal/Clear take it for Eden's signature only

KEvent::KEvent(std::string name_) : name{std::move(name_)}, impl{std::make_shared<zn::KEvent>()} {}
KEvent::~KEvent() = default;

Result KEvent::Signal() {
    zn::kernel_signal(static_cast<zn::KEvent*>(impl.get()));
    return ResultSuccess;
}

Result KEvent::Clear() {
    zn::kernel_clear(static_cast<zn::KEvent*>(impl.get()));
    return ResultSuccess;
}

u32 KEvent::CreateHandle() {
    return zn::kernel_add_handle(std::static_pointer_cast<zn::KEvent>(impl));
}

u32 KReadableEvent::CreateHandle() { return parent.CreateHandle(); }

void KEvent::MarkGpuFence() { static_cast<zn::KEvent*>(impl.get())->gpu_fence = true; }

Core::Memory::Memory& KProcess::GetMemory() const { return zn::gpu::system().ApplicationMemory(); }
u64 KProcess::GetProgramId() const { return 0x0100F2C0115B6000ull; }

KProcess& KProcess::Application() {
    static KProcess process;
    return process;
}

}  // namespace Kernel

namespace Core {
Kernel::KernelCore& System::Kernel() {
    static Kernel::KernelCore kernel;
    return kernel;
}

Service::SM::ServiceManager& System::ServiceManager() {
    static Service::SM::ServiceManager sm;
    return sm;
}
}  // namespace Core

namespace Service::KernelHelpers {

ServiceContext::ServiceContext(Core::System& system, std::string) : kernel{system.Kernel()} {}
ServiceContext::~ServiceContext() = default;

Kernel::KEvent* ServiceContext::CreateEvent(std::string&& name) {
    auto* e = new Kernel::KEvent(std::move(name));
    if (e->GetName().starts_with("NVCTRL::NvEvent")) e->MarkGpuFence();  // syncpoint waits (see zn::KEvent)
    return e;
}
void ServiceContext::CloseEvent(Kernel::KEvent* event) { delete event; }

}  // namespace Service::KernelHelpers

// ---------------------------------------------------------------- device table (Eden's Module)
namespace Service::Nvidia {

EventInterface::EventInterface(Module& module_) : module{module_} {}
EventInterface::~EventInterface() = default;

Kernel::KEvent* EventInterface::CreateEvent(std::string name) {
    return module.service_context.CreateEvent(std::move(name));
}

void EventInterface::FreeEvent(Kernel::KEvent* event) { module.service_context.CloseEvent(event); }

Module::Module(Core::System& system)
    : container{system.Host1x()}, service_context{system, "nvdrv"}, events_interface{*this} {
    auto& ev = events_interface;
    builders["/dev/nvhost-as-gpu"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_as_gpu>(system, *this, container)).first;
    };
    builders["/dev/nvhost-gpu"] = [this, &system, &ev](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_gpu>(system, ev, container)).first;
    };
    builders["/dev/nvhost-ctrl-gpu"] = [this, &system, &ev](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_ctrl_gpu>(system, ev)).first;
    };
    builders["/dev/nvmap"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvmap>(system, container)).first;
    };
    builders["/dev/nvdisp_disp0"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvdisp_disp0>(system, container)).first;
    };
    builders["/dev/nvhost-ctrl"] = [this, &system, &ev](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_ctrl>(system, ev, container)).first;
    };
    builders["/dev/nvhost-nvdec"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_nvdec>(system, container)).first;
    };
    builders["/dev/nvhost-nvjpg"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_nvjpg>(system)).first;
    };
    builders["/dev/nvhost-vic"] = [this, &system](DeviceFD fd) {
        return open_files.emplace(fd, std::make_shared<Devices::nvhost_vic>(system, container)).first;
    };
}

Module::~Module() = default;

NvResult Module::VerifyFD(DeviceFD fd) const {
    if (fd < 0) return NvResult::InvalidState;
    return open_files.contains(fd) ? NvResult::Success : NvResult::NotImplemented;
}

DeviceFD Module::Open(const std::string& device_name, NvCore::SessionId session_id) {
    auto it = builders.find(device_name);
    if (it == builders.end()) {
        zn::log("nvdrv: unknown device %s", device_name.c_str());
        return INVALID_NVDRV_FD;
    }
    const DeviceFD fd = next_fd++;
    it->second(fd)->second->OnOpen(session_id, fd);
    return fd;
}

NvResult Module::Ioctl1(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output) {
    if (NvResult r = VerifyFD(fd); r != NvResult::Success) return r;
    return open_files.at(fd)->Ioctl1(fd, command, input, output);
}

NvResult Module::Ioctl2(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<const u8> inline_input,
                        std::span<u8> output) {
    if (NvResult r = VerifyFD(fd); r != NvResult::Success) return r;
    return open_files.at(fd)->Ioctl2(fd, command, input, inline_input, output);
}

NvResult Module::Ioctl3(DeviceFD fd, Ioctl command, std::span<const u8> input, std::span<u8> output,
                        std::span<u8> inline_output) {
    if (NvResult r = VerifyFD(fd); r != NvResult::Success) return r;
    return open_files.at(fd)->Ioctl3(fd, command, input, output, inline_output);
}

NvResult Module::Close(DeviceFD fd) {
    if (NvResult r = VerifyFD(fd); r != NvResult::Success) return r;
    auto it = open_files.find(fd);
    it->second->OnClose(fd);
    open_files.erase(it);
    return NvResult::Success;
}

NvResult Module::QueryEvent(DeviceFD fd, u32 event_id, Kernel::KEvent*& event) {
    if (NvResult r = VerifyFD(fd); r != NvResult::Success) return r;
    event = open_files.at(fd)->QueryEvent(event_id);
    return event ? NvResult::Success : NvResult::BadParameter;
}

}  // namespace Service::Nvidia

// ---------------------------------------------------------------- IPC front end
namespace zn {

namespace {

namespace Nv = ::Service::Nvidia;

// One driver for the process, shared by every nvdrv session. Eden's devices aren't thread-safe across calls,
// and guest threads call in concurrently, so calls are serialized. (Ioctls never block: waits go through events.)
struct Driver {
    std::mutex lock;
    std::shared_ptr<Nv::Module> module;
    Nv::NvCore::SessionId session{};
    bool initialized = false;
};

Driver& driver() {
    static Driver d;
    return d;
}

std::shared_ptr<Nv::Module> module() {
    auto& d = driver();
    std::lock_guard l(d.lock);
    if (!d.module) d.module = std::make_shared<Nv::Module>(gpu::system());
    return d.module;
}

std::vector<u8> read_guest(IpcCtx& ctx, const Buffer& b) {
    std::vector<u8> v(b.size);
    if (b.size) std::memcpy(v.data(), ctx.m + b.addr, b.size);
    return v;
}

void write_guest(IpcCtx& ctx, const Buffer& b, const std::vector<u8>& v) {
    if (b.size) std::memcpy(ctx.m + b.addr, v.data(), std::min<size_t>(b.size, v.size()));
}

void push_result(IpcCtx& ctx, Nv::NvResult r) { ctx.push<u32>(static_cast<u32>(r)); }

// Failed ioctls are rare and tell a lot (the driver in the game carries on with whatever it got): log them
void note_ioctl(s32 fd, Nv::Ioctl command, Nv::NvResult r, IpcCtx* ctx = nullptr) {
    if (r != Nv::NvResult::Success)
        log("nvdrv: ioctl %#010x on fd %d failed: %#x", (unsigned)command.raw, fd, (unsigned)r);
}

class Nvdrv : public Service {
public:
    explicit Nvdrv(const char* name) : Service(name) {
        on(0, "Open", [](IpcCtx& ctx) {
            auto& d = driver();
            std::lock_guard l(d.lock);
            const auto raw = read_guest(ctx, ctx.in_buffer());
            const std::string dev(raw.begin(), std::find(raw.begin(), raw.end(), u8{0}));
            if (!d.initialized) {
                ctx.push<s32>(0);
                return push_result(ctx, Nv::NvResult::NotInitialized);
            }
            if (dev == "/dev/nvhost-prof-gpu") {  // not openable on retail units
                ctx.push<s32>(0);
                return push_result(ctx, Nv::NvResult::NotSupported);
            }
            const Nv::DeviceFD fd = d.module->Open(dev, d.session);
            ctx.push<s32>(fd);
            push_result(ctx, fd != Nv::INVALID_NVDRV_FD ? Nv::NvResult::Success : Nv::NvResult::FileOperationFailed);
        });
        on(1, "Ioctl", [](IpcCtx& ctx) {
            const auto fd = ctx.pop<s32>();
            const auto command = ctx.pop<Nv::Ioctl>();
            const Buffer out_b = ctx.out_buffer();
            const auto in = read_guest(ctx, ctx.in_buffer());
            std::vector<u8> out(out_b.size);
            auto& d = driver();
            std::lock_guard l(d.lock);
            const auto r = d.module->Ioctl1(fd, command, in, out);
            if (r == Nv::NvResult::Success) write_guest(ctx, out_b, out);
            note_ioctl(fd, command, r, &ctx);
            push_result(ctx, r);
        });
        on(2, "Close", [](IpcCtx& ctx) {
            const auto fd = ctx.pop<s32>();
            auto& d = driver();
            std::lock_guard l(d.lock);
            push_result(ctx, d.module->Close(fd));
        });
        on(3, "Initialize", [](IpcCtx& ctx) {
            // The transfer memory is the driver's work buffer on hardware; the HLE driver doesn't need it.
            auto& d = driver();
            std::lock_guard l(d.lock);
            if (!d.initialized) {
                d.session = d.module->GetContainer().OpenSession(&Kernel::KProcess::Application());
                d.initialized = true;
            }
            push_result(ctx, Nv::NvResult::Success);
        });
        on(4, "QueryEvent", [](IpcCtx& ctx) {
            const auto fd = ctx.pop<s32>();
            const auto event_id = ctx.pop<u32>();
            auto& d = driver();
            std::lock_guard l(d.lock);
            Kernel::KEvent* event = nullptr;
            const auto r = d.module->QueryEvent(fd, event_id, event);
            if (r == Nv::NvResult::Success) ctx.push_copy(event->CreateHandle());
            push_result(ctx, r);
        });
        on(6, "GetStatus", [](IpcCtx& ctx) { push_result(ctx, Nv::NvResult::Success); });
        on(8, "SetAruid", [](IpcCtx& ctx) { push_result(ctx, Nv::NvResult::Success); });
        on(9, "DumpGraphicsMemoryInfo", [](IpcCtx&) {});
        on(11, "Ioctl2", [](IpcCtx& ctx) {
            const auto fd = ctx.pop<s32>();
            const auto command = ctx.pop<Nv::Ioctl>();
            const Buffer out_b = ctx.out_buffer();
            const auto in = read_guest(ctx, ctx.in_buffer(0));
            const auto inline_in = read_guest(ctx, ctx.in_buffer(1));
            std::vector<u8> out(out_b.size);
            auto& d = driver();
            std::lock_guard l(d.lock);
            const auto r = d.module->Ioctl2(fd, command, in, inline_in, out);
            if (r == Nv::NvResult::Success) write_guest(ctx, out_b, out);
            note_ioctl(fd, command, r, &ctx);
            push_result(ctx, r);
        });
        on(12, "Ioctl3", [](IpcCtx& ctx) {
            const auto fd = ctx.pop<s32>();
            const auto command = ctx.pop<Nv::Ioctl>();
            const Buffer out_b = ctx.out_buffer(0), inline_b = ctx.out_buffer(1);
            const auto in = read_guest(ctx, ctx.in_buffer());
            std::vector<u8> out(out_b.size), inline_out(inline_b.size);
            auto& d = driver();
            std::lock_guard l(d.lock);
            const auto r = d.module->Ioctl3(fd, command, in, out, inline_out);
            note_ioctl(fd, command, r, &ctx);
            if (r == Nv::NvResult::Success) {
                write_guest(ctx, out_b, out);
                write_guest(ctx, inline_b, inline_out);
            }
            push_result(ctx, r);
        });
        on(13, "SetGraphicsFirmwareMemoryMarginEnabled", [](IpcCtx&) {});
    }
};

std::shared_ptr<Service> make_nvdrv(const char* name) {
    module();
    return std::make_shared<Nvdrv>(name);
}

ServiceRegistrar reg("nvdrv", [] { return make_nvdrv("nvdrv"); });
ServiceRegistrar reg_a("nvdrv:a", [] { return make_nvdrv("nvdrv:a"); });
ServiceRegistrar reg_s("nvdrv:s", [] { return make_nvdrv("nvdrv:s"); });
ServiceRegistrar reg_t("nvdrv:t", [] { return make_nvdrv("nvdrv:t"); });

}  // namespace

}  // namespace zn

namespace Service::Nvidia {
std::shared_ptr<NVDRV> NVDRV::ZonaiteGetService(const std::string&) { return std::make_shared<NVDRV>(); }
std::shared_ptr<Module> NVDRV::GetModule() const { return zn::module(); }
}  // namespace Service::Nvidia
