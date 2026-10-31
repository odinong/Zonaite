// Horizon IPC (HIPC framing + CMIF payload) server side: services implemented in C++ answer requests that guest
// code sends with svc SendSyncRequest. Message layout: switchbrew "IPC Marshalling".
#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "zonaite/guest.h"
#include "zonaite/kernel.h"

namespace zn {

struct Buffer {
    uint64_t addr = 0, size = 0;
};

class Service;

// One request being handled: parsed inputs, and the response being built.
struct IpcCtx {
    zg::Ctx* c;
    uint8_t* m;
    uint32_t command = 0;
    uint64_t pid = 0;
    std::vector<uint32_t> copy_handles, move_handles;
    std::vector<Buffer> x, a, b, w, cbuf;  // pointer (X), send (A), receive (B), exchange (W), receive list (C)
    std::vector<uint32_t> in_objects;       // domain input object ids
    const uint8_t* in = nullptr;           // command arguments
    size_t in_size = 0, in_pos = 0;
    // response
    uint32_t result = 0;
    std::vector<uint8_t> out;
    std::vector<uint32_t> out_copy, out_move;
    std::vector<std::shared_ptr<Service>> out_objects;
    bool is_domain = false;

    template <typename T> T pop() {
        in_pos = (in_pos + alignof(T) - 1) & ~(alignof(T) - 1);
        T v{};
        if (in_pos + sizeof(T) <= in_size) std::memcpy(&v, in + in_pos, sizeof(T));
        in_pos += sizeof(T);
        return v;
    }
    template <typename T> void push(const T& v) {
        size_t at = (out.size() + alignof(T) - 1) & ~(alignof(T) - 1);
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &v, sizeof(T));
    }
    void push_copy(uint32_t h) { out_copy.push_back(h); }
    void push_move(uint32_t h) { out_move.push_back(h); }
    void push_object(std::shared_ptr<Service> s) { out_objects.push_back(std::move(s)); }
    // first buffer of a kind: A or X for input, B or C for output (the SDK picks either "auto" form)
    Buffer in_buffer(size_t i = 0) const { return i < a.size() && a[i].addr ? a[i] : (i < x.size() ? x[i] : Buffer{}); }
    Buffer out_buffer(size_t i = 0) const {
        return i < b.size() && b[i].addr ? b[i] : (i < cbuf.size() ? cbuf[i] : Buffer{});
    }
};

using Handler = std::function<void(IpcCtx&)>;

class Service : public std::enable_shared_from_this<Service> {
public:
    explicit Service(std::string name) : name_(std::move(name)) {}
    virtual ~Service() = default;
    const std::string& name() const { return name_; }
    virtual void dispatch(IpcCtx& ctx);  // looks up handlers_; unknown commands are fatal with the name

protected:
    void on(uint32_t cmd, const char* what, Handler h) { handlers_[cmd] = {what, std::move(h)}; }
    std::map<uint32_t, std::pair<const char*, Handler>> handlers_;
    std::string name_;
};

// Domain state, shared by a session and its clones (CloneCurrentObject keeps the domain)
struct DomainState {
    bool domain = false;
    std::vector<std::shared_ptr<Service>> objects;  // domain object ids are index + 1
};

// Kernel object behind a session handle
struct KSession : KObject {
    std::shared_ptr<Service> service;
    std::shared_ptr<DomainState> state = std::make_shared<DomainState>();
};

uint32_t make_session(std::shared_ptr<Service> s);
uint32_t ipc_send_sync_request(zg::Ctx* c, uint8_t* m, uint32_t handle);
uint32_t ipc_connect_to_named_port(uint8_t* m, uint64_t name_ptr, uint32_t* out_handle);

// Registry of named services handed out by sm:
std::shared_ptr<Service> open_service(const std::string& name);

}  // namespace zn
