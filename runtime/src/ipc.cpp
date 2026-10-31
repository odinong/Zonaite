// HIPC/CMIF message parsing and responses, sessions, domains, control commands, and the sm: service manager.
#include <cstdlib>
#include <cstdio>

#include "zonaite/ipc.h"
#include "zonaite/runtime.h"

namespace zn {

using namespace zg;

constexpr uint32_t kSfci = 0x49434653;  // "SFCI": CMIF request
constexpr uint32_t kSfco = 0x4F434653;  // "SFCO": CMIF response
constexpr uint32_t kResultSessionClosed = 0xF601;
constexpr uint32_t kResultUnknownCommand = 0x25A0B;  // sf (module 10) desc 221

uint32_t make_session(std::shared_ptr<Service> s) {
    auto k = std::make_shared<KSession>();
    k->service = std::move(s);
    return kernel_add_handle(k);
}

void Service::dispatch(IpcCtx& ctx) {
    auto it = handlers_.find(ctx.command);
    if (it == handlers_.end())
        fatal(ctx.c, "service %s: unimplemented command %u", name_.c_str(), ctx.command);
    it->second.second(ctx);
}

static Buffer read_buffer_desc(uint8_t* tls, size_t& p) {
    uint32_t w0 = ld32(tls, p), w1 = ld32(tls, p + 4), w2 = ld32(tls, p + 8);
    p += 12;
    Buffer b;
    b.size = w0 | ((uint64_t)((w2 >> 24) & 0xF) << 32);
    b.addr = w1 | ((uint64_t)((w2 >> 28) & 0xF) << 32) | ((uint64_t)((w2 >> 2) & 0x7) << 36);
    return b;
}

static Buffer read_pointer_desc(uint8_t* tls, size_t& p) {
    uint32_t w0 = ld32(tls, p), w1 = ld32(tls, p + 4);
    p += 8;
    Buffer b;
    b.size = w0 >> 16;
    b.addr = w1 | ((uint64_t)((w0 >> 12) & 0xF) << 32) | ((uint64_t)((w0 >> 6) & 0x7) << 36);
    return b;
}

// Writes the response into the TLS buffer: header, handle descriptor, aligned raw data.
static void write_response(uint8_t* tls, IpcCtx& ctx, const std::vector<uint8_t>& raw) {
    bool handles = !ctx.out_copy.empty() || !ctx.out_move.empty();
    size_t p = 8;
    std::memset(tls, 0, 0x100);
    if (handles) {
        st32(tls, p, (uint32_t)(ctx.out_copy.size() << 1) | (uint32_t)(ctx.out_move.size() << 5));
        p += 4;
        for (uint32_t h : ctx.out_copy) { st32(tls, p, h); p += 4; }
        for (uint32_t h : ctx.out_move) { st32(tls, p, h); p += 4; }
    }
    size_t raw_at = (p + 15) & ~15ull;
    uint32_t raw_words = (uint32_t)(4 + (raw.size() + 3) / 4);  // includes the 16-byte alignment allowance
    st32(tls, 0, 0);
    st32(tls, 4, raw_words | (handles ? 0x80000000u : 0));
    if (raw_at + raw.size() > 0x100) fatal(ctx.c, "IPC response too large (%zu bytes)", raw.size());
    std::memcpy(tls + raw_at, raw.data(), raw.size());
}

static void cmif_raw(IpcCtx& ctx, std::vector<uint8_t>& raw) {
    raw.resize(16);
    st32(raw.data(), 0, kSfco);
    st32(raw.data(), 4, 0);
    st32(raw.data(), 8, ctx.result);
    st32(raw.data(), 12, 0);
    raw.insert(raw.end(), ctx.out.begin(), ctx.out.end());
}

uint32_t ipc_send_sync_request(Ctx* c, uint8_t* m, uint32_t handle) {
    auto obj = kernel_get(handle);
    auto* session = dynamic_cast<KSession*>(obj.get());
    if (!session) fatal(c, "SendSyncRequest on handle %#x, which is not a session", handle);
    uint8_t* tls = m + c->tpidrro_el0;
    uint32_t h0 = ld32(tls, 0), h1 = ld32(tls, 4);
    uint32_t type = h0 & 0xFFFF;
    IpcCtx ctx{c, m};
    size_t p = 8;
    if (h1 & 0x80000000u) {  // handle descriptor
        uint32_t hd = ld32(tls, p);
        p += 4;
        if (hd & 1) { ctx.pid = ld64(tls, p); p += 8; }
        for (uint32_t i = 0; i < ((hd >> 1) & 0xF); i++) { ctx.copy_handles.push_back(ld32(tls, p)); p += 4; }
        for (uint32_t i = 0; i < ((hd >> 5) & 0xF); i++) { ctx.move_handles.push_back(ld32(tls, p)); p += 4; }
    }
    for (uint32_t i = 0; i < ((h0 >> 16) & 0xF); i++) ctx.x.push_back(read_pointer_desc(tls, p));
    for (uint32_t i = 0; i < ((h0 >> 20) & 0xF); i++) ctx.a.push_back(read_buffer_desc(tls, p));
    for (uint32_t i = 0; i < ((h0 >> 24) & 0xF); i++) ctx.b.push_back(read_buffer_desc(tls, p));
    for (uint32_t i = 0; i < ((h0 >> 28) & 0xF); i++) ctx.w.push_back(read_buffer_desc(tls, p));
    size_t raw_at = (p + 15) & ~15ull;
    uint32_t raw_words = h1 & 0x3FF;
    size_t raw_end = p + raw_words * 4;
    uint32_t cflags = (h1 >> 10) & 0xF;
    if (cflags >= 2) {  // receive list follows the raw data
        size_t q = raw_end;
        for (uint32_t i = 0; i < (cflags == 2 ? 1u : cflags - 2); i++) {
            uint32_t w0 = ld32(tls, q), w1 = ld32(tls, q + 4);
            q += 8;
            ctx.cbuf.push_back(Buffer{w0 | ((uint64_t)(w1 & 0xFFFF) << 32), w1 >> 16});
        }
    }
    if (type == 2 || type == 15) {  // Close (CMIF) / Close (TIPC)
        kernel_close_handle(handle);
        std::memset(tls, 0, 16);
        return 0;
    }
    if (type >= 16) {  // TIPC: command = type - 16, arguments right after the headers, reply = result + outputs
        ctx.command = type - 16;
        ctx.in = tls + p;
        ctx.in_size = raw_words * 4;
        session->service->dispatch(ctx);
        for (auto& o : ctx.out_objects) ctx.push_move(make_session(o));
        bool handles = !ctx.out_copy.empty() || !ctx.out_move.empty();
        std::vector<uint8_t> raw(4);
        st32(raw.data(), 0, ctx.result);
        raw.insert(raw.end(), ctx.out.begin(), ctx.out.end());
        std::memset(tls, 0, 0x100);
        size_t o = 8;
        if (handles) {
            st32(tls, o, (uint32_t)(ctx.out_copy.size() << 1) | (uint32_t)(ctx.out_move.size() << 5));
            o += 4;
            for (uint32_t h : ctx.out_copy) { st32(tls, o, h); o += 4; }
            for (uint32_t h : ctx.out_move) { st32(tls, o, h); o += 4; }
        }
        st32(tls, 0, type);
        st32(tls, 4, (uint32_t)((raw.size() + 3) / 4) | (handles ? 0x80000000u : 0));
        std::memcpy(tls + o, raw.data(), raw.size());
        return 0;
    }
    if (type != 4 && type != 5 && type != 6 && type != 7)
        fatal(c, "IPC: unsupported message type %u to %s", type, session->service->name().c_str());

    // domain header (requests to a domain session)
    std::shared_ptr<Service> target = session->service;
    size_t q = raw_at;
    if (session->state->domain && (type == 4 || type == 6)) {
        uint8_t dcmd = tls[q];
        uint8_t nobj = tls[q + 1];
        uint16_t payload = ld16(tls, q + 2);
        uint32_t object_id = ld32(tls, q + 4);
        q += 16;
        for (uint32_t i = 0; i < nobj; i++) ctx.in_objects.push_back(ld32(tls, q + payload + 4 * i));
        if (dcmd == 2) {  // close a domain object
            if (object_id && object_id <= session->state->objects.size()) session->state->objects[object_id - 1].reset();
            ctx.is_domain = true;
            std::vector<uint8_t> raw(16, 0);
            std::vector<uint8_t> body;
            cmif_raw(ctx, body);
            raw.insert(raw.end(), body.begin(), body.end());
            write_response(tls, ctx, raw);
            return 0;
        }
        if (!object_id || object_id > session->state->objects.size() || !session->state->objects[object_id - 1])
            fatal(c, "IPC: bad domain object id %u on %s", object_id, session->service->name().c_str());
        target = session->state->objects[object_id - 1];
        ctx.is_domain = true;
    }
    if (ld32(tls, q) != kSfci) {
        for (int row = 0; row < 4; row++) {
            char line[128];
            int n = std::snprintf(line, sizeof line, "  tls+%02x:", row * 16);
            for (int i = 0; i < 4; i++) n += std::snprintf(line + n, sizeof line - n, " %08x", ld32(tls, row * 16 + i * 4));
            log("%s", line);
        }
        fatal(c, "IPC: missing SFCI magic at +%#zx in request to %s (type %u, raw words %u)", q,
              target->name().c_str(), type, raw_words);
    }
    ctx.command = ld32(tls, q + 8);
    ctx.in = tls + q + 16;
    ctx.in_size = raw_end > q + 16 ? raw_end - (q + 16) : 0;

    if (type == 5 || type == 7) {  // control
        log("IPC control %u on handle %#x (%s, domain=%d)", ctx.command, handle, session->service->name().c_str(),
            session->state->domain);
        switch (ctx.command) {
            case 0:  // ConvertCurrentObjectToDomain
                session->state->domain = true;
                session->state->objects.push_back(session->service);
                ctx.push<uint32_t>((uint32_t)session->state->objects.size());
                break;
            case 2:  // CloneCurrentObject
            case 4: {  // CloneCurrentObjectEx: a new session onto the same object, sharing the domain
                auto k = std::make_shared<KSession>();
                k->service = session->service;
                k->state = session->state;
                ctx.push_move(kernel_add_handle(k));
                break;
            }
            case 3:  // QueryPointerBufferSize
                ctx.push<uint16_t>(0x8000);
                break;
            default:
                fatal(c, "IPC: unsupported control command %u", ctx.command);
        }
    } else {
        target->dispatch(ctx);
    }

    // output objects: domain ids on a domain session, new sessions (move handles) otherwise
    std::vector<uint32_t> out_ids;
    for (auto& o : ctx.out_objects) {
        if (session->state->domain) {
            session->state->objects.push_back(o);
            out_ids.push_back((uint32_t)session->state->objects.size());
        } else {
            ctx.push_move(make_session(o));
        }
    }
    std::vector<uint8_t> body;
    cmif_raw(ctx, body);
    std::vector<uint8_t> raw;
    if (ctx.is_domain) {  // domain response header: output object count, then the payload, then the ids
        raw.resize(16, 0);
        st32(raw.data(), 0, (uint32_t)out_ids.size());
        raw.insert(raw.end(), body.begin(), body.end());
        for (uint32_t id : out_ids) {
            size_t at = raw.size();
            raw.resize(at + 4);
            st32(raw.data(), at, id);
        }
    } else {
        raw = std::move(body);
    }
    write_response(tls, ctx, raw);
    return 0;
}

// ---- sm: (service manager)
class ServiceManager : public Service {
public:
    ServiceManager() : Service("sm:") {
        on(0, "RegisterClient", [](IpcCtx&) {});
        on(1, "GetServiceHandle", [](IpcCtx& ctx) {
            uint64_t raw = ctx.pop<uint64_t>();
            char name[9] = {};
            std::memcpy(name, &raw, 8);
            auto s = open_service(name);
            if (!s) fatal(ctx.c, "sm: GetServiceHandle(\"%s\"): service not implemented", name);
            log("sm: opened %s", name);
            ctx.push_move(make_session(s));
        });
    }
};

uint32_t ipc_connect_to_named_port(uint8_t* m, uint64_t name_ptr, uint32_t* out_handle) {
    char name[16] = {};
    for (int i = 0; i < 12 && m[name_ptr + i]; i++) name[i] = (char)m[name_ptr + i];
    std::string n = name;
    if (n != "sm:") {
        log("ConnectToNamedPort(\"%s\"): unknown port", name);
        return kResultNotFound;
    }
    *out_handle = make_session(std::make_shared<ServiceManager>());
    return 0;
}

}  // namespace zn
