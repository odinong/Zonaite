// Guest CPU state and helpers used by the generated code. Included by every generated translation unit, so keep
// it header-only and cheap to parse.
#pragma once
#include <atomic>
#include <cmath>
#if defined(_WIN32)
#include <intrin.h>
#endif
#include <cstdint>
#include <cstring>

namespace zg {

struct alignas(16) V {
    uint64_t lo, hi;
};

struct Ctx {
    uint64_t x[32];  // x[31] is never read as a register (ZR/SP are decided per operand); kept for indexing
    uint64_t sp;
    uint64_t pc;     // test mode: next PC after the instruction; otherwise the last PC recorded for diagnostics
    V v[32];
    uint8_t n, z, c, vf;  // NZCV as separate bytes
    uint32_t fpcr, fpsr;
    uint64_t tpidr_el0, tpidrro_el0;
    uint64_t excl_addr;   // exclusive monitor: address and value(s) seen by the last load-exclusive
    uint64_t excl_val[2];
    uint32_t excl_size;   // bytes per element, 0 = no reservation
    uint32_t excl_pair;
    uint64_t excl_gen;    // the monitor generation of the reserved granule when it was loaded
};

using GuestFn = void (*)(Ctx*, uint8_t*);

// Debug probes on hooked functions (config/hooks.txt, kind "probe"): log arguments on entry, x0 on return.

// Provided by the runtime.
void call_indirect(Ctx* c, uint8_t* m, uint64_t target);
void dispatch_indirect(Ctx* c, uint8_t* m);  // tail-call target: jumps to c->pc (same signature as guest functions)
void jt_miss(Ctx* c, uint8_t* m, uint64_t pc, uint64_t target);
void guest_trap(Ctx* c, uint8_t* m, uint64_t pc, uint32_t word);
void guest_svc(Ctx* c, uint8_t* m, uint32_t imm);
void guest_brk(Ctx* c, uint8_t* m, uint64_t pc, uint32_t imm);
uint64_t sysreg_read(Ctx* c, uint32_t enc);
void sysreg_write(Ctx* c, uint32_t enc, uint64_t value);
void sys_op(Ctx* c, uint8_t* m, uint32_t enc, uint64_t value);  // DC/IC/TLBI etc.

// ---- memory (little-endian guest on a little-endian host: plain unaligned copies)
template <typename T> static inline T ld(uint8_t* m, uint64_t a) { T v; std::memcpy(&v, m + a, sizeof(T)); return v; }
template <typename T> static inline void st(uint8_t* m, uint64_t a, T v) { std::memcpy(m + a, &v, sizeof(T)); }
static inline uint8_t ld8(uint8_t* m, uint64_t a) { return m[a]; }
static inline uint16_t ld16(uint8_t* m, uint64_t a) { return ld<uint16_t>(m, a); }
static inline uint32_t ld32(uint8_t* m, uint64_t a) { return ld<uint32_t>(m, a); }
static inline uint64_t ld64(uint8_t* m, uint64_t a) { return ld<uint64_t>(m, a); }
static inline V ld128(uint8_t* m, uint64_t a) { return ld<V>(m, a); }
static inline void st8(uint8_t* m, uint64_t a, uint8_t v) { m[a] = v; }
static inline void st16(uint8_t* m, uint64_t a, uint16_t v) { st<uint16_t>(m, a, v); }
static inline void st32(uint8_t* m, uint64_t a, uint32_t v) { st<uint32_t>(m, a, v); }
static inline void st64(uint8_t* m, uint64_t a, uint64_t v) { st<uint64_t>(m, a, v); }
static inline void st128(uint8_t* m, uint64_t a, V v) { st<V>(m, a, v); }

// ---- flags
static inline uint64_t add64f(Ctx* c, uint64_t a, uint64_t b, uint64_t cin) {
    unsigned __int128 u = (unsigned __int128)a + b + cin;
    uint64_t r = (uint64_t)u;
    __int128 s = (__int128)(int64_t)a + (int64_t)b + (int64_t)cin;
    c->n = (uint8_t)(r >> 63); c->z = r == 0; c->c = (uint8_t)(u >> 64); c->vf = s != (__int128)(int64_t)r;
    return r;
}
static inline uint32_t add32f(Ctx* c, uint32_t a, uint32_t b, uint32_t cin) {
    uint64_t u = (uint64_t)a + b + cin;
    uint32_t r = (uint32_t)u;
    int64_t s = (int64_t)(int32_t)a + (int32_t)b + (int64_t)cin;
    c->n = (uint8_t)(r >> 31); c->z = r == 0; c->c = (uint8_t)(u >> 32); c->vf = s != (int64_t)(int32_t)r;
    return r;
}
static inline uint64_t adc64(uint64_t a, uint64_t b, uint64_t cin) { return a + b + cin; }
static inline void logic64f(Ctx* c, uint64_t r) { c->n = (uint8_t)(r >> 63); c->z = r == 0; c->c = 0; c->vf = 0; }
static inline void logic32f(Ctx* c, uint32_t r) { c->n = (uint8_t)(r >> 31); c->z = r == 0; c->c = 0; c->vf = 0; }
static inline void set_nzcv(Ctx* c, uint32_t nzcv) {
    c->n = (nzcv >> 3) & 1; c->z = (nzcv >> 2) & 1; c->c = (nzcv >> 1) & 1; c->vf = nzcv & 1;
}
static inline uint64_t get_nzcv(const Ctx* c) {
    return ((uint64_t)c->n << 31) | ((uint64_t)c->z << 30) | ((uint64_t)c->c << 29) | ((uint64_t)c->vf << 28);
}
static inline bool cond(const Ctx* c, int code) {
    bool r;
    switch (code >> 1) {
        case 0: r = c->z; break;                          // EQ / NE
        case 1: r = c->c; break;                          // CS / CC
        case 2: r = c->n; break;                          // MI / PL
        case 3: r = c->vf; break;                         // VS / VC
        case 4: r = c->c && !c->z; break;                 // HI / LS
        case 5: r = c->n == c->vf; break;                 // GE / LT
        case 6: r = c->n == c->vf && !c->z; break;        // GT / LE
        default: return true;                             // AL / NV (both always)
    }
    return (code & 1) ? !r : r;
}

// ---- bit operations
static inline uint64_t ror64(uint64_t v, unsigned s) { s &= 63; return s ? (v >> s) | (v << (64 - s)) : v; }
static inline uint32_t ror32(uint32_t v, unsigned s) { s &= 31; return s ? (v >> s) | (v << (32 - s)) : v; }
static inline uint64_t asr64(uint64_t v, unsigned s) { return (uint64_t)((int64_t)v >> (s & 63)); }
static inline uint32_t asr32(uint32_t v, unsigned s) { return (uint32_t)((int32_t)v >> (s & 31)); }
static inline uint64_t clz64(uint64_t v) { return v ? __builtin_clzll(v) : 64; }
static inline uint32_t clz32(uint32_t v) { return v ? __builtin_clz(v) : 32; }
// CLS: leading bits equal to the sign bit, excluding it = CLZ over 63 (31) bits of x[63:1] ^ x[62:0]
static inline uint64_t cls64(uint64_t v) { return clz64((v >> 1) ^ (v & 0x7FFFFFFFFFFFFFFFull)) - 1; }
static inline uint32_t cls32(uint32_t v) { return clz32((v >> 1) ^ (v & 0x7FFFFFFFu)) - 1; }
static inline uint64_t rbit64(uint64_t v) {
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    return __builtin_bswap64(v);
}
static inline uint32_t rbit32(uint32_t v) { return (uint32_t)(rbit64(v) >> 32); }
static inline uint64_t sdiv64(uint64_t a, uint64_t b) {
    if (b == 0) return 0;
    if ((int64_t)a == INT64_MIN && (int64_t)b == -1) return a;
    return (uint64_t)((int64_t)a / (int64_t)b);
}
static inline uint32_t sdiv32(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    if ((int32_t)a == INT32_MIN && (int32_t)b == -1) return a;
    return (uint32_t)((int32_t)a / (int32_t)b);
}
static inline uint64_t udiv64(uint64_t a, uint64_t b) { return b ? a / b : 0; }
static inline uint32_t udiv32(uint32_t a, uint32_t b) { return b ? a / b : 0; }
static inline uint64_t smulh(uint64_t a, uint64_t b) { return (uint64_t)(((__int128)(int64_t)a * (int64_t)b) >> 64); }
static inline uint64_t umulh(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }

uint32_t crc32_update(uint32_t crc, uint64_t value, int bytes, bool castagnoli);

// ---- exclusive monitor and atomics (host atomics on guest memory)
template <typename T> static inline std::atomic_ref<T> aref(uint8_t* m, uint64_t a) {
    return std::atomic_ref<T>(*reinterpret_cast<T*>(m + a));
}

// Global exclusive monitor. A store-exclusive must fail if another store-exclusive hit the reserved location after
// the load-exclusive, even if the value is back to what was loaded: a plain compare-and-swap lets lock-free free
// lists hand one block to two threads (the ABA problem), which tore TotK's command buffers. Every 64-byte granule
// hashes to a generation counter that each successful store-exclusive bumps by 2; bit 0 locks the counter while a
// store-exclusive commits. Unrelated granules sharing a counter only cause spurious failures, which the
// architecture allows (the guest retries).
constexpr size_t kExclStripes = 4096;
inline std::atomic<uint64_t> g_excl_gen[kExclStripes];
static inline std::atomic<uint64_t>& excl_gen(uint64_t a) { return g_excl_gen[(a >> 6) & (kExclStripes - 1)]; }

// the generation to remember at a load-exclusive (waits out a store-exclusive committing in that granule)
static inline uint64_t excl_mark(uint64_t a) {
    uint64_t g;
    while ((g = excl_gen(a).load(std::memory_order_acquire)) & 1) {}
    return g;
}

// Runs `commit` (the compare-and-swap) if the granule's generation is still `gen`; bumps it if `commit` succeeds.
template <typename F> static inline uint32_t excl_commit(uint64_t a, uint64_t gen, F&& commit) {
    auto& g = excl_gen(a);
    uint64_t expected = gen;
    if (!g.compare_exchange_strong(expected, gen | 1, std::memory_order_acq_rel)) return 1;
    const bool ok = commit();
    g.store(ok ? gen + 2 : gen, std::memory_order_release);
    return ok ? 0 : 1;
}

template <typename T> static inline T ldx(Ctx* c, uint8_t* m, uint64_t a) {
    c->excl_gen = excl_mark(a);
    T v = aref<T>(m, a).load(std::memory_order_seq_cst);
    c->excl_addr = a; c->excl_val[0] = v; c->excl_size = sizeof(T); c->excl_pair = 0;
    return v;
}
template <typename T> static inline uint32_t stx(Ctx* c, uint8_t* m, uint64_t a, T v) {
    if (c->excl_size != sizeof(T) || c->excl_addr != a || c->excl_pair) { c->excl_size = 0; return 1; }
    c->excl_size = 0;
    T expected = (T)c->excl_val[0];
    return excl_commit(a, c->excl_gen, [&] {
        return aref<T>(m, a).compare_exchange_strong(expected, v, std::memory_order_seq_cst);
    });
}
static inline void clrex(Ctx* c) { c->excl_size = 0; }

// 128-bit compare-and-swap on guest memory (cmpxchg16b). Returns true on success; `expected` gets the old value.
static inline bool cas128(uint8_t* p, uint64_t expected[2], uint64_t lo, uint64_t hi) {
#if defined(_WIN32)
    return _InterlockedCompareExchange128((volatile long long*)p, (long long)hi, (long long)lo,
                                          (long long*)expected) != 0;
#else
    unsigned __int128 e = ((unsigned __int128)expected[1] << 64) | expected[0];
    unsigned __int128 v = ((unsigned __int128)hi << 64) | lo;
    bool ok = __atomic_compare_exchange_n((unsigned __int128*)p, &e, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    expected[0] = (uint64_t)e; expected[1] = (uint64_t)(e >> 64);
    return ok;
#endif
}

// LDXP / STXP: 32-bit pairs are one 64-bit unit, 64-bit pairs one 128-bit unit
static inline void ldxp32(Ctx* c, uint8_t* m, uint64_t a, uint64_t& t1, uint64_t& t2) {
    c->excl_gen = excl_mark(a);
    uint64_t v = aref<uint64_t>(m, a).load(std::memory_order_seq_cst);
    c->excl_addr = a; c->excl_val[0] = v; c->excl_size = 8; c->excl_pair = 1;
    t1 = (uint32_t)v; t2 = v >> 32;
}
static inline uint32_t stxp32(Ctx* c, uint8_t* m, uint64_t a, uint64_t lo, uint64_t hi) {
    if (c->excl_size != 8 || !c->excl_pair || c->excl_addr != a) { c->excl_size = 0; return 1; }
    c->excl_size = 0;
    uint64_t e = c->excl_val[0];
    return excl_commit(a, c->excl_gen, [&] {
        return aref<uint64_t>(m, a).compare_exchange_strong(e, ((uint64_t)(uint32_t)hi << 32) | (uint32_t)lo);
    });
}
static inline void ldxp64(Ctx* c, uint8_t* m, uint64_t a, uint64_t& t1, uint64_t& t2) {
    c->excl_gen = excl_mark(a);
    t1 = ld64(m, a); t2 = ld64(m, a + 8);
    c->excl_addr = a; c->excl_val[0] = t1; c->excl_val[1] = t2; c->excl_size = 16; c->excl_pair = 1;
}
static inline uint32_t stxp64(Ctx* c, uint8_t* m, uint64_t a, uint64_t lo, uint64_t hi) {
    if (c->excl_size != 16 || !c->excl_pair || c->excl_addr != a) { c->excl_size = 0; return 1; }
    c->excl_size = 0;
    uint64_t e[2] = {c->excl_val[0], c->excl_val[1]};
    return excl_commit(a, c->excl_gen, [&] { return cas128(m + a, e, lo, hi); });
}

}  // namespace zg

#include "zonaite/guest_fp.h"
#include "zonaite/guest_simd.h"
