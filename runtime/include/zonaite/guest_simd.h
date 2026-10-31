// Advanced SIMD lane primitives for the generated code. A V register is viewed as lanes of T; operations that
// work on a 64-bit vector (Q=0) process the low half and zero the high half, as on hardware.
#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace zg {

template <typename T> struct L {
    static constexpr int N = 16 / (int)sizeof(T);
    T e[16 / sizeof(T)];
};
template <typename T> static inline L<T> lanes(const V& v) { L<T> l; std::memcpy(l.e, &v, 16); return l; }
template <typename T> static inline V pack(const L<T>& l) { V v; std::memcpy(&v, l.e, 16); return v; }
template <typename T> static inline T lane_get(const V& v, int i) { T t; std::memcpy(&t, (const uint8_t*)&v + i * sizeof(T), sizeof(T)); return t; }
template <typename T> static inline V lane_set(V v, int i, T t) { std::memcpy((uint8_t*)&v + i * sizeof(T), &t, sizeof(T)); return v; }
static inline V vzero() { return V{0, 0}; }
static inline V vhalf(V v, bool q) { if (!q) v.hi = 0; return v; }
static inline int nl(int esize, bool q) { return (q ? 16 : 8) / esize; }

// reinterpret a lane's bits (integer <-> float lanes of the same width)
template <typename F, typename I> static inline F bitcast_lane(I i) {
    static_assert(sizeof(F) == sizeof(I), "lane width");
    F f; std::memcpy(&f, &i, sizeof f); return f;
}
template <typename I, typename F> static inline I bits_lane(F f) {
    static_assert(sizeof(F) == sizeof(I), "lane width");
    I i; std::memcpy(&i, &f, sizeof i); return i;
}

// ---- saturation
template <typename T> static inline T sat(__int128 v) {
    __int128 lo = (__int128)std::numeric_limits<T>::min(), hi = (__int128)std::numeric_limits<T>::max();
    return (T)(v < lo ? lo : v > hi ? hi : v);
}
template <typename T> using Wide = std::conditional_t<std::is_signed_v<T>, __int128, __int128>;

// ---- element-wise
template <typename T, typename F> static inline V vmap1(V a, bool q, F f) {
    L<T> x = lanes<T>(a), r{};
    for (int i = 0; i < nl(sizeof(T), q); i++) r.e[i] = (T)f(x.e[i]);
    return pack(r);
}
template <typename T, typename F> static inline V vmap2(V a, V b, bool q, F f) {
    L<T> x = lanes<T>(a), y = lanes<T>(b), r{};
    for (int i = 0; i < nl(sizeof(T), q); i++) r.e[i] = (T)f(x.e[i], y.e[i]);
    return pack(r);
}
template <typename T, typename F> static inline V vmap3(V d, V a, V b, bool q, F f) {
    L<T> z = lanes<T>(d), x = lanes<T>(a), y = lanes<T>(b), r{};
    for (int i = 0; i < nl(sizeof(T), q); i++) r.e[i] = (T)f(z.e[i], x.e[i], y.e[i]);
    return pack(r);
}
// by element: f(d_i, a_i, b[idx])
template <typename T, typename F> static inline V vmap_elem(V d, V a, V b, int idx, bool q, F f) {
    T s = lane_get<T>(b, idx);
    L<T> z = lanes<T>(d), x = lanes<T>(a), r{};
    for (int i = 0; i < nl(sizeof(T), q); i++) r.e[i] = (T)f(z.e[i], x.e[i], s);
    return pack(r);
}
// pairwise over the concatenation a:b
template <typename T, typename F> static inline V vpair(V a, V b, bool q, F f) {
    int n = nl(sizeof(T), q);
    T cat[32];
    L<T> x = lanes<T>(a), y = lanes<T>(b), r{};
    for (int i = 0; i < n; i++) { cat[i] = x.e[i]; cat[n + i] = y.e[i]; }
    for (int i = 0; i < n; i++) r.e[i] = (T)f(cat[2 * i], cat[2 * i + 1]);
    return pack(r);
}
// across lanes: fold with f
template <typename T, typename A, typename F> static inline A vreduce(V a, bool q, A init, F f) {
    L<T> x = lanes<T>(a);
    A acc = init;
    for (int i = 0; i < nl(sizeof(T), q); i++) acc = f(acc, x.e[i]);
    return acc;
}
// FP across-lane reductions are pairwise trees on hardware (FMAXV etc.)
template <typename T, typename F> static inline T vreduce_tree(V a, bool q, F f) {
    L<T> x = lanes<T>(a);
    int n = nl(sizeof(T), q);
    T t[16];
    for (int i = 0; i < n; i++) t[i] = x.e[i];
    while (n > 1) { for (int i = 0; i < n / 2; i++) t[i] = f(t[2 * i], t[2 * i + 1]); n /= 2; }
    return t[0];
}

// ---- widening / narrowing. "upper" selects the high 64 bits of the narrow sources (the "2" forms).
template <typename TS, typename TD, typename F> static inline V vwiden1(V a, bool upper, F f) {
    L<TS> x = lanes<TS>(a);
    L<TD> r{};
    int off = upper ? L<TD>::N : 0;
    for (int i = 0; i < L<TD>::N; i++) r.e[i] = (TD)f(x.e[off + i]);
    return pack(r);
}
template <typename TS, typename TD, typename F> static inline V vwiden2(V a, V b, bool upper, F f) {
    L<TS> x = lanes<TS>(a), y = lanes<TS>(b);
    L<TD> r{};
    int off = upper ? L<TD>::N : 0;
    for (int i = 0; i < L<TD>::N; i++) r.e[i] = (TD)f(x.e[off + i], y.e[off + i]);
    return pack(r);
}
// W forms: a is already wide
template <typename TS, typename TD, typename F> static inline V vwidenw(V a, V b, bool upper, F f) {
    L<TD> x = lanes<TD>(a), r{};
    L<TS> y = lanes<TS>(b);
    int off = upper ? L<TD>::N : 0;
    for (int i = 0; i < L<TD>::N; i++) r.e[i] = (TD)f(x.e[i], y.e[off + i]);
    return pack(r);
}
// accumulate into wide d
template <typename TS, typename TD, typename F> static inline V vwidenacc(V d, V a, V b, bool upper, F f) {
    L<TS> x = lanes<TS>(a), y = lanes<TS>(b);
    L<TD> z = lanes<TD>(d), r{};
    int off = upper ? L<TD>::N : 0;
    for (int i = 0; i < L<TD>::N; i++) r.e[i] = (TD)f(z.e[i], x.e[off + i], y.e[off + i]);
    return pack(r);
}
template <typename TS, typename TD, typename F> static inline V vwiden_elem(V d, V a, V b, int idx, bool upper, F f) {
    TS s = lane_get<TS>(b, idx);
    L<TS> x = lanes<TS>(a);
    L<TD> z = lanes<TD>(d), r{};
    int off = upper ? L<TD>::N : 0;
    for (int i = 0; i < L<TD>::N; i++) r.e[i] = (TD)f(z.e[i], x.e[off + i], s);
    return pack(r);
}
// narrow: wide lanes of a -> narrow results in the low half (upper=false, high half zero) or the high half
// (upper=true, low half kept from dold)
template <typename TS, typename TD, typename F> static inline V vnarrow(V dold, V a, bool upper, F f) {
    L<TS> x = lanes<TS>(a);
    L<TD> r = upper ? lanes<TD>(dold) : L<TD>{};
    int off = upper ? L<TS>::N : 0;
    for (int i = 0; i < L<TS>::N; i++) r.e[off + i] = (TD)f(x.e[i]);
    if (!upper) for (int i = L<TS>::N; i < L<TD>::N; i++) r.e[i] = 0;
    return pack(r);
}
template <typename TS, typename TD, typename F> static inline V vnarrow2(V dold, V a, V b, bool upper, F f) {
    L<TS> x = lanes<TS>(a), y = lanes<TS>(b);
    L<TD> r = upper ? lanes<TD>(dold) : L<TD>{};
    int off = upper ? L<TS>::N : 0;
    for (int i = 0; i < L<TS>::N; i++) r.e[off + i] = (TD)f(x.e[i], y.e[i]);
    if (!upper) for (int i = L<TS>::N; i < L<TD>::N; i++) r.e[i] = 0;
    return pack(r);
}
// pairwise widening add (SADDLP/UADDLP), with optional accumulate (SADALP)
template <typename TS, typename TD> static inline V vaddlp(V d, V a, bool q, bool acc) {
    L<TS> x = lanes<TS>(a);
    L<TD> z = lanes<TD>(d), r{};
    int n = nl(sizeof(TD), q);
    for (int i = 0; i < n; i++) r.e[i] = (TD)((acc ? z.e[i] : 0) + (TD)x.e[2 * i] + (TD)x.e[2 * i + 1]);
    return pack(r);
}

// ---- shifts
// SSHL/USHL family: shift by the signed low byte of b; negative shifts right. rounding / saturating variants.
template <typename T> static inline T vshr_big(T a, bool round) {  // right shift by more than the width
    if (round) return 0;
    return std::is_signed_v<T> ? (T)(a < 0 ? -1 : 0) : (T)0;
}
template <typename T> static inline T vshl(T a, int64_t s, bool round) {
    const int bits = sizeof(T) * 8;
    if (s >= 0) return s >= bits ? 0 : (T)((std::make_unsigned_t<T>)a << s);
    int64_t r = -s;
    if (r > bits) return vshr_big(a, round);
    __int128 v = (__int128)a;
    if (round) v += (__int128)1 << (r - 1);
    return (T)(v >> r);
}
template <typename T> static inline T vqshl(T a, int64_t s, bool round) {
    const int bits = sizeof(T) * 8;
    if (s >= 0) {
        if (a == 0) return 0;
        if (s >= bits + 1) return a > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        return sat<T>((__int128)a << s);
    }
    int64_t r = -s;
    if (r > bits) return vshr_big(a, round);
    __int128 v = (__int128)a;
    if (round) v += (__int128)1 << (r - 1);
    return sat<T>(v >> r);
}
// immediate rounding shift right
template <typename T> static inline T vrshr(T a, int s) { return (T)((((__int128)a) + ((__int128)1 << (s - 1))) >> s); }

// ---- permutes / tables
template <typename T> static inline V vdup(T t, bool q) { L<T> r{}; for (int i = 0; i < nl(sizeof(T), q); i++) r.e[i] = t; return pack(r); }
template <typename T> static inline V vzip(V a, V b, bool q, int part) {
    L<T> x = lanes<T>(a), y = lanes<T>(b), r{};
    int n = nl(sizeof(T), q), base = part * n / 2;
    for (int i = 0; i < n / 2; i++) { r.e[2 * i] = x.e[base + i]; r.e[2 * i + 1] = y.e[base + i]; }
    return pack(r);
}
template <typename T> static inline V vuzp(V a, V b, bool q, int part) {
    L<T> x = lanes<T>(a), y = lanes<T>(b), r{};
    int n = nl(sizeof(T), q);
    T cat[32];
    for (int i = 0; i < n; i++) { cat[i] = x.e[i]; cat[n + i] = y.e[i]; }
    for (int i = 0; i < n; i++) r.e[i] = cat[2 * i + part];
    return pack(r);
}
template <typename T> static inline V vtrn(V a, V b, bool q, int part) {
    L<T> x = lanes<T>(a), y = lanes<T>(b), r{};
    int n = nl(sizeof(T), q);
    for (int i = 0; i < n / 2; i++) { r.e[2 * i] = x.e[2 * i + part]; r.e[2 * i + 1] = y.e[2 * i + part]; }
    return pack(r);
}
static inline V vext(V a, V b, int idx, bool q) {
    uint8_t cat[32], r[16] = {0};
    int n = q ? 16 : 8;
    std::memcpy(cat, &a, n);
    std::memcpy(cat + n, &b, n);
    for (int i = 0; i < n; i++) r[i] = cat[idx + i];
    V v; std::memcpy(&v, r, 16); return v;
}
static inline V vtbl(const V* tbl, int regs, V idx, V dold, bool q, bool tbx) {
    uint8_t t[64], ix[16], r[16];
    for (int i = 0; i < regs; i++) std::memcpy(t + 16 * i, &tbl[i], 16);
    std::memcpy(ix, &idx, 16);
    std::memcpy(r, &dold, 16);
    int n = q ? 16 : 8;
    for (int i = 0; i < n; i++) r[i] = ix[i] < regs * 16 ? t[ix[i]] : (tbx ? r[i] : 0);
    if (!q) std::memset(r + 8, 0, 8);
    V v; std::memcpy(&v, r, 16); return v;
}
template <typename T> static inline V vrev(V a, bool q, int group_bytes) {  // reverse T lanes in each group
    L<T> x = lanes<T>(a), r{};
    int n = nl(sizeof(T), q), g = group_bytes / (int)sizeof(T);
    for (int i = 0; i < n; i++) r.e[i] = x.e[(i / g) * g + (g - 1 - i % g)];
    return pack(r);
}

// ---- bit select (BSL/BIT/BIF): result = (a & mask) | (b & ~mask)
static inline V vsel(V mask, V a, V b, bool q) {
    return vhalf(V{(a.lo & mask.lo) | (b.lo & ~mask.lo), (a.hi & mask.hi) | (b.hi & ~mask.hi)}, q);
}

// ---- integer helpers used by lane lambdas
template <typename T> static inline T vabd(T a, T b) { return a > b ? (T)(a - b) : (T)(b - a); }
template <typename T> static inline T vhadd(T a, T b, bool round) { return (T)(((__int128)a + b + (round ? 1 : 0)) >> 1); }
template <typename T> static inline T vhsub(T a, T b) { return (T)(((__int128)a - b) >> 1); }
template <typename T> static inline T vqdmulh(T a, T b, bool round) {
    __int128 p = 2 * (__int128)a * b + (round ? ((__int128)1 << (sizeof(T) * 8 - 1)) : 0);
    return sat<T>(p >> (sizeof(T) * 8));
}
static inline uint8_t pmul8(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    for (int i = 0; i < 8; i++) if (b & (1 << i)) r ^= (uint8_t)(a << i);
    return r;
}
static inline uint16_t pmull8(uint8_t a, uint8_t b) {
    uint16_t r = 0;
    for (int i = 0; i < 8; i++) if (b & (1 << i)) r ^= (uint16_t)((uint16_t)a << i);
    return r;
}
static inline V pmull64(uint64_t a, uint64_t b) {
    unsigned __int128 r = 0;
    for (int i = 0; i < 64; i++) if ((b >> i) & 1) r ^= (unsigned __int128)a << i;
    return V{(uint64_t)r, (uint64_t)(r >> 64)};
}
template <typename T> static inline T vcls(T a) {
    using UT = std::make_unsigned_t<T>;
    const int bits = sizeof(T) * 8;
    UT u = (UT)a, s = (UT)(u >> (bits - 1)) ? (UT)~u : u;
    int n = 0;
    for (int i = bits - 2; i >= 0 && !((s >> i) & 1); i--) n++;
    return (T)n;
}
template <typename T> static inline T vclz(T a) {
    using UT = std::make_unsigned_t<T>;
    const int bits = sizeof(T) * 8;
    UT u = (UT)a;
    int n = 0;
    for (int i = bits - 1; i >= 0 && !((u >> i) & 1); i--) n++;
    return (T)n;
}

// ---- unsigned reciprocal estimates (URECPE / URSQRTE)
static inline uint32_t urecpe(uint32_t a) {
    if (!(a >> 31)) return 0xFFFFFFFFu;
    return recip_estimate_u(a >> 23) << 23;
}
static inline uint32_t ursqrte(uint32_t a) {
    if (!(a >> 30)) return 0xFFFFFFFFu;
    return rsqrt_estimate_u(a >> 23) << 23;
}

// ---- crypto (FEAT_AES, FEAT_SHA1, FEAT_SHA256): implemented in guest_support.cpp
V aese(V d, V n);
V aesd(V d, V n);
V aesmc(V n);
V aesimc(V n);
V sha1c(V d, uint32_t n, V m);
V sha1p(V d, uint32_t n, V m);
V sha1m(V d, uint32_t n, V m);
V sha1su0(V d, V n, V m);
V sha1su1(V d, V n);
uint32_t sha1h(uint32_t n);
V sha256h(V d, V n, V m, bool part2);
V sha256su0(V d, V n);
V sha256su1(V d, V n, V m);

}  // namespace zg
