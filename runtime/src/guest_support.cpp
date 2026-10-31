// Out-of-line helpers for the generated code.
#include "zonaite/guest.h"

namespace zg {

namespace {
struct CrcTables {
    uint32_t ieee[256], castagnoli[256];
    CrcTables() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t a = i, b = i;
            for (int k = 0; k < 8; k++) {
                a = (a >> 1) ^ ((a & 1) ? 0xEDB88320u : 0);  // CRC-32 (reflected 0x04C11DB7)
                b = (b >> 1) ^ ((b & 1) ? 0x82F63B78u : 0);  // CRC-32C (reflected 0x1EDC6F41)
            }
            ieee[i] = a;
            castagnoli[i] = b;
        }
    }
};
const CrcTables kCrc;
}  // namespace

// CRC32{B,H,W,X} / CRC32C*: no pre/post inversion, the guest does that itself.
uint32_t crc32_update(uint32_t crc, uint64_t value, int bytes, bool castagnoli) {
    const uint32_t* t = castagnoli ? kCrc.castagnoli : kCrc.ieee;
    for (int i = 0; i < bytes; i++) crc = (crc >> 8) ^ t[(crc ^ (uint32_t)(value >> (8 * i))) & 0xFF];
    return crc;
}

// ---------------------------------------------------------------- AES (state bytes column-major, as ARM)
namespace {
struct AesTables {
    uint8_t sbox[256], inv[256];
    AesTables() {
        uint8_t p = 1, q = 1;
        do {  // generate the S-box from the multiplicative inverse and the affine transform
            p = (uint8_t)(p ^ (p << 1) ^ ((p & 0x80) ? 0x1B : 0));
            q ^= (uint8_t)(q << 1); q ^= (uint8_t)(q << 2); q ^= (uint8_t)(q << 4);
            if (q & 0x80) q ^= 0x09;
            uint8_t x = (uint8_t)(q ^ (uint8_t)((q << 1) | (q >> 7)) ^ (uint8_t)((q << 2) | (q >> 6)) ^
                                  (uint8_t)((q << 3) | (q >> 5)) ^ (uint8_t)((q << 4) | (q >> 4)));
            sbox[p] = (uint8_t)(x ^ 0x63);
        } while (p != 1);
        sbox[0] = 0x63;
        for (int i = 0; i < 256; i++) inv[sbox[i]] = (uint8_t)i;
    }
};
const AesTables kAes;
inline uint8_t xtime(uint8_t b) { return (uint8_t)((b << 1) ^ ((b & 0x80) ? 0x1B : 0)); }
inline uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t r = 0;
    while (b) { if (b & 1) r ^= a; a = xtime(a); b >>= 1; }
    return r;
}
inline void to_bytes(V v, uint8_t* b) { std::memcpy(b, &v, 16); }
inline V from_bytes(const uint8_t* b) { V v; std::memcpy(&v, b, 16); return v; }
inline uint32_t rol32(uint32_t x, int s) { return (x << s) | (x >> (32 - s)); }
inline uint32_t ror32_(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }
inline uint32_t w32(V v, int i) { return lane_get<uint32_t>(v, i); }
}  // namespace

V aese(V d, V n) {
    uint8_t s[16], r[16];
    to_bytes(V{d.lo ^ n.lo, d.hi ^ n.hi}, s);
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) r[row + 4 * c] = kAes.sbox[s[row + 4 * ((c + row) % 4)]];
    return from_bytes(r);
}
V aesd(V d, V n) {
    uint8_t s[16], r[16];
    to_bytes(V{d.lo ^ n.lo, d.hi ^ n.hi}, s);
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) r[row + 4 * c] = kAes.inv[s[row + 4 * ((c - row + 4) % 4)]];
    return from_bytes(r);
}
V aesmc(V n) {
    uint8_t s[16], r[16];
    to_bytes(n, s);
    for (int c = 0; c < 4; c++) {
        uint8_t* a = s + 4 * c;
        r[4 * c + 0] = (uint8_t)(gmul(a[0], 2) ^ gmul(a[1], 3) ^ a[2] ^ a[3]);
        r[4 * c + 1] = (uint8_t)(a[0] ^ gmul(a[1], 2) ^ gmul(a[2], 3) ^ a[3]);
        r[4 * c + 2] = (uint8_t)(a[0] ^ a[1] ^ gmul(a[2], 2) ^ gmul(a[3], 3));
        r[4 * c + 3] = (uint8_t)(gmul(a[0], 3) ^ a[1] ^ a[2] ^ gmul(a[3], 2));
    }
    return from_bytes(r);
}
V aesimc(V n) {
    uint8_t s[16], r[16];
    to_bytes(n, s);
    for (int c = 0; c < 4; c++) {
        uint8_t* a = s + 4 * c;
        r[4 * c + 0] = (uint8_t)(gmul(a[0], 14) ^ gmul(a[1], 11) ^ gmul(a[2], 13) ^ gmul(a[3], 9));
        r[4 * c + 1] = (uint8_t)(gmul(a[0], 9) ^ gmul(a[1], 14) ^ gmul(a[2], 11) ^ gmul(a[3], 13));
        r[4 * c + 2] = (uint8_t)(gmul(a[0], 13) ^ gmul(a[1], 9) ^ gmul(a[2], 14) ^ gmul(a[3], 11));
        r[4 * c + 3] = (uint8_t)(gmul(a[0], 11) ^ gmul(a[1], 13) ^ gmul(a[2], 9) ^ gmul(a[3], 14));
    }
    return from_bytes(r);
}

// ---------------------------------------------------------------- SHA1 / SHA256 (ARM ARM pseudocode)
namespace {
template <int Kind> V sha1_hash(V xv, uint32_t y, V w) {
    uint32_t x[4] = {w32(xv, 0), w32(xv, 1), w32(xv, 2), w32(xv, 3)};
    for (int e = 0; e < 4; e++) {
        uint32_t b = x[1], c = x[2], d = x[3], t;
        if (Kind == 0) t = (b & c) | (~b & d);           // choose
        else if (Kind == 1) t = b ^ c ^ d;               // parity
        else t = (b & c) | ((b | c) & d);                // majority
        y = y + rol32(x[0], 5) + t + w32(w, e);
        x[1] = rol32(x[1], 30);
        uint32_t top = x[3];                             // <Y, X> = ROL(Y:X, 32)
        x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = y; y = top;
    }
    V r{}; for (int i = 0; i < 4; i++) r = lane_set<uint32_t>(r, i, x[i]);
    return r;
}
}  // namespace
V sha1c(V d, uint32_t n, V m) { return sha1_hash<0>(d, n, m); }
V sha1p(V d, uint32_t n, V m) { return sha1_hash<1>(d, n, m); }
V sha1m(V d, uint32_t n, V m) { return sha1_hash<2>(d, n, m); }
uint32_t sha1h(uint32_t n) { return rol32(n, 30); }
V sha1su0(V d, V n, V m) { return V{d.hi ^ d.lo ^ m.lo, n.lo ^ d.hi ^ m.hi}; }
V sha1su1(V d, V n) {
    uint32_t t[4];
    V y{(n.lo >> 32) | (n.hi << 32), n.hi >> 32};  // Vn >> 32
    for (int i = 0; i < 4; i++) t[i] = w32(d, i) ^ w32(y, i);
    uint32_t r[4] = {rol32(t[0], 1), rol32(t[1], 1), rol32(t[2], 1), rol32(t[3], 1) ^ rol32(t[0], 2)};
    V v{}; for (int i = 0; i < 4; i++) v = lane_set<uint32_t>(v, i, r[i]);
    return v;
}
V sha256h(V d, V n, V m, bool part2) {
    uint32_t x[4], y[4];
    V xv = part2 ? n : d, yv = part2 ? d : n;
    for (int i = 0; i < 4; i++) { x[i] = w32(xv, i); y[i] = w32(yv, i); }
    for (int e = 0; e < 4; e++) {
        uint32_t chs = (y[0] & y[1]) | (~y[0] & y[2]);
        uint32_t maj = (x[0] & x[1]) | ((x[0] | x[1]) & x[2]);
        uint32_t s1 = ror32_(y[0], 6) ^ ror32_(y[0], 11) ^ ror32_(y[0], 25);
        uint32_t s0 = ror32_(x[0], 2) ^ ror32_(x[0], 13) ^ ror32_(x[0], 22);
        uint32_t t1 = y[3] + s1 + chs + w32(m, e);
        x[3] = t1 + x[3];
        y[3] = t1 + s0 + maj;
        uint32_t ytop = y[3], xtop = x[3];               // <Y, X> = ROL(Y:X, 32)
        y[3] = y[2]; y[2] = y[1]; y[1] = y[0]; y[0] = xtop;
        x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = ytop;
    }
    V r{}; for (int i = 0; i < 4; i++) r = lane_set<uint32_t>(r, i, part2 ? y[i] : x[i]);
    return r;
}
V sha256su0(V d, V n) {
    uint32_t t[4] = {w32(d, 1), w32(d, 2), w32(d, 3), w32(n, 0)};
    V r{};
    for (int e = 0; e < 4; e++) {
        uint32_t elt = ror32_(t[e], 7) ^ ror32_(t[e], 18) ^ (t[e] >> 3);
        r = lane_set<uint32_t>(r, e, elt + w32(d, e));
    }
    return r;
}
V sha256su1(V d, V n, V m) {
    uint32_t t0[4] = {w32(n, 1), w32(n, 2), w32(n, 3), w32(m, 0)};
    uint32_t t1[2] = {w32(m, 2), w32(m, 3)};
    uint32_t r[4];
    for (int e = 0; e < 2; e++) {
        uint32_t elt = ror32_(t1[e], 17) ^ ror32_(t1[e], 19) ^ (t1[e] >> 10);
        r[e] = elt + w32(d, e) + t0[e];
    }
    for (int e = 2; e < 4; e++) {
        uint32_t x = r[e - 2];
        uint32_t elt = ror32_(x, 17) ^ ror32_(x, 19) ^ (x >> 10);
        r[e] = elt + w32(d, e) + t0[e];
    }
    V v{}; for (int i = 0; i < 4; i++) v = lane_set<uint32_t>(v, i, r[i]);
    return v;
}

}  // namespace zg
