// Scalar floating point with AArch64 semantics (FPCR: round to nearest, FZ=0, DN=0, AHP=0).
// Fast path is plain IEEE arithmetic; a NaN result goes through the ARM NaN rules. FPSR exception flags are not
// tracked (games don't read them).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace zg {

template <typename T> struct FBits;
template <> struct FBits<float> { using U = uint32_t; static constexpr U kQuiet = 0x00400000u, kDefault = 0x7FC00000u; };
template <> struct FBits<double> {
    using U = uint64_t;
    static constexpr U kQuiet = 0x0008000000000000ull, kDefault = 0x7FF8000000000000ull;
};

template <typename T> static inline typename FBits<T>::U fbits(T v) { typename FBits<T>::U u; std::memcpy(&u, &v, sizeof v); return u; }
template <typename T> static inline T ffrom(typename FBits<T>::U u) { T v; std::memcpy(&v, &u, sizeof v); return v; }
static inline float f32(uint32_t u) { return ffrom<float>(u); }
static inline double f64(uint64_t u) { return ffrom<double>(u); }
static inline uint32_t u32f(float f) { return fbits(f); }
static inline uint64_t u64f(double d) { return fbits(d); }

template <typename T> static inline bool is_snan(T v) { return v != v && !(fbits(v) & FBits<T>::kQuiet); }
template <typename T> static inline T quiet(T v) { return ffrom<T>(fbits(v) | FBits<T>::kQuiet); }
template <typename T> static inline T default_nan() { return ffrom<T>(FBits<T>::kDefault); }

// ARM FPProcessNaNs: first signalling NaN (quietened), else first quiet NaN, else default NaN (invalid op).
template <typename T> static inline T nan2(T a, T b) {
    if (is_snan(a)) return quiet(a);
    if (is_snan(b)) return quiet(b);
    if (a != a) return a;
    if (b != b) return b;
    return default_nan<T>();
}
template <typename T> static inline T nan3(T a, T b, T c) {
    if (is_snan(a)) return quiet(a);
    if (is_snan(b)) return quiet(b);
    if (is_snan(c)) return quiet(c);
    if (a != a) return a;
    if (b != b) return b;
    if (c != c) return c;
    return default_nan<T>();
}
template <typename T> static inline T nan1(T a) { return a != a ? quiet(a) : default_nan<T>(); }

#define ZG_FP_BIN(name, expr)                                                                       \
    template <typename T> static inline T name(T a, T b) {                                          \
        T r = (expr);                                                                               \
        if (__builtin_expect(r != r, 0)) r = nan2(a, b);                                           \
        return r;                                                                                   \
    }
ZG_FP_BIN(fp_add, a + b)
ZG_FP_BIN(fp_sub, a - b)
ZG_FP_BIN(fp_mul, a * b)
ZG_FP_BIN(fp_div, a / b)
#undef ZG_FP_BIN

template <typename T> static inline T fp_sqrt(T a) {
    T r = std::sqrt(a);
    if (__builtin_expect(r != r, 0)) r = nan1(a);
    return r;
}
template <typename T> static inline T fp_neg(T a) { return ffrom<T>(fbits(a) ^ (fbits(T(-0.0)))); }
template <typename T> static inline T fp_abs(T a) { return ffrom<T>(fbits(a) & ~fbits(T(-0.0))); }
template <typename T> static inline T fp_abd(T a, T b) { return fp_abs(fp_sub(a, b)); }

// FMULX: like FMUL but inf * 0 = +-2.0
template <typename T> static inline T fp_mulx(T a, T b) {
    if ((std::isinf(a) && b == 0) || (a == 0 && std::isinf(b))) {
        if (a != a || b != b) return nan2(a, b);
        return std::signbit(a) != std::signbit(b) ? T(-2.0) : T(2.0);
    }
    return fp_mul(a, b);
}

// a + n*m fused. ARM: if the addend is a quiet NaN and n*m is inf*0, the result is the default NaN.
template <typename T> static inline T fp_fma(T a, T n, T m) {
    T r = std::fma(n, m, a);
    if (__builtin_expect(r != r, 0)) {
        bool inf_zero = (std::isinf(n) && m == 0) || (n == 0 && std::isinf(m));
        if (a != a && !is_snan(a) && inf_zero && !(n != n) && !(m != m)) return default_nan<T>();
        return nan3(a, n, m);
    }
    return r;
}

// FMIN/FMAX: NaN propagates (ARM rules); -0 < +0.
template <typename T> static inline T fp_max(T a, T b) {
    if (a != a || b != b) return nan2(a, b);
    if (a == 0 && b == 0) return std::signbit(a) ? b : a;
    return a > b ? a : b;
}
template <typename T> static inline T fp_min(T a, T b) {
    if (a != a || b != b) return nan2(a, b);
    if (a == 0 && b == 0) return std::signbit(a) ? a : b;
    return a < b ? a : b;
}
// FMAXNM/FMINNM: a single quiet NaN loses to a number.
template <typename T> static inline T fp_maxnm(T a, T b) {
    bool qa = a != a && !is_snan(a), qb = b != b && !is_snan(b);
    if (qa && !(b != b)) a = ffrom<T>(fbits(T(-INFINITY)));
    else if (qb && !(a != a)) b = ffrom<T>(fbits(T(-INFINITY)));
    return fp_max(a, b);
}
template <typename T> static inline T fp_minnm(T a, T b) {
    bool qa = a != a && !is_snan(a), qb = b != b && !is_snan(b);
    if (qa && !(b != b)) a = T(INFINITY);
    else if (qb && !(a != a)) b = T(INFINITY);
    return fp_min(a, b);
}

// Compare -> NZCV (FCMP/FCMPE): unordered = 0011, equal = 0110, less = 1000, greater = 0010
template <typename T> static inline uint32_t fp_cmp(T a, T b) {
    if (a != a || b != b) return 0x3;
    if (a == b) return 0x6;
    if (a < b) return 0x8;
    return 0x2;
}

// ---- rounding to integral (FRINT*): mode 0 nearest-even, 1 +inf, 2 -inf, 3 zero, 4 nearest-away
template <typename T> static inline T fp_round(T a, int mode) {
    if (a != a) return nan1(a);
    if (std::isinf(a) || a == 0) return a;
    T r;
    switch (mode) {
        case 0: r = std::nearbyint(a); break;  // host default mode is round-to-nearest-even
        case 1: r = std::ceil(a); break;
        case 2: r = std::floor(a); break;
        case 3: r = std::trunc(a); break;
        default: r = std::round(a); break;
    }
    if (r == 0) r = std::signbit(a) ? T(-0.0) : T(0.0);
    return r;
}

// ---- float -> integer with ARM saturation (NaN -> 0)
template <typename I, typename T> static inline I fp_to_int(T a, int mode) {
    if (a != a) return 0;
    T r = fp_round(a, mode);
    const int bits = (int)sizeof(I) * 8 - (std::numeric_limits<I>::is_signed ? 1 : 0);
    const T hi = std::ldexp(T(1), bits);  // 2^31, 2^32, 2^63 or 2^64: exactly representable
    if (r >= hi) return std::numeric_limits<I>::max();
    if (std::numeric_limits<I>::is_signed) {
        if (r < -hi) return std::numeric_limits<I>::min();
    } else if (r < 0) {
        return 0;
    }
    return (I)r;
}

// ---- half precision (IEEE, AHP = 0), round to nearest even
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, frac = h & 0x3FF;
    if (exp == 0x1F) return f32(sign | 0x7F800000u | (frac ? 0x00400000u | (frac << 13) : 0));  // NaN quietened
    if (exp == 0) {
        if (!frac) return f32(sign);
        float v = std::ldexp((float)frac, -24);
        return sign ? -v : v;
    }
    return f32(sign | ((exp + 112) << 23) | (frac << 13));
}
template <typename T> static inline uint16_t to_half(T a) {
    using U = typename FBits<T>::U;
    const int mbits = sizeof(T) == 4 ? 23 : 52;
    U u = fbits(a);
    uint16_t sign = (uint16_t)((u >> (sizeof(T) * 8 - 1)) << 15);
    if (a != a) return (uint16_t)(sign | 0x7E00 | (uint16_t)((u >> (mbits - 10)) & 0x1FF));
    if (std::isinf(a)) return (uint16_t)(sign | 0x7C00);
    T m = std::fabs(a);
    if (m == 0) return sign;
    int e;
    T f = std::frexp(m, &e);  // m = f * 2^e, f in [0.5, 1)
    int exp = e - 1 + 15;     // biased half exponent
    uint32_t mant;
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
    if (exp <= 0) {  // subnormal: value / 2^-24, rounded to nearest even
        T scaled = std::ldexp(m, 24);
        mant = (uint32_t)std::nearbyint(scaled);
        if (mant >= 0x400) return (uint16_t)(sign | (1 << 10) | (mant - 0x400));
        return (uint16_t)(sign | mant);
    }
    T scaled = std::ldexp(f, 11);  // [1024, 2048)
    mant = (uint32_t)std::nearbyint(scaled);
    if (mant == 0x800) { mant = 0x400; exp++; }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3FF));
}

// ---- FCVT between float and double (ARM keeps the NaN payload's top bits, quietened)
static inline double f32_to_f64(float a) { return a != a ? f64((uint64_t)(u32f(a) & 0x80000000u) << 32 | 0x7FF8000000000000ull | ((uint64_t)(u32f(a) & 0x003FFFFFu) << 29)) : (double)a; }
static inline float f64_to_f32(double a) { return a != a ? f32((uint32_t)(u64f(a) >> 32 & 0x80000000u) | 0x7FC00000u | (uint32_t)((u64f(a) >> 29) & 0x003FFFFFu)) : (float)a; }

// ---- reciprocal / reciprocal square root estimates (ARM ARM RecipEstimate / RecipSqrtEstimate)
static inline uint32_t recip_estimate_u(uint32_t a) {  // a in [256, 511] -> estimate in [256, 511]
    a = a * 2 + 1;
    uint32_t b = (1u << 19) / a;
    return (b + 1) / 2;
}
static inline uint32_t rsqrt_estimate_u(uint32_t a) {  // a in [128, 511]
    if (a < 256) a = a * 2 + 1;
    else { a = (a >> 1) << 1; a = (a + 1) * 2; }
    uint32_t b = 512;
    while (a * (b + 1) * (b + 1) < (1u << 28)) b++;
    return (b + 1) / 2;
}
// FPRecipEstimate / FPRSqrtEstimate, worked on the double layout (52-bit fraction) as the ARM ARM does.
// ebits/fbits describe T; bias_r = 2*bias - 1 (253 / 2045), bias_s = 3*bias - 1 (380 / 3068).
template <typename T> static inline T fp_recpe(T a) {
    using U = typename FBits<T>::U;
    const int fb = sizeof(T) == 4 ? 23 : 52, ebits = sizeof(T) == 4 ? 8 : 11;
    const int64_t bias_r = sizeof(T) == 4 ? 253 : 2045;
    U u = fbits(a);
    U sign = u >> (fb + ebits);
    if (a != a) return nan1(a);
    if (std::isinf(a)) return ffrom<T>(sign << (fb + ebits));
    if (a == 0) return ffrom<T>((sign << (fb + ebits)) | ((((U)1 << ebits) - 1) << fb));
    if (std::fabs(a) < std::ldexp(T(1), sizeof(T) == 4 ? -128 : -1024))  // overflows: inf (round to nearest)
        return ffrom<T>((sign << (fb + ebits)) | ((((U)1 << ebits) - 1) << fb));
    uint64_t frac = (uint64_t)(u & (((U)1 << fb) - 1)) << (52 - fb);
    int64_t exp = (int64_t)((u >> fb) & (((U)1 << ebits) - 1));
    if (exp == 0) {
        if (!((frac >> 51) & 1)) { exp = -1; frac = (frac & ((1ull << 50) - 1)) << 2; }
        else frac = (frac & ((1ull << 51) - 1)) << 1;
    }
    uint32_t scaled = 0x100 | (uint32_t)((frac >> 44) & 0xFF);
    int64_t rexp = bias_r - exp;
    uint32_t est = recip_estimate_u(scaled);
    frac = (uint64_t)(est & 0xFF) << 44;
    if (rexp == 0) frac = (1ull << 51) | (frac >> 1);
    else if (rexp == -1) { frac = (1ull << 50) | (frac >> 2); rexp = 0; }
    return ffrom<T>((sign << (fb + ebits)) | ((U)rexp << fb) | (U)(frac >> (52 - fb)));
}
template <typename T> static inline T fp_rsqrte(T a) {
    using U = typename FBits<T>::U;
    const int fb = sizeof(T) == 4 ? 23 : 52, ebits = sizeof(T) == 4 ? 8 : 11;
    const int64_t bias_s = sizeof(T) == 4 ? 380 : 3068;
    U u = fbits(a);
    U sign = u >> (fb + ebits);
    if (a != a) return nan1(a);
    if (a == 0) return ffrom<T>((sign << (fb + ebits)) | ((((U)1 << ebits) - 1) << fb));
    if (sign) return default_nan<T>();
    if (std::isinf(a)) return T(0);
    uint64_t frac = (uint64_t)(u & (((U)1 << fb) - 1)) << (52 - fb);
    int64_t exp = (int64_t)((u >> fb) & (((U)1 << ebits) - 1));
    if (exp == 0) {
        while (!((frac >> 51) & 1)) { frac = (frac << 1) & ((1ull << 52) - 1); exp--; }
        frac = (frac << 1) & ((1ull << 52) - 1);
    }
    uint32_t scaled = (exp & 1) ? (0x80 | (uint32_t)((frac >> 45) & 0x7F)) : (0x100 | (uint32_t)((frac >> 44) & 0xFF));
    int64_t rexp = (bias_s - exp) / 2;
    uint32_t est = rsqrt_estimate_u(scaled);
    return ffrom<T>(((U)rexp << fb) | (U)(((uint64_t)(est & 0xFF) << 44) >> (52 - fb)));
}
template <typename T> static inline T fp_recps(T a, T b) {  // 2 - a*b fused; inf*0 -> 2
    if ((std::isinf(a) && b == 0) || (a == 0 && std::isinf(b))) {
        if (a != a || b != b) return nan2(fp_neg(a), b);
        return T(2.0);
    }
    T r = std::fma(-a, b, T(2.0));
    if (r != r) r = nan2(fp_neg(a), b);  // ARM negates op1 before NaN processing
    return r;
}
template <typename T> static inline T fp_rsqrts(T a, T b) {  // (3 - a*b) / 2 fused; inf*0 -> 1.5
    if ((std::isinf(a) && b == 0) || (a == 0 && std::isinf(b))) {
        if (a != a || b != b) return nan2(fp_neg(a), b);
        return T(1.5);
    }
    T r = std::fma(-a, b, T(3.0)) / T(2.0);
    if (r != r) r = nan2(fp_neg(a), b);
    return r;
}

}  // namespace zg
