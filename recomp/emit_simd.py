"""AArch64 SIMD & floating point -> C++ (scalar FP, Advanced SIMD, crypto, structure loads/stores).

Scalar FP uses the helpers in runtime/include/zonaite/guest_fp.h (ARM NaN/min/max/conversion rules); Advanced
SIMD uses the lane templates in guest_simd.h. Half-precision arithmetic (FEAT_FP16) is not on the A57 and is
reported Unimplemented so tests skip it; half <-> single/double conversions are base and supported.
"""
from .emit import Undefined, Unimplemented, rx, wx, U

FTYPE = {0: ("float", 4), 1: ("double", 8)}


# ---------------------------------------------------------------- scalar register access

def sread(n, ftype):
    """Scalar FP register n as a float/double expression."""
    if ftype == "float":
        return f"f32((uint32_t)c->v[{n}].lo)"
    return f"f64(c->v[{n}].lo)"


def swrite(d, ftype, expr):
    """Write a float/double scalar, zeroing the rest of V[d]."""
    if ftype == "float":
        return f"c->v[{d}] = V{{(uint64_t)u32f({expr}), 0}};"
    return f"c->v[{d}] = V{{u64f({expr}), 0}};"


def ftype_of(w):
    t = (w >> 22) & 3
    if t == 3:
        raise Unimplemented("half-precision arithmetic (FEAT_FP16)")
    if t == 2:
        raise Undefined("fp type 2")
    return FTYPE[t][0]


def vfp_expand_imm(imm8, ftype):
    """VFPExpandImm: the 8-bit FMOV immediate as raw bits."""
    sign, b6, low6 = imm8 >> 7, (imm8 >> 6) & 1, imm8 & 0x3F
    if ftype == "float":
        exp = ((b6 ^ 1) << 7) | ((0x1F if b6 else 0) << 2) | ((imm8 >> 4) & 3)
        return (sign << 31) | (exp << 23) | ((imm8 & 0xF) << 19)
    exp = ((b6 ^ 1) << 10) | ((0xFF if b6 else 0) << 2) | ((imm8 >> 4) & 3)
    return (sign << 63) | (exp << 52) | ((imm8 & 0xF) << 48)


# ---------------------------------------------------------------- entry

def emit(w, pc, flow):
    if w & 0x5F000000 == 0x1E000000:
        return scalar_fp(w)
    if w & 0x5F000000 == 0x1F000000:
        if (w >> 31) & 1 or (w >> 29) & 1:
            raise Undefined("fp dp3")
        return fp_dp3(w)
    from . import emit_neon
    return emit_neon.emit(w)


def scalar_fp(w):
    rd, rn, rm = w & 31, (w >> 5) & 31, (w >> 16) & 31
    if w & 0x5F20FC00 == 0x1E200000:
        return fp_int_conv(w)
    if w & 0x5F200000 == 0x1E000000:  # bit 31 is sf here, not M
        if (w >> 29) & 1:
            raise Undefined("fixed conv S")
        return fp_fixed_conv(w)
    if (w >> 31) & 1 or (w >> 29) & 1:
        raise Undefined("scalar fp M/S")
    if w & 0x5F207C00 == 0x1E204000:  # data processing, 1 source
        return fp_dp1(w)
    t = ftype_of(w)
    if w & 0x5F203C00 == 0x1E202000:  # compare
        if (w >> 14) & 3 or w & 7:
            raise Undefined("fcmp op/opcode2")
        b = ("0.0f" if t == "float" else "0.0") if (w >> 3) & 1 else sread(rm, t)
        return [f"set_nzcv(c, fp_cmp({sread(rn, t)}, {b}));"]
    if w & 0x5F201C00 == 0x1E201000:  # immediate
        if (w >> 5) & 31:
            raise Undefined("fmov imm5")
        bits = vfp_expand_imm((w >> 13) & 0xFF, t)
        if t == "float":
            return [f"c->v[{rd}] = V{{{bits:#x}ull, 0}};"]
        return [f"c->v[{rd}] = V{{{bits:#x}ull, 0}};"]
    if w & 0x5F200C00 == 0x1E200400:  # conditional compare
        code, nzcv = (w >> 12) & 15, w & 15
        return [f"if (cond(c, {code})) set_nzcv(c, fp_cmp({sread(rn, t)}, {sread(rm, t)})); "
                f"else set_nzcv(c, {nzcv}u);"]
    if w & 0x5F200C00 == 0x1E200800:  # data processing, 2 source
        opcode = (w >> 12) & 15
        a, b = sread(rn, t), sread(rm, t)
        ops = {0: f"fp_mul({a}, {b})", 1: f"fp_div({a}, {b})", 2: f"fp_add({a}, {b})", 3: f"fp_sub({a}, {b})",
               4: f"fp_max({a}, {b})", 5: f"fp_min({a}, {b})", 6: f"fp_maxnm({a}, {b})",
               7: f"fp_minnm({a}, {b})", 8: f"fp_neg(fp_mul({a}, {b}))"}
        if opcode not in ops:
            raise Undefined("fp dp2 opcode")
        return [swrite(rd, t, ops[opcode])]
    if w & 0x5F200C00 == 0x1E200C00:  # conditional select
        code = (w >> 12) & 15
        return [f"c->v[{rd}] = V{{cond(c, {code}) ? c->v[{rn}].lo{' & 0xFFFFFFFFull' if t == 'float' else ''} : "
                f"c->v[{rm}].lo{' & 0xFFFFFFFFull' if t == 'float' else ''}, 0}};"]
    raise Undefined("scalar fp")


def fp_dp1(w):
    rd, rn = w & 31, (w >> 5) & 31
    ty, opcode = (w >> 22) & 3, (w >> 15) & 0x3F
    if 4 <= opcode <= 7:  # FCVT between precisions (type 3 = half is base here)
        if opcode == 6 or ty == 2 or {4: 0, 5: 1, 7: 3}[opcode] == ty:
            raise Undefined("fcvt same/invalid type")
        src = {0: sread(rn, "float"), 1: sread(rn, "double"), 3: f"half_to_float((uint16_t)c->v[{rn}].lo)"}[ty]
        dst = {4: "float", 5: "double", 7: "half"}[opcode]
        if dst == "half":
            return [f"c->v[{rd}] = V{{(uint64_t)to_half({src}), 0}};"]
        if dst == "float":
            return [swrite(rd, "float", src if ty == 3 else f"f64_to_f32({src})")]
        return [swrite(rd, "double", f"f32_to_f64({src})" if ty == 0 else f"(double){src}")]
    if 16 <= opcode <= 19:
        raise Unimplemented("frint32/64 (FEAT_FRINTTS)")
    t = ftype_of(w)
    a = sread(rn, t)
    ops = {0: a, 1: f"fp_abs({a})", 2: f"fp_neg({a})", 3: f"fp_sqrt({a})", 8: f"fp_round({a}, 0)",
           9: f"fp_round({a}, 1)", 10: f"fp_round({a}, 2)", 11: f"fp_round({a}, 3)", 12: f"fp_round({a}, 4)",
           14: f"fp_round({a}, 0)", 15: f"fp_round({a}, 0)"}
    if opcode == 0:  # FMOV register: raw bits, no NaN processing
        mask = " & 0xFFFFFFFFull" if t == "float" else ""
        return [f"c->v[{rd}] = V{{c->v[{rn}].lo{mask}, 0}};"]
    if opcode not in ops:
        raise Undefined("fp dp1 opcode")
    return [swrite(rd, t, ops[opcode])]


def fp_dp3(w):
    rd, rn, rm, ra = w & 31, (w >> 5) & 31, (w >> 16) & 31, (w >> 10) & 31
    t = ftype_of(w)
    o1, o0 = (w >> 21) & 1, (w >> 15) & 1
    n, m, a = sread(rn, t), sread(rm, t), sread(ra, t)
    expr = {(0, 0): f"fp_fma({a}, {n}, {m})", (0, 1): f"fp_fma({a}, fp_neg({n}), {m})",
            (1, 0): f"fp_fma(fp_neg({a}), fp_neg({n}), {m})", (1, 1): f"fp_fma(fp_neg({a}), {n}, {m})"}[(o1, o0)]
    return [swrite(rd, t, expr)]


ROUND_OF = {"N": 0, "P": 1, "M": 2, "Z": 3, "A": 4}


def fp_int_conv(w):
    sf, ty, rmode, opcode = w >> 31, (w >> 22) & 3, (w >> 19) & 3, (w >> 16) & 7
    rd, rn = w & 31, (w >> 5) & 31
    if (w >> 29) & 1:
        raise Undefined("fp<->int S")
    if ty == 1 and rmode == 3 and opcode == 6:
        raise Unimplemented("fjcvtzs (FEAT_JSCVT)")
    if opcode in (6, 7):  # FMOV between general and FP registers
        if ty == 3:
            raise Unimplemented("fmov half")
        if rmode == 0 and ((sf == 0 and ty == 0) or (sf == 1 and ty == 1)):
            if opcode == 6:
                return [wx(rd, sf, "c->v[%d].lo" % rn if sf else "(uint32_t)c->v[%d].lo" % rn)]
            return [f"c->v[{rd}] = V{{{rx(rn, sf)}, 0}};" if sf else f"c->v[{rd}] = V{{(uint64_t){rx(rn, 0)}, 0}};"]
        if rmode == 1 and sf == 1 and ty == 2:  # top half of a 128-bit register
            if opcode == 6:
                return [wx(rd, 1, f"c->v[{rn}].hi")]
            return [f"c->v[{rd}].hi = {rx(rn, 1)};"]
        raise Undefined("fmov combination")
    if ty == 3:
        raise Unimplemented("half <-> int (FEAT_FP16)")
    if ty == 2:
        raise Undefined("fp type 2")
    t = FTYPE[ty][0]
    if opcode in (2, 3):  # SCVTF / UCVTF (rmode must be 0)
        if rmode:
            raise Undefined("cvtf rmode")
        src = rx(rn, sf)
        cast = ("int64_t" if sf else "int32_t") if opcode == 2 else ("uint64_t" if sf else "uint32_t")
        return [swrite(rd, t, f"({t})({cast}){src}")]
    if opcode in (0, 1, 4, 5):
        if opcode in (4, 5) and rmode:
            raise Undefined("fcvta rmode")
        mode = 4 if opcode in (4, 5) else {0: 0, 1: 1, 2: 2, 3: 3}[rmode]
        signed = opcode in (0, 4)
        itype = ("int64_t" if sf else "int32_t") if signed else ("uint64_t" if sf else "uint32_t")
        return [wx(rd, sf, f"({U(sf)})fp_to_int<{itype}>({sread(rn, t)}, {mode})")]
    raise Undefined("fp<->int opcode")


def fp_fixed_conv(w):
    sf, ty, rmode, opcode, scale = w >> 31, (w >> 22) & 3, (w >> 19) & 3, (w >> 16) & 7, (w >> 10) & 0x3F
    rd, rn = w & 31, (w >> 5) & 31
    if ty == 3:
        raise Unimplemented("half fixed point (FEAT_FP16)")
    if ty == 2 or (not sf and scale < 32):
        raise Undefined("fixed conv")
    t = FTYPE[ty][0]
    fbits = 64 - scale
    if rmode == 3 and opcode in (0, 1):  # FCVTZS / FCVTZU, fixed point
        itype = ("int64_t" if sf else "int32_t") if opcode == 0 else ("uint64_t" if sf else "uint32_t")
        return [wx(rd, sf, f"({U(sf)})fp_to_int<{itype}>(std::ldexp({sread(rn, t)}, {fbits}), 3)")]
    if rmode == 0 and opcode in (2, 3):  # SCVTF / UCVTF, fixed point
        cast = ("int64_t" if sf else "int32_t") if opcode == 2 else ("uint64_t" if sf else "uint32_t")
        return [swrite(rd, t, f"std::ldexp(({t})({cast}){rx(rn, sf)}, -{fbits})")]
    raise Undefined("fixed conv opcode")


def ls_structure(w):
    from . import emit_neon
    return emit_neon.ls_structure(w)
