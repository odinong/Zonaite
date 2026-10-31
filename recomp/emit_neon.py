"""Advanced SIMD (NEON) -> C++: vector and scalar forms, crypto, structure loads/stores.

Each operation is written once as a lane expression over x, y (sources) and d (destination lane, for
accumulating forms); `Lanes` wraps it as a whole-vector op (vmap*) or as a scalar op on lane 0.
Lane templates live in runtime/include/zonaite/guest_simd.h.
"""
from .emit import Undefined, Unimplemented, rx, rsp, wx, U

UT = {0: "uint8_t", 1: "uint16_t", 2: "uint32_t", 3: "uint64_t"}
ST = {0: "int8_t", 1: "int16_t", 2: "int32_t", 3: "int64_t"}
FT = {0: "float", 1: "double"}


def it(size, signed):
    return (ST if signed else UT)[size]


def ones(t):
    return f"({t})~({t})0" if t[0] == "u" else f"({t})-1"


def fmask(ft, cond):
    """FP compare result: all ones or zero, as a float/double lane."""
    if ft == "float":
        return f"f32(({cond}) ? 0xFFFFFFFFu : 0u)"
    return f"f64(({cond}) ? ~0ull : 0ull)"


class Lanes:
    """Emits an operation either on whole vectors (vector=True) or on lane 0 only (scalar forms)."""

    def __init__(self, rd, rn, rm, q, vector):
        self.rd, self.rn, self.rm, self.q, self.vector = rd, rn, rm, q, vector

    def V(self, n):
        return f"c->v[{n}]"

    def un(self, t, body, src=None):
        n = self.rn if src is None else src
        if self.vector:
            return f"c->v[{self.rd}] = vmap1<{t}>(c->v[{n}], {self.qs}, []({t} x) {{ return ({t})({body}); }});"
        return (f"{{ {t} x = lane_get<{t}>(c->v[{n}], 0); "
                f"c->v[{self.rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({t})({body})); }}")

    def bin(self, t, body):
        if self.vector:
            return (f"c->v[{self.rd}] = vmap2<{t}>(c->v[{self.rn}], c->v[{self.rm}], {self.qs}, "
                    f"[]({t} x, {t} y) {{ return ({t})({body}); }});")
        return (f"{{ {t} x = lane_get<{t}>(c->v[{self.rn}], 0), y = lane_get<{t}>(c->v[{self.rm}], 0); "
                f"c->v[{self.rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({t})({body})); }}")

    def acc(self, t, body):
        """d = f(d, x, y)."""
        if self.vector:
            return (f"c->v[{self.rd}] = vmap3<{t}>(c->v[{self.rd}], c->v[{self.rn}], c->v[{self.rm}], {self.qs}, "
                    f"[]({t} d, {t} x, {t} y) {{ return ({t})({body}); }});")
        return (f"{{ {t} d = lane_get<{t}>(c->v[{self.rd}], 0), x = lane_get<{t}>(c->v[{self.rn}], 0), "
                f"y = lane_get<{t}>(c->v[{self.rm}], 0); "
                f"c->v[{self.rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({t})({body})); }}")

    def acc1(self, t, body):
        """d = f(d, x) (accumulating one-source forms: SSRA, SUQADD, ...)."""
        if self.vector:
            return (f"c->v[{self.rd}] = vmap2<{t}>(c->v[{self.rd}], c->v[{self.rn}], {self.qs}, "
                    f"[]({t} d, {t} x) {{ return ({t})({body}); }});")
        return (f"{{ {t} d = lane_get<{t}>(c->v[{self.rd}], 0), x = lane_get<{t}>(c->v[{self.rn}], 0); "
                f"c->v[{self.rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({t})({body})); }}")

    @property
    def qs(self):
        return "true" if self.q else "false"


def emit(w):
    # crypto
    if w & 0xFF3E0C00 == 0x4E280800:
        return crypto_aes(w)
    if w & 0xFF208C00 == 0x5E000000:
        return crypto_sha3(w)
    if w & 0xFF3E0C00 == 0x5E280800:
        return crypto_sha2(w)
    # scalar forms (bit 28 = 1 within the SIMD space: 01x1 1110 ...)
    if w & 0xDFE08400 == 0x5E000400:
        return scalar_copy(w)
    if w & 0xDF3E0C00 == 0x5E200800:
        return two_misc(w, scalar=True)
    if w & 0xDF3E0C00 == 0x5E300800:
        return scalar_pairwise(w)
    if w & 0xDF200C00 == 0x5E200000:
        return three_diff(w, scalar=True)
    if w & 0xDF200400 == 0x5E200400:
        return three_same(w, scalar=True)
    if w & 0xDF800400 == 0x5F000400:
        return shift_imm(w, scalar=True)
    if w & 0xDF000400 == 0x5F000000:
        return by_element(w, scalar=True)
    # vector forms
    if w & 0xBF208C00 == 0x0E000000:
        return table_lookup(w)
    if w & 0xBF208C00 == 0x0E000800:
        return permute(w)
    if w & 0xBF208400 == 0x2E000000:
        return extract(w)
    if w & 0x9FE08400 == 0x0E000400:
        return copy(w)
    if w & 0x9F3E0C00 == 0x0E200800:
        return two_misc(w, scalar=False)
    if w & 0x9F3E0C00 == 0x0E300800:
        return across(w)
    if w & 0x9F200C00 == 0x0E200000:
        return three_diff(w, scalar=False)
    if w & 0x9F200400 == 0x0E200400:
        return three_same(w, scalar=False)
    if w & 0x9FF80400 == 0x0F000400:
        return mod_imm(w)
    if w & 0x9F800400 == 0x0F000400:
        return shift_imm(w, scalar=False)
    if w & 0x9F000400 == 0x0F000000:
        return by_element(w, scalar=False)
    raise Unimplemented("advanced simd group")


def fields(w):
    return w & 31, (w >> 5) & 31, (w >> 16) & 31, (w >> 30) & 1, (w >> 29) & 1, (w >> 22) & 3


# ---------------------------------------------------------------- three same

def three_same(w, scalar):
    rd, rn, rm, q, u, size = fields(w)
    opcode = (w >> 11) & 31
    if opcode >= 0x18:
        return three_same_fp(w, scalar)
    if scalar:
        q = 1
    L = Lanes(rd, rn, rm, q, not scalar)
    t = it(size, not u)
    allowed64 = {1, 5, 6, 7, 8, 9, 10, 11, 16, 17}
    if size == 3 and (not q or opcode not in allowed64) and opcode not in (3, 23):
        raise Undefined("three same 64-bit")
    if scalar and opcode not in {1, 5, 6, 7, 8, 9, 10, 11, 16, 17, 22}:
        raise Undefined("scalar three same opcode")
    if scalar and opcode in (6, 7, 16, 17, 8, 10) and size != 3:
        raise Undefined("scalar three same size")
    if opcode == 3:  # logical
        a, b, d = f"c->v[{rn}]", f"c->v[{rm}]", f"c->v[{rd}]"
        qs = "true" if q else "false"
        if not u:
            expr = {0: f"V{{{a}.lo & {b}.lo, {a}.hi & {b}.hi}}", 1: f"V{{{a}.lo & ~{b}.lo, {a}.hi & ~{b}.hi}}",
                    2: f"V{{{a}.lo | {b}.lo, {a}.hi | {b}.hi}}", 3: f"V{{{a}.lo | ~{b}.lo, {a}.hi | ~{b}.hi}}"}[size]
            return [f"c->v[{rd}] = vhalf({expr}, {qs});"]
        expr = {0: f"vhalf(V{{{a}.lo ^ {b}.lo, {a}.hi ^ {b}.hi}}, {qs})", 1: f"vsel({d}, {a}, {b}, {qs})",
                2: f"vsel({b}, {a}, {d}, {qs})", 3: f"vsel({b}, {d}, {a}, {qs})"}[size]
        return [f"c->v[{rd}] = {expr};"]
    ops = {
        0: lambda: L.bin(t, f"vhadd<{t}>(x, y, false)"),
        1: lambda: L.bin(t, f"sat<{t}>((__int128)x + y)"),
        2: lambda: L.bin(t, f"vhadd<{t}>(x, y, true)"),
        4: lambda: L.bin(t, f"vhsub<{t}>(x, y)"),
        5: lambda: L.bin(t, f"sat<{t}>((__int128)x - y)"),
        6: lambda: L.bin(t, f"x > y ? {ones(t)} : 0"),
        7: lambda: L.bin(t, f"x >= y ? {ones(t)} : 0"),
        8: lambda: L.bin(t, f"vshl<{t}>(x, (int8_t)y, false)"),
        9: lambda: L.bin(t, f"vqshl<{t}>(x, (int8_t)y, false)"),
        10: lambda: L.bin(t, f"vshl<{t}>(x, (int8_t)y, true)"),
        11: lambda: L.bin(t, f"vqshl<{t}>(x, (int8_t)y, true)"),
        12: lambda: L.bin(t, "x > y ? x : y"),
        13: lambda: L.bin(t, "x < y ? x : y"),
        14: lambda: L.bin(t, f"vabd<{t}>(x, y)"),
        15: lambda: L.acc(t, f"d + vabd<{t}>(x, y)"),
        16: lambda: L.bin(UT[size], "x - y" if u else "x + y"),
        17: lambda: L.bin(UT[size], f"x == y ? {ones(UT[size])} : 0" if u else f"(x & y) ? {ones(UT[size])} : 0"),
        18: lambda: L.acc(UT[size], "d - x * y" if u else "d + x * y"),
        20: lambda: pairwise(rd, rn, rm, q, t, "x > y ? x : y"),
        21: lambda: pairwise(rd, rn, rm, q, t, "x < y ? x : y"),
    }
    if opcode == 19:
        if u:
            if size:
                raise Undefined("pmul size")
            return [L.bin("uint8_t", "pmul8(x, y)")]
        return [L.bin(UT[size], "x * y")]
    if opcode == 22:
        if size in (0, 3):
            raise Undefined("sqdmulh size")
        return [L.bin(ST[size], f"vqdmulh<{ST[size]}>(x, y, {'true' if u else 'false'})")]
    if opcode == 23:
        if u or (size == 3 and not q):
            raise Undefined("addp")
        return [pairwise(rd, rn, rm, q, UT[size], "x + y")]
    if opcode not in ops:
        raise Undefined("three same opcode")
    return [ops[opcode]()]


def pairwise(rd, rn, rm, q, t, body):
    return (f"c->v[{rd}] = vpair<{t}>(c->v[{rn}], c->v[{rm}], {'true' if q else 'false'}, "
            f"[]({t} x, {t} y) {{ return ({t})({body}); }});")


def three_same_fp(w, scalar):
    rd, rn, rm, q, u, size = fields(w)
    opcode = (w >> 11) & 31
    a, sz = size >> 1, size & 1
    if scalar:
        q = 1
    if sz and not q:
        raise Undefined("fp three same 1d")
    t = FT[sz]
    L = Lanes(rd, rn, rm, q, not scalar)
    key = (u, a, opcode)
    ops = {
        (0, 0, 24): ("bin", "fp_maxnm(x, y)"), (0, 0, 25): ("acc", "fp_fma(d, x, y)"),
        (0, 0, 26): ("bin", "fp_add(x, y)"), (0, 0, 27): ("bin", "fp_mulx(x, y)"),
        (0, 0, 28): ("bin", fmask(t, "x == y")), (0, 0, 30): ("bin", "fp_max(x, y)"),
        (0, 0, 31): ("bin", "fp_recps(x, y)"),
        (0, 1, 24): ("bin", "fp_minnm(x, y)"), (0, 1, 25): ("acc", "fp_fma(d, fp_neg(x), y)"),
        (0, 1, 26): ("bin", "fp_sub(x, y)"), (0, 1, 30): ("bin", "fp_min(x, y)"),
        (0, 1, 31): ("bin", "fp_rsqrts(x, y)"),
        (1, 0, 24): ("pair", "fp_maxnm(x, y)"), (1, 0, 26): ("pair", "fp_add(x, y)"),
        (1, 0, 27): ("bin", "fp_mul(x, y)"), (1, 0, 28): ("bin", fmask(t, "x >= y")),
        (1, 0, 29): ("bin", fmask(t, "fp_abs(x) >= fp_abs(y)")), (1, 0, 30): ("pair", "fp_max(x, y)"),
        (1, 0, 31): ("bin", "fp_div(x, y)"),
        (1, 1, 24): ("pair", "fp_minnm(x, y)"), (1, 1, 26): ("bin", "fp_abd(x, y)"),
        (1, 1, 28): ("bin", fmask(t, "x > y")), (1, 1, 29): ("bin", fmask(t, "fp_abs(x) > fp_abs(y)")),
        (1, 1, 30): ("pair", "fp_min(x, y)"),
    }
    scalar_ok = {(0, 0, 27), (0, 0, 28), (0, 0, 31), (0, 1, 31), (1, 0, 28), (1, 0, 29), (1, 1, 26), (1, 1, 28),
                 (1, 1, 29)}
    if key not in ops or (scalar and key not in scalar_ok):
        if opcode in (0x19, 0x1D) and not scalar:
            raise Unimplemented("fmlal/fmlsl (FEAT_FHM)")
        raise Undefined("fp three same")
    kind, body = ops[key]
    if kind == "pair":
        return [pairwise(rd, rn, rm, q, t, body)]
    return [getattr(L, kind)(t, body)]


# ---------------------------------------------------------------- three different (long / wide / narrow-high)

def three_diff(w, scalar):
    rd, rn, rm, q, u, size = fields(w)
    opcode = (w >> 12) & 15
    up = "true" if q else "false"
    if scalar:
        if u or size not in (1, 2) or opcode not in (9, 11, 13):
            raise Undefined("scalar three different")
        ts, td = ST[size], ST[size + 1]
        body = {9: "sat<TD>((__int128)d + sat<TD>(2 * (__int128)x * y))",
                11: "sat<TD>((__int128)d - sat<TD>(2 * (__int128)x * y))",
                13: "sat<TD>(2 * (__int128)x * y)"}[opcode].replace("TD", td)
        return [f"{{ {ts} x = lane_get<{ts}>(c->v[{rn}], 0), y = lane_get<{ts}>(c->v[{rm}], 0); "
                f"{td} d = lane_get<{td}>(c->v[{rd}], 0); "
                f"c->v[{rd}] = lane_set<{td}>(V{{0, 0}}, 0, ({td})({body})); }}"]
    if opcode == 14 and not u:  # PMULL / PMULL2
        if size == 0:
            return [f"c->v[{rd}] = vwiden2<uint8_t, uint16_t>(c->v[{rn}], c->v[{rm}], {up}, "
                    f"[](uint8_t x, uint8_t y) {{ return pmull8(x, y); }});"]
        if size == 3:
            half = "hi" if q else "lo"
            return [f"c->v[{rd}] = pmull64(c->v[{rn}].{half}, c->v[{rm}].{half});"]
        raise Undefined("pmull size")
    if size == 3:
        raise Undefined("three different size")
    signed = not u
    ts, td = it(size, signed), it(size + 1, signed)
    tdu = UT[size + 1]
    a, b, d = f"c->v[{rn}]", f"c->v[{rm}]", f"c->v[{rd}]"

    def widen2(body):
        return f"c->v[{rd}] = vwiden2<{ts}, {td}>({a}, {b}, {up}, []({ts} x, {ts} y) {{ return ({td})({body}); }});"

    def widenacc(body):
        return (f"c->v[{rd}] = vwidenacc<{ts}, {td}>({d}, {a}, {b}, {up}, "
                f"[]({td} d, {ts} x, {ts} y) {{ return ({td})({body}); }});")

    if opcode in (0, 2):
        return [widen2(f"({td})x {'+' if opcode == 0 else '-'} ({td})y")]
    if opcode in (1, 3):
        return [f"c->v[{rd}] = vwidenw<{ts}, {td}>({a}, {b}, {up}, "
                f"[]({td} x, {ts} y) {{ return ({td})(x {'+' if opcode == 1 else '-'} ({td})y); }});"]
    if opcode in (4, 6):  # ADDHN/RADDHN/SUBHN/RSUBHN: wide sources, narrow high halves
        tw, tn = UT[size + 1], UT[size]
        bits = 8 << size
        rnd = f" + (({tw})1 << {bits - 1})" if u else ""
        op = "+" if opcode == 4 else "-"
        return [f"c->v[{rd}] = vnarrow2<{tw}, {tn}>({d}, {a}, {b}, {up}, "
                f"[]({tw} x, {tw} y) {{ return ({tn})((({tw})(x {op} y){rnd}) >> {bits}); }});"]
    wabd = f"(x > y ? ({td})x - ({td})y : ({td})y - ({td})x)"  # in the wide type: |-128 - 127| = 255
    if opcode == 5:
        return [widenacc(f"d + {wabd}")]
    if opcode == 7:
        return [widen2(wabd)]
    if opcode in (8, 10):
        return [widenacc(f"d {'+' if opcode == 8 else '-'} ({td})x * ({td})y")]
    if opcode == 12:
        return [widen2(f"({td})x * ({td})y")]
    if not u and opcode in (9, 11, 13) and size in (1, 2):
        tds = ST[size + 1]
        if opcode == 13:
            return [widen2(f"sat<{tds}>(2 * (__int128)x * y)")]
        op = "+" if opcode == 9 else "-"
        return [widenacc(f"sat<{tds}>((__int128)d {op} sat<{tds}>(2 * (__int128)x * y))")]
    del tdu
    raise Undefined("three different opcode")


# ---------------------------------------------------------------- two-register misc

def two_misc(w, scalar):
    rd, rn, _, q, u, size = fields(w)
    opcode = (w >> 12) & 31
    if scalar:
        q = 1
    L = Lanes(rd, rn, 0, q, not scalar)
    qs = "true" if q else "false"
    a = size >> 1
    if opcode >= 0x0C and (opcode >= 0x16 or opcode in (0x0C, 0x0D, 0x0E, 0x0F)) and not (opcode in (0x12, 0x13, 0x14) ):
        fp = two_misc_fp(w, scalar, L, q)
        if fp is not None:
            return fp
    t = it(size, not u)
    tu, tsg = UT[size], ST[size]
    if scalar and opcode not in (3, 7, 8, 9, 10, 11, 18, 20) or (scalar and opcode == 18 and not u):
        raise Undefined("scalar two misc")  # XTN has no scalar form; SQXTUN (U=1) does
    if scalar and opcode in (8, 9, 10, 11) and size != 3:
        raise Undefined("scalar two misc size")
    if not scalar and size == 3 and not q and opcode not in (18, 19, 20):
        raise Undefined("two misc 1d")
    if not u:
        if opcode == 0:
            if size == 3:
                raise Undefined("rev64 size")
            return [f"c->v[{rd}] = vrev<{tu}>(c->v[{rn}], {qs}, 8);"]
        if opcode == 1:
            if size:
                raise Undefined("rev16 size")
            return [f"c->v[{rd}] = vrev<uint8_t>(c->v[{rn}], {qs}, 2);"]
        if opcode in (2, 6):
            if size == 3:
                raise Undefined("saddlp size")
            return [f"c->v[{rd}] = vaddlp<{tsg}, {ST[size + 1]}>(c->v[{rd}], c->v[{rn}], {qs}, "
                    f"{'true' if opcode == 6 else 'false'});"]
        if opcode == 3:
            return [L.acc1(tsg, f"sat<{tsg}>((__int128)d + ({tu})x)")]
        if opcode == 4:
            if size == 3:
                raise Undefined("cls size")
            return [L.un(tsg, f"vcls<{tsg}>(x)")]
        if opcode == 5:
            if size:
                raise Undefined("cnt size")
            return [L.un("uint8_t", "__builtin_popcount(x)")]
        if opcode == 7:
            return [L.un(tsg, f"sat<{tsg}>(x < 0 ? -(__int128)x : (__int128)x)")]
        if opcode == 8:
            return [L.un(tsg, f"x > 0 ? {ones(tsg)} : 0")]
        if opcode == 9:
            return [L.un(tsg, f"x == 0 ? {ones(tsg)} : 0")]
        if opcode == 10:
            return [L.un(tsg, f"x < 0 ? {ones(tsg)} : 0")]
        if opcode == 11:
            return [L.un(tu, f"({tsg})x < 0 ? ({tu})(0 - x) : x")]
        if opcode in (18, 20):  # XTN / SQXTN
            return narrow_un(rd, rn, q, size, scalar,
                             "x" if opcode == 18 else "sat<TD>((__int128)x)", src_signed=opcode == 20,
                             dst_signed=opcode == 20)
    else:
        if opcode == 0:
            if size >= 2:
                raise Undefined("rev32 size")
            return [f"c->v[{rd}] = vrev<{tu}>(c->v[{rn}], {qs}, 4);"]
        if opcode in (2, 6):
            if size == 3:
                raise Undefined("uaddlp size")
            return [f"c->v[{rd}] = vaddlp<{tu}, {UT[size + 1]}>(c->v[{rd}], c->v[{rn}], {qs}, "
                    f"{'true' if opcode == 6 else 'false'});"]
        if opcode == 3:
            return [L.acc1(tu, f"sat<{tu}>((__int128)d + ({tsg})x)")]
        if opcode == 4:
            if size == 3:
                raise Undefined("clz size")
            return [L.un(tu, f"vclz<{tu}>(x)")]
        if opcode == 5:
            if size == 0:
                return [f"c->v[{rd}] = vhalf(V{{~c->v[{rn}].lo, ~c->v[{rn}].hi}}, {qs});"]
            if size == 1:
                return [L.un("uint8_t", "(uint8_t)(rbit32(x) >> 24)")]
            raise Undefined("not/rbit size")
        if opcode == 7:
            return [L.un(tsg, f"sat<{tsg}>(-(__int128)x)")]
        if opcode == 8:
            return [L.un(tsg, f"x >= 0 ? {ones(tsg)} : 0")]
        if opcode == 9:
            return [L.un(tsg, f"x <= 0 ? {ones(tsg)} : 0")]
        if opcode == 11:
            return [L.un(tu, "0 - x")]
        if opcode == 18:  # SQXTUN
            return narrow_un(rd, rn, q, size, scalar, "sat<TD>((__int128)x)", src_signed=True, dst_signed=False)
        if opcode == 19 and not scalar:  # SHLL
            if size == 3:
                raise Undefined("shll size")
            return [f"c->v[{rd}] = vwiden1<{tu}, {UT[size + 1]}>(c->v[{rn}], {qs}, "
                    f"[]({tu} x) {{ return ({UT[size + 1]})(({UT[size + 1]})x << {8 << size}); }});"]
        if opcode == 20:  # UQXTN
            return narrow_un(rd, rn, q, size, scalar, "sat<TD>((__int128)x)", src_signed=False, dst_signed=False)
    del a, t
    raise Undefined("two misc opcode")


def narrow_un(rd, rn, q, size, scalar, body, src_signed, dst_signed):
    if size == 3:
        raise Undefined("narrow size")
    ts, td = it(size + 1, src_signed), it(size, dst_signed)
    body = body.replace("TD", td)
    if scalar:
        return [f"{{ {ts} x = lane_get<{ts}>(c->v[{rn}], 0); "
                f"c->v[{rd}] = lane_set<{td}>(V{{0, 0}}, 0, ({td})({body})); }}"]
    return [f"c->v[{rd}] = vnarrow<{ts}, {td}>(c->v[{rd}], c->v[{rn}], {'true' if q else 'false'}, "
            f"[]({ts} x) {{ return ({td})({body}); }});"]


def two_misc_fp(w, scalar, L, q):
    rd, rn, _, _, u, size = fields(w)
    opcode = (w >> 12) & 31
    a, sz = size >> 1, size & 1
    qs = "true" if q else "false"
    if opcode in (0x16, 0x17) and not scalar:  # FCVTN/FCVTXN (narrow) and FCVTL (widen)
        if opcode == 0x16:
            if u:
                raise Unimplemented("fcvtxn")
            if a:
                return None
            if sz:
                return [f"c->v[{rd}] = vnarrow<double, float>(c->v[{rd}], c->v[{rn}], {qs}, "
                        f"[](double x) {{ return f64_to_f32(x); }});"]
            return [f"c->v[{rd}] = vnarrow<float, uint16_t>(c->v[{rd}], c->v[{rn}], {qs}, "
                    f"[](float x) {{ return to_half(x); }});"]
        if u or a:
            return None
        if sz:
            return [f"c->v[{rd}] = vwiden1<float, double>(c->v[{rn}], {qs}, [](float x) {{ return f32_to_f64(x); }});"]
        return [f"c->v[{rd}] = vwiden1<uint16_t, float>(c->v[{rn}], {qs}, "
                f"[](uint16_t x) {{ return half_to_float(x); }});"]
    if opcode == 0x16 and scalar and u and sz:
        raise Unimplemented("fcvtxn scalar")
    if opcode in (0x1E, 0x1F) and not a:
        raise Unimplemented("frint32/64 (FEAT_FRINTTS)")
    if opcode not in (0x0C, 0x0D, 0x0E, 0x0F, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1F):
        return None
    if opcode in (0x0C, 0x0D, 0x0E, 0x0F) and not a:
        return None  # integer ops with these opcodes (cmgt/cmeq/cmlt/abs zero forms are opcodes 8-11)
    if sz and not q:
        raise Undefined("fp two misc 1d")
    t = FT[sz]
    it_s = "int32_t" if not sz else "int64_t"
    it_u = "uint32_t" if not sz else "uint64_t"
    key = (u, a, opcode)
    zero = "0.0f" if t == "float" else "0.0"
    conv = lambda signed, mode: (  # noqa: E731
        f"bitcast_lane<{t}>(fp_to_int<{it_s if signed else it_u}>(x, {mode}))")
    ops = {
        (0, 0, 0x18): f"fp_round(x, 0)", (0, 0, 0x19): f"fp_round(x, 2)",
        (0, 0, 0x1A): conv(True, 0), (0, 0, 0x1B): conv(True, 2), (0, 0, 0x1C): conv(True, 4),
        (0, 0, 0x1D): f"({t})(bits_lane<{it_s}>(x))",
        (0, 1, 0x0C): fmask(t, f"x > {zero}"), (0, 1, 0x0D): fmask(t, f"x == {zero}"),
        (0, 1, 0x0E): fmask(t, f"x < {zero}"), (0, 1, 0x0F): "fp_abs(x)",
        (0, 1, 0x18): "fp_round(x, 1)", (0, 1, 0x19): "fp_round(x, 3)",
        (0, 1, 0x1A): conv(True, 1), (0, 1, 0x1B): conv(True, 3), (0, 1, 0x1D): "fp_recpe(x)",
        (1, 0, 0x18): "fp_round(x, 4)", (1, 0, 0x19): "fp_round(x, 0)",
        (1, 0, 0x1A): conv(False, 0), (1, 0, 0x1B): conv(False, 2), (1, 0, 0x1C): conv(False, 4),
        (1, 0, 0x1D): f"({t})(bits_lane<{it_u}>(x))",
        (1, 1, 0x0C): fmask(t, f"x >= {zero}"), (1, 1, 0x0D): fmask(t, f"x <= {zero}"),
        (1, 1, 0x0F): "fp_neg(x)", (1, 1, 0x19): "fp_round(x, 0)",
        (1, 1, 0x1A): conv(False, 1), (1, 1, 0x1B): conv(False, 3), (1, 1, 0x1D): "fp_rsqrte(x)",
        (1, 1, 0x1F): "fp_sqrt(x)",
    }
    if key in ((0, 1, 0x1C), (1, 1, 0x1C)):  # URECPE / URSQRTE (32-bit integer lanes, vector only)
        if sz or scalar:
            raise Undefined("urecpe size")
        fn = "urecpe" if not u else "ursqrte"
        return [L.un("uint32_t", f"{fn}(x)")]
    if key == (0, 1, 0x1F) and scalar:  # FRECPX
        raise Unimplemented("frecpx")
    if key not in ops:
        raise Undefined("fp two misc")
    if scalar and opcode in (0x18, 0x19, 0x0F, 0x1F) or (scalar and key == (1, 1, 0x0F)):
        raise Undefined("scalar fp two misc")
    return [L.un(t, ops[key])]


# ---------------------------------------------------------------- across lanes

def across(w):
    rd, rn, _, q, u, size = fields(w)
    opcode = (w >> 12) & 31
    qs = "true" if q else "false"
    if opcode in (0x0C, 0x0F):  # FMAXNMV/FMINNMV/FMAXV/FMINV
        if not u:
            raise Unimplemented("fp16 across lanes")
        if size & 1 or not q:
            raise Undefined("fp across size")
        mn = size >> 1
        fn = {(0x0C, 0): "fp_maxnm", (0x0C, 1): "fp_minnm", (0x0F, 0): "fp_max", (0x0F, 1): "fp_min"}[(opcode, mn)]
        return [f"c->v[{rd}] = V{{(uint64_t)u32f(vreduce_tree<float>(c->v[{rn}], true, "
                f"[](float x, float y) {{ return {fn}(x, y); }})), 0}};"]
    if size == 3 or (size == 2 and not q):
        raise Undefined("across size")
    t = it(size, not u)
    if opcode == 3:  # SADDLV/UADDLV
        acc = "int64_t" if not u else "uint64_t"
        td = it(size + 1, not u)
        return [f"c->v[{rd}] = lane_set<{td}>(V{{0, 0}}, 0, ({td})vreduce<{t}, {acc}>(c->v[{rn}], {qs}, ({acc})0, "
                f"[]({acc} a, {t} x) {{ return a + x; }}));"]
    if opcode in (0x0A, 0x1A):
        better = ">" if opcode == 0x0A else "<"
        return [f"c->v[{rd}] = lane_set<{t}>(V{{0, 0}}, 0, vreduce<{t}, {t}>(c->v[{rn}], {qs}, lane_get<{t}>(c->v[{rn}], 0), "
                f"[]({t} a, {t} x) {{ return x {better} a ? x : a; }}));"]
    if opcode == 0x1B and not u:
        tu = UT[size]
        return [f"c->v[{rd}] = lane_set<{tu}>(V{{0, 0}}, 0, vreduce<{tu}, {tu}>(c->v[{rn}], {qs}, ({tu})0, "
                f"[]({tu} a, {tu} x) {{ return ({tu})(a + x); }}));"]
    raise Undefined("across opcode")


def scalar_pairwise(w):
    rd, rn, _, _, u, size = fields(w)
    opcode = (w >> 12) & 31
    if opcode == 0x1B and not u:  # ADDP (scalar): d = n[0] + n[1]
        if size != 3:
            raise Undefined("addp scalar size")
        return [f"c->v[{rd}] = V{{c->v[{rn}].lo + c->v[{rn}].hi, 0}};"]
    if not u:
        raise Unimplemented("fp16 scalar pairwise")
    sz = size & 1
    t = FT[sz]
    fn = {(0x0C, 0): "fp_maxnm", (0x0D, 0): "fp_add", (0x0F, 0): "fp_max", (0x0C, 1): "fp_minnm",
          (0x0F, 1): "fp_min"}.get((opcode, size >> 1))
    if fn is None:
        raise Undefined("scalar pairwise")
    if sz:
        return [f"c->v[{rd}] = V{{u64f({fn}(f64(c->v[{rn}].lo), f64(c->v[{rn}].hi))), 0}};"]
    return [f"c->v[{rd}] = V{{(uint64_t)u32f({fn}(f32((uint32_t)c->v[{rn}].lo), "
            f"f32((uint32_t)(c->v[{rn}].lo >> 32)))), 0}};"]


# ---------------------------------------------------------------- copy / insert / move

def lowest_size(imm5):
    for s in range(4):
        if imm5 & (1 << s):
            return s
    return None


def copy(w):
    rd, rn, _, q, op, _ = fields(w)
    imm5, imm4 = (w >> 16) & 31, (w >> 11) & 15
    size = lowest_size(imm5)
    if size is None:
        raise Undefined("copy imm5")
    t = UT[size]
    idx = imm5 >> (size + 1)
    qs = "true" if q else "false"
    if op:
        if not q:
            raise Undefined("ins element q")
        src = imm4 >> size
        return [f"c->v[{rd}] = lane_set<{t}>(c->v[{rd}], {idx}, lane_get<{t}>(c->v[{rn}], {src}));"]
    if imm4 == 0:  # DUP element
        if size == 3 and not q:
            raise Undefined("dup 1d")
        return [f"c->v[{rd}] = vdup<{t}>(lane_get<{t}>(c->v[{rn}], {idx}), {qs});"]
    if imm4 == 1:  # DUP general
        if size == 3 and not q:
            raise Undefined("dup 1d")
        return [f"c->v[{rd}] = vdup<{t}>(({t}){rx(rn, 1)}, {qs});"]
    if imm4 == 3:  # INS general
        if not q:
            raise Undefined("ins q")
        return [f"c->v[{rd}] = lane_set<{t}>(c->v[{rd}], {idx}, ({t}){rx(rn, 1)});"]
    if imm4 == 5:  # SMOV
        if size == 3 or (size == 2 and not q):
            raise Undefined("smov size")
        return [wx(rd, q, f"(uint64_t)(int64_t)lane_get<{ST[size]}>(c->v[{rn}], {idx})")]
    if imm4 == 7:  # UMOV
        if (q and size != 3) or (not q and size == 3):
            raise Undefined("umov size")
        return [wx(rd, q, f"(uint64_t)lane_get<{t}>(c->v[{rn}], {idx})")]
    raise Undefined("copy imm4")


def scalar_copy(w):
    rd, rn = w & 31, (w >> 5) & 31
    imm5, imm4, op = (w >> 16) & 31, (w >> 11) & 15, (w >> 29) & 1
    size = lowest_size(imm5)
    if op or imm4 or size is None:
        raise Undefined("scalar copy")
    t = UT[size]
    return [f"c->v[{rd}] = lane_set<{t}>(V{{0, 0}}, 0, lane_get<{t}>(c->v[{rn}], {imm5 >> (size + 1)}));"]


# ---------------------------------------------------------------- modified immediate

def mod_imm(w):
    rd, _, _, q, op, _ = fields(w)
    cmode, o2 = (w >> 12) & 15, (w >> 11) & 1
    imm8 = (((w >> 16) & 7) << 5) | ((w >> 5) & 31)
    if o2:
        raise Unimplemented("fmov fp16 immediate")
    qs = "true" if q else "false"

    def rep(value, bits):
        r = 0
        for i in range(64 // bits):
            r |= value << (i * bits)
        return r

    c3 = cmode >> 1
    if c3 <= 3:
        imm = rep(imm8 << (8 * c3), 32)
    elif c3 <= 5:
        imm = rep(imm8 << (8 * (c3 - 4)), 16)
    elif c3 == 6:
        imm = rep((imm8 << 8) | 0xFF if not cmode & 1 else (imm8 << 16) | 0xFFFF, 32)
    elif not cmode & 1 and not op:
        imm = rep(imm8, 8)
    elif not cmode & 1 and op:
        imm = 0
        for i in range(8):
            if (imm8 >> i) & 1:
                imm |= 0xFF << (8 * i)
    elif not op:  # FMOV (vector, immediate) single
        from .emit_simd import vfp_expand_imm
        imm = rep(vfp_expand_imm(imm8, "float"), 32)
    else:  # FMOV double
        if not q:
            raise Undefined("fmov 2d q")
        from .emit_simd import vfp_expand_imm
        imm = vfp_expand_imm(imm8, "double")
    imm &= 0xFFFFFFFFFFFFFFFF
    orr_bic = (cmode < 12 and cmode & 1)
    if orr_bic:
        d = f"c->v[{rd}]"
        if op:  # BIC
            return [f"c->v[{rd}] = vhalf(V{{{d}.lo & {(~imm) & 0xFFFFFFFFFFFFFFFF:#x}ull, "
                    f"{d}.hi & {(~imm) & 0xFFFFFFFFFFFFFFFF:#x}ull}}, {qs});"]
        return [f"c->v[{rd}] = vhalf(V{{{d}.lo | {imm:#x}ull, {d}.hi | {imm:#x}ull}}, {qs});"]
    if op and cmode != 14 and cmode != 15:  # MVNI
        imm = ~imm & 0xFFFFFFFFFFFFFFFF
    return [f"c->v[{rd}] = V{{{imm:#x}ull, {f'{imm:#x}ull' if q else '0'}}};"]


# ---------------------------------------------------------------- shift by immediate

def shift_imm(w, scalar):
    rd, rn, _, q, u, _ = fields(w)
    immh, immb, opcode = (w >> 19) & 15, (w >> 16) & 7, (w >> 11) & 31
    if immh == 0:
        raise Undefined("shift immh")
    size = immh.bit_length() - 1  # 0..3
    esize = 8 << size
    imm = (immh << 3) | immb
    shr, shl = 2 * esize - imm, imm - esize
    if scalar:
        q = 1
    L = Lanes(rd, rn, 0, q, not scalar)
    qs = "true" if q else "false"
    narrowing = opcode in (16, 17, 18, 19)
    widening = opcode == 20
    if narrowing or widening:
        if size == 3:
            raise Undefined("narrow/widen shift size")
    elif size == 3 and not q:
        raise Undefined("shift 1d")
    if scalar and size != 3 and opcode in (0, 2, 4, 6, 8, 10):
        raise Undefined("scalar shift size")
    t = it(size, not u)
    tu, tsg = UT[size], ST[size]
    if opcode == 0:
        return [L.un(t, f"(__int128)x >> {shr}")]
    if opcode == 2:
        return [L.acc1(t, f"d + ({t})((__int128)x >> {shr})")]
    if opcode == 4:
        return [L.un(t, f"vrshr<{t}>(x, {shr})")]
    if opcode == 6:
        return [L.acc1(t, f"d + vrshr<{t}>(x, {shr})")]
    if opcode == 8 and u:  # SRI
        mask = (((1 << esize) - 1) >> shr) if shr < esize else 0
        return [L.acc1(tu, f"(d & ({tu}){~mask & ((1 << esize) - 1):#x}ull) | "
                           f"(({tu})((__int128)x >> {shr}) & ({tu}){mask:#x}ull)")]
    if opcode == 10:
        if not u:
            return [L.un(tu, f"({tu})(x << {shl})")]
        mask = ((1 << esize) - 1) << shl & ((1 << esize) - 1)
        return [L.acc1(tu, f"(d & ({tu}){~mask & ((1 << esize) - 1):#x}ull) | ({tu})(x << {shl})")]
    if opcode == 12 and u:  # SQSHLU
        return [L.un(tsg, f"({tsg})sat<{tu}>((__int128)x << {shl})")]
    if opcode == 14:
        return [L.un(t, f"sat<{t}>((__int128)x << {shl})")]
    if narrowing:
        # source lanes are 2*esize; shift amount is relative to the narrow esize
        sig_src = {16: not u and False, 17: False, 18: not u, 19: not u}[opcode]
        if opcode in (16, 17):
            sig_src = bool(u)  # SQSHRUN/SQRSHRUN take signed sources; SHRN/RSHRN don't care
        ts = it(size + 1, sig_src or (opcode in (18, 19) and not u))
        rnd = opcode in (17, 19)
        val = f"(((__int128)x{f' + ((__int128)1 << {shr - 1})' if rnd else ''}) >> {shr})"
        if opcode in (16, 17) and not u:
            td = UT[size]
            body = f"({td}){val}"
        elif opcode in (16, 17):
            td = UT[size]
            body = f"sat<{td}>({val})"
        else:
            td = it(size, not u)
            body = f"sat<{td}>({val})"
        if scalar:
            if opcode in (16, 17) and not u:
                raise Undefined("scalar shrn")
            return [f"{{ {ts} x = lane_get<{ts}>(c->v[{rn}], 0); "
                    f"c->v[{rd}] = lane_set<{td}>(V{{0, 0}}, 0, ({td})({body})); }}"]
        return [f"c->v[{rd}] = vnarrow<{ts}, {td}>(c->v[{rd}], c->v[{rn}], {qs}, "
                f"[]({ts} x) {{ return ({td})({body}); }});"]
    if widening:  # SSHLL / USHLL
        if scalar:
            raise Undefined("scalar sshll")
        td = it(size + 1, not u)
        return [f"c->v[{rd}] = vwiden1<{t}, {td}>(c->v[{rn}], {qs}, "
                f"[]({t} x) {{ return ({td})(({td})x << {shl}); }});"]
    if opcode in (28, 31):  # fixed-point conversions
        if size < 2:
            raise Unimplemented("fp16 fixed point vector")
        ft = FT[size - 2]
        fbits = shr
        if opcode == 28:
            src = f"bits_lane<{t}>(x)"
            return [L.un(ft, f"std::ldexp(({ft}){src}, -{fbits})")]
        itype = it(size, not u)
        return [L.un(ft, f"bitcast_lane<{ft}>(fp_to_int<{itype}>(std::ldexp(x, {fbits}), 3))")]
    raise Undefined("shift opcode")


# ---------------------------------------------------------------- by element

def by_element(w, scalar):
    rd, rn, _, q, u, size = fields(w)
    opcode = (w >> 12) & 15
    l, m, h = (w >> 21) & 1, (w >> 20) & 1, (w >> 11) & 1
    rm = (w >> 16) & 15
    if scalar:
        q = 1
    qs = "true" if q else "false"
    fp_ops = {(0, 1): "fp_fma(d, x, y)", (0, 5): "fp_fma(d, fp_neg(x), y)", (0, 9): "fp_mul(x, y)",
              (1, 9): "fp_mulx(x, y)"}
    if (u, opcode) in fp_ops:
        if size < 2:
            raise Unimplemented("fp16 by element")
        sz = size & 1
        rm |= m << 4
        if sz:
            if l or not q:
                raise Undefined("fp by element 2d")
            idx = h
        else:
            idx = (h << 1) | l
        t = FT[sz]
        body = fp_ops[(u, opcode)]
        if scalar:
            return [f"{{ {t} d = lane_get<{t}>(c->v[{rd}], 0), x = lane_get<{t}>(c->v[{rn}], 0), "
                    f"y = lane_get<{t}>(c->v[{rm}], {idx}); c->v[{rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({body})); }}"]
        return [f"c->v[{rd}] = vmap_elem<{t}>(c->v[{rd}], c->v[{rn}], c->v[{rm}], {idx}, {qs}, "
                f"[]({t} d, {t} x, {t} y) {{ return ({t})({body}); }});"]
    if size == 1:
        idx = (h << 2) | (l << 1) | m
    elif size == 2:
        idx = (h << 1) | l
        rm |= m << 4
    else:
        raise Undefined("by element size")
    up = qs
    if not u and opcode in (12, 13) or (u and opcode == 13 and False):  # SQDMULH / SQRDMULH
        t = ST[size]
        body = f"vqdmulh<{t}>(x, y, {'true' if opcode == 13 else 'false'})"
        if scalar:
            return [f"{{ {t} x = lane_get<{t}>(c->v[{rn}], 0), y = lane_get<{t}>(c->v[{rm}], {idx}); "
                    f"c->v[{rd}] = lane_set<{t}>(V{{0, 0}}, 0, ({t})({body})); }}"]
        return [f"c->v[{rd}] = vmap_elem<{t}>(c->v[{rd}], c->v[{rn}], c->v[{rm}], {idx}, {qs}, "
                f"[]({t} d, {t} x, {t} y) {{ return ({t})({body}); }});"]
    if scalar:
        if u and opcode in (13, 15):
            raise Unimplemented("sqrdmlah (FEAT_RDM)")
        if u or opcode not in (3, 7, 11):
            raise Undefined("scalar by element")
        ts, td = ST[size], ST[size + 1]
        body = {3: f"sat<{td}>((__int128)d + sat<{td}>(2 * (__int128)x * y))",
                7: f"sat<{td}>((__int128)d - sat<{td}>(2 * (__int128)x * y))",
                11: f"sat<{td}>(2 * (__int128)x * y)"}[opcode]
        return [f"{{ {ts} x = lane_get<{ts}>(c->v[{rn}], 0), y = lane_get<{ts}>(c->v[{rm}], {idx}); "
                f"{td} d = lane_get<{td}>(c->v[{rd}], 0); "
                f"c->v[{rd}] = lane_set<{td}>(V{{0, 0}}, 0, ({td})({body})); }}"]
    if opcode in (0, 4, 8) and (u or opcode == 8):  # MLA / MLS / MUL
        if opcode == 8 and u:
            raise Unimplemented("fmlal2 by element (FEAT_FHM)")
        t = UT[size]
        body = {0: "d + x * y", 4: "d - x * y", 8: "x * y"}[opcode]
        return [f"c->v[{rd}] = vmap_elem<{t}>(c->v[{rd}], c->v[{rn}], c->v[{rm}], {idx}, {qs}, "
                f"[]({t} d, {t} x, {t} y) {{ return ({t})({body}); }});"]
    long_ops = {2: "d + ({TD})x * ({TD})y", 6: "d - ({TD})x * ({TD})y", 10: "({TD})x * ({TD})y"}
    if opcode in long_ops:
        ts, td = it(size, not u), it(size + 1, not u)
        body = long_ops[opcode].replace("{TD}", td)
        return [f"c->v[{rd}] = vwiden_elem<{ts}, {td}>(c->v[{rd}], c->v[{rn}], c->v[{rm}], {idx}, {up}, "
                f"[]({td} d, {ts} x, {ts} y) {{ return ({td})({body}); }});"]
    if not u and opcode in (3, 7, 11):
        ts, td = ST[size], ST[size + 1]
        body = {3: f"sat<{td}>((__int128)d + sat<{td}>(2 * (__int128)x * y))",
                7: f"sat<{td}>((__int128)d - sat<{td}>(2 * (__int128)x * y))",
                11: f"sat<{td}>(2 * (__int128)x * y)"}[opcode]
        return [f"c->v[{rd}] = vwiden_elem<{ts}, {td}>(c->v[{rd}], c->v[{rn}], c->v[{rm}], {idx}, {up}, "
                f"[]({td} d, {ts} x, {ts} y) {{ return ({td})({body}); }});"]
    if u and opcode in (13, 15):
        raise Unimplemented("sqrdmlah (FEAT_RDM)")
    if opcode == 14:
        raise Unimplemented("dot product (FEAT_DotProd)")
    if (u and opcode in (1, 3, 5, 7, 8, 12)) or (not u and opcode in (0, 4)):
        raise Unimplemented("fcmla / fmlal by element (FEAT_FCMA, FEAT_FHM)")
    raise Undefined("by element opcode")


# ---------------------------------------------------------------- permutes, ext, tbl

def permute(w):
    rd, rn, rm, q, _, size = fields(w)
    opcode = (w >> 12) & 7
    if size == 3 and not q:
        raise Undefined("permute 1d")
    t = UT[size]
    qs = "true" if q else "false"
    fn = {1: ("vuzp", 0), 2: ("vtrn", 0), 3: ("vzip", 0), 5: ("vuzp", 1), 6: ("vtrn", 1), 7: ("vzip", 1)}.get(opcode)
    if fn is None:
        raise Undefined("permute opcode")
    return [f"c->v[{rd}] = {fn[0]}<{t}>(c->v[{rn}], c->v[{rm}], {qs}, {fn[1]});"]


def extract(w):
    rd, rn, rm, q, _, op2 = fields(w)
    imm4 = (w >> 11) & 15
    if op2 or (not q and imm4 & 8):
        raise Undefined("ext")
    return [f"c->v[{rd}] = vext(c->v[{rn}], c->v[{rm}], {imm4}, {'true' if q else 'false'});"]


def table_lookup(w):
    rd, rn, rm, q, _, op2 = fields(w)
    length, tbx = (w >> 13) & 3, (w >> 12) & 1
    if op2:
        raise Undefined("tbl op2")
    regs = ", ".join(f"c->v[{(rn + i) % 32}]" for i in range(length + 1))
    return [f"{{ V t_[{length + 1}] = {{{regs}}}; c->v[{rd}] = vtbl(t_, {length + 1}, c->v[{rm}], c->v[{rd}], "
            f"{'true' if q else 'false'}, {'true' if tbx else 'false'}); }}"]


# ---------------------------------------------------------------- crypto

def crypto_aes(w):
    if (w >> 22) & 3:
        raise Undefined("crypto size")
    rd, rn = w & 31, (w >> 5) & 31
    opcode = (w >> 12) & 31
    fn = {4: f"aese(c->v[{rd}], c->v[{rn}])", 5: f"aesd(c->v[{rd}], c->v[{rn}])", 6: f"aesmc(c->v[{rn}])",
          7: f"aesimc(c->v[{rn}])"}.get(opcode)
    if fn is None:
        raise Undefined("aes opcode")
    return [f"c->v[{rd}] = {fn};"]


def crypto_sha3(w):
    if (w >> 22) & 3:
        raise Undefined("crypto size")
    rd, rn, rm = w & 31, (w >> 5) & 31, (w >> 16) & 31
    opcode = (w >> 12) & 7
    d, n, m = f"c->v[{rd}]", f"c->v[{rn}]", f"c->v[{rm}]"
    fn = {0: f"sha1c({d}, (uint32_t){n}.lo, {m})", 1: f"sha1p({d}, (uint32_t){n}.lo, {m})",
          2: f"sha1m({d}, (uint32_t){n}.lo, {m})", 3: f"sha1su0({d}, {n}, {m})",
          4: f"sha256h({d}, {n}, {m}, false)", 5: f"sha256h({d}, {n}, {m}, true)",
          6: f"sha256su1({d}, {n}, {m})"}.get(opcode)
    if fn is None:
        raise Undefined("sha3 opcode")
    return [f"c->v[{rd}] = {fn};"]


def crypto_sha2(w):
    if (w >> 22) & 3:
        raise Undefined("crypto size")
    rd, rn = w & 31, (w >> 5) & 31
    opcode = (w >> 12) & 31
    if opcode == 0:
        return [f"c->v[{rd}] = V{{(uint64_t)sha1h((uint32_t)c->v[{rn}].lo), 0}};"]
    if opcode == 1:
        return [f"c->v[{rd}] = sha1su1(c->v[{rd}], c->v[{rn}]);"]
    if opcode == 2:
        return [f"c->v[{rd}] = sha256su0(c->v[{rd}], c->v[{rn}]);"]
    raise Undefined("sha2 opcode")


# ---------------------------------------------------------------- structure loads / stores

def ls_structure(w):
    q, load = (w >> 30) & 1, (w >> 22) & 1
    rt, rn, rm = w & 31, (w >> 5) & 31, (w >> 16) & 31
    post = (w >> 23) & 1
    single = (w >> 24) & 1
    if not post and rm:
        raise Undefined("structure: Rm must be 0 without post-index")
    base = rsp(rn, 1)
    if not single:
        opcode, size = (w >> 12) & 15, (w >> 10) & 3
        layout = {0: (4, 1), 2: (1, 4), 4: (3, 1), 6: (1, 3), 7: (1, 1), 8: (2, 1), 10: (1, 2)}.get(opcode)
        if layout is None or (w >> 21) & 1:
            raise Undefined("structure opcode")
        selem, rpt = layout
        if size == 3 and not q and selem > 1:
            raise Undefined("structure 1d")
        t = UT[size]
        esize = 1 << size
        elems = (16 if q else 8) // esize
        nregs = selem * rpt
        regs = [(rt + i) % 32 for i in range(nregs)]
        lines = [f"uint64_t a_ = {base};"]
        if load:
            lines.append(f"V t_[{nregs}] = {{{', '.join('V{0, 0}' for _ in regs)}}};")
        else:
            lines.append(f"V t_[{nregs}] = {{{', '.join(f'c->v[{r}]' for r in regs)}}};")
        lines.append(f"uint64_t o_ = 0;")
        lines.append(f"for (int r_ = 0; r_ < {rpt}; r_++) for (int e_ = 0; e_ < {elems}; e_++) "
                     f"for (int s_ = 0; s_ < {selem}; s_++) {{ ")
        if load:
            lines.append(f"t_[r_ * {selem} + s_] = lane_set<{t}>(t_[r_ * {selem} + s_], e_, ld<{t}>(m, a_ + o_)); ")
        else:
            lines.append(f"st<{t}>(m, a_ + o_, lane_get<{t}>(t_[r_ * {selem} + s_], e_)); ")
        lines.append(f"o_ += {esize}; }}")
        if load:
            lines += [f"c->v[{r}] = t_[{i}];" for i, r in enumerate(regs)]
        total = nregs * (16 if q else 8)
    else:
        opcode, s, size, r = (w >> 13) & 7, (w >> 12) & 1, (w >> 10) & 3, (w >> 21) & 1
        selem = (((opcode & 1) << 1) | r) + 1
        scale = opcode >> 1
        if scale == 3:  # LDnR
            if not load or s:
                raise Undefined("ldnr")
            t, esize = UT[size], 1 << size
            regs = [(rt + i) % 32 for i in range(selem)]
            qs = "true" if q else "false"
            lines = [f"uint64_t a_ = {base};"]
            lines += [f"c->v[{reg}] = vdup<{t}>(ld<{t}>(m, a_ + {i * esize}), {qs});" for i, reg in enumerate(regs)]
            total = selem * esize
        else:
            if scale == 0:
                index, esize = (q << 3) | (s << 2) | size, 1
            elif scale == 1:
                if size & 1:
                    raise Undefined("single half size")
                index, esize = (q << 2) | (s << 1) | (size >> 1), 2
            elif size == 0:
                index, esize = (q << 1) | s, 4
            elif size == 1 and not s:
                index, esize = q, 8
            else:
                raise Undefined("single word/double size")
            t = UT[{1: 0, 2: 1, 4: 2, 8: 3}[esize]]
            regs = [(rt + i) % 32 for i in range(selem)]
            lines = [f"uint64_t a_ = {base};"]
            for i, reg in enumerate(regs):
                if load:
                    lines.append(f"c->v[{reg}] = lane_set<{t}>(c->v[{reg}], {index}, ld<{t}>(m, a_ + {i * esize}));")
                else:
                    lines.append(f"st<{t}>(m, a_ + {i * esize}, lane_get<{t}>(c->v[{reg}], {index}));")
            total = selem * esize
    if post:
        target = "c->sp" if rn == 31 else f"c->x[{rn}]"
        inc = f"{total}" if rm == 31 else rx(rm, 1)
        lines.append(f"{target} = a_ + {inc};")
    return ["{ " + " ".join(lines) + " }"]
