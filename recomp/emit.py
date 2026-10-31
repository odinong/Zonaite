"""AArch64 -> C++ instruction emitter.

emit(word, pc, flow) returns C++ statements for one instruction. Control transfers go through `flow`, so the same
semantics serve generated functions (goto / calls / musttail) and single-instruction tests (store the next PC).
Decoding follows the ARM ARM's top-level groups; anything unallocated raises Undefined, anything not written yet
raises Unimplemented, so coverage gaps are explicit.
"""
from . import a64


class Undefined(Exception):
    """Unallocated encoding: executes as a guest trap."""


class Unimplemented(Exception):
    """Valid encoding the emitter does not handle yet."""


# ---------------------------------------------------------------- control flow back ends

class TestFlow:
    """Single-instruction tests: every control transfer stores the next PC and returns."""

    def jump(self, target):
        return f"c->pc = {target:#x}ull; return;"

    def cond_jump(self, cond, target):
        return f"if ({cond}) {{ c->pc = {target:#x}ull; return; }}"

    def call(self, target, ret):
        return f"c->x[30] = {ret:#x}ull; c->pc = {target:#x}ull; return;"

    def call_indirect(self, expr, ret):
        return f"{{ uint64_t t_ = {expr}; c->x[30] = {ret:#x}ull; c->pc = t_; return; }}"

    def branch_indirect(self, expr, pc):
        return f"c->pc = {expr}; return;"

    def ret(self, expr):
        return f"c->pc = {expr}; return;"

    def trap(self, pc, word):
        return f"guest_trap(c, m, {pc:#x}ull, {word:#x}u); c->pc = {pc + 4:#x}ull; return;"


# ---------------------------------------------------------------- operand helpers

MASK = {32: 0xFFFFFFFF, 64: 0xFFFFFFFFFFFFFFFF}


def U(sf):
    return "uint64_t" if sf else "uint32_t"


def S(sf):
    return "int64_t" if sf else "int32_t"


def rx(n, sf):
    """Read X/W register n, register 31 = zero."""
    if n == 31:
        return "0ull" if sf else "0u"
    return f"c->x[{n}]" if sf else f"(uint32_t)c->x[{n}]"


def rsp(n, sf):
    """Read X/W register n, register 31 = SP."""
    if n == 31:
        return "c->sp" if sf else "(uint32_t)c->sp"
    return rx(n, sf)


def wx(n, sf, expr):
    """Write X/W register n (32-bit writes zero-extend), register 31 = discard."""
    if n == 31:
        return f"(void)({expr});"
    return f"c->x[{n}] = (uint64_t)({U(sf)})({expr});"


def wsp(n, sf, expr):
    if n == 31:
        return f"c->sp = (uint64_t)({U(sf)})({expr});"
    return wx(n, sf, expr)


def lit(v, sf):
    return f"{v & MASK[64 if sf else 32]:#x}{'ull' if sf else 'u'}"


def shifted(n, sf, shift, amount):
    v = rx(n, sf)
    if amount == 0:
        return v
    bits = 64 if sf else 32
    if shift == 0:
        return f"(({U(sf)})({v} << {amount}))"
    if shift == 1:
        return f"(({U(sf)})({v} >> {amount}))"
    if shift == 2:
        return f"(({U(sf)})(({S(sf)}){v} >> {amount}))"
    return f"ror{bits}({v}, {amount})"


def extended(n, option, amount, sf):
    """Rm extended by option (UXTB..SXTX) then shifted left; result has the operation's width."""
    base = rx(n, 1)
    e = {0: f"(uint64_t)(uint8_t){base}", 1: f"(uint64_t)(uint16_t){base}", 2: f"(uint64_t)(uint32_t){base}",
         3: base, 4: f"(uint64_t)(int64_t)(int8_t){base}", 5: f"(uint64_t)(int64_t)(int16_t){base}",
         6: f"(uint64_t)(int64_t)(int32_t){base}", 7: base}[option]
    if amount:
        e = f"({e} << {amount})"
    return f"(({U(sf)}){e})"


# ---------------------------------------------------------------- entry point

def emit(w, pc, flow):
    op0 = (w >> 25) & 0xF
    if op0 in (0b1000, 0b1001):
        return dp_imm(w, pc, flow)
    if op0 in (0b1010, 0b1011):
        return branch_sys(w, pc, flow)
    if op0 & 0b0101 == 0b0100:
        return loadstore(w, pc, flow)
    if op0 in (0b0101, 0b1101):
        return dp_reg(w, pc, flow)
    if op0 in (0b0111, 0b1111):
        from . import emit_simd
        return emit_simd.emit(w, pc, flow)
    raise Undefined(f"{w:#010x}")


# ---------------------------------------------------------------- data processing, immediate

def dp_imm(w, pc, flow):
    sf = w >> 31
    rd, rn = w & 31, (w >> 5) & 31
    op = (w >> 23) & 7
    if op in (0, 1):  # ADR / ADRP
        imm = a64.sext(((w >> 3) & 0x1FFFFC) | ((w >> 29) & 3), 21)
        if w >> 31:
            value = ((pc & ~0xFFF) + (imm << 12)) & MASK[64]
        else:
            value = (pc + imm) & MASK[64]
        return [wx(rd, 1, lit(value, 1))]
    if op == 2:  # add/sub immediate
        sub, setf = (w >> 30) & 1, (w >> 29) & 1
        imm = (w >> 10) & 0xFFF
        if (w >> 22) & 1:
            imm <<= 12
        a = rsp(rn, sf)
        bits = 64 if sf else 32
        if setf:
            b = lit(~imm if sub else imm, sf)
            return [wx(rd, sf, f"add{bits}f(c, {a}, {b}, {1 if sub else 0})")]
        return [wsp(rd, sf, f"{a} {'-' if sub else '+'} {lit(imm, sf)}")]
    if op == 4:  # logical immediate
        n = (w >> 22) & 1
        if not sf and n:
            raise Undefined("logical imm N=1 with sf=0")
        mask = a64.decode_bit_masks(n, (w >> 10) & 0x3F, (w >> 16) & 0x3F, 64 if sf else 32)
        if mask is None:
            raise Undefined("logical imm reserved pattern")
        opc = (w >> 29) & 3
        a = rx(rn, sf)
        expr = {0: f"{a} & {lit(mask, sf)}", 1: f"{a} | {lit(mask, sf)}", 2: f"{a} ^ {lit(mask, sf)}",
                3: f"{a} & {lit(mask, sf)}"}[opc]
        if opc == 3:
            return [f"{{ {U(sf)} r_ = {expr}; logic{64 if sf else 32}f(c, r_); {wx(rd, sf, 'r_')} }}"]
        return [wsp(rd, sf, expr)]
    if op == 5:  # move wide
        opc, hw = (w >> 29) & 3, (w >> 21) & 3
        if opc == 1 or (not sf and hw >= 2):
            raise Undefined("move wide")
        imm = ((w >> 5) & 0xFFFF) << (16 * hw)
        if opc == 0:
            return [wx(rd, sf, lit(~imm, sf))]
        if opc == 2:
            return [wx(rd, sf, lit(imm, sf))]
        keep = ~(0xFFFF << (16 * hw))
        return [wx(rd, sf, f"({rx(rd, sf)} & {lit(keep, sf)}) | {lit(imm, sf)}")]
    if op == 6:  # bitfield
        opc, n = (w >> 29) & 3, (w >> 22) & 1
        if opc == 3 or n != sf:
            raise Undefined("bitfield")
        r, s = (w >> 16) & 0x3F, (w >> 10) & 0x3F
        bits = 64 if sf else 32
        if not sf and (r >= 32 or s >= 32):
            raise Undefined("bitfield 32-bit immr/imms")
        return [bitfield(opc, sf, bits, r, s, rd, rn)]
    if op == 7:  # extract
        if (w >> 29) & 3 or (w >> 21) & 1 or ((w >> 22) & 1) != sf:
            raise Undefined("extr")
        rm, lsb = (w >> 16) & 31, (w >> 10) & 0x3F
        bits = 64 if sf else 32
        if not sf and lsb >= 32:
            raise Undefined("extr lsb")
        if lsb == 0:
            return [wx(rd, sf, rx(rm, sf))]
        return [wx(rd, sf, f"({rx(rm, sf)} >> {lsb}) | ({U(sf)})({rx(rn, sf)} << {bits - lsb})")]
    raise Undefined("dp imm")


def bitfield(opc, sf, bits, r, s, rd, rn):
    src = rx(rn, sf)
    t = U(sf)
    if s >= r:  # extract bits s..r to the bottom
        width = s - r + 1
        fmask = lit((1 << width) - 1, sf)
        field = f"(({src} >> {r}) & {fmask})"
        if opc == 0:  # SBFM: sign-extend the field
            sh = bits - width
            return wx(rd, sf, f"({t})(({S(sf)})({t})({field} << {sh}) >> {sh})")
        if opc == 2:  # UBFM
            return wx(rd, sf, field)
        return wx(rd, sf, f"({rx(rd, sf)} & {lit(~((1 << width) - 1), sf)}) | {field}")  # BFM (BFXIL)
    width, pos = s + 1, bits - r  # field of s+1 bits moved up to bit (bits - r)
    fmask = lit((1 << width) - 1, sf)
    if opc == 0:
        sh = bits - width
        sext = f"(({S(sf)})({t})({src} << {sh}) >> {sh})"
        return wx(rd, sf, f"({t})((({t}){sext}) << {pos})")
    if opc == 2:
        return wx(rd, sf, f"({t})(({src} & {fmask}) << {pos})")
    keep = lit(~(((1 << width) - 1) << pos), sf)
    return wx(rd, sf, f"({rx(rd, sf)} & {keep}) | ({t})(({src} & {fmask}) << {pos})")  # BFM (BFI)


# ---------------------------------------------------------------- branches, exceptions, system

SYSREG_NZCV, SYSREG_FPCR, SYSREG_FPSR = 0xDA10, 0xDA20, 0xDA21
SYSREG_TPIDR, SYSREG_TPIDRRO = 0xDE82, 0xDE83


def branch_sys(w, pc, flow):
    if w & 0x7C000000 == 0x14000000:  # B / BL
        target = (pc + (a64.sext(w & 0x03FFFFFF, 26) << 2)) & MASK[64]
        if w >> 31:
            return [flow.call(target, pc + 4)]
        return [flow.jump(target)]
    if w & 0x7E000000 == 0x34000000:  # CBZ / CBNZ
        sf, nz, rt = w >> 31, (w >> 24) & 1, w & 31
        target = (pc + (a64.sext((w >> 5) & 0x7FFFF, 19) << 2)) & MASK[64]
        return [flow.cond_jump(f"{rx(rt, sf)} {'!=' if nz else '=='} 0", target)]
    if w & 0x7E000000 == 0x36000000:  # TBZ / TBNZ
        bit = ((w >> 31) << 5) | ((w >> 19) & 31)
        nz, rt = (w >> 24) & 1, w & 31
        target = (pc + (a64.sext((w >> 5) & 0x3FFF, 14) << 2)) & MASK[64]
        return [flow.cond_jump(f"(({rx(rt, 1)} >> {bit}) & 1) {'!=' if nz else '=='} 0", target)]
    if w & 0xFF000010 == 0x54000000:  # B.cond
        target = (pc + (a64.sext((w >> 5) & 0x7FFFF, 19) << 2)) & MASK[64]
        code = w & 15
        if code >= 14:
            return [flow.jump(target)]
        return [flow.cond_jump(f"cond(c, {code})", target)]
    if w & 0xFE000000 == 0xD6000000:  # branch to register
        opc, op2, op3, op4 = (w >> 21) & 15, (w >> 16) & 31, (w >> 10) & 0x3F, w & 31
        rn = (w >> 5) & 31
        if op2 != 31 or op3 != 0 or op4 != 0 or opc > 2:
            raise Undefined("branch register (PAC/ERET)")
        if opc == 0:
            return [flow.branch_indirect(rx(rn, 1), pc)]
        if opc == 1:
            return [flow.call_indirect(rx(rn, 1), pc + 4)]
        return [flow.ret(rx(rn, 1))]
    if w & 0xFF000000 == 0xD4000000:  # exception generation
        opc, op2, ll = (w >> 21) & 7, (w >> 2) & 7, w & 3
        imm = (w >> 5) & 0xFFFF
        if op2 == 0 and opc == 0 and ll == 1:
            return [f"guest_svc(c, m, {imm:#x}u);"]
        if op2 == 0 and opc in (1, 2) and ll == 0:  # BRK / HLT
            return [f"guest_brk(c, m, {pc:#x}ull, {imm:#x}u);", flow.trap(pc, w)]
        raise Undefined("exception generation")
    if w & 0xFFC00000 == 0xD5000000:
        return system(w, pc, flow)
    raise Undefined("branch/system")


def system(w, pc, flow):
    l, op0, op1 = (w >> 21) & 1, (w >> 19) & 3, (w >> 16) & 7
    crn, crm, op2, rt = (w >> 12) & 15, (w >> 8) & 15, (w >> 5) & 7, w & 31
    if op0 == 0 and l == 0 and crn == 2 and rt == 31:
        return ["/* hint */"]
    if op0 == 0 and l == 0 and crn == 3 and rt == 31:
        if op2 == 2:
            return ["clrex(c);"]
        if op2 in (4, 5):
            return ["std::atomic_thread_fence(std::memory_order_seq_cst);"]
        if op2 == 6 or (op2 == 7 and crm == 0):
            return ["/* isb / sb */"]
        raise Undefined("barrier")
    if op0 == 0 and l == 0 and crn == 4 and rt == 31:
        raise Unimplemented("msr (immediate)")
    enc = (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
    if op0 == 1:  # SYS / SYSL (DC ZVA, DC CIVAC, IC IVAU, ...)
        if l:
            raise Undefined("sysl")
        return [f"sys_op(c, m, {enc:#x}u, {rx(rt, 1)});"]
    if op0 >= 2:
        if l:  # MRS
            if enc == SYSREG_NZCV:
                return [wx(rt, 1, "get_nzcv(c)")]
            if enc == SYSREG_FPCR:
                return [wx(rt, 1, "(uint64_t)c->fpcr")]
            if enc == SYSREG_FPSR:
                return [wx(rt, 1, "(uint64_t)c->fpsr")]
            if enc == SYSREG_TPIDR:
                return [wx(rt, 1, "c->tpidr_el0")]
            if enc == SYSREG_TPIDRRO:
                return [wx(rt, 1, "c->tpidrro_el0")]
            return [wx(rt, 1, f"sysreg_read(c, {enc:#x}u)")]
        v = rx(rt, 1)
        if enc == SYSREG_NZCV:
            return [f"set_nzcv(c, (uint32_t)({v} >> 28));"]
        if enc == SYSREG_FPCR:
            return [f"c->fpcr = (uint32_t){v};"]
        if enc == SYSREG_FPSR:
            return [f"c->fpsr = (uint32_t){v};"]
        if enc == SYSREG_TPIDR:
            return [f"c->tpidr_el0 = {v};"]
        return [f"sysreg_write(c, {enc:#x}u, {v});"]
    raise Undefined("system")


# ---------------------------------------------------------------- loads and stores

INT_TYPES = {0: ("uint8_t", "int8_t", 8), 1: ("uint16_t", "int16_t", 16), 2: ("uint32_t", "int32_t", 32),
             3: ("uint64_t", "int64_t", 64)}


def vreg_write(t, nbytes, expr):
    """Write a scalar of nbytes into V[t], zeroing the rest of the register."""
    if nbytes == 16:
        return f"c->v[{t}] = {expr};"
    return f"c->v[{t}] = V{{(uint64_t)({expr}), 0}};"


def vreg_read(t, nbytes):
    if nbytes == 16:
        return f"c->v[{t}]"
    return {1: f"(uint8_t)c->v[{t}].lo", 2: f"(uint16_t)c->v[{t}].lo", 4: f"(uint32_t)c->v[{t}].lo",
            8: f"c->v[{t}].lo"}[nbytes]


def ld_expr(nbytes, addr):
    return {1: f"ld8(m, {addr})", 2: f"ld16(m, {addr})", 4: f"ld32(m, {addr})", 8: f"ld64(m, {addr})",
            16: f"ld128(m, {addr})"}[nbytes]


def st_stmt(nbytes, addr, value):
    return {1: f"st8(m, {addr}, (uint8_t)({value}));", 2: f"st16(m, {addr}, (uint16_t)({value}));",
            4: f"st32(m, {addr}, (uint32_t)({value}));", 8: f"st64(m, {addr}, (uint64_t)({value}));",
            16: f"st128(m, {addr}, {value});"}[nbytes]


def single_transfer(kind, rt, nbytes, signed_to, vector, addr):
    """Statements for one load/store of register rt at C expression addr (already evaluated into a variable)."""
    if kind == "prfm":
        return []
    if vector:
        if kind == "load":
            return [vreg_write(rt, nbytes, ld_expr(nbytes, addr))]
        return [st_stmt(nbytes, addr, vreg_read(rt, nbytes))]
    if kind == "store":
        return [st_stmt(nbytes, addr, rx(rt, 1))]
    value = ld_expr(nbytes, addr)
    if signed_to:
        signed = {1: "int8_t", 2: "int16_t", 4: "int32_t"}[nbytes]
        return [wx(rt, signed_to == 64, f"({U(signed_to == 64)})(int64_t)({signed}){value}")]
    return [wx(rt, 1, f"(uint64_t){value}")]


def ls_kind(size, vector, opc):
    """(kind, bytes, sign-extend-to bits) for the single-register load/store forms."""
    if vector:
        if opc in (0, 1):
            return ("store" if opc == 0 else "load"), 1 << size, None
        if size == 0:
            return ("store" if opc == 2 else "load"), 16, None
        raise Undefined("vector ldr/str")
    nbytes = 1 << size
    if opc == 0:
        return "store", nbytes, None
    if opc == 1:
        return "load", nbytes, None
    if opc == 2:
        if size == 3:
            return "prfm", 8, None
        return "load", nbytes, 64
    if size >= 2:
        raise Undefined("ldrs* to 32 bits of a word")
    return "load", nbytes, 32


def with_writeback(rn, base_expr, offset, pre, body):
    """addr = base (+ offset if pre); body; base += offset (writeback)."""
    out = [f"uint64_t a_ = {base_expr}{' + ' + offset if pre and offset != '0' else ''};"]
    out += body
    target = "c->sp" if rn == 31 else f"c->x[{rn}]"
    new = "a_" if pre else f"a_ + {offset}"
    out.append(f"{target} = {new};")
    return ["{ " + " ".join(out) + " }"]


def loadstore(w, pc, flow):
    rt, rn = w & 31, (w >> 5) & 31
    # load/store exclusive, load-acquire/store-release, compare-and-swap
    if w & 0x3F000000 == 0x08000000:
        return ls_exclusive(w, pc)
    # load register (literal)
    if w & 0x3B000000 == 0x18000000:
        opc, vector = w >> 30, (w >> 26) & 1
        addr = (pc + (a64.sext((w >> 5) & 0x7FFFF, 19) << 2)) & MASK[64]
        if vector:
            if opc == 3:
                raise Undefined("ldr literal vector")
            nbytes = {0: 4, 1: 8, 2: 16}[opc]
            return [vreg_write(rt, nbytes, ld_expr(nbytes, lit(addr, 1)))]
        if opc == 3:
            return ["/* prfm literal */"]
        if opc == 2:
            return [wx(rt, 1, f"(uint64_t)(int64_t)(int32_t)ld32(m, {lit(addr, 1)})")]
        return [wx(rt, 1, f"(uint64_t){ld_expr(4 if opc == 0 else 8, lit(addr, 1))}")]
    # load/store pair
    if w & 0x3A000000 == 0x28000000:
        return ls_pair(w)
    # SIMD structure loads/stores
    if w & 0xBF000000 in (0x0C000000, 0x0D000000):
        from . import emit_simd
        return emit_simd.ls_structure(w)
    # load/store register: immediate, register offset, atomics
    if w & 0x3B000000 in (0x38000000, 0x39000000):
        size, vector, opc = w >> 30, (w >> 26) & 1, (w >> 22) & 3
        base = rsp(rn, 1)
        if w & 0x01000000:  # unsigned scaled offset
            kind, nbytes, signed_to = ls_kind(size, vector, opc)
            off = ((w >> 10) & 0xFFF) * nbytes
            body = single_transfer(kind, rt, nbytes, signed_to, vector, "a_")
            return ["{ uint64_t a_ = " + base + (f" + {off:#x}" if off else "") + "; " + " ".join(body) + " }"]
        if not (w >> 21) & 1:  # 9-bit signed immediate forms
            mode = (w >> 10) & 3
            imm = a64.sext((w >> 12) & 0x1FF, 9)
            kind, nbytes, signed_to = ls_kind(size, vector, opc)
            if mode == 2 and vector:
                raise Undefined("ldtr vector")
            if kind == "prfm" and mode != 0:
                raise Undefined("prfm only has the unscaled immediate form")
            off = f"{imm & MASK[64]:#x}ull"
            body = single_transfer(kind, rt, nbytes, signed_to, vector, "a_")
            if mode in (0, 2):  # unscaled (LDUR) / unprivileged (LDTR: same at EL0)
                return ["{ uint64_t a_ = " + base + f" + {off}; " + " ".join(body) + " }"]
            return with_writeback(rn, base, off, mode == 3, body)
        if (w >> 10) & 3 == 2:  # register offset
            option, s = (w >> 13) & 7, (w >> 12) & 1
            if not option & 2:
                raise Undefined("register offset option")
            kind, nbytes, signed_to = ls_kind(size, vector, opc)
            scale = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[nbytes]
            index = extended((w >> 16) & 31, option, scale if s else 0, 1)
            body = single_transfer(kind, rt, nbytes, signed_to, vector, "a_")
            return ["{ uint64_t a_ = " + base + f" + {index}; " + " ".join(body) + " }"]
        if (w >> 10) & 3 == 0 and not vector:
            return ls_atomic(w)
        raise Undefined("load/store register")
    raise Undefined("load/store")


def ls_pair(w):
    opc, vector, mode, load = w >> 30, (w >> 26) & 1, (w >> 23) & 3, (w >> 22) & 1
    rt, rn, rt2 = w & 31, (w >> 5) & 31, (w >> 10) & 31
    imm7 = a64.sext((w >> 15) & 0x7F, 7)
    if vector:
        if opc == 3:
            raise Undefined("ldp vector opc")
        nbytes, signed = 4 << opc, False
    else:
        if opc == 3 or (opc == 1 and not load):
            raise Undefined("ldp/stp opc")  # STGP (MTE) is not available
        if opc == 1 and mode == 0:
            raise Undefined("ldnp sw")
        nbytes, signed = (4 if opc < 2 else 8), opc == 1
    off = (imm7 * nbytes) & MASK[64]
    base = rsp(rn, 1)
    body = []
    if load:
        if vector:
            body += [f"auto t1_ = {ld_expr(nbytes, 'a_')}; auto t2_ = {ld_expr(nbytes, f'a_ + {nbytes}')};",
                     vreg_write(rt, nbytes, "t1_"), vreg_write(rt2, nbytes, "t2_")]
        else:
            conv = "(uint64_t)(int64_t)(int32_t)" if signed else "(uint64_t)"
            body += [f"uint64_t t1_ = {conv}{ld_expr(nbytes, 'a_')}; "
                     f"uint64_t t2_ = {conv}{ld_expr(nbytes, f'a_ + {nbytes}')};",
                     wx(rt, 1, "t1_"), wx(rt2, 1, "t2_")]
    else:
        if vector:
            body += [st_stmt(nbytes, "a_", vreg_read(rt, nbytes)),
                     st_stmt(nbytes, f"a_ + {nbytes}", vreg_read(rt2, nbytes))]
        else:
            body += [f"uint64_t t1_ = {rx(rt, 1)}; uint64_t t2_ = {rx(rt2, 1)};",
                     st_stmt(nbytes, "a_", "t1_"), st_stmt(nbytes, f"a_ + {nbytes}", "t2_")]
    if mode in (0, 2):  # no-allocate / signed offset
        return ["{ uint64_t a_ = " + base + (f" + {off:#x}ull" if off else "") + "; " + " ".join(body) + " }"]
    return with_writeback(rn, base, f"{off:#x}ull", mode == 3, body)


def ls_exclusive(w, pc):
    size, o2, l, o1, o0 = w >> 30, (w >> 23) & 1, (w >> 22) & 1, (w >> 21) & 1, (w >> 15) & 1
    rs, rt2, rn, rt = (w >> 16) & 31, (w >> 10) & 31, (w >> 5) & 31, w & 31
    nbytes = 1 << size
    utype = INT_TYPES[size][0]
    addr = rsp(rn, 1)
    fence = "std::atomic_thread_fence(std::memory_order_seq_cst);"
    if o2 == 0 and o1 == 0:  # LDXR/LDAXR, STXR/STLXR
        if l:
            out = [wx(rt, 1, f"(uint64_t)ldx<{utype}>(c, m, {addr})")]
            return out + ([fence] if o0 else [])
        return ([fence] if o0 else []) + [
            wx(rs, 0, f"stx<{utype}>(c, m, {addr}, ({utype}){rx(rt, 1)})")]
    if o2 == 0 and o1 == 1:
        if size >= 2:  # LDXP/LDAXP, STXP/STLXP
            bits = 32 if size == 2 else 64
            if l:
                out = ["{ uint64_t t1_, t2_; " + f"ldxp{bits}(c, m, {addr}, t1_, t2_); "
                       + wx(rt, 1, "t1_") + " " + wx(rt2, 1, "t2_") + " }"]
                return out + ([fence] if o0 else [])
            return ([fence] if o0 else []) + [
                wx(rs, 0, f"stxp{bits}(c, m, {addr}, {rx(rt, 1)}, {rx(rt2, 1)})")]
        if rt2 != 31 or rs & 1 or rt & 1:
            raise Undefined("casp registers")
        # CASP: compare (Rs, Rs+1) with the memory pair, store (Rt, Rt+1) if equal; Rs pair gets the old values
        if size == 0:
            return ["{ " + f"uint64_t e_ = (uint64_t)(uint32_t){rx(rs, 1)} | ((uint64_t)(uint32_t){rx(rs + 1, 1)} << 32); "
                    f"aref<uint64_t>(m, {addr}).compare_exchange_strong(e_, (uint64_t)(uint32_t){rx(rt, 1)} | "
                    f"((uint64_t)(uint32_t){rx(rt + 1, 1)} << 32)); "
                    + wx(rs, 1, "(uint32_t)e_") + " " + wx(rs + 1, 1, "e_ >> 32") + " }"]
        return ["{ " + f"uint64_t e_[2] = {{{rx(rs, 1)}, {rx(rs + 1, 1)}}}; "
                f"cas128(m + {addr}, e_, {rx(rt, 1)}, {rx(rt + 1, 1)}); "
                + wx(rs, 1, "e_[0]") + " " + wx(rs + 1, 1, "e_[1]") + " }"]
    if o2 == 1 and o1 == 0:  # LDAR / STLR (and LORegion variants, same at EL0)
        if l:
            return [wx(rt, 1, f"(uint64_t)aref<{utype}>(m, {addr}).load(std::memory_order_seq_cst)")]
        return [f"aref<{utype}>(m, {addr}).store(({utype}){rx(rt, 1)}, std::memory_order_seq_cst);"]
    if rt2 == 31:  # CAS{A}{L}{B,H}
        return ["{ " + f"{utype} e_ = ({utype}){rx(rs, 1)}; "
                f"aref<{utype}>(m, {addr}).compare_exchange_strong(e_, ({utype}){rx(rt, 1)}); "
                + wx(rs, 1, "(uint64_t)e_") + " }"]
    raise Undefined("exclusive")


def ls_atomic(w):
    size, a, r, rs = w >> 30, (w >> 23) & 1, (w >> 22) & 1, (w >> 16) & 31
    o3, opc, rn, rt = (w >> 15) & 1, (w >> 12) & 7, (w >> 5) & 31, w & 31
    utype, stype, _ = INT_TYPES[size]
    addr = rsp(rn, 1)
    if o3 == 1 and opc == 0:  # SWP
        return ["{ " + f"{utype} o_ = aref<{utype}>(m, {addr}).exchange(({utype}){rx(rs, 1)}); "
                + wx(rt, 1, "(uint64_t)o_") + " }"]
    if o3 == 1 and opc == 4 and rs == 31 and a == 1 and r == 0:  # LDAPR (acquire only)
        return [wx(rt, 1, f"(uint64_t)aref<{utype}>(m, {addr}).load(std::memory_order_seq_cst)")]
    if o3:
        raise Undefined("atomic op")
    op = {0: "o_ + v_", 1: "o_ & ~v_", 2: "o_ ^ v_", 3: "o_ | v_",
          4: f"(({stype})o_ > ({stype})v_ ? o_ : v_)", 5: f"(({stype})o_ < ({stype})v_ ? o_ : v_)",
          6: "(o_ > v_ ? o_ : v_)", 7: "(o_ < v_ ? o_ : v_)"}[opc]
    return ["{ " + f"{utype} v_ = ({utype}){rx(rs, 1)}; auto r_ = aref<{utype}>(m, {addr}); "
            f"{utype} o_ = r_.load(); while (!r_.compare_exchange_weak(o_, ({utype})({op}))) {{}} "
            + wx(rt, 1, "(uint64_t)o_") + " }"]


# ---------------------------------------------------------------- data processing, register

def dp_reg(w, pc, flow):
    sf = w >> 31
    rd, rn, rm = w & 31, (w >> 5) & 31, (w >> 16) & 31
    bits = 64 if sf else 32
    t = U(sf)
    if w & 0x1F000000 == 0x0A000000:  # logical, shifted register
        opc, shift, neg, imm6 = (w >> 29) & 3, (w >> 22) & 3, (w >> 21) & 1, (w >> 10) & 0x3F
        if not sf and imm6 >= 32:
            raise Undefined("logical shift amount")
        b = shifted(rm, sf, shift, imm6)
        if neg:
            b = f"({t})~{b}"
        a = rx(rn, sf)
        expr = {0: f"{a} & {b}", 1: f"{a} | {b}", 2: f"{a} ^ {b}", 3: f"{a} & {b}"}[opc]
        if opc == 3:
            return [f"{{ {t} r_ = {expr}; logic{bits}f(c, r_); {wx(rd, sf, 'r_')} }}"]
        return [wx(rd, sf, expr)]
    if w & 0x1F200000 == 0x0B000000:  # add/sub, shifted register
        sub, setf, shift, imm6 = (w >> 30) & 1, (w >> 29) & 1, (w >> 22) & 3, (w >> 10) & 0x3F
        if shift == 3 or (not sf and imm6 >= 32):
            raise Undefined("add/sub shift")
        return addsub(sf, sub, setf, rx(rn, sf), shifted(rm, sf, shift, imm6), rd, False)
    if w & 0x1F200000 == 0x0B200000:  # add/sub, extended register
        sub, setf, opt = (w >> 30) & 1, (w >> 29) & 1, (w >> 22) & 3
        option, imm3 = (w >> 13) & 7, (w >> 10) & 7
        if opt or imm3 > 4:
            raise Undefined("add/sub extended")
        return addsub(sf, sub, setf, rsp(rn, sf), extended(rm, option, imm3, sf), rd, not setf)
    if w & 0x1FE0FC00 == 0x1A000000:  # ADC / SBC
        sub, setf = (w >> 30) & 1, (w >> 29) & 1
        b = rx(rm, sf)
        if sub:
            b = f"({t})~{b}"
        if setf:
            return [wx(rd, sf, f"add{bits}f(c, {rx(rn, sf)}, {b}, c->c)")]
        return [wx(rd, sf, f"{rx(rn, sf)} + {b} + ({t})c->c")]
    if w & 0x1FE00000 == 0x1A400000:  # conditional compare
        sub, setf = (w >> 30) & 1, (w >> 29) & 1
        if not setf or (w >> 10) & 1 or (w >> 4) & 1:
            raise Undefined("ccmp")
        code, nzcv = (w >> 12) & 15, w & 15
        b = lit(rm, sf) if (w >> 11) & 1 else rx(rm, sf)
        if sub:
            b = f"({t})~{b}"
        return [f"if (cond(c, {code})) (void)add{bits}f(c, {rx(rn, sf)}, {b}, {1 if sub else 0}); "
                f"else set_nzcv(c, {nzcv}u);"]
    if w & 0x1FE00000 == 0x1A800000:  # conditional select
        op, setf, op2 = (w >> 30) & 1, (w >> 29) & 1, (w >> 10) & 3
        if setf or op2 > 1:
            raise Undefined("csel")
        code = (w >> 12) & 15
        b = rx(rm, sf)
        b = {(0, 0): b, (0, 1): f"({t})({b} + 1)", (1, 0): f"({t})~{b}", (1, 1): f"({t})(0 - {b})"}[(op, op2)]
        return [wx(rd, sf, f"cond(c, {code}) ? {rx(rn, sf)} : {b}")]
    if w & 0x5FE00000 == 0x1AC00000:  # data processing, 2 source
        if (w >> 29) & 1:
            raise Undefined("dp2 S")
        opcode = (w >> 10) & 0x3F
        a, b = rx(rn, sf), rx(rm, sf)
        simple = {2: f"udiv{bits}({a}, {b})", 3: f"sdiv{bits}({a}, {b})", 8: f"({t})({a} << ({b} & {bits - 1}))",
                  9: f"({t})({a} >> ({b} & {bits - 1}))", 10: f"asr{bits}({a}, (unsigned)({b} & {bits - 1}))",
                  11: f"ror{bits}({a}, (unsigned)({b} & {bits - 1}))"}
        if opcode in simple:
            return [wx(rd, sf, simple[opcode])]
        if 16 <= opcode <= 23:  # CRC32{B,H,W,X}, CRC32C*
            sz, castagnoli = opcode & 3, bool(opcode & 4)
            if (sz == 3) != bool(sf):
                raise Undefined("crc32 size")
            return [wx(rd, 0, f"crc32_update((uint32_t){rx(rn, 0)}, {rx(rm, 1)}, {1 << sz}, "
                              f"{'true' if castagnoli else 'false'})")]
        raise Undefined("dp2 opcode")
    if w & 0x5FE00000 == 0x5AC00000:  # data processing, 1 source
        if (w >> 29) & 1 or (w >> 16) & 31:
            raise Undefined("dp1")
        opcode = (w >> 10) & 0x3F
        a = rx(rn, sf)
        if opcode == 0:
            return [wx(rd, sf, f"rbit{bits}({a})")]
        if opcode == 1:  # REV16
            m1 = lit(0x00FF00FF00FF00FF, sf)
            return [wx(rd, sf, f"(({a} >> 8) & {m1}) | ({t})(({a} & {m1}) << 8)")]
        if opcode == 2:  # REV (32-bit) / REV32 (64-bit)
            if sf:
                return [wx(rd, 1, f"((uint64_t)__builtin_bswap32((uint32_t)({a} >> 32)) << 32) | "
                                  f"__builtin_bswap32((uint32_t){a})")]
            return [wx(rd, 0, f"__builtin_bswap32({a})")]
        if opcode == 3:
            if not sf:
                raise Undefined("rev64 on w")
            return [wx(rd, 1, f"__builtin_bswap64({a})")]
        if opcode == 4:
            return [wx(rd, sf, f"clz{bits}({a})")]
        if opcode == 5:
            return [wx(rd, sf, f"cls{bits}({a})")]
        raise Undefined("dp1 opcode")
    if w & 0x1F000000 == 0x1B000000:  # data processing, 3 source
        op54, op31, o0, ra = (w >> 29) & 3, (w >> 21) & 7, (w >> 15) & 1, (w >> 10) & 31
        if op54:
            raise Undefined("dp3 op54")
        if op31 == 0:
            op = "-" if o0 else "+"
            return [wx(rd, sf, f"{rx(ra, sf)} {op} ({t})({rx(rn, sf)} * {rx(rm, sf)})")]
        if not sf:
            raise Undefined("dp3 long on w")
        if op31 in (1, 5):  # SMADDL/SMSUBL, UMADDL/UMSUBL
            conv = "(uint64_t)(int64_t)(int32_t)" if op31 == 1 else "(uint64_t)(uint32_t)"
            op = "-" if o0 else "+"
            return [wx(rd, 1, f"{rx(ra, 1)} {op} {conv}{rx(rn, 1)} * {conv}{rx(rm, 1)}")]
        if op31 in (2, 6) and not o0:
            return [wx(rd, 1, f"{'smulh' if op31 == 2 else 'umulh'}({rx(rn, 1)}, {rx(rm, 1)})")]
        raise Undefined("dp3")
    raise Undefined("dp reg")


def addsub(sf, sub, setf, a, b, rd, rd_is_sp):
    bits = 64 if sf else 32
    t = U(sf)
    if setf:
        if sub:
            b = f"({t})~{b}"
        return [wx(rd, sf, f"add{bits}f(c, {a}, {b}, {1 if sub else 0})")]
    expr = f"{a} {'-' if sub else '+'} {b}"
    return [wsp(rd, sf, expr) if rd_is_sp else wx(rd, sf, expr)]
