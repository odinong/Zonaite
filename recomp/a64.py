"""AArch64 decoding helpers for analysis: control flow, address formation and the loads used by jump tables.

The emitter has the full decoder; this module only covers what function discovery needs. Every helper takes
the instruction word (and its address where PC-relative) and returns None when the word is not that instruction.
"""

NOP = 0xD503201F


def sext(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def b_target(insn, pc):
    """B imm26."""
    if insn & 0xFC000000 == 0x14000000:
        return pc + (sext(insn & 0x03FFFFFF, 26) << 2)
    return None


def bl_target(insn, pc):
    """BL imm26."""
    if insn & 0xFC000000 == 0x94000000:
        return pc + (sext(insn & 0x03FFFFFF, 26) << 2)
    return None


def cond_target(insn, pc):
    """B.cond, CBZ/CBNZ, TBZ/TBNZ: returns the taken target (fallthrough is pc + 4)."""
    if insn & 0xFF000010 == 0x54000000:
        return pc + (sext((insn >> 5) & 0x7FFFF, 19) << 2)
    if insn & 0x7E000000 == 0x34000000:
        return pc + (sext((insn >> 5) & 0x7FFFF, 19) << 2)
    if insn & 0x7E000000 == 0x36000000:
        return pc + (sext((insn >> 5) & 0x3FFF, 14) << 2)
    return None


def bcond_code(insn):
    if insn & 0xFF000010 == 0x54000000:
        return insn & 0xF
    return None


def br_reg(insn):
    if insn & 0xFFFFFC1F == 0xD61F0000:
        return (insn >> 5) & 31
    return None


def blr_reg(insn):
    if insn & 0xFFFFFC1F == 0xD63F0000:
        return (insn >> 5) & 31
    return None


def is_ret(insn):
    return insn & 0xFFFFFC1F == 0xD65F0000


def is_trap(insn):
    """BRK, HLT, UDF: execution never continues past these."""
    return insn & 0xFFE0001F in (0xD4200000, 0xD4400000) or insn & 0xFFFF0000 == 0


def svc_imm(insn):
    if insn & 0xFFE0001F == 0xD4000001:
        return (insn >> 5) & 0xFFFF
    return None


def adrp(insn, pc):
    """ADRP Xd, label -> (rd, page address)."""
    if insn & 0x9F000000 == 0x90000000:
        imm = sext(((insn >> 3) & 0x1FFFFC) | ((insn >> 29) & 3), 21) << 12
        return insn & 31, (pc & ~0xFFF) + imm
    return None


def adr(insn, pc):
    """ADR Xd, label -> (rd, address)."""
    if insn & 0x9F000000 == 0x10000000:
        imm = sext(((insn >> 3) & 0x1FFFFC) | ((insn >> 29) & 3), 21)
        return insn & 31, pc + imm
    return None


def add_imm(insn):
    """ADD Xd|SP, Xn|SP, #imm{, lsl #12} (64-bit) -> (rd, rn, imm)."""
    if insn & 0xFF800000 == 0x91000000:
        imm = (insn >> 10) & 0xFFF
        if insn & (1 << 22):
            imm <<= 12
        return insn & 31, (insn >> 5) & 31, imm
    return None


def ldr_uimm64(insn):
    """LDR Xt, [Xn, #imm] (unsigned offset) -> (rt, rn, byte offset)."""
    if insn & 0xFFC00000 == 0xF9400000:
        return insn & 31, (insn >> 5) & 31, ((insn >> 10) & 0xFFF) << 3
    return None


def cmp_imm(insn):
    """CMP Wn|Xn, #imm (SUBS with Rd = ZR) -> (rn, imm)."""
    if insn & 0x7F800000 == 0x71000000 and insn & 31 == 31:
        imm = (insn >> 10) & 0xFFF
        if insn & (1 << 22):
            imm <<= 12
        return (insn >> 5) & 31, imm
    return None


# Register-offset loads used to read jump table entries: mask/value -> (entry size, signed)
_TABLE_LOADS = (
    (0xFFE00C00, 0x38600800, 1, False),  # LDRB  Wt, [Xn, Rm]
    (0xFFE00C00, 0x78600800, 2, False),  # LDRH  Wt, [Xn, Rm{, lsl #1}]
    (0xFFE00C00, 0xB8600800, 4, False),  # LDR   Wt, [Xn, Rm{, lsl #2}]
    (0xFFE00C00, 0xB8A00800, 4, True),   # LDRSW Xt, [Xn, Rm{, lsl #2}]
    (0xFFE00C00, 0x38A00800, 1, True),   # LDRSB Xt
    (0xFFE00C00, 0x78A00800, 2, True),   # LDRSH Xt
    (0xFFE00C00, 0x38E00800, 1, True),   # LDRSB Wt
    (0xFFE00C00, 0x78E00800, 2, True),   # LDRSH Wt
)


def table_load(insn):
    """Register-offset byte/half/word load -> (rt, rn, rm, entry size, signed)."""
    for mask, value, size, signed in _TABLE_LOADS:
        if insn & mask == value:
            return insn & 31, (insn >> 5) & 31, (insn >> 16) & 31, size, signed
    return None


def add_reg(insn):
    """ADD Xd, Xn, Rm{, shift|extend #amount} (64-bit) -> (rd, rn, rm, left shift)."""
    if insn & 0xFF200000 == 0x8B000000:  # shifted register, LSL only for our purposes
        if (insn >> 22) & 3 == 0:
            return insn & 31, (insn >> 5) & 31, (insn >> 16) & 31, (insn >> 10) & 63
        return None
    if insn & 0xFFE00000 == 0x8B200000:  # extended register (sxtw/uxtw/sxtb/...)
        return insn & 31, (insn >> 5) & 31, (insn >> 16) & 31, (insn >> 10) & 7
    return None


def decode_bit_masks(n, imms, immr, datasize):
    """The logical-immediate value for N:immr:imms (ARM ARM DecodeBitMasks, wmask only)."""
    combined = (n << 6) | (~imms & 0x3F)
    length = combined.bit_length() - 1
    if length < 1:
        return None
    esize = 1 << length
    levels = esize - 1
    s, r = imms & levels, immr & levels
    if s == levels:
        return None
    welem = (1 << (s + 1)) - 1
    welem = ((welem >> r) | (welem << (esize - r))) & ((1 << esize) - 1)  # ROR
    value = 0
    for i in range(datasize // esize):
        value |= welem << (i * esize)
    return value


def and_imm(insn):
    """AND Wd|Xd, Wn|Xn, #bitmask -> (rd, rn, mask)."""
    if insn & 0x7F800000 == 0x12000000:
        sf = insn >> 31
        n = (insn >> 22) & 1
        if not sf and n:
            return None
        mask = decode_bit_masks(n, (insn >> 10) & 0x3F, (insn >> 16) & 0x3F, 64 if sf else 32)
        if mask is None:
            return None
        return insn & 31, (insn >> 5) & 31, mask
    return None


def _ldst_pair_regs(insn):
    """(Rt, Rt2, is_load) for 64-bit LDP/STP in any addressing mode, else None."""
    if insn & 0xFE000000 == 0xA8000000 and (insn >> 23) & 3 in (1, 2, 3):
        return insn & 31, (insn >> 10) & 31, bool(insn & (1 << 22))
    return None


def loads_lr(insn):
    """LDP with x30 as either register, or LDR x30 (unsigned offset / pre / post index)."""
    p = _ldst_pair_regs(insn)
    if p is not None:
        return p[2] and 30 in (p[0], p[1])
    if insn & 0xFFC00000 == 0xF9400000 or insn & 0xFFE00400 == 0xF8400400:
        return insn & 31 == 30
    return False


def saves_lr(insn):
    """STP with x30 as either register, or STR x30 (unsigned offset / pre / post index)."""
    p = _ldst_pair_regs(insn)
    if p is not None:
        return not p[2] and 30 in (p[0], p[1])
    if insn & 0xFFC00000 == 0xF9000000 or insn & 0xFFE00400 == 0xF8000400:
        return insn & 31 == 30
    return False


def writes_reg(insn, reg):
    """Conservative: True if the instruction may write general register `reg` (0-30).

    Errs on the side of True (unknown encodings, SIMD moves to general registers), so callers that stop a backward
    search at a write never step over a redefinition. Calls clobber x0-x18 and x30 (AAPCS64 + Horizon's x18).
    """
    if insn & 0xFC000000 == 0x94000000 or insn & 0xFFFFFC1F == 0xD63F0000:  # BL / BLR
        return reg <= 18 or reg == 30
    op0 = (insn >> 25) & 0xF
    if op0 in (0b1010, 0b1011):  # branches, exceptions, system (MRS writes Rt)
        return insn & 0xFFF00000 == 0xD5300000 and insn & 31 == reg
    if op0 in (0b1000, 0b1001):  # data processing, immediate
        return insn & 31 == reg
    if op0 in (0b0101, 0b1101):  # data processing, register
        return insn & 31 == reg
    if op0 in (0b0111, 0b1111):  # SIMD/FP: only moves and conversions to general registers write one
        return simd_writes_gpr(insn) and insn & 31 == reg
    if op0 & 0b0101 == 0b0100:  # loads and stores
        rt, rn, rt2 = insn & 31, (insn >> 5) & 31, (insn >> 10) & 31
        if is_prefetch(insn):
            return False
        if insn & 0xBF000000 in (0x0C000000, 0x0D000000):  # SIMD structure load/store: only writeback
            return bool(insn & (1 << 23)) and rn == reg
        if insn & 0x3B000000 == 0x18000000:  # load literal
            return not insn & (1 << 26) and rt == reg
        if insn & 0x3A000000 == 0x28000000:  # load/store pair
            load = bool(insn & (1 << 22))
            vector = bool(insn & (1 << 26))
            writeback = (insn >> 23) & 3 in (1, 3)
            return (load and not vector and reg in (rt, rt2)) or (writeback and rn == reg)
        if insn & 0x3F000000 == 0x08000000:  # exclusives / load-acquire: conservative
            return reg in (rt, rt2, (insn >> 16) & 31)
        if insn & 0x3B200000 == 0x38000000 or insn & 0x3B000000 == 0x39000000 or insn & 0x3B200C00 == 0x38200800:
            opc = (insn >> 22) & 3
            vector = bool(insn & (1 << 26))
            load = opc != 0 and not vector
            writeback = insn & 0x3B000000 == 0x38000000 and (insn >> 10) & 1 and not insn & (1 << 21)
            return (load and rt == reg) or bool(writeback and rn == reg)
        if insn & 0x3B200C00 == 0x38200000:  # atomic memory ops (LSE): Rt receives the old value
            return rt == reg
        return True
    return True


def simd_writes_gpr(insn):
    """True for the SIMD/FP instructions whose destination is a general register."""
    if insn & 0xBFE0FC00 in (0x0E003C00, 0x0E002C00):  # UMOV, SMOV
        return True
    if insn & 0x5F20FC00 == 0x1E200000:  # FP <-> integer conversion (scalar)
        opcode = (insn >> 16) & 7
        return opcode in (0, 1, 4, 5, 6)  # FCVT[NPMZA][SU] to int, FMOV to general
    if insn & 0x5F200000 == 0x1E000000:  # FP <-> fixed-point conversion
        return (insn >> 16) & 7 in (0, 1)  # FCVTZS/FCVTZU with fbits
    return False


def mov_reg(insn):
    """MOV Xd, Xm (ORR Xd, XZR, Xm) -> (rd, rm)."""
    if insn & 0xFFE0FFE0 == 0xAA0003E0:
        return insn & 31, (insn >> 16) & 31
    return None


def mov_reg_any(insn):
    """MOV Xd, Xm or MOV Wd, Wm (ORR with the zero register) -> (rd, rm). For index registers: the 32-bit form
    zero-extends, which keeps a bounds check on the source valid (unlike for addresses)."""
    if insn & 0x7FE0FFE0 == 0x2A0003E0:
        return insn & 31, (insn >> 16) & 31
    return None


def is_prefetch(insn):
    """PRFM (immediate, register, literal) and PRFUM: Rt is the prefetch operation, not a register."""
    return (insn & 0xFFC00000 == 0xF9800000 or insn & 0xFFE00C00 == 0xF8A00800
            or insn & 0xFFE00C00 == 0xF8800000 or insn & 0xFF000000 == 0xD8000000)


def loads_from_sp(insn):
    """LDR/LDP (any 64-bit integer form) with SP as the base register: a stack reload."""
    if (insn >> 5) & 31 != 31:
        return False
    p = _ldst_pair_regs(insn)
    if p is not None:
        return p[2]
    return insn & 0xFFC00000 == 0xF9400000 or insn & 0xFFE00400 == 0xF8400400
