"""Function discovery for the loaded image.

Seeds, strongest first:
  FDE     unwind table entries (exact start and end)
  EXPORT  exported function symbols
  RELOC   relocated pointers into .text (vtables, callback tables, init_array)
  CALL    BL targets
  TAIL    B targets outside the current function
  ADDR    ADRP+ADD / ADR addresses into .text outside the current function (function pointers)
  GAP     first instruction of an unexplained code run in a gap between functions (last resort, logged)

FDE functions keep their unwind range. Others are traced (recursive descent) up to the next known function.
Jump tables (BR after an indexed table load) are resolved to intra-function targets. Everything that points into
the middle of a function, and every BR that is neither a jump table, a PLT stub nor an indirect tail call, is
recorded so it can be explained.
"""
import bisect
import collections
import hashlib
import struct

from . import a64, ehframe
from .image import R_AARCH64_JUMP_SLOT, STT_FUNC

FDE, EXPORT, RELOC, CALL, TAIL, ADDR, GAP = (1 << i for i in range(7))
FLAG_NAMES = {FDE: "fde", EXPORT: "export", RELOC: "reloc", CALL: "call", TAIL: "tail",
              ADDR: "addr", GAP: "gap"}

NORETURN_PREFIXES = ("abort", "exit", "_exit", "__cxa_throw", "__cxa_rethrow", "__cxa_bad_", "_ZSt9terminatev",
                     "__stack_chk_fail", "_ZN2nn4diag6detail9AbortImpl", "_ZN2nn4diag6detail10VAbortImpl",
                     "_ZN2nn6detail21UnexpectedDefaultImpl", "_ZSt20__throw_", "_ZSt17__throw_",
                     "_ZSt16__throw_", "_ZSt19__throw_", "_ZSt18__throw_", "_ZSt21__throw_", "_ZSt24__throw_",
                     "_ZNSt3__120__throw_", "_ZNSt3__122__throw_", "_ZNSt3__121__throw_")

LDP_FP_LR_POST_MASK, LDP_FP_LR_POST = 0xFFC07FFF, 0xA8C07BFD   # ldp x29, x30, [sp], #imm
LDP_FP_LR_OFF_MASK, LDP_FP_LR_OFF = 0xFFC07FFF, 0xA9407BFD     # ldp x29, x30, [sp, #imm]


class Function:
    __slots__ = ("start", "end", "flags", "fde", "name", "kind", "target", "jump_tables", "module")

    def __init__(self, start, module, flags=0, end=None, fde=False):
        self.start = start
        self.end = end
        self.flags = flags
        self.fde = fde
        self.name = None
        self.kind = "code"      # code | plt | plt0
        self.target = None      # plt: resolved destination
        self.jump_tables = []   # (br address, [targets])
        self.module = module


class Analysis:
    def __init__(self, image, log=print):
        self.image = image
        self.log = log
        self.words = memoryview(image.mem).cast("I")
        self.funcs = {}             # start -> Function
        self.fde_starts = []        # sorted FDE starts
        self.fde_ends = []
        self.names = {}             # address -> symbol name (functions)
        self.noreturn = set()
        self.refs_mid = collections.Counter()   # kind -> count of references into a function's middle
        self.refs_mid_samples = collections.defaultdict(list)
        self.misaligned_relocs = []
        self.br_kinds = collections.Counter()
        self.br_unknown = []
        self.cond_tails = 0
        self.stats = collections.Counter()
        self._queue = []
        self.got_names = {}         # GOT slot -> symbol name (JUMP_SLOT relocations)
        self.data_in_text = []      # (start, end): module headers, never code

    # -- helpers
    def word(self, addr):
        return self.words[(addr - self.image.base) >> 2]

    def module_text(self, addr):
        for m in self.image.modules:
            s, e = m.segments["text"]
            if s <= addr < e:
                return m
        return None

    def fde_containing(self, addr):
        i = bisect.bisect_right(self.fde_starts, addr) - 1
        if i >= 0 and addr < self.fde_ends[i]:
            return self.fde_starts[i]
        return None

    def add_start(self, addr, flag, ref_kind=None, ref_from=None):
        """Register a function start. References into the middle of an FDE function are recorded, not added."""
        if addr & 3:
            self._mid(ref_kind or FLAG_NAMES[flag], addr, ref_from, "misaligned")
            return None
        m = self.module_text(addr)
        if m is None:
            return None
        f = self.funcs.get(addr)
        if f is not None:
            f.flags |= flag
            return f
        owner = self.fde_containing(addr)
        if owner is not None:
            self._mid(ref_kind or FLAG_NAMES[flag], addr, ref_from, f"{owner:#x}+{addr - owner:#x}")
            return None
        f = Function(addr, m, flag)
        self.funcs[addr] = f
        self._queue.append(f)
        return f

    def _mid(self, kind, addr, ref_from, where):
        self.refs_mid[kind] += 1
        samples = self.refs_mid_samples[kind]
        if len(samples) < 20:
            samples.append((ref_from, addr, where))

    # -- seeds
    def seed(self):
        for m in self.image.modules:
            # Text starts with the module header (a zero word, the MOD0 offset, then MOD0): data, not an entry
            # point. Horizon enters through rtld, which calls exported entry symbols (seeded as EXPORT below).
            ts = m.segments["text"][0]
            self.data_in_text.append((ts, ts + m.u32(4) + 0x1C))
        for m in self.image.modules:
            for fde in ehframe.parse_module(self.image, m):
                f = Function(fde.start, m, FDE, fde.end, fde=True)
                self.funcs[fde.start] = f
                self._queue.append(f)
        self.fde_starts = sorted(a for a, f in self.funcs.items() if f.fde)
        self.fde_ends = [self.funcs[a].end for a in self.fde_starts]
        for m in self.image.modules:
            for s in m.symbols:
                if s.defined and s.name and s.addr is not None and m.in_text(s.addr):
                    self.names.setdefault(s.addr, s.name)
                    if s.type == STT_FUNC:
                        self.add_start(s.addr, EXPORT, "export", None)
                    if s.name.startswith(NORETURN_PREFIXES):
                        self.noreturn.add(s.addr)
            for r in m.relocs:
                if r.type == R_AARCH64_JUMP_SLOT:
                    self.got_names[r.where] = r.sym.name
                if r.value is not None and self.in_header(r.value):
                    self.stats["reloc_to_module_header"] += 1  # module base / MOD0 pointers
                    continue
                if r.value is not None and r.value & 3 and self.module_text(r.value) is not None:
                    self.stats["reloc_into_text_misaligned"] += 1  # tagged pointers (e.g. base | 1): not code
                    self.misaligned_relocs.append((r.where, r.value))
                    continue
                if r.value is not None and self.module_text(r.value) is not None:
                    self.stats["reloc_into_text"] += 1
                    self.add_start(r.value, RELOC, "reloc", r.where)

    # -- PLT stubs: adrp x16, page; ldr x17, [x16, #off]; add x16, x16, #off; br x17
    def classify_plt(self, f):
        w0 = self.word(f.start)
        p = a64.adrp(w0, f.start)
        if p is None or p[0] != 16:
            return False
        l = a64.ldr_uimm64(self.word(f.start + 4))
        ad = a64.add_imm(self.word(f.start + 8))
        if l is None or ad is None or l[:2] != (17, 16) or ad[:2] != (16, 16) or a64.br_reg(self.word(f.start + 12)) != 17:
            return False
        got = p[1] + l[2]
        target = self.image.u64(got)
        f.kind = "plt"
        f.target = target
        f.end = f.start + 16
        sym = self.got_names.get(got)
        if sym:
            f.name = f.name or (sym + "@plt")
        if target in self.noreturn:
            self.noreturn.add(f.start)
        return True

    # -- jump tables
    def _nearest_def(self, reg, before_pc, lo, through_calls=False):
        """(pc, insn) of the closest instruction before `before_pc` (in address order, back to `lo`) that may write
        `reg`. Address order is not the control flow graph, so callers validate what they derive from it.
        through_calls ignores call clobbers (a call laid out in between is often on another path)."""
        pc = before_pc - 4
        crossed_exit = False  # passed (backwards) a ret/b/br: code before it does not fall through to here
        while pc >= lo:
            w = self.word(pc)
            if a64.is_ret(w) or a64.b_target(w, pc) is not None or a64.br_reg(w) is not None:
                crossed_exit = True
            if through_calls and (a64.bl_target(w, pc) is not None or a64.blr_reg(w) is not None):
                pc -= 4
                continue
            if a64.writes_reg(w, reg):
                if through_calls and crossed_exit and a64.loads_from_sp(w):
                    pc -= 4  # an epilogue's register restore on another path
                    continue
                return pc, w
            pc -= 4
        return None

    def match_jump_table(self, br_pc, reg, lo, hi=None):
        """Recognize `table load + add anchor + br`. Returns (anchor, table, idx_reg, size, signed, shift, load_pc)
        or None when the BR is not table-shaped."""
        d = self._nearest_def(reg, br_pc, lo)
        if d is None:
            return None
        add_pc, w = d
        r = a64.add_reg(w)
        if r is None or r[0] != reg:
            return None
        _, anchor_reg, val_reg, shift = r
        load = None
        for want, other in ((val_reg, anchor_reg), (anchor_reg, val_reg)):  # second: add xD, xLoaded, xBase
            d = self._nearest_def(want, add_pc, lo)
            if d is not None:
                t = a64.table_load(d[1])
                if t is not None and t[0] == want:
                    load = (d[0],) + t
                    anchor_reg = other
                    break
        if load is None:
            return None
        load_pc, _, table_reg, idx_reg, size, signed = load
        anchor = self._reg_address(anchor_reg, add_pc, lo)
        table = self._reg_address(table_reg, load_pc, lo)
        direct = anchor is not None and table is not None
        if not direct:
            anchor = anchor if anchor is not None else self._reg_address(anchor_reg, add_pc, lo, True)
            table = table if table is not None else self._reg_address(table_reg, load_pc, lo, True)
            if anchor is None:
                return None
            if table is None and hi is not None:  # base set once, far away (a callee-saved register)
                table = self._unique_address(table_reg, lo, hi)
            if table is None:  # base reloaded from a spill slot: candidates, validated by the caller
                f = self.funcs.get(lo)
                table = [a for _, a in self._rodata_addresses(lo, f.end if f is not None and f.end else br_pc)]
                if not table:
                    return None
        return anchor, table, idx_reg, size, signed, shift, load_pc, direct, table_reg

    def resolve_jump_table(self, br_pc, reg, lo, hi):
        """Targets of a table-driven BR: (targets, exact). None if the BR is not table-shaped; ([], False) if it is
        but no plausible entries were found.

        exact: the entry count comes from a bounds check and every entry is valid. Otherwise (no check, e.g. a
        switch whose default is unreachable, or a check that overshoots) entries are read until the first one
        that leaves the function. The emitter dispatches on the computed address at run time and reports any
        address missing from the list, so an over-long list only adds labels and a short one fails loudly.
        """
        m = self.match_jump_table(br_pc, reg, lo, hi)
        if m is None:
            return None
        anchor, table, idx_reg, size, signed, shift, load_pc, direct, table_reg = m
        bound = self._table_bound(idx_reg, load_pc, lo, hi)
        if isinstance(table, list):  # candidate bases: the nearest one whose entries all validate
            if bound is None:
                return [], False
            # First the tables the function forms in the table register itself (several, when paths that set
            # different tables share this dispatch): all of those whose entries validate.
            merged = set()
            for cand in self._all_addresses(table_reg, lo, hi):
                ts = self._read_table(cand, anchor, size, signed, shift, bound, lo, hi)
                if len(ts) == bound:
                    merged |= set(ts)
            if merged:
                return sorted(merged), False
            for cand in reversed(table):
                ts = self._read_table(cand, anchor, size, signed, shift, bound, lo, hi)
                if len(ts) == bound:
                    return ts, False
            return [], False
        targets = self._read_table(table, anchor, size, signed, shift,
                                   bound if bound is not None and 0 < bound <= 65536 else 1024, lo, hi)
        exact = direct and bound is not None and len(targets) == bound
        if bound is not None and 0 < bound <= 65536:
            # Tail-merged dispatch: other paths may reach this block with another table in the same register (the
            # nearest one in address order need not be valid here at all). Take every table the function loads
            # into it whose `bound` entries all land in the function.
            merged = set(targets) if exact else set()
            # Candidates: what the function forms in the table register, and (bases passed in registers or
            # reloaded from the stack) every read-only address it forms anywhere. Extra labels cost nothing:
            # the dispatch compares the computed address, and every 4-aligned address is an instruction.
            candidates = self._all_addresses(table_reg, lo, hi) | {a for _, a in self._rodata_addresses(lo, hi)}
            for other in candidates - ({table} if exact else set()):
                ts = self._read_table(other, anchor, size, signed, shift, bound, lo, hi)
                if len(ts) == bound:
                    merged |= set(ts)
            if merged:
                targets = sorted(merged)
                exact = exact or len(merged) >= bound
        if len(targets) < 2 and not exact:
            return [], False
        return targets, exact

    def _read_table(self, table, anchor, size, signed, shift, limit, lo, hi):
        """Leading entries of a table whose targets are aligned and inside [lo, hi)."""
        fmt = {(1, False): "<B", (1, True): "<b", (2, False): "<H", (2, True): "<h", (4, False): "<I",
               (4, True): "<i"}[(size, signed)]
        targets = []
        for i in range(limit):
            off = table + i * size - self.image.base
            if off + size > len(self.image.mem):
                break
            e = struct.unpack_from(fmt, self.image.mem, off)[0]
            if size == 4 and not signed and shift == 0:
                e = a64.sext(e, 32)  # sxtw in the add
            t = (anchor + (e << shift)) & 0xFFFFFFFFFFFFFFFF
            if t & 3 or not lo <= t < hi:
                break
            targets.append(t)
        return targets

    def _reg_address(self, reg, before_pc, lo, through_calls=False, depth=0):
        """Address in `reg` at `before_pc` if its nearest definition is ADR, ADRP, or ADD #imm on an ADRP."""
        d = self._nearest_def(reg, before_pc, lo, through_calls)
        if d is None:
            return None
        pc, w = d
        r = a64.adr(w, pc)
        if r is not None:
            return r[1]
        p = a64.adrp(w, pc)
        if p is not None:
            return p[1]
        r = a64.add_imm(w)
        if r is not None and r[0] == reg:
            base = self._nearest_def(r[1], pc, lo, through_calls)
            if base is not None:
                p = a64.adrp(base[1], base[0])
                if p is not None and p[0] == r[1]:
                    return p[1] + r[2]
            return None
        mv = a64.mov_reg(w)
        if mv is not None and mv[0] == reg and depth < 8:
            return self._reg_address(mv[1], pc, lo, through_calls, depth + 1)
        return None

    def _unique_address(self, reg, lo, hi):
        """The address `reg` holds throughout [lo, hi) if every write to it there forms that same address
        (ADR, ADRP, ADD #imm on an ADRP), ignoring restores from the stack (epilogues on other paths). Compilers
        keep a jump table base hoisted in a callee-saved register this way, set once and used by several
        dispatches (sometimes with different anchors over the same entries)."""
        values = set()
        for pc in range(lo, hi, 4):
            w = self.word(pc)
            if not a64.writes_reg(w, reg) or a64.loads_from_sp(w):
                continue
            p = a64.adrp(w, pc)
            if p is not None and p[0] == reg:
                nxt = a64.add_imm(self.word(pc + 4))
                if nxt is not None and nxt[0] == reg and nxt[1] == reg:
                    continue  # the ADD that follows forms the value
                values.add(p[1])
                continue
            v = self._reg_address(reg, pc + 4, lo)
            if v is None:
                return None
            values.add(v)
        return values.pop() if len(values) == 1 else None

    def _all_addresses(self, reg, lo, hi):
        """Every address formed in `reg` within [lo, hi) by ADR, ADRP, or ADRP + ADD #imm."""
        out = set()
        for pc in range(lo, hi, 4):
            w = self.word(pc)
            r = a64.adr(w, pc)
            if r is not None and r[0] == reg:
                out.add(r[1])
                continue
            r = a64.add_imm(w)
            if r is not None and r[0] == reg:
                v = self._reg_address(reg, pc + 4, lo)
                if v is not None:
                    out.add(v)
        return out

    def _rodata_addresses(self, lo, hi):
        """Every ADRP+ADD address into a read-only segment formed in [lo, hi), in address order."""
        out = []
        last = {}
        for pc in range(lo, hi, 4):
            w = self.word(pc)
            p = a64.adrp(w, pc)
            if p is not None:
                last[p[0]] = p[1]
                continue
            r = a64.add_imm(w)
            if r is not None and r[1] in last:
                addr = last[r[1]] + r[2]
                m = self.image.module_at(addr)
                if m is not None and m.segments["ro"][0] <= addr < m.segments["ro"][1]:
                    out.append((pc, addr))
        return out

    def _table_bound(self, idx_reg, load_pc, lo, hi):
        """Entry count from the bounds check guarding the dispatch block.

        The dispatch block (straight-line code ending at the table load) is reached either by falling through a
        `cmp idx, #n; b.hi/b.hs default`, or by a branch into it: `cmp idx, #n; b.ls/b.lo dispatch`. A mask
        `and idx, x, #(2^k - 1)` on the straight-line path bounds it too.
        """
        pc = load_pc - 4
        chain_start = load_pc
        while pc >= lo:
            w = self.word(pc)
            if a64.b_target(w, pc) is not None or a64.is_ret(w) or a64.br_reg(w) is not None or a64.is_trap(w):
                break
            mv = a64.mov_reg_any(w)
            if mv is not None and mv[0] == idx_reg:  # `cmp w23, #3; b.hi ..; mov w8, w23`: the check is on w23
                idx_reg = mv[1]
                chain_start = pc
                pc -= 4
                continue
            c = a64.cmp_imm(w)
            if c is not None and c[0] == idx_reg:
                n = self._cmp_count(pc, c[1], chain_start, load_pc)
                if n is not None:
                    return n
            a = a64.and_imm(w)
            if a is not None and a[0] == idx_reg:
                mask = a[2]  # 0xffffffff and wider masks are zero-extensions, not bounds
                return mask + 1 if mask & (mask + 1) == 0 and mask < 0x10000 else None
            chain_start = pc
            pc -= 4
        # branches into the dispatch chain from elsewhere in the function
        for q in range(lo, hi, 4):
            w = self.word(q)
            t = a64.cond_target(w, q)
            if t is not None and chain_start <= t <= load_pc:
                for k in (1, 2, 3):
                    c = a64.cmp_imm(self.word(q - 4 * k))
                    if c is not None and c[0] == idx_reg:
                        cond = a64.bcond_code(w)
                        n = {9: c[1] + 1, 3: c[1]}.get(cond)  # b.ls / b.lo into the dispatch block
                        if n is not None:
                            return n
                        break
            t = a64.b_target(w, q)
            if t is not None and chain_start <= t <= load_pc:
                # `cmp idx, #n; b.hi default; b dispatch`
                for k in (1, 2, 3, 4):
                    c = a64.cmp_imm(self.word(q - 4 * k))
                    if c is not None and c[0] == idx_reg:
                        n = self._cmp_count(q - 4 * k, c[1], None, None)
                        if n is not None:
                            return n
                        break
        return None

    def _cmp_count(self, cmp_pc, imm, chain_start, load_pc):
        """Entry count implied by the conditional branch following `cmp idx, #imm` at cmp_pc."""
        for k in (1, 2, 3):
            w = self.word(cmp_pc + 4 * k)
            cond = a64.bcond_code(w)
            if cond is None:
                continue
            if cond in (8, 2):  # b.hi / b.hs leave the range: fallthrough is the dispatch path
                return imm + 1 if cond == 8 else imm
            if cond in (9, 3) and chain_start is not None:  # b.ls / b.lo jump into the dispatch path
                t = a64.cond_target(w, cmp_pc + 4 * k)
                if chain_start <= t <= load_pc:
                    return imm + 1 if cond == 9 else imm
            return None
        return None

    def _frame_restored(self, f, pc):
        """True if LR was reloaded (frame torn down) on the straight path before `pc`: an indirect tail call."""
        q = pc - 4
        while q >= f.start:
            w = self.word(q)
            if a64.loads_lr(w):
                return True
            if a64.bl_target(w, q) is not None or a64.blr_reg(w) is not None or a64.saves_lr(w):
                return False
            q -= 4
        return None  # reached the start: no frame on this path

    def classify_br(self, f, pc, reg, lo, hi):
        jt = self.resolve_jump_table(pc, reg, lo, hi)
        targets = None if jt is None else jt[0]
        if targets:
            f.jump_tables.append((pc, targets))
            self.br_kinds["jump_table" if jt[1] else "jump_table_heuristic"] += 1
            return targets
        if targets == []:
            self.br_kinds["jump_table_unresolved"] += 1
            self.br_unknown.append((f.start, pc, reg, "jump_table_unresolved"))
            return None
        # computed goto: the register only ever holds ADR'd labels inside this function
        labels = sorted({r[1] for q in range(lo, hi, 4) for r in [a64.adr(self.word(q), q)]
                         if r is not None and r[0] == reg and lo <= r[1] < hi})
        d = self._nearest_def(reg, pc, lo)
        if labels and d is not None and a64.adr(d[1], d[0]) is not None or labels and d is None:
            f.jump_tables.append((pc, labels))
            self.br_kinds["computed_goto"] += 1
            return labels
        restored = self._frame_restored(f, pc)
        if restored:
            self.br_kinds["indirect_tail_after_epilogue"] += 1
            return None
        if restored is None and not any(a64.saves_lr(self.word(q)) for q in range(f.start, pc, 4)):
            self.br_kinds["indirect_tail_leaf"] += 1
            return None
        if reg in (16, 17):
            self.br_kinds["veneer_x16_x17"] += 1
            return None
        self.br_kinds["unknown"] += 1
        self.br_unknown.append((f.start, pc, reg, "unknown"))
        return None

    # -- scanning one function
    def scan(self, f):
        if f.kind == "code" and not f.fde and self.classify_plt(f):
            self.add_start(f.target, CALL, "plt_target", f.start)
            return
        if f.fde:
            self._scan_linear(f)
        else:
            self._trace(f)

    def _refs(self, f, pc, w, last_adrp, lo, hi):
        """Calls, tail branches and address formation at one instruction; returns branch info for tracing.

        last_adrp maps register -> (pc, page) for ADRPs whose register has not been overwritten since, so an
        `add xD, xN, #lo12` only pairs with an ADRP that still owns xN."""
        if last_adrp and a64.add_imm(w) is None and a64.adrp(w, pc) is None:
            for reg in [r for r in last_adrp if a64.writes_reg(w, r)]:
                del last_adrp[reg]
            if a64.is_ret(w) or a64.b_target(w, pc) is not None or a64.br_reg(w) is not None:
                last_adrp.clear()  # the next instruction starts a block entered from elsewhere
        t = a64.bl_target(w, pc)
        if t is not None:
            self.stats["bl"] += 1
            self.add_start(t, CALL, "bl", pc)
            return "call", t
        t = a64.b_target(w, pc)
        if t is not None:
            if lo <= t < hi and t != f.start:
                return "jump", t
            self.add_start(t, TAIL, "b_tail", pc)
            return "tail", t
        t = a64.cond_target(w, pc)
        if t is not None:
            if not lo <= t < hi:
                self.cond_tails += 1
                self.add_start(t, TAIL, "cond_tail", pc)
            return "cond", t
        p = a64.adrp(w, pc)
        if p is not None:
            last_adrp[p[0]] = (pc, p[1])
            return None, None
        r = a64.add_imm(w)
        if r is not None:
            prev = last_adrp.get(r[1])
            if prev is not None:
                addr = prev[1] + r[2]
                if not lo <= addr < hi and self.module_text(addr) is not None:
                    self.stats["addr_into_text"] += 1
                    self.add_start(addr, ADDR, "adrp_add", pc)
            last_adrp.pop(r[0], None)
            return None, None
        r = a64.adr(w, pc)
        if r is not None:
            if not lo <= r[1] < hi and self.module_text(r[1]) is not None:
                self.stats["addr_into_text"] += 1
                self.add_start(r[1], ADDR, "adr", pc)
            return None, None
        return None, None

    def _scan_linear(self, f):
        last_adrp = {}
        lo, hi = f.start, f.end
        pc = lo
        while pc < hi:
            w = self.word(pc)
            kind, _ = self._refs(f, pc, w, last_adrp, lo, hi)
            if kind is None:
                reg = a64.br_reg(w)
                if reg is not None:
                    self.classify_br(f, pc, reg, lo, hi)
            pc += 4

    def _next_start_after(self, addr):
        """Upper bound for tracing a function without an FDE: the next known function start."""
        i = bisect.bisect_right(self._sorted_starts, addr)
        nxt = self._sorted_starts[i] if i < len(self._sorted_starts) else None
        m = self.module_text(addr)
        end = m.segments["text"][1]
        return min(nxt, end) if nxt is not None else end

    def _trace(self, f):
        lo = f.start
        hi = self._next_start_after(f.start)
        seen = set()
        todo = [lo]
        last_adrp = {}
        while todo:
            pc = todo.pop()
            while lo <= pc < hi and pc not in seen:
                seen.add(pc)
                w = self.word(pc)
                kind, t = self._refs(f, pc, w, last_adrp, lo, hi)
                if kind == "jump":
                    todo.append(t)
                    break
                if kind == "tail":
                    break
                if kind == "cond":
                    if lo <= t < hi:
                        todo.append(t)
                elif kind == "call" and t in self.noreturn:
                    break
                elif a64.is_ret(w) or a64.is_trap(w):
                    break
                else:
                    reg = a64.br_reg(w)
                    if reg is not None:
                        targets = self.classify_br(f, pc, reg, lo, hi)
                        if targets:
                            todo.extend(targets)
                        break
                pc += 4
        f.end = (max(seen) + 4) if seen else lo + 4

    # -- gaps
    def gaps(self):
        """Uncovered text ranges per module: list of (module, start, end)."""
        out = []
        for m in self.image.modules:
            ts, te = m.segments["text"]
            cur = ts
            for s in self._sorted_starts:
                if s < ts or s >= te:
                    continue
                if s > cur:
                    out.append((m, cur, s))
                cur = max(cur, self.funcs[s].end)
            if cur < te:
                out.append((m, cur, te))
        return out

    def is_padding(self, w):
        return w == 0 or w == a64.NOP or a64.is_trap(w)

    def in_header(self, addr):
        return any(s <= addr < e for s, e in self.data_in_text)

    def sweep_gaps(self):
        new = 0
        for m, s, e in self.gaps():
            pc = s
            while pc < e and (self.is_padding(self.word(pc)) or self.in_header(pc)):
                pc += 4
            if pc < e:
                if self.add_start(pc, GAP, "gap", None) is not None:
                    new += 1
        return new

    # -- driver
    def run(self):
        self.seed()
        passes = 0
        while True:
            passes += 1
            while self._queue:
                batch, self._queue = self._queue, []
                self._sorted_starts = sorted(self.funcs)
                batch.sort(key=lambda x: (not x.fde, x.start))  # FDE functions first: they bound the others
                for f in batch:
                    self.scan(f)
            self._sorted_starts = sorted(self.funcs)
            new = self.sweep_gaps()
            self.log(f"pass {passes}: {len(self.funcs)} functions, {new} gap seeds")
            if not new:
                break
        for a, f in self.funcs.items():
            f.name = f.name or self.names.get(a)
        return self

    # -- results
    def digest(self):
        h = hashlib.sha256()
        for a in sorted(self.funcs):
            f = self.funcs[a]
            h.update(struct.pack("<QQI", a, f.end, f.flags))
            for br, ts in f.jump_tables:
                h.update(struct.pack(f"<Q{len(ts)}Q", br, *ts))
        return h.hexdigest()

    def summary(self):
        mods = {}
        for m in self.image.modules:
            ts, te = m.segments["text"]
            fs = [f for f in self.funcs.values() if f.module is m]
            covered = sum(min(f.end, te) - f.start for f in fs)
            gap_bytes = padding = 0
            for gm, s, e in self.gaps():
                if gm is not m:
                    continue
                for pc in range(s, e, 4):
                    if self.is_padding(self.word(pc)) or self.in_header(pc):
                        padding += 4
                    else:
                        gap_bytes += 4
            by_flag = collections.Counter()
            for f in fs:
                for bit, name in FLAG_NAMES.items():
                    if f.flags & bit:
                        by_flag[name] += 1
            mods[m.name] = {
                "functions": len(fs),
                "fde": sum(1 for f in fs if f.fde),
                "traced": sum(1 for f in fs if not f.fde and f.kind == "code"),
                "plt_stubs": sum(1 for f in fs if f.kind == "plt"),
                "only_from_gap_sweep": sum(1 for f in fs if f.flags == GAP),
                "named": sum(1 for f in fs if f.name),
                "jump_tables": sum(len(f.jump_tables) for f in fs),
                "text_bytes": te - ts,
                "covered_bytes": covered,
                "uncovered_padding_bytes": padding,
                "uncovered_code_bytes": gap_bytes,
                "seed_sources": dict(sorted(by_flag.items())),
            }
        return {
            "digest": self.digest(),
            "modules": mods,
            "refs_into_middle": dict(self.refs_mid),
            "br": dict(self.br_kinds),
            "conditional_tail_branches": self.cond_tails,
            "stats": dict(self.stats),
        }
