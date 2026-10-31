"""Generate C++ for whole guest functions from the analysis.

Each translation unit holds a slice of functions and declares only the functions it references, so no unit parses
a 300k-line prototype header. recomp_table.cpp holds the sorted address -> function table for indirect calls.

Hooks (config/hooks.txt, see load_hooks) replace a guest function with host code. The original is still generated,
as f_<addr>_orig, so a hook can call it; every caller and the indirect-call table reach the hook.
"""
import os
import re

from . import a64, emit as E

FN_PREFIX = "f_"


def write_if_changed(path, text):
    """Keep the file (and its timestamp, so the compile is skipped) when the content is unchanged."""
    try:
        with open(path, encoding="utf-8", newline="\n") as fp:
            if fp.read() == text:
                return False
    except OSError:
        pass
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(text)
    return True


def fname(addr):
    return f"{FN_PREFIX}{addr:x}"


HOOK_KINDS = ("replace", "wrap")


def load_hooks(path):
    """Hook table: one `ADDRESS NAME [KIND]` per line, '#' comments. KIND:
      replace (default)  f_ADDR calls the host function zn_hook_NAME(Ctx*, uint8_t*) instead of the original
      wrap               the same; documents that the host function calls f_ADDR_orig itself
    Returns {address: (name, kind)}."""
    hooks = {}
    if not os.path.exists(path):
        return hooks
    with open(path, encoding="utf-8") as fp:
        for n, line in enumerate(fp, 1):
            line = line.split("#", 1)[0].split()
            if not line:
                continue
            if len(line) not in (2, 3) or not re.fullmatch(r"\w+", line[1]) or \
                    (len(line) == 3 and line[2] not in HOOK_KINDS):
                raise ValueError(f"{path}:{n}: expected ADDRESS NAME [{'|'.join(HOOK_KINDS)}]")
            hooks[int(line[0], 16)] = (line[1], line[2] if len(line) == 3 else "replace")
    return hooks


class FuncFlow:
    """Control flow back end for one function."""

    def __init__(self, gen, f):
        self.gen, self.f = gen, f
        self.refs = gen.refs  # functions referenced by the current unit
        self.jt = {pc: targets for pc, targets in f.jump_tables}

    def target_fn(self, addr):
        """(host function name or None, resolved guest address) for a call/tail-call target. PLT stubs resolve to
        their destination; a stub of an undefined weak symbol resolves to 0, as rtld leaves it, and is reached
        through call_indirect (which reports the branch to 0 if it ever happens)."""
        g = self.gen.funcs.get(addr)
        if g is not None and g.kind == "plt":
            addr = g.target
        g = self.gen.funcs.get(addr)
        if g is None or g.kind != "code":
            return None, addr
        self.refs.add(addr)
        return fname(addr), addr

    def inside(self, t):
        return self.f.start <= t < self.f.end

    def tail(self, target):
        fn, resolved = self.target_fn(target)
        if fn is None:
            return f"c->pc = {resolved:#x}ull; [[clang::musttail]] return dispatch_indirect(c, m);"
        return f"[[clang::musttail]] return {fn}(c, m);"

    def jump(self, target):
        if self.inside(target):
            self.gen.labels.add(target)
            return f"goto L_{target:x};"
        return self.tail(target)

    def cond_jump(self, cond, target):
        return f"if ({cond}) {{ {self.jump(target)} }}"

    def call(self, target, ret):
        fn, resolved = self.target_fn(target)
        if fn is None:
            return f"c->x[30] = {ret:#x}ull; call_indirect(c, m, {resolved:#x}ull);"
        return f"c->x[30] = {ret:#x}ull; {fn}(c, m);"

    def call_indirect(self, expr, ret):
        return f"{{ uint64_t t_ = {expr}; c->x[30] = {ret:#x}ull; call_indirect(c, m, t_); }}"

    def branch_indirect(self, expr, pc):
        targets = self.jt.get(pc)
        if targets:
            cases = []
            for t in sorted(set(targets)):
                self.gen.labels.add(t)
                cases.append(f"case {t:#x}ull: goto L_{t:x};")
            return (f"{{ uint64_t t_ = {expr}; switch (t_) {{ {' '.join(cases)} "
                    f"default: jt_miss(c, m, {pc:#x}ull, t_); return; }} }}")
        return f"c->pc = {expr}; [[clang::musttail]] return dispatch_indirect(c, m);"

    def ret(self, expr):
        return "return;"

    def trap(self, pc, word):
        return f"guest_trap(c, m, {pc:#x}ull, {word:#x}u); return;"


class Generator:
    def __init__(self, image, analysis, hooks=None):
        self.image = image
        self.funcs = analysis.funcs
        self.hooks = hooks or {}
        self.words = memoryview(image.mem).cast("I")
        self.refs = set()
        self.labels = set()

    def word(self, addr):
        return self.words[(addr - self.image.base) >> 2]

    def function(self, f):
        """C++ source of one function. Labels are collected while emitting, then inserted."""
        self.labels = set()
        flow = FuncFlow(self, f)
        body = []
        last_falls = True
        for pc in range(f.start, f.end, 4):
            w = self.word(pc)
            try:
                stmts = E.emit(w, pc, flow)
                last_falls = not (a64.is_ret(w) or a64.b_target(w, pc) is not None or a64.br_reg(w) is not None
                                  or a64.is_trap(w))
            except E.Undefined:
                stmts = [flow.trap(pc, w)]
                last_falls = False
            body.append((pc, w, stmts))
        hook = self.hooks.get(f.start)
        out = []
        if hook:  # the hook takes the function's name; the original follows as f_<addr>_orig
            name, kind = hook
            orig = fname(f.start) + "_orig"
            out.append(f"void {orig}(Ctx* c, uint8_t* m);")
            out.append(f"void zn_hook_{name}(Ctx* c, uint8_t* m);")
            out.append(f"void {fname(f.start)}(Ctx* c, uint8_t* m) {{ "
                       f"[[clang::musttail]] return zn_hook_{name}(c, m); }}")
            out.append(f"void {orig}(Ctx* c, uint8_t* m) {{")
        else:
            out.append(f"void {fname(f.start)}(Ctx* c, uint8_t* m) {{")
        for pc, w, stmts in body:
            if pc in self.labels:
                out.append(f"L_{pc:x}:")
            out.append(f"    /* {pc:x}: {w:08x} */ " + " ".join(stmts))
        if last_falls:  # runs off the end: continue in the next function, as the hardware would
            out.append("    " + flow.tail(f.end))
        out.append("}")
        return "\n".join(out)

    def unit(self, funcs):
        """A translation unit: includes, extern declarations of everything referenced, then the functions."""
        self.refs = set()
        bodies = [self.function(f) for f in funcs]
        defined = {f.start for f in funcs}
        decls = [f"void {fname(a)}(Ctx*, uint8_t*);" for a in sorted(self.refs - defined)]
        decls += [f"void {fname(f.start)}(Ctx*, uint8_t*);" for f in funcs]
        head = ['#include "zonaite/guest.h"', "using namespace zg;", "#pragma clang diagnostic ignored \"-Wunused-label\""]
        return "\n".join(head + decls + [""] + bodies) + "\n"

    def code_functions(self):
        return [self.funcs[a] for a in sorted(self.funcs) if self.funcs[a].kind == "code"]

    def write(self, out_dir, unit_insns=80000, huge=6000, log=print):
        """Write translation units balanced by instruction count. Functions over `huge` instructions get a unit
        of their own, compiled at -O1: clang's -O2 is superlinear on them (one 18k-instruction function takes
        206 s at -O2, 57 s at -O1). Writes units.txt: unit name, optimization level, instruction count."""
        os.makedirs(out_dir, exist_ok=True)
        funcs = self.code_functions()
        units = []

        def flush(fs, n, opt):
            name = f"recomp_{len(units):04d}.cpp"
            write_if_changed(os.path.join(out_dir, name), self.unit(fs))
            units.append((name, opt, n))

        batch, count = [], 0
        for f in funcs:
            n = (f.end - f.start) // 4
            if n > huge:
                flush([f], n, "O1")
                continue
            batch.append(f)
            count += n
            if count >= unit_insns:
                flush(batch, count, "O2")
                batch, count = [], 0
        if batch:
            flush(batch, count, "O2")
        table = ['#include "zonaite/guest.h"', "using namespace zg;"]
        table += [f"void {fname(f.start)}(Ctx*, uint8_t*);" for f in funcs]
        table += ["namespace zg {", "struct FnEntry { uint64_t addr; GuestFn fn; };",
                  "extern const FnEntry kFunctions[] = {"]
        table += [f"    {{{f.start:#x}ull, {fname(f.start)}}}," for f in funcs]
        table += ["};", f"extern const unsigned kFunctionCount = {len(funcs)};", "}"]
        write_if_changed(os.path.join(out_dir, "recomp_table.cpp"), "\n".join(table) + "\n")
        units.append(("recomp_table.cpp", "O1", 0))
        with open(os.path.join(out_dir, "units.txt"), "w") as fp:
            for name, opt, n in units:
                fp.write(f"{name}\t{opt}\t{n}\n")
        log(f"wrote {len(units)} units for {len(funcs)} functions to {out_dir}")
        return units
