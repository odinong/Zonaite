"""Load the game's NSO modules into one guest address space and link them, like Horizon's rtld would.

Layout (fixed, no ASLR): modules are placed from LOAD_BASE in Horizon's load order (main, subsdk0..9, sdk),
each aligned to MODULE_ALIGN. rtld is not loaded: its only job is relocating and linking the other modules, which
this loader does ahead of time.

Every relocation is applied and kept (Module.relocs). Symbol imports resolve against the exports of all modules in
load order (Horizon's global lookup); anything left unresolved is listed in Image.unresolved.
"""
import os
import struct

from .lz4 import block_decompress

LOAD_BASE = 0x08000000
MODULE_ALIGN = 0x00100000
LOAD_ORDER = ["main"] + [f"subsdk{i}" for i in range(10)] + ["sdk"]

R_AARCH64_ABS64 = 257
R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027
RELOC_NAMES = {R_AARCH64_ABS64: "ABS64", R_AARCH64_GLOB_DAT: "GLOB_DAT", R_AARCH64_JUMP_SLOT: "JUMP_SLOT",
               R_AARCH64_RELATIVE: "RELATIVE"}

DT_NULL, DT_PLTRELSZ, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_RELA, DT_RELASZ = 0, 2, 4, 5, 6, 7, 8
DT_INIT, DT_FINI, DT_JMPREL, DT_INIT_ARRAY, DT_INIT_ARRAYSZ = 12, 13, 23, 25, 27

STB_LOCAL, STB_GLOBAL, STB_WEAK = 0, 1, 2
STT_NOTYPE, STT_OBJECT, STT_FUNC = 0, 1, 2


class Symbol:
    __slots__ = ("index", "name", "bind", "type", "shndx", "value", "size", "addr")

    def __init__(self, index, name, info, shndx, value, size, base):
        self.index = index
        self.name = name
        self.bind = info >> 4
        self.type = info & 15
        self.shndx = shndx
        self.value = value
        self.size = size
        self.addr = base + value if shndx != 0 else None

    @property
    def defined(self):
        return self.shndx != 0


class Reloc:
    __slots__ = ("where", "type", "sym", "addend", "value")

    def __init__(self, where, rtype, sym, addend):
        self.where = where      # guest address patched
        self.type = rtype
        self.sym = sym          # Symbol or None
        self.addend = addend
        self.value = None       # guest value written (None if unresolved)


class Module:
    def __init__(self, name, base, nso_path):
        self.name = name
        self.base = base
        with open(nso_path, "rb") as f:
            blob = f.read()
        if blob[:4] != b"NSO0":
            raise ValueError(f"{nso_path}: not an NSO")
        flags = struct.unpack_from("<I", blob, 0xC)[0]
        self.build_id = blob[0x40:0x54].hex().upper()
        self.segments = {}  # name -> (start, end) guest addresses
        image = bytearray()
        for k, seg in enumerate(("text", "ro", "data")):
            file_off, mem_off, size = struct.unpack_from("<III", blob, 0x10 + k * 0x10)
            file_size = struct.unpack_from("<I", blob, 0x60 + k * 4)[0]
            raw = blob[file_off:file_off + file_size]
            data = block_decompress(raw, size) if flags & (1 << k) else raw[:size]
            if mem_off < len(image):
                raise ValueError(f"{name}: segment {seg} overlaps the previous one")
            image.extend(b"\0" * (mem_off - len(image)))
            image.extend(data)
            self.segments[seg] = (base + mem_off, base + mem_off + size)
        bss = struct.unpack_from("<I", blob, 0x3C)[0]
        data_end = self.segments["data"][1] - base
        image.extend(b"\0" * (data_end + bss - len(image)))
        self.segments["bss"] = (base + data_end, base + data_end + bss)
        self.image = image
        self.size = (len(image) + 0xFFF) & ~0xFFF
        self.symbols = []
        self.exports = {}
        self.relocs = []
        self.init_array = []
        self.dynamic = {}
        self._parse_dynamic()

    # -- little-endian readers on the module's own image (offsets are module-relative)
    def u32(self, off):
        return struct.unpack_from("<I", self.image, off)[0]

    def u64(self, off):
        return struct.unpack_from("<Q", self.image, off)[0]

    def _parse_dynamic(self):
        mod0 = self.u32(4)
        if self.image[mod0:mod0 + 4] != b"MOD0":
            raise ValueError(f"{self.name}: MOD0 header not found")
        dyn_rel, _, _, eh_start, eh_end, _ = struct.unpack_from("<iiiiii", self.image, mod0 + 4)
        self.eh_frame_hdr = (self.base + mod0 + eh_start, self.base + mod0 + eh_end)
        off = mod0 + dyn_rel
        tags = {}
        while True:
            tag, val = struct.unpack_from("<qQ", self.image, off)
            off += 16
            if tag == DT_NULL:
                break
            tags.setdefault(tag, []).append(val)
        self.dynamic = tags
        strtab = tags[DT_STRTAB][0]
        symtab = tags[DT_SYMTAB][0]
        nsyms = self.u32(tags[DT_HASH][0] + 4)  # nchain == number of symbols

        def cstr(o):
            end = self.image.index(b"\0", strtab + o)
            return self.image[strtab + o:end].decode("utf-8", "replace")

        for i in range(nsyms):
            st_name, st_info, _, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", self.image,
                                                                                symtab + i * 24)
            sym = Symbol(i, cstr(st_name) if st_name else "", st_info, st_shndx, st_value, st_size, self.base)
            self.symbols.append(sym)
            if sym.defined and sym.name and sym.bind in (STB_GLOBAL, STB_WEAK):
                self.exports.setdefault(sym.name, sym)
        for start_tag, size_tag in ((DT_RELA, DT_RELASZ), (DT_JMPREL, DT_PLTRELSZ)):
            if start_tag not in tags:
                continue
            start, size = tags[start_tag][0], tags[size_tag][0]
            for o in range(start, start + size, 24):
                r_off, r_info, r_addend = struct.unpack_from("<QQq", self.image, o)
                rtype, rsym = r_info & 0xFFFFFFFF, r_info >> 32
                self.relocs.append(Reloc(self.base + r_off, rtype, self.symbols[rsym] if rsym else None, r_addend))
        if DT_INIT_ARRAY in tags:
            start, size = tags[DT_INIT_ARRAY][0], tags[DT_INIT_ARRAYSZ][0]
            self.init_array_range = (self.base + start, self.base + start + size)
        else:
            self.init_array_range = None

    def contains(self, addr):
        return self.base <= addr < self.base + self.size

    def in_text(self, addr):
        s, e = self.segments["text"]
        return s <= addr < e


class Image:
    """All modules mapped into one flat guest address range [LOAD_BASE, end)."""

    def __init__(self, exefs_dir, order=None):
        names = [n for n in (order or LOAD_ORDER) if os.path.isfile(os.path.join(exefs_dir, n))]
        if not names or names[0] != "main":
            raise ValueError(f"{exefs_dir}: no 'main' NSO")
        self.modules = []
        addr = LOAD_BASE
        for n in names:
            m = Module(n, addr, os.path.join(exefs_dir, n))
            self.modules.append(m)
            addr = (addr + m.size + MODULE_ALIGN - 1) & ~(MODULE_ALIGN - 1)
        self.base = LOAD_BASE
        self.end = self.modules[-1].base + self.modules[-1].size
        self.mem = bytearray(self.end - self.base)
        for m in self.modules:
            self.mem[m.base - self.base:m.base - self.base + len(m.image)] = m.image
        self.unresolved = []  # (module, Reloc) whose symbol no module defines
        self._link()

    def module_at(self, addr):
        for m in self.modules:
            if m.contains(addr):
                return m
        return None

    def lookup(self, name):
        for m in self.modules:
            s = m.exports.get(name)
            if s is not None and s.bind == STB_GLOBAL:
                return s
        for m in self.modules:  # weak definitions only if no strong one exists
            s = m.exports.get(name)
            if s is not None:
                return s
        return None

    def u32(self, addr):
        return struct.unpack_from("<I", self.mem, addr - self.base)[0]

    def u64(self, addr):
        return struct.unpack_from("<Q", self.mem, addr - self.base)[0]

    def put64(self, addr, value):
        struct.pack_into("<Q", self.mem, addr - self.base, value & 0xFFFFFFFFFFFFFFFF)

    def _link(self):
        for m in self.modules:
            for r in m.relocs:
                if r.type == R_AARCH64_RELATIVE:
                    r.value = m.base + r.addend
                elif r.type in (R_AARCH64_ABS64, R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT):
                    target = r.sym if r.sym.defined and r.sym.bind == STB_LOCAL else self.lookup(r.sym.name)
                    if target is None and r.sym.defined:
                        target = r.sym
                    if target is None:
                        if r.sym.bind != STB_WEAK:
                            self.unresolved.append((m, r))
                        r.value = 0  # unresolved weak symbols are null, like rtld
                        if r.sym.bind != STB_WEAK:
                            r.value = None
                            continue
                    else:
                        r.value = target.addr + r.addend
                else:
                    raise ValueError(f"{m.name}: unsupported relocation type {r.type} at {r.where:#x}")
                self.put64(r.where, r.value)

    def text_ranges(self):
        return [(m, *m.segments["text"]) for m in self.modules]
