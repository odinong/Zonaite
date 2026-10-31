"""Parse .eh_frame_hdr + .eh_frame to get every FDE's code range (and LSDA pointer).

Each FDE covers exactly one function (or one cold part of one), so its [pc_begin, pc_begin + pc_range) is the most
reliable function boundary there is.
"""
import struct

DW_EH_PE_omit = 0xFF
DW_EH_PE_absptr, DW_EH_PE_uleb128, DW_EH_PE_udata2, DW_EH_PE_udata4, DW_EH_PE_udata8 = 0x00, 0x01, 0x02, 0x03, 0x04
DW_EH_PE_sleb128, DW_EH_PE_sdata2, DW_EH_PE_sdata4, DW_EH_PE_sdata8 = 0x09, 0x0A, 0x0B, 0x0C
DW_EH_PE_pcrel, DW_EH_PE_datarel = 0x10, 0x30
DW_EH_PE_indirect = 0x80


class Fde:
    __slots__ = ("addr", "start", "end", "lsda", "personality")

    def __init__(self, addr, start, end, lsda, personality):
        self.addr = addr
        self.start = start
        self.end = end
        self.lsda = lsda
        self.personality = personality


class _Reader:
    def __init__(self, image, addr):
        self.image = image
        self.addr = addr

    def u8(self):
        v = self.image.mem[self.addr - self.image.base]
        self.addr += 1
        return v

    def fixed(self, fmt, size):
        v = struct.unpack_from(fmt, self.image.mem, self.addr - self.image.base)[0]
        self.addr += size
        return v

    def uleb(self):
        result = shift = 0
        while True:
            b = self.u8()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return result

    def sleb(self):
        result = shift = 0
        while True:
            b = self.u8()
            result |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                if b & 0x40:
                    result -= 1 << shift
                return result

    def cstr(self):
        start = self.addr - self.image.base
        end = self.image.mem.index(b"\0", start)
        self.addr += end - start + 1
        return bytes(self.image.mem[start:end]).decode("ascii")

    def encoded(self, enc, datarel_base=0):
        if enc == DW_EH_PE_omit:
            return None
        pos = self.addr
        fmt = enc & 0x0F
        if fmt == DW_EH_PE_absptr or fmt == DW_EH_PE_udata8:
            v = self.fixed("<Q", 8)
        elif fmt == DW_EH_PE_udata4:
            v = self.fixed("<I", 4)
        elif fmt == DW_EH_PE_sdata4:
            v = self.fixed("<i", 4)
        elif fmt == DW_EH_PE_udata2:
            v = self.fixed("<H", 2)
        elif fmt == DW_EH_PE_sdata2:
            v = self.fixed("<h", 2)
        elif fmt == DW_EH_PE_sdata8:
            v = self.fixed("<q", 8)
        elif fmt == DW_EH_PE_uleb128:
            v = self.uleb()
        elif fmt == DW_EH_PE_sleb128:
            v = self.sleb()
        else:
            raise ValueError(f"unsupported pointer encoding {enc:#x} at {pos:#x}")
        app = enc & 0x70
        if app == DW_EH_PE_pcrel:
            v += pos
        elif app == DW_EH_PE_datarel:
            v += datarel_base
        elif app != 0:
            raise ValueError(f"unsupported pointer application {enc:#x} at {pos:#x}")
        v &= 0xFFFFFFFFFFFFFFFF
        if enc & DW_EH_PE_indirect:
            v = self.image.u64(v)
        return v


def _parse_cie(image, addr):
    r = _Reader(image, addr)
    length = r.fixed("<I", 4)
    if length == 0xFFFFFFFF:
        raise ValueError(f"64-bit CIE at {addr:#x} not supported")
    r.fixed("<I", 4)  # CIE id (0)
    version = r.u8()
    aug = r.cstr()
    r.uleb()  # code alignment
    r.sleb()  # data alignment
    if version == 1:
        r.u8()
    else:
        r.uleb()
    fde_enc, lsda_enc, personality = DW_EH_PE_absptr, DW_EH_PE_omit, None
    if aug.startswith("z"):
        r.uleb()  # augmentation data length
        for ch in aug[1:]:
            if ch == "R":
                fde_enc = r.u8()
            elif ch == "L":
                lsda_enc = r.u8()
            elif ch == "P":
                personality = r.encoded(r.u8())
            elif ch in "SB":
                pass
            else:
                raise ValueError(f"unknown CIE augmentation {aug!r} at {addr:#x}")
    return fde_enc, lsda_enc, personality


def parse_module(image, module):
    """Return the module's FDEs sorted by start address, via the binary search table in .eh_frame_hdr."""
    hdr, _ = module.eh_frame_hdr
    r = _Reader(image, hdr)
    version = r.u8()
    if version != 1:
        raise ValueError(f"{module.name}: eh_frame_hdr version {version}")
    ptr_enc, count_enc, table_enc = r.u8(), r.u8(), r.u8()
    r.encoded(ptr_enc, hdr)  # eh_frame start
    count = r.encoded(count_enc, hdr)
    cies = {}
    fdes = []
    for _ in range(count):
        r.encoded(table_enc, hdr)  # initial location (same as the FDE's pc_begin)
        fde_addr = r.encoded(table_enc, hdr)
        f = _Reader(image, fde_addr)
        length = f.fixed("<I", 4)
        if length == 0xFFFFFFFF:
            raise ValueError(f"64-bit FDE at {fde_addr:#x} not supported")
        cie_ptr_pos = f.addr
        cie_addr = cie_ptr_pos - f.fixed("<I", 4)
        if cie_addr not in cies:
            cies[cie_addr] = _parse_cie(image, cie_addr)
        fde_enc, lsda_enc, personality = cies[cie_addr]
        start = f.encoded(fde_enc)
        rng = f.encoded(fde_enc & 0x0F)
        lsda = None
        if lsda_enc != DW_EH_PE_omit:
            f.uleb()
            lsda = f.encoded(lsda_enc)
            if lsda == 0:
                lsda = None
        fdes.append(Fde(fde_addr, start, start + rng, lsda, personality))
    fdes.sort(key=lambda x: x.start)
    return fdes
