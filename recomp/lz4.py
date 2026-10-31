"""Raw LZ4 block decompression (no frame header), as used by NSO segments."""


def block_decompress(src, out_size):
    dst = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0 or off > len(dst):
            raise ValueError(f"bad LZ4 match offset {off} at input {i}")
        ml = token & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(dst) - off
        if ml <= off:
            dst += dst[start:start + ml]
        else:  # overlapping copy repeats the last `off` bytes
            reps, rem = divmod(ml, off)
            chunk = bytes(dst[start:])
            dst += chunk * reps + chunk[:rem]
    if len(dst) != out_size:
        raise ValueError(f"LZ4 output is {len(dst)} bytes, header says {out_size}")
    return bytes(dst)
