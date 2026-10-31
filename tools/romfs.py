#!/usr/bin/env python3
"""Build a RomFS index from an extracted RomFS folder: the metadata the SDK's RomFS driver expects.

The game reads its RomFS as one storage (fsp-srv OpenDataStorageByCurrentProcess) and parses the RomFS format
itself. The runtime presents a virtual image: header + directory/file tables from this index, and the data area
mapped onto the extracted files.

Image layout produced (offsets in the virtual storage):
  0x000            header (0x50 bytes, padded to 0x200)
  0x200            file data, each file 16-byte aligned, in table order
  tables_offset    dir hash table, dir meta table, file hash table, file meta table

Outputs (derived from the dump, kept in build/game):
  romfs_meta.bin   header (0x200 bytes) followed by the tables blob (which belongs at tables_offset)
  romfs_files.txt  one line per file: data offset (absolute), size, path relative to the RomFS root
"""
import os
import struct
import sys

EMPTY = 0xFFFFFFFF
DATA_OFFSET = 0x200


def path_hash(parent, name):
    h = parent ^ 123456789
    for b in name:
        h = ((h >> 5) | (h << 27)) & 0xFFFFFFFF
        h ^= b
    return h


def bucket_count(n):
    if n < 3:
        return 3
    if n < 19:
        return n | 1
    c = n
    while any(c % p == 0 for p in (2, 3, 5, 7, 11, 13, 17)):
        c += 1
    return c


def align(v, a):
    return (v + a - 1) & ~(a - 1)


def scan(root):
    """Directories (breadth-first) and files, children sorted by UTF-8 name."""
    dirs = [{"rel": "", "name": b"", "parent": 0, "subdirs": [], "files": []}]
    queue = [0]
    while queue:
        di = queue.pop(0)
        d = dirs[di]
        full = os.path.join(root, d["rel"])
        entries = sorted(os.scandir(full), key=lambda e: e.name.encode("utf-8"))
        for e in entries:
            rel = (d["rel"] + "/" + e.name) if d["rel"] else e.name
            if e.is_dir(follow_symlinks=False):
                dirs.append({"rel": rel, "name": e.name.encode("utf-8"), "parent": di, "subdirs": [], "files": []})
                d["subdirs"].append(len(dirs) - 1)
                queue.append(len(dirs) - 1)
            elif e.is_file(follow_symlinks=False):
                d["files"].append({"rel": rel, "name": e.name.encode("utf-8"), "size": e.stat().st_size})
    return dirs


def build(root, out_dir):
    dirs = scan(root)
    files = [f for d in dirs for f in d["files"]]
    for d_index, d in enumerate(dirs):
        for f in d["files"]:
            f["dir"] = d_index
    # meta offsets
    off = 0
    for d in dirs:
        d["off"] = off
        off += 0x18 + align(len(d["name"]), 4)
    dir_meta_size = off
    off = 0
    for f in files:
        f["off"] = off
        off += 0x20 + align(len(f["name"]), 4)
    file_meta_size = off
    # data layout
    pos = DATA_OFFSET
    for f in files:
        pos = align(pos, 16)
        f["data"] = pos
        pos += f["size"]
    tables_offset = align(pos, 16)
    # hash tables
    dir_buckets = [EMPTY] * bucket_count(len(dirs))
    for d in dirs:
        parent_off = dirs[d["parent"]]["off"]
        b = path_hash(parent_off, d["name"]) % len(dir_buckets)
        d["hash_next"] = dir_buckets[b]
        dir_buckets[b] = d["off"]
    file_buckets = [EMPTY] * bucket_count(len(files))
    for f in files:
        parent_off = dirs[f["dir"]]["off"]
        b = path_hash(parent_off, f["name"]) % len(file_buckets)
        f["hash_next"] = file_buckets[b]
        file_buckets[b] = f["off"]
    # meta tables
    dir_meta = bytearray()
    for d in dirs:
        parent = dirs[d["parent"]]
        siblings = parent["subdirs"] if d["off"] else []
        idx = siblings.index(dirs.index(d)) if d["off"] else -1
        sibling = dirs[siblings[idx + 1]]["off"] if d["off"] and idx + 1 < len(siblings) else EMPTY
        child_dir = dirs[d["subdirs"][0]]["off"] if d["subdirs"] else EMPTY
        child_file = d["files"][0]["off"] if d["files"] else EMPTY
        dir_meta += struct.pack("<IIIIII", parent["off"], sibling, child_dir, child_file, d["hash_next"],
                                len(d["name"]))
        dir_meta += d["name"] + b"\0" * (align(len(d["name"]), 4) - len(d["name"]))
    file_meta = bytearray()
    for d in dirs:
        for i, f in enumerate(d["files"]):
            sibling = d["files"][i + 1]["off"] if i + 1 < len(d["files"]) else EMPTY
            file_meta += struct.pack("<IIQQII", d["off"], sibling, f["data"] - DATA_OFFSET, f["size"],
                                     f["hash_next"], len(f["name"]))
            file_meta += f["name"] + b"\0" * (align(len(f["name"]), 4) - len(f["name"]))
    assert len(dir_meta) == dir_meta_size and len(file_meta) == file_meta_size
    dir_hash = struct.pack(f"<{len(dir_buckets)}I", *dir_buckets)
    file_hash = struct.pack(f"<{len(file_buckets)}I", *file_buckets)
    tables = bytearray()
    t_dir_hash = tables_offset
    tables += dir_hash
    t_dir_meta = tables_offset + len(tables)
    tables += dir_meta
    t_file_hash = tables_offset + len(tables)
    tables += file_hash
    t_file_meta = tables_offset + len(tables)
    tables += file_meta
    header = struct.pack("<10Q", 0x50, t_dir_hash, len(dir_hash), t_dir_meta, len(dir_meta), t_file_hash,
                         len(file_hash), t_file_meta, len(file_meta), DATA_OFFSET)
    header += b"\0" * (DATA_OFFSET - len(header))
    with open(os.path.join(out_dir, "romfs_meta.bin"), "wb") as f:
        f.write(header + tables)
    with open(os.path.join(out_dir, "romfs_files.txt"), "w", encoding="utf-8") as f:
        f.write(f"# tables_offset {tables_offset} total_size {tables_offset + len(tables)}\n")
        for fi in files:
            f.write(f"{fi['data']}\t{fi['size']}\t{fi['rel']}\n")
    return len(dirs), len(files), tables_offset + len(tables)


if __name__ == "__main__":
    nd, nf, total = build(sys.argv[1], sys.argv[2])
    print(f"RomFS index: {nd} directories, {nf} files, virtual image {total / 1e9:.2f} GB")
