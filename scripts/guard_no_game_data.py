#!/usr/bin/env python3
"""Block game data from commits and release packages.

usage:
  guard_no_game_data.py                       check files staged in git (runs as the pre-commit hook)
  guard_no_game_data.py --dir PACKAGE_DIR      check every file under a directory (run on the release package)
  options:
    --dump GAME_DIR      also hash every file of the user's dump and block exact copies of any of them
                         (defaults to the ZONAITE_DUMP environment variable when set)
    --allow-big NAME     file name allowed to exceed the size limit (repeatable: the runtime exe, bundled tools)
    --max-mb N           size limit for other files (default 8)
    --block-name GLOB    extra names to block (repeatable)

Exit code 1 (and a list of offending files) when something looks like game data; 0 otherwise.

Blocks: Switch executable/container/asset extensions, ExeFS file names (they have no extension), key files,
generated recompiler output, anything over the size limit that isn't allowed, files containing an NSO/NPDM/NCA
header, Discord webhook URLs, and (with --dump) byte-identical copies of the user's files.
"""
import argparse
import fnmatch
import hashlib
import os
import subprocess
import sys

BLOCKED_EXTS = {
    # Switch executables, containers and metadata
    ".nso", ".nro", ".nrr", ".npdm", ".nca", ".ncz", ".nsp", ".nsz", ".xci", ".xcz", ".nacp", ".cnmt", ".tik", ".cert",
    # keys
    ".keys",
    # Switch / Nintendo engine assets (TotK uses zstd-compressed versions of most of these)
    ".zs", ".pack", ".sarc", ".szs", ".bfres", ".bntx", ".bars", ".bwav", ".bfstm", ".bfsar", ".msbt", ".msbp",
    ".byml", ".bgyml", ".byaml", ".rstbl", ".bfarc", ".blarc", ".bflyt", ".bflan", ".bphsh", ".bphcl", ".bphnm",
    ".bkres", ".ainb", ".mc", ".bstar", ".bcett", ".esetb", ".genvb", ".ta", ".sbfres", ".sbactorpack",
    # other consoles, in case a sibling game's file slips in
    ".rpx", ".rpl", ".elf", ".dol", ".xex", ".iso", ".wbfs", ".rvz", ".wua",
    # memory dumps and saves made while testing
    ".bin", ".sav", ".dmp",
}
DEFAULT_BLOCKED_NAMES = [
    # ExeFS contents
    "main", "rtld", "sdk", "subsdk*", "main.npdm",
    # key files under any name
    "prod.keys", "title.keys", "dev.keys", "*keys.txt",
    # generated code and built game module
    "recomp_*.c", "recomp_*.cpp", "recomp_table.*", "*_game.dll", "*_game.so",
    # test output
    "memdump_*", "actor_*.bin",
]
BLOCKED_DIRS = {"generated", "exefs", "romfs", "dump", ".local"}
# Secrets that must never be committed, matched in file contents
SECRET_PATTERNS = (b"discord.com/api/webhooks/", b"discordapp.com/api/webhooks/")
# Header magics of Switch formats, checked at offset 0 of every file.
MAGICS = {b"NSO0": "NSO executable", b"META": "NPDM metadata", b"PFS0": "NSP/PFS0 container", b"MOD0": "module header",
          b"NRO0": "NRO executable", b"SARC": "SARC archive", b"Yaz0": "Yaz0 archive", b"BNTX": "texture archive",
          b"FRES": "model archive", b"\x28\xb5\x2f\xfd": "zstd-compressed data"}


def staged_files():
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], capture_output=True, text=True)
    return [f for f in out.stdout.splitlines() if f.strip()]


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def magic_of(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return None
    return MAGICS.get(head)


def secret_in(path):
    try:
        with open(path, "rb") as f:
            data = f.read(4 << 20)
    except OSError:
        return None
    for pat in SECRET_PATTERNS:
        i = data.find(pat)
        # a URL with an id after the prefix is a real webhook; the bare prefix (as in this file) is not
        if i >= 0 and data[i + len(pat):i + len(pat) + 1].isdigit():
            return "a Discord webhook URL"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir")
    ap.add_argument("--dump", default=os.environ.get("ZONAITE_DUMP"))
    ap.add_argument("--allow-big", action="append", default=[])
    ap.add_argument("--max-mb", type=float, default=8)
    ap.add_argument("--block-name", action="append", default=[])
    a = ap.parse_args()

    if a.dir:
        files = [os.path.join(r, n) for r, _, ns in os.walk(a.dir) for n in ns]
    else:
        files = staged_files()
    names = DEFAULT_BLOCKED_NAMES + a.block_name

    dump_hashes = {}
    if a.dump and os.path.isdir(a.dump):
        for r, _, ns in os.walk(a.dump):
            for n in ns:
                p = os.path.join(r, n)
                try:
                    if os.path.getsize(p) >= 4096:
                        dump_hashes[sha1(p)] = os.path.relpath(p, a.dump)
                except OSError:
                    pass

    bad = []
    for f in files:
        if not os.path.isfile(f):
            continue
        base = os.path.basename(f)
        ext = os.path.splitext(base)[1].lower()
        parts = {p.lower() for p in f.replace("\\", "/").split("/")[:-1]}
        size = os.path.getsize(f)
        magic = magic_of(f)
        if ext in BLOCKED_EXTS:
            bad.append(f"{f}: game-data extension {ext}")
        elif any(fnmatch.fnmatch(base.lower(), g.lower()) for g in names):
            bad.append(f"{f}: game, key or generated file name")
        elif parts & BLOCKED_DIRS:
            bad.append(f"{f}: inside a dump/generated directory ({', '.join(sorted(parts & BLOCKED_DIRS))})")
        elif magic:
            bad.append(f"{f}: starts with a {magic} header")
        elif secret_in(f):
            bad.append(f"{f}: contains {secret_in(f)} (secrets belong in .local/)")
        elif size > a.max_mb * 1024 * 1024 and base not in a.allow_big:
            bad.append(f"{f}: {size / 1048576:.1f} MB (over {a.max_mb} MB and not allowed)")
        elif dump_hashes and size >= 4096 and sha1(f) in dump_hashes:
            bad.append(f"{f}: identical to the game's {dump_hashes[sha1(f)]}")
    if bad:
        print("game data found, nothing should be committed or packaged:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print(f"ok: {len(files)} file(s) checked" + (f" against {len(dump_hashes)} dump file(s)" if dump_hashes else ""))


if __name__ == "__main__":
    main()
