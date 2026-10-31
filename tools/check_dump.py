#!/usr/bin/env python3
"""Check a dump folder: which supported version of the game it is (config/targets/*.json), and that it is
complete and unmodified.

usage:
  check_dump.py DUMP_DIR [--json]

DUMP_DIR holds the extracted ExeFS and RomFS as `Exefs/` + `RomFs/` (Ryujinx's names) or `exefs/` + `romfs/`.
The version comes from the build ID in ExeFS/main; the files with pinned hashes must match them, and the RomFS
version marker must name the same version (a RomFS with an update applied next to an ExeFS without it, or the
other way round, is refused). Exit code 0 when it passes; 1 with a reason per problem otherwise. --json prints
{"version", "label", "verified", "problems"} for the launcher.
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_dir(base, *names):
    for n in names:
        p = os.path.join(base, n)
        if os.path.isdir(p):
            return p
    return None


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def targets():
    """Every supported version, newest first"""
    out = []
    for path in glob.glob(os.path.join(ROOT, "config", "targets", "*.json")):
        with open(path) as f:
            out.append(json.load(f))
    return sorted(out, key=lambda t: [int(x) for x in t["version"].split(".")], reverse=True)


def target(version):
    for t in targets():
        if t["version"] == version:
            return t
    sys.exit(f"no such version: {version} (config/targets)")


def nso_build_id(path):
    """The module ID in an NSO header (its first 20 bytes are the build ID)"""
    with open(path, "rb") as f:
        head = f.read(0x60)
    if len(head) < 0x60 or head[:4] != b"NSO0":
        return None
    return head[0x40:0x54].hex().upper()


def check(dump, version=None):
    """(target, problems): the version the dump is (or the one asked for), and what is wrong with it"""
    problems = []
    exefs = find_dir(dump, "Exefs", "exefs", "ExeFS")
    romfs = find_dir(dump, "RomFs", "romfs", "RomFS")
    found = None
    if not exefs:
        problems.append(f"no ExeFS folder (Exefs/ or exefs/) in {dump}")
    else:
        main = os.path.join(exefs, "main")
        build = nso_build_id(main) if os.path.isfile(main) else None
        if not build:
            problems.append("ExeFS/main is missing or not a game executable")
        else:
            found = next((t for t in targets() if t["exefs"]["main"]["build_id"] == build), None)
            if not found:
                problems.append(f"ExeFS/main is not a supported version of the game (build {build[:16]}); "
                                f"supported: " + ", ".join(t["version"] for t in targets()))
    t = found
    if version:
        t = target(version)
        if found and found is not t and found["version"] != version:
            problems.append(f"this dump is version {found['version']}, not {version}")
    if t and exefs and found:
        for name, want in t["exefs"].items():
            p = os.path.join(exefs, name)
            if not os.path.isfile(p):
                problems.append(f"ExeFS/{name} is missing")
            elif "sha1" in want and sha1(p) != want["sha1"]:
                problems.append(f"ExeFS/{name} is not the one from version {t['version']} (modified?)")
    if not romfs:
        problems.append(f"no RomFS folder (RomFs/ or romfs/) in {dump}")
    elif t:
        marker = t["romfs_version_marker"]
        p = os.path.join(romfs, *marker["file"].split("/"))
        try:
            with open(p, encoding="ascii", errors="replace") as f:
                lines = [l.strip() for l in f]
        except OSError:
            lines = None
        if lines is None:
            problems.append(f"RomFS/{marker['file']} is missing")
        elif marker["version_line"] not in lines or marker["build_id_line"] not in lines:
            problems.append(f"the RomFS is not from version {t['version']} (extracted with a different update "
                            f"applied than the ExeFS?)")
    return t, problems


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        sys.exit(2)
    t, problems = check(args[0])
    if "--json" in sys.argv:
        print(json.dumps({"version": t["version"] if t else None, "label": t.get("label") if t else None,
                          "verified": bool(t and t.get("verified")), "problems": problems}))
        sys.exit(1 if problems else 0)
    if problems:
        for p in problems:
            print("error: " + p)
        sys.exit(1)
    note = "" if t.get("verified") else " (not yet verified on a real dump of this version)"
    print(f"ok: {t['title']} {t['version']} ({t['program_id']}){note}")


if __name__ == "__main__":
    main()
