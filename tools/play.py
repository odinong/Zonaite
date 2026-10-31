#!/usr/bin/env python3
"""Start Zonaite, building the game from your dump first when needed. The launcher (launcher/) runs this; it
also works on its own.

usage:
  play.py --add-dump DIR             check a dump folder and remember it for its version (prints JSON)
  play.py [--version V] [--prepare] [--rebuild]

A dump is the folder holding your extracted Exefs and RomFs. Each supported version (config/targets) can have
one; they are remembered in user/dumps.json. Without --version the newest version with a dump is played. The
first run of a version (and any run after an update that changes the recompiler, see tools/build_id.py)
recompiles the game from its dump into build/game-<version>, which takes several minutes, then links
build/zonaite-<version>.exe on this PC. --prepare stops before starting the game. Nothing derived from the game
ever leaves your machine.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_id  # noqa: E402
import check_dump  # noqa: E402
import versions  # noqa: E402

DUMPS = os.path.join(ROOT, "user", "dumps.json")


def load_dumps():
    try:
        with open(DUMPS) as f:
            return json.load(f)
    except (OSError, ValueError):
        dumps = {}
    old = os.path.join(ROOT, "user", "dump.txt")  # before versions: one 1.2.1 dump
    if os.path.isfile(old):
        dumps["1.2.1"] = open(old).read().strip()
    return dumps


def save_dumps(dumps):
    os.makedirs(os.path.dirname(DUMPS), exist_ok=True)
    with open(DUMPS, "w") as f:
        json.dump(dumps, f, indent=2)


def add_dump(path):
    """Checks a dump and remembers it under its version; prints the result as JSON for the launcher"""
    path = os.path.abspath(path)
    t, problems = check_dump.check(path)
    result = {"version": t["version"] if t else None, "label": t.get("label") if t else None,
              "verified": bool(t and t.get("verified")), "problems": problems, "path": path}
    if t and not problems:
        dumps = load_dumps()
        dumps[t["version"]] = path
        save_dumps(dumps)
    print(json.dumps(result))
    return 0 if not problems else 1


def newest(paths):
    return max((os.path.getmtime(p) for p in paths), default=0)


def prepare(version, dump, rebuild):
    """Recompiles and links the version when needed; returns the exe"""
    game = versions.game_dir(version)
    stamp = os.path.join(game, "BUILD_ID")
    wanted = build_id.compute(ROOT)
    have = open(stamp).read().strip() if os.path.isfile(stamp) else None
    if rebuild or have != wanted:
        print(f"Preparing version {version}: recompiling it from your dump (several minutes the first time)...",
              flush=True)
        if subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_game.py"), dump]).returncode:
            sys.exit("The game build failed (see above).")
        with open(stamp, "w") as f:
            f.write(wanted + "\n")
    exe = versions.exe(version)
    lib_dir = os.path.join(ROOT, "lib")
    package = os.path.isdir(lib_dir)  # a release package; a repository checkout links in tools/build_runtime.py
    inputs = glob.glob(os.path.join(game, "*.obj")) + (glob.glob(os.path.join(lib_dir, "*.lib")) if package else [])
    if not os.path.isfile(exe) or os.path.getmtime(exe) < newest(inputs):
        print("Linking Zonaite...", flush=True)
        cmd = [sys.executable, os.path.join(ROOT, "tools", "link_game.py"), "--version", version]
        if package:
            cmd += ["--lib-dir", lib_dir]
        if subprocess.run(cmd).returncode:
            sys.exit("Linking failed (see above).")
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add-dump", metavar="DIR")
    ap.add_argument("--version")
    ap.add_argument("--prepare", action="store_true", help="build and link if needed, but don't start the game")
    ap.add_argument("--rebuild", action="store_true", help="recompile the game even if it is up to date")
    a = ap.parse_args()
    if a.add_dump:
        return add_dump(a.add_dump)

    dumps = load_dumps()
    version = a.version or next((t["version"] for t in check_dump.targets() if t["version"] in dumps), None)
    if not version or version not in dumps:
        sys.exit("No dump for " + (f"version {version}" if version else "any version") +
                 " yet: add one with play.py --add-dump <folder> (or in the launcher)")
    dump = dumps[version]
    exe = prepare(version, dump, a.rebuild)
    if a.prepare:
        print("ready", flush=True)
        return 0
    return subprocess.run([exe, versions.game_dir(version), dump], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
