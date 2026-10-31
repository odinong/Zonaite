#!/usr/bin/env python3
"""Make a release package: Zonaite's runtime (prebuilt libraries), the recompiler and the build scripts that turn
the player's own dump into the game on their PC. The package holds no game code, no game data and no keys.

usage:
  package.py [--out DIR] [--zip] [--check-dump DUMP_DIR]

Steps: compile the runtime (tools/build_runtime.py --no-link), archive its objects into lib/runtime.lib, copy the
GPU/shader/third-party libraries, the recompiler (recomp/), the game build scripts, the runtime headers the
generated code includes, config/ and the licences; write BUILD_ID (a hash of everything that decides the generated
code) and check that the copied recompiler hashes the same; then run scripts/guard_no_game_data.py over the
package (--check-dump also blocks byte-identical copies of any file of that dump). If the guard fires the
package is deleted.

On the player's machine, Zonaite.exe (the launcher, launcher/) sets up their dumps and lets them pick a version;
tools/play.py then builds that version from its dump the first time (and whenever BUILD_ID changes), links
build/zonaite-<version>.exe there (tools/link_game.py --lib-dir lib) and the launcher starts it.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import low_priority  # noqa: E402,F401 (builds yield the CPU to whatever else is running)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_id  # noqa: E402
import build_runtime  # noqa: E402
import link_game  # noqa: E402

# Files a package carries, relative to the repository (globs)
PACKAGE_FILES = ["LICENSE", "README.md", "THIRD_PARTY.md", "VERSION", "docs/DEVELOPMENT.md",
                 "recomp/*.py", "config/hooks.txt", "config/targets/*.json",
                 "tools/__init__.py", "tools/build_game.py", "tools/check_dump.py", "tools/romfs.py",
                 "tools/link_game.py", "tools/play.py", "tools/build_id.py", "tools/low_priority.py",
                 "tools/versions.py",
                 "runtime/include/zonaite/*.h",
                 "scripts/guard_no_game_data.py",
                 "res/fonts/*", "third_party/imgui/LICENSE.txt"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "release"))
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--check-dump")
    a = ap.parse_args()
    version = open(os.path.join(ROOT, "VERSION")).read().strip()
    name = f"Zonaite-{version}"
    pkg = os.path.join(a.out, name)
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    os.makedirs(os.path.join(pkg, "lib"))

    # the runtime, compiled and archived (no game code: it only reads build/game at run time)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_runtime.py"), "--no-link"])
    if r.returncode:
        sys.exit("package: runtime build failed")
    llvm_lib = shutil.which("llvm-lib") or r"C:\Program Files\LLVM\bin\llvm-lib.exe"
    rsp = os.path.join(a.out, "runtime_lib.rsp")
    with open(rsp, "w") as f:
        for o in build_runtime.runtime_objects():
            f.write(f'"{o.replace(os.sep, "/")}"\n')
    r = subprocess.run([llvm_lib, f"/out:{os.path.join(pkg, 'lib', 'runtime.lib')}", f"@{rsp}"])
    os.remove(rsp)
    if r.returncode:
        sys.exit("package: archiving the runtime failed")
    _, libs = link_game.repo_inputs()
    for lib in libs:
        shutil.copy2(lib, os.path.join(pkg, "lib", os.path.basename(lib)))

    for pattern in PACKAGE_FILES:
        matches = glob.glob(os.path.join(ROOT, pattern))
        if not matches:
            sys.exit(f"package: nothing matches {pattern}")
        for src in matches:
            if os.path.isdir(src):
                continue
            dst = os.path.join(pkg, os.path.relpath(src, ROOT))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    repo_id, pkg_id = build_id.compute(ROOT), build_id.compute(pkg)
    if repo_id != pkg_id:
        shutil.rmtree(pkg)
        sys.exit(f"package: the packaged recompiler's build ID {pkg_id} differs from the repository's {repo_id}")
    with open(os.path.join(pkg, "BUILD_ID"), "w") as f:
        f.write(repo_id + "\n")
    # the launcher (launcher/, no game code): first-run setup, the version choice, then it starts the game
    if subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_launcher.py"),
                       "--out", os.path.join(pkg, "Zonaite.exe")]).returncode:
        shutil.rmtree(pkg)
        sys.exit("package: the launcher build failed")

    guard = [sys.executable, os.path.join(ROOT, "scripts", "guard_no_game_data.py"), "--dir", pkg]
    for lib in os.listdir(os.path.join(pkg, "lib")):
        guard += ["--allow-big", lib]  # Zonaite's and third-party libraries (licences in THIRD_PARTY.md)
    if a.check_dump:
        guard += ["--dump", a.check_dump]
    if subprocess.run(guard).returncode:
        shutil.rmtree(pkg)
        sys.exit("package: the guard fired; package deleted")

    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(pkg) for f in fs)
    print(f"package: {pkg} ({size / 1e6:.0f} MB), build ID {repo_id}")
    if a.zip:
        archive = shutil.make_archive(pkg, "zip", a.out, name)
        print(f"package: {archive} ({os.path.getsize(archive) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
