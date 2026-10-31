#!/usr/bin/env python3
"""Recompile the game: analyze the dump, generate C++, compile every unit in parallel.

usage:
  build_game.py DUMP_DIR [--jobs N] [--unit-insns 80000] [--skip-generate]

The version comes from the dump (tools/check_dump.py). Outputs (derived from the game, git-ignored):
generated/game-<version>/*.cpp and build/game-<version>/ (objects, the memory image, the RomFS index); each
version builds next to the others. tools/link_game.py --version links it.
"""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import low_priority  # noqa: E402,F401 (builds yield the CPU to whatever else is running)
sys.path.insert(0, ROOT)
from recomp.analyze import Analysis  # noqa: E402
from recomp.generate import Generator, load_hooks  # noqa: E402
from recomp.image import Image  # noqa: E402
from tools.check_dump import check, find_dir  # noqa: E402
from tools import versions  # noqa: E402

CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
# -fwrapv / -fno-strict-aliasing: guest arithmetic wraps (two's complement) and guest memory is untyped; without
# them clang may optimize generated code on assumptions the game breaks (signed overflow, type-based aliasing).
FLAGS = ["-std=c++20", "-mcx16", "-fno-exceptions", "-fno-rtti", "-w", "-fwrapv", "-fno-strict-aliasing"]
HEADERS_MTIME = max(os.path.getmtime(os.path.join(ROOT, "runtime", "include", "zonaite", h))
                    for h in ("guest.h", "guest_fp.h", "guest_simd.h"))


def write_image(img, out_dir, version):
    """The relocated guest memory image and its description, loaded by the runtime at startup (replaces rtld)."""
    import json
    with open(os.path.join(out_dir, "image.bin"), "wb") as f:
        f.write(img.mem)
    sdk = next(m for m in img.modules if m.name == "sdk")
    desc = {
        "version": version,
        "base": img.base,
        "size": len(img.mem),
        "modules": [{"name": m.name, "base": m.base, "size": m.size,
                     "segments": {k: list(v) for k, v in m.segments.items()},
                     "init": m.base + m.dynamic[12][0] if 12 in m.dynamic else None,
                     "fini": m.base + m.dynamic[13][0] if 13 in m.dynamic else None} for m in img.modules],
        "symbols": {s.name: s.addr for m in img.modules for s in m.symbols
                    if s.defined and s.name and s.addr is not None},
        "entry": sdk.exports["_ZN2nn4init5StartEmmPFvvES2_S2_"].addr,
    }
    with open(os.path.join(out_dir, "image.json"), "w") as f:
        json.dump(desc, f)


def compile_unit(src, obj, opt):
    t0 = time.time()
    if os.path.exists(obj) and os.path.getmtime(obj) > max(os.path.getmtime(src), HEADERS_MTIME):
        return src, 0, "", 0.0  # up to date
    r = subprocess.run([CLANGXX, *FLAGS, f"-{opt}", "-I", os.path.join(ROOT, "runtime", "include"), "-c", src,
                        "-o", obj], capture_output=True, text=True)
    return src, r.returncode, r.stderr[-2000:], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--unit-insns", type=int, default=80000)
    ap.add_argument("--skip-generate", action="store_true")
    a = ap.parse_args()
    target, problems = check(a.dump)
    if problems:
        sys.exit("\n".join("error: " + p for p in problems))
    version = target["version"]
    print(f"version {version}", flush=True)
    src_dir = versions.generated_dir(version)
    obj_dir = versions.game_dir(version)
    os.makedirs(obj_dir, exist_ok=True)
    t_start = time.time()
    if not a.skip_generate:
        img = Image(find_dir(a.dump, "Exefs", "exefs", "ExeFS"))
        an = Analysis(img, log=lambda *x: None).run()
        print(f"analysis: {len(an.funcs)} functions ({time.time() - t_start:.0f}s)", flush=True)
        hooks = load_hooks(os.path.join(ROOT, "config", "hooks.txt"))
        Generator(img, an, hooks).write(src_dir, a.unit_insns, log=lambda m: print(m, flush=True))
        print(f"hooks: {len(hooks)}", flush=True)
        write_image(img, obj_dir, version)
        from tools import romfs
        nd, nf, _ = romfs.build(find_dir(a.dump, "RomFs", "romfs", "RomFS"), obj_dir)
        print(f"RomFS index: {nd} directories, {nf} files", flush=True)
        print(f"generated in {time.time() - t_start:.0f}s", flush=True)
    with open(os.path.join(src_dir, "units.txt")) as fp:
        units = [(p[0], p[1], int(p[2])) for p in (line.split() for line in fp)]
    current = {u for u, _, _ in units}
    for d, ext in ((src_dir, ".cpp"), (obj_dir, ".obj")):  # units that no longer exist
        for f in os.listdir(d):
            if f.endswith(ext) and f[:-len(ext)] + ".cpp" not in current:
                os.remove(os.path.join(d, f))
    units.sort(key=lambda u: -(u[2] * (4 if u[1] == "O2" else 1)))  # longest first
    t_compile = time.time()
    failed = []
    done = 0
    slowest = []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(compile_unit, os.path.join(src_dir, u), os.path.join(obj_dir, u[:-4] + ".obj"), opt)
                for u, opt, _ in units]
        for fut in concurrent.futures.as_completed(futs):
            src, rc, err, dt = fut.result()
            done += 1
            slowest.append((dt, os.path.basename(src)))
            if rc:
                failed.append((src, err))
            if done % 10 == 0 or done == len(units):
                print(f"compiled {done}/{len(units)} ({time.time() - t_compile:.0f}s), {len(failed)} failed",
                      flush=True)
    slowest.sort(reverse=True)
    total = sum(os.path.getsize(os.path.join(obj_dir, f)) for f in os.listdir(obj_dir) if f.endswith(".obj"))
    print(f"compile: {time.time() - t_compile:.0f}s wall with {a.jobs} jobs; objects {total / 1e6:.0f} MB; "
          f"slowest units: " + ", ".join(f"{n} {t:.0f}s" for t, n in slowest[:5]))
    for src, err in failed[:5]:
        print(f"\nFAILED {src}:\n{err}")
    print(f"total {time.time() - t_start:.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
