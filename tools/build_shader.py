#!/usr/bin/env python3
"""Build Eden's shader recompiler (SPIR-V backend) with sirit and the needed parts of Eden's common library into
build/shader/shader.lib, plus the shader translation tool.

usage:
  build_shader.py [--jobs N]
"""
import argparse
import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import low_priority  # noqa: E402,F401 (builds yield the CPU to whatever else is running)
TP = os.path.join(ROOT, "third_party")
CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
LLVM_LIB = os.path.join(os.path.dirname(CLANGXX), "llvm-lib.exe")
INCLUDES = [
    os.path.join(TP, "eden", "src"),
    os.path.join(TP, "sirit", "include"),
    os.path.join(TP, "sirit", "src"),
    os.path.join(TP, "deps", "spirv-headers", "include"),
    os.path.join(TP, "deps", "fmt", "include"),
    os.path.join(TP, "deps", "boost"),
]
DEFINES = ["FMT_HEADER_ONLY", "NOMINMAX", "WIN32_LEAN_AND_MEAN", "_CRT_SECURE_NO_WARNINGS",
           "BOOST_ALL_NO_LIB", "_USE_MATH_DEFINES"]
FLAGS = ["-std=c++20", "-O2", "-mcx16", "-w", "-fms-extensions"] + [f"-D{d}" for d in DEFINES] + \
        [x for i in INCLUDES for x in ("-I", i)]

COMMON = ["assert.cpp", "logging.cpp", "settings.cpp", "settings_common.cpp", "fs/fs.cpp", "fs/file.cpp",
          "fs/path_util.cpp", "fs/fs_util.cpp", "string_util.cpp", "time_zone.cpp"]


def sources():
    s = sorted(glob.glob(os.path.join(TP, "eden", "src", "shader_recompiler", "**", "*.cpp"), recursive=True))
    s += [os.path.join(TP, "eden", "src", "common", c) for c in COMMON]
    s += sorted(glob.glob(os.path.join(TP, "sirit", "src", "**", "*.cpp"), recursive=True))
    return s


def compile_one(src, obj):
    if os.path.exists(obj) and os.path.getmtime(obj) > os.path.getmtime(src):
        return src, 0, ""
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    r = subprocess.run([CLANGXX, *FLAGS, "-c", src, "-o", obj], capture_output=True, text=True)
    return src, r.returncode, r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    out = os.path.join(ROOT, "build", "shader")
    srcs = sources()
    objs = [os.path.join(out, "obj", os.path.relpath(s, TP).replace(os.sep, "_")[:-4] + ".obj") for s in srcs]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(compile_one, srcs, objs))
    bad = [(s, e) for s, rc, e in results if rc]
    print(f"{len(srcs)} sources, {len(bad)} failed")
    for s, e in bad[:a.show]:
        errs = [l for l in e.splitlines() if "error" in l][:4]
        print(f"\n{os.path.relpath(s, TP)}:\n  " + "\n  ".join(errs))
    if bad:
        sys.exit(1)
    lib = os.path.join(out, "shader.lib")
    rsp = os.path.join(out, "lib.rsp")
    with open(rsp, "w") as f:
        f.write("\n".join(f'"{o}"' for o in objs))
    subprocess.run([LLVM_LIB, f"/OUT:{lib}", f"@{rsp}"], check=True)
    print(f"built {lib}")


if __name__ == "__main__":
    main()
