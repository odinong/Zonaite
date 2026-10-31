#!/usr/bin/env python3
"""Build Eden's video_core (GPU emulation + Vulkan renderer) into build/gpu/gpu.lib.

usage:
  build_gpu.py [--jobs N] [--show N] [--only SUBSTRING]

Needs: tools/build_host_shaders.py output (build/gpu/include), the Vulkan SDK headers, third_party/deps
(VMA, xbyak, Boost, fmt, SPIRV-Headers). Objects are rebuilt when their source is newer, or any header Zonaite
changes (its overrides, and the Eden headers tools/eden_patches.py patches): a header that changes a class layout
must not leave objects built against the old one.
Zonaite's replacements for Eden's core (Core::System and friends) live in runtime/src/gpu_*.cpp and are built
with the runtime, not here.
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
EDEN = os.path.join(TP, "eden")
CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
LLVM_LIB = os.path.join(os.path.dirname(CLANGXX), "llvm-lib.exe")
VULKAN_SDK = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.363.0")
EDEN_SRC_CMAKE = os.path.join(EDEN, "src")  # Eden's CMake source lists (vendored with the sources)
INCLUDES = [
    os.path.join(ROOT, "runtime", "gpu", "override_video"),  # Zonaite's replacements for video_core headers
    os.path.join(EDEN, "src"),
    os.path.join(ROOT, "build", "gpu", "include"),
    os.path.join(ROOT, "runtime", "include"),  # zonaite/config.h, zonaite/gpu.h (frame_interp.cpp)
    os.path.join(VULKAN_SDK, "Include"),
    os.path.join(TP, "deps", "vma", "include"),
    os.path.join(TP, "deps", "xbyak"),
    os.path.join(EDEN, "externals", "stb"),
    os.path.join(EDEN, "externals", "bc_decoder"),
    os.path.join(TP, "sirit", "include"),
    os.path.join(TP, "deps", "spirv-headers", "include"),
    os.path.join(TP, "deps", "fmt", "include"),
    os.path.join(TP, "deps", "boost"),
    os.path.join(TP, "deps", "ffmpeg", "include"),
    os.path.join(TP, "deps", "opus", "include", "opus"),  # audio_core includes <opus.h>
]
DEFINES = ["FMT_HEADER_ONLY", "NOMINMAX", "WIN32_LEAN_AND_MEAN", "UNICODE", "_UNICODE", "_CRT_SECURE_NO_WARNINGS",
           "_CRT_NONSTDC_NO_DEPRECATE", "_SCL_SECURE_NO_WARNINGS", "_WIN32_WINNT=0x0A00", "WINVER=0x0A00",
           "BOOST_ALL_NO_LIB", "_USE_MATH_DEFINES", "ARCHITECTURE_x86_64=1", 'BUILD_ID="zonaite"',
           "HAS_LSFG"]  # Eden's frame generation hooks, driving Zonaite's own interpolation (runtime/gpu/frame_interp.cpp)
FLAGS = ["-std=c++20", "-O2", "-mcx16", "-msse4.2", "-w", "-fms-extensions", "-g"] + \
        [f"-D{d}" for d in DEFINES] + [x for i in INCLUDES for x in ("-I", i)]

# OpenGL renderer (not vendored); ReShade post-processing (optional in Eden: ENABLE_RESHADE)
EXCLUDE = ["renderer_opengl", "post_processing", os.path.join("present", "post_process.cpp"),
           # Eden's frame generation runs Lossless Scaling's shaders; Zonaite has its own (frame_interp.cpp)
           os.path.join("present", "frame_gen.cpp"), os.path.join("present", "lsfg_"),
           os.path.join("video_core", "frame_gen", "")]


def listed_in_cmake(path):
    """True if the nearest CMakeLists.txt up the tree lists this source. The tree also holds dead files (e.g.
    host1x/codecs/codec.cpp, which no longer matches its headers) and sub-targets (gpu_logging) with their own
    lists."""
    d = os.path.dirname(path)
    while len(d) >= len(EDEN_SRC_CMAKE):
        cm = os.path.join(d, "CMakeLists.txt")
        if os.path.exists(cm):
            with open(cm, encoding="utf-8") as f:
                return os.path.relpath(path, d).replace(os.sep, "/") in f.read()
        d = os.path.dirname(d)
    return False


# Eden common sources video_core needs beyond those already in build/shader/shader.lib
COMMON = ["dynamic_library.cpp", "multi_level_page_table.cpp", "cityhash.cpp", "thread.cpp",
          "sparse_large_vector.cpp", "cpu_features.cpp", "x64/rdtsc.cpp", "windows/timer_resolution.cpp",
          "steady_clock.cpp", "host_memory.cpp", "address_space.cpp", "stream.cpp"]
# Eden core sources compiled unmodified (the rest of core is replaced by Zonaite's shims in runtime/gpu)
CORE = ["frontend/emu_window.cpp", "frontend/framebuffer_layout.cpp"]
ANDROID_ONLY = ["freedreno_debug.cpp"]


def sources(only):
    vc = os.path.join(EDEN, "src", "video_core")
    s = [p for p in glob.glob(os.path.join(vc, "**", "*.cpp"), recursive=True)
         if not any(e in p for e in EXCLUDE + ANDROID_ONLY) and listed_in_cmake(p)]
    s += [os.path.join(EDEN, "src", "common", c) for c in COMMON]
    s += [os.path.join(EDEN, "src", "core", c) for c in CORE]
    s += [os.path.join(EDEN, "externals", "bc_decoder", "bc_decoder.cpp"),
          os.path.join(EDEN, "externals", "stb", "stb_dxt.cpp"),
          os.path.join(ROOT, "runtime", "gpu", "vma_impl.cpp"),
          os.path.join(ROOT, "runtime", "gpu", "frame_interp.cpp"),
          os.path.join(ROOT, "runtime", "gpu", "picture.cpp"),
          os.path.join(ROOT, "runtime", "gpu", "scene_capture.cpp")]
    s = sorted(s)
    return [p for p in s if only in p] if only else s


def zonaite_headers_mtime():
    import eden_patches
    headers = glob.glob(os.path.join(ROOT, "runtime", "gpu", "override_video", "**", "*.h"), recursive=True)
    headers += [os.path.join(EDEN, rel) for rel, _, _ in eden_patches.PATCHES if rel.endswith(".h")]
    return max(os.path.getmtime(h) for h in headers if os.path.exists(h))


HEADERS_MTIME = None


def compile_one(src, obj):
    if os.path.exists(obj) and os.path.getmtime(obj) > max(os.path.getmtime(src), HEADERS_MTIME):
        return src, 0, ""
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    r = subprocess.run([CLANGXX, *FLAGS, "-c", src, "-o", obj], capture_output=True, text=True)
    return src, r.returncode, r.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = os.path.join(ROOT, "build", "gpu")
    global HEADERS_MTIME
    HEADERS_MTIME = zonaite_headers_mtime()
    srcs = sources(a.only)
    objs = [os.path.join(out, "obj", os.path.relpath(s, TP).replace(os.sep, "_")[:-4] + ".obj") for s in srcs]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(compile_one, srcs, objs))
    bad = [(s, e) for s, rc, e in results if rc]
    print(f"{len(srcs)} sources, {len(bad)} failed")
    reasons = {}
    for s, e in bad:
        first = next((l for l in e.splitlines() if "error" in l), e[:200])
        reasons.setdefault(first.split("error:")[-1].strip()[:150], []).append(os.path.relpath(s, EDEN))
    for r, files in sorted(reasons.items(), key=lambda kv: -len(kv[1]))[:a.show]:
        print(f"  {len(files):4d}  {r}\n        e.g. {files[0]}")
    if bad or a.only:
        sys.exit(1 if bad else 0)
    lib = os.path.join(out, "gpu.lib")
    rsp = os.path.join(out, "lib.rsp")
    with open(rsp, "w") as f:
        f.write("\n".join(f'"{o}"' for o in objs))
    subprocess.run([LLVM_LIB, f"/OUT:{lib}", f"@{rsp}"], check=True)
    print(f"built {lib}")


if __name__ == "__main__":
    main()
