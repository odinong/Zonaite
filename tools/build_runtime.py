#!/usr/bin/env python3
"""Compile the runtime and link it with each recompiled game version and Eden's GPU into
build/zonaite-<version>.exe (the link is tools/link_game.py).

usage:
  build_runtime.py [--jobs N] [--debug] [--no-link]

Needs build/game-<version>/*.obj (tools/build_game.py), build/gpu/gpu.lib (tools/build_gpu.py) and build/shader/shader.lib
(tools/build_shader.py). runtime/src is Zonaite's own code; runtime/gpu talks to Eden and is compiled with Eden's
include paths and flags. Objects are rebuilt only when their sources or headers changed.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_gpu  # noqa: E402
import link_game  # noqa: E402
import versions  # noqa: E402

CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
RUNTIME_INCLUDE = os.path.join(ROOT, "runtime", "include")
FLAGS = ["-std=c++20", "-mcx16", "-fno-exceptions", "-Wall", "-Wno-unused-function", "-I", RUNTIME_INCLUDE]
# runtime/gpu: Eden's flags (exceptions on: Eden's renderer uses them) plus Zonaite's headers
# runtime/gpu/override comes first: it replaces the Eden kernel/service headers nvdrv includes (see its README.md)
GPU_OVERRIDE = os.path.join(ROOT, "runtime", "gpu", "override")
IMGUI = os.path.join(ROOT, "third_party", "imgui")
GPU_FLAGS = (["-I", GPU_OVERRIDE] + [f for f in build_gpu.FLAGS if f != "-O2"] +
             ["-I", RUNTIME_INCLUDE, "-I", IMGUI, "-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES"])
# Dear ImGui (MIT), for the settings menu (runtime/gpu/ui.cpp)
IMGUI_SOURCES = [os.path.join(IMGUI, f) for f in ("imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp",
                                                  "imgui_widgets.cpp", "backends/imgui_impl_win32.cpp",
                                                  "backends/imgui_impl_vulkan.cpp")]
# Eden sources compiled unmodified with GPU_FLAGS: the nvdrv devices and their core (runtime/gpu/nvdrv.cpp is the glue)
EDEN_SRC = os.path.join(build_gpu.EDEN, "src")
EDEN_WITH_OVERRIDES = sorted(glob.glob(os.path.join(EDEN_SRC, "core", "hle", "service", "nvdrv", "devices", "*.cpp")) +
                             glob.glob(os.path.join(EDEN_SRC, "core", "hle", "service", "nvdrv", "core", "*.cpp"))) +                       [os.path.join(EDEN_SRC, "common", "adpf.cpp"), os.path.join(EDEN_SRC, "common", "stb.cpp")]
# nvnflinger's buffer queues and compositor (runtime/gpu/vi.cpp is the glue). Left out: the IPC front end
# (hos_binder_driver.cpp, Zonaite has its own) and the service process setup (nvnflinger.cpp).
NVNFLINGER = os.path.join(EDEN_SRC, "core", "hle", "service", "nvnflinger")
EDEN_WITH_OVERRIDES += sorted(p for p in glob.glob(os.path.join(NVNFLINGER, "*.cpp")) +
                              glob.glob(os.path.join(NVNFLINGER, "ui", "*.cpp"))
                              if os.path.basename(p) not in ("hos_binder_driver.cpp", "nvnflinger.cpp"))
# audio_core (runtime/gpu/audio.cpp is the glue), minus the cubeb and SDL sinks and their selection (Zonaite has its
# own WASAPI sink, runtime/gpu/audio_sink.cpp)
AUDIO_CORE = os.path.join(EDEN_SRC, "audio_core")
EDEN_WITH_OVERRIDES += sorted(p for p in glob.glob(os.path.join(AUDIO_CORE, "**", "*.cpp"), recursive=True)
                              if build_gpu.listed_in_cmake(p) and
                              os.path.basename(p) not in ("cubeb_sink.cpp", "sdl3_sink.cpp", "sink_details.cpp"))


def newest_header():
    """Zonaite's headers, and the Eden headers it overrides or patches (tools/build_gpu.py)"""
    global _NEWEST
    if _NEWEST is None:
        _NEWEST = max(max(os.path.getmtime(p) for p in glob.glob(os.path.join(RUNTIME_INCLUDE, "zonaite", "*.h"))),
                      build_gpu.zonaite_headers_mtime())
    return _NEWEST


_NEWEST = None


def compile_one(src, obj, flags, opt):
    if os.path.exists(obj) and os.path.getmtime(obj) > max(os.path.getmtime(src), newest_header()):
        return src, 0, ""
    r = subprocess.run([CLANGXX, *flags, opt, "-g", "-c", src, "-o", obj], capture_output=True, text=True)
    return src, r.returncode, r.stderr[-3000:]


def jobs_for(obj_dir):
    """(source, object, flags) for every runtime source"""
    jobs = [(s, os.path.join(obj_dir, os.path.basename(s)[:-4] + ".obj"), FLAGS)
            for s in sorted(glob.glob(os.path.join(ROOT, "runtime", "src", "*.cpp")))]
    jobs += [(s, os.path.join(obj_dir, "gpu_" + os.path.basename(s)[:-4] + ".obj"), GPU_FLAGS)
             for s in sorted(glob.glob(os.path.join(ROOT, "runtime", "gpu", "*.cpp")))
             if not s.endswith("vma_impl.cpp")]  # vma_impl is part of gpu.lib
    jobs += [(s, os.path.join(obj_dir, "eden_" + os.path.relpath(s, EDEN_SRC).replace(os.sep, "_")[:-4] + ".obj"),
              GPU_FLAGS) for s in EDEN_WITH_OVERRIDES]
    jobs += [(s, os.path.join(obj_dir, "imgui_" + os.path.basename(s)[:-4] + ".obj"), GPU_FLAGS)
             for s in IMGUI_SOURCES]
    return jobs


def runtime_objects():
    """The runtime's object files, as this script builds them (tools/package.py archives them)"""
    return [j[1] for j in jobs_for(os.path.join(ROOT, "build", "runtime"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-link", action="store_true", help="compile only (tools/package.py)")
    a = ap.parse_args()
    obj_dir = os.path.join(ROOT, "build", "runtime")
    os.makedirs(obj_dir, exist_ok=True)
    opt = "-O0" if a.debug else "-O2"
    jobs = jobs_for(obj_dir)
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda j: compile_one(j[0], j[1], j[2], opt), jobs))
    bad = [(s, e) for s, rc, e in results if rc]
    for s, e in bad:
        print(f"FAILED {s}\n{e}")
    if bad:
        sys.exit(1)
    if a.no_link:
        return
    _, libs = link_game.repo_inputs()
    for version in versions.built():  # every version recompiled here: build/zonaite-<version>.exe
        link_game.link([j[1] for j in jobs], libs, version)


if __name__ == "__main__":
    main()
