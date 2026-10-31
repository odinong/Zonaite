#!/usr/bin/env python3
"""Build the launcher (launcher/): build/Zonaite.exe. It holds no game code, so the release package ships it
(as Zonaite.exe in the package's root).

usage:
  build_launcher.py [--out EXE]

Needs clang (LLVM) and the Vulkan SDK's headers (VULKAN_SDK); Vulkan itself is loaded at run time.
"""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import low_priority  # noqa: E402,F401 (builds yield the CPU to whatever else is running)

CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
VULKAN_SDK = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.363.0")
IMGUI = os.path.join(ROOT, "third_party", "imgui")
SOURCES = [os.path.join(ROOT, "launcher", "launcher.cpp"), os.path.join(ROOT, "launcher", "gfx.cpp")] + \
          [os.path.join(IMGUI, f) for f in ("imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp", "imgui_widgets.cpp",
                                            "backends/imgui_impl_win32.cpp", "backends/imgui_impl_vulkan.cpp")]
FLAGS = ["-std=c++20", "-O2", "-DUNICODE", "-D_UNICODE", "-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES",
         "-I", IMGUI, "-I", os.path.join(VULKAN_SDK, "Include")]
LIBS = ["user32", "gdi32", "shell32", "ole32", "imm32", "dwmapi", "advapi32"]


def compile_one(src, obj):
    if os.path.exists(obj) and os.path.getmtime(obj) > os.path.getmtime(src) and \
            not src.startswith(os.path.join(ROOT, "launcher")):
        return src, 0, ""
    r = subprocess.run([CLANGXX, *FLAGS, "-c", src, "-o", obj], capture_output=True, text=True)
    return src, r.returncode, r.stderr[-3000:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "Zonaite.exe"))
    a = ap.parse_args()
    obj_dir = os.path.join(ROOT, "build", "launcher")
    os.makedirs(obj_dir, exist_ok=True)
    objs = [os.path.join(obj_dir, os.path.basename(s)[:-4] + ".obj") for s in SOURCES]
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        results = list(pool.map(compile_one, SOURCES, objs))
    bad = [(s, e) for s, rc, e in results if rc]
    for s, e in bad:
        print(f"FAILED {s}:\n{e}")
    if bad:
        sys.exit(1)
    r = subprocess.run([CLANGXX, "-fuse-ld=lld", *objs, "-o", a.out, "-Wl,/SUBSYSTEM:WINDOWS",
                        "-Wl,/ENTRY:wWinMainCRTStartup"] + [f"-l{l}" for l in LIBS], capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout[-4000:] + r.stderr[-4000:])
    print(f"built {a.out}")


if __name__ == "__main__":
    main()
