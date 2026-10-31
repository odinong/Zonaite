#!/usr/bin/env python3
"""Copy the parts of Eden that Zonaite builds from the local checkout into third_party/eden.

usage:
  vendor_eden.py [EDEN_CHECKOUT]   (default third_party/eden-src)

Copies the parts in PARTS (minus SKIP_DIRS: OpenGL backends, the CPU JIT, the debugger) with licence headers
intact, plus Eden's LICENSE, and writes third_party/eden/VENDORED.txt with the upstream commit. Re-run after
updating the checkout.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PARTS = ["src/common", "src/shader_recompiler", "src/video_core", "src/core", "src/audio_core", "src/hid_core",
         "externals/stb",
         "externals/bc_decoder", "externals/FidelityFX-FSR"]
SKIP_DIRS = {os.path.normpath(p) for p in ("src/shader_recompiler/backend/glasm",
                                           "src/shader_recompiler/backend/glsl", "src/common/android",
                                           "src/common/linux", "src/common/apple",
                                           "src/video_core/renderer_opengl", "src/core/debugger")}
EXTENSIONS = (".cpp", ".h", ".hpp", ".inc", ".c", ".comp", ".frag", ".vert", ".glsl", ".in", "CMakeLists.txt")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "third_party", "eden-src")
    dst = os.path.join(ROOT, "third_party", "eden")
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    files = 0
    for part in PARTS:
        for root, dirs, names in os.walk(os.path.join(src, part)):
            rel_root = os.path.relpath(root, src)
            dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(rel_root, d)) not in SKIP_DIRS]
            for n in names:
                if not n.endswith(EXTENSIONS):
                    continue
                out = os.path.join(dst, rel_root, n)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(os.path.join(root, n), out)
                files += 1
    shutil.copy2(os.path.join(src, "LICENSE.txt"), os.path.join(dst, "LICENSE.txt"))
    with open(os.path.join(dst, "VENDORED.txt"), "w") as f:
        f.write(f"Eden emulator, https://github.com/eden-emulator/mirror\ncommit {commit}\n")
        f.write("parts: " + ", ".join(PARTS) + "\n")
        f.write("skipped: " + ", ".join(sorted(p.replace(os.sep, "/") for p in SKIP_DIRS)) + "\n")
        f.write("licence: GPL-3.0-or-later (see LICENSE.txt and the SPDX header in each file)\n")
    print(f"vendored {files} files from {commit[:10]} into {dst}")
    import eden_patches  # Zonaite's few changes to Eden, re-applied on every vendoring
    eden_patches.apply(dst)


if __name__ == "__main__":
    main()
