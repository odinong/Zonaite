#!/usr/bin/env python3
"""Link Zonaite: the runtime, Eden's GPU and the third-party libraries with the recompiled game objects
(build/game/*.obj from tools/build_game.py) into build/zonaite.exe.

usage:
  link_game.py [--lib-dir DIR] [--out EXE]

From a repository checkout (tools/build_runtime.py calls this) the runtime's objects come from build/runtime and
the libraries from their build folders. In a release package they all come prebuilt in lib/ (--lib-dir): the
package carries no game code, so the exe that contains it is only ever linked on the player's machine.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import low_priority  # noqa: E402,F401 (builds yield the CPU to whatever else is running)
import versions  # noqa: E402
CLANGXX = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
CLANG_RT = glob.glob(os.path.join(os.path.dirname(os.path.dirname(CLANGXX)), "lib", "clang", "*", "lib", "windows",
                                  "clang_rt.builtins-x86_64.lib"))
SYSTEM_LIBS = ["user32", "gdi32", "shell32", "ole32", "ntdll", "advapi32", "winmm", "ws2_32", "bcrypt", "secur32",
               "mfplat", "mfuuid", "strmiids", "dwmapi", "dbghelp"]
FFMPEG_LIBS = ["avcodec", "avfilter", "swscale", "swresample", "avutil"]


def repo_inputs():
    """Runtime objects (as tools/build_runtime.py builds them) and library paths in a repository checkout"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build_runtime  # noqa: E402 (it imports this module too)
    objs = build_runtime.runtime_objects()
    libs = [os.path.join(ROOT, "build", "gpu", "gpu.lib"), os.path.join(ROOT, "build", "shader", "shader.lib")]
    libs += [os.path.join(ROOT, "third_party", "deps", "ffmpeg", "lib", f"{n}.lib") for n in FFMPEG_LIBS]
    libs += [os.path.join(ROOT, "third_party", "deps", "opus", "lib", "opus.lib")]
    return objs, libs


def package_inputs(lib_dir):
    """The same, prebuilt in a release package's lib/ (runtime.lib holds the runtime's objects)"""
    names = ["runtime.lib", "gpu.lib", "shader.lib"] + [f"{n}.lib" for n in FFMPEG_LIBS] + ["opus.lib"]
    libs = [os.path.join(lib_dir, n) for n in names]
    missing = [l for l in libs if not os.path.isfile(l)]
    if missing:
        sys.exit("link_game: missing " + ", ".join(missing))
    # runtime.lib is linked whole: nothing in the game objects refers to most of the runtime (main, services)
    return [], libs


def link(objs, libs, version, exe=None, whole=None):
    exe = exe or versions.exe(version)
    game = sorted(glob.glob(os.path.join(versions.game_dir(version), "*.obj")))
    if not game:
        sys.exit(f"no game objects for version {version}: run tools/build_game.py first")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    rsp = os.path.join(os.path.dirname(exe), "link.rsp")
    with open(rsp, "w") as f:
        for o in objs + game + libs + CLANG_RT:
            f.write(f'"{o.replace(os.sep, "/")}"\n')
    extra = [f"-Wl,/WHOLEARCHIVE:{whole.replace(os.sep, '/')}"] if whole else []
    t0 = time.time()
    r = subprocess.run([CLANGXX, "-fuse-ld=lld", "-g", f"@{rsp}", "-o", exe, "-Wl,/STACK:268435456"] + extra +
                       [f"-l{l}" for l in SYSTEM_LIBS], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-6000:], r.stderr[-6000:])
        sys.exit(1)
    print(f"linked {exe} ({os.path.getsize(exe) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib-dir")
    ap.add_argument("--version", required=True, help="the game version to link (build/game-<version>)")
    ap.add_argument("--out", help="default: build/zonaite-<version>.exe")
    a = ap.parse_args()
    if a.lib_dir:
        objs, libs = package_inputs(a.lib_dir)
        link(objs, libs, a.version, a.out, whole=os.path.join(a.lib_dir, "runtime.lib"))
    else:
        objs, libs = repo_inputs()
        link(objs, libs, a.version, a.out)


if __name__ == "__main__":
    main()
