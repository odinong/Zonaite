#!/usr/bin/env python3
"""Generate video_core's host shader headers (Eden's host_shaders/CMakeLists.txt, in Python).

For each shader source in third_party/eden/src/video_core/host_shaders:
  NAME_EXT.h      the GLSL source as a string (not for vulkan_* files)
  NAME_EXT_spv.h  SPIR-V from glslangValidator (not for opengl_* files)
plus the integer variants of the MSAA conversion shaders, and Zonaite's own compute shaders in runtime/gpu/shaders
(frame interpolation, NAME_comp_spv.h). Output: build/gpu/include/video_core/host_shaders.
"""
import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "eden", "src", "video_core", "host_shaders")
FSR = os.path.join(ROOT, "third_party", "eden", "externals", "FidelityFX-FSR", "ffx-fsr")
OUT = os.path.join(ROOT, "build", "gpu", "include", "video_core", "host_shaders")
ZN_SHADERS = os.path.join(ROOT, "runtime", "gpu", "shaders")  # Zonaite's own (frame interpolation)
GLSLANG = shutil.which("glslangValidator") or r"C:\VulkanSDK\1.4.363.0\Bin\glslangValidator.exe"
GLSL_FLAGS = ["-DUseUniformBlock=1"]
TARGET = ["--target-env", "spirv1.3"]
INCLUDE_ONLY = {"fidelityfx_fsr.frag"}  # GLSL_INCLUDES in Eden's CMake: included by other shaders, not compiled
VARIANTS = [("convert_msaa_to_non_msaa.frag", "sint", "isampler2DMS", "ivec4"),
            ("convert_msaa_to_non_msaa.frag", "uint", "usampler2DMS", "uvec4"),
            ("convert_non_msaa_to_msaa.frag", "sint", "isampler2D", "ivec4"),
            ("convert_non_msaa_to_msaa.frag", "uint", "usampler2D", "uvec4")]


def string_header(src, out_name):
    """The source as a std::string_view, one raw-string chunk per line (as StringShaderHeader.cmake does)."""
    name = os.path.basename(src).replace(".", "_").upper()
    with open(src, encoding="utf-8") as f:
        lines = f.read().split("\n")
    contents = " ".join(f'R"({line}\n)"' for line in lines)
    with open(os.path.join(SRC, "source_shader.h.in"), encoding="utf-8") as f:
        template = f.read()
    text = template.replace("@CONTENTS_NAME@", name).replace("@CONTENTS@", contents)
    with open(os.path.join(OUT, out_name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def spirv_header(src, var, out_name, extra=()):
    out = os.path.join(OUT, out_name)
    deps = [src] + glob.glob(os.path.join(os.path.dirname(src), "*.glsl"))
    if os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(d) for d in deps):
        return None
    r = subprocess.run([GLSLANG, "-V", "--quiet", f"-I{FSR}", *GLSL_FLAGS, *extra, "--variable-name", var,
                        "-o", out, src, *TARGET], capture_output=True, text=True)
    return None if r.returncode == 0 else f"{os.path.basename(src)}: {r.stdout[-800:]}{r.stderr[-800:]}"


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for src in sorted(glob.glob(os.path.join(SRC, "*"))):
        name = os.path.basename(src)
        if not name.endswith((".comp", ".frag", ".vert", ".glsl")) or name in INCLUDE_ONLY:
            continue
        base = name.replace(".", "_")
        if not name.startswith("vulkan"):
            string_header(src, base + ".h")
        if not name.startswith("opengl") and not name.endswith(".glsl"):
            jobs.append((src, base.upper() + "_SPV", base + "_spv.h", ()))
    for fname, suffix, sampler, texel in VARIANTS:
        stem, ext = fname.split(".")
        vname = f"{stem}_{suffix}_{ext}"
        jobs.append((os.path.join(SRC, fname), vname.upper() + "_SPV", vname + "_spv.h",
                     (f"-DSAMPLER_TYPE={sampler}", f"-DTEXEL_TYPE={texel}")))
    for src in sorted(glob.glob(os.path.join(ZN_SHADERS, "*.comp"))):
        base = os.path.basename(src).replace(".", "_")
        jobs.append((src, base.upper() + "_SPV", base + "_spv.h", (f"-I{ZN_SHADERS}",)))
    for f in ("ffx_a.h", "ffx_fsr1.h"):  # FidelityFX headers as strings too
        string_header(os.path.join(FSR, f), f.replace(".", "_") + ".h")
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        errors = [e for e in pool.map(lambda j: spirv_header(*j), jobs) if e]
    for e in errors:
        print("FAILED", e)
    print(f"host shaders: {len(jobs)} SPIR-V headers, {len(errors)} failed -> {OUT}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
