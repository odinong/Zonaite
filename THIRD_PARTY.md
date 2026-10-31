# Third-party code

Zonaite is licensed under the GNU General Public License v3.0 or later (see `LICENSE`), because it reuses GPLv3
code from the Eden emulator (a continuation of yuzu). Everything below is compatible with that.

| Component | Used for | Source and pinned version | Licence |
|---|---|---|---|
| Eden `shader_recompiler`, `video_core` (and the parts of `common` they need) | Maxwell shader translation to SPIR-V; GPU emulation and Vulkan renderer | github.com/eden-emulator/mirror @ `072da775164ab258f2bbe62eecf935a99e7f36f1` | GPL-3.0-or-later (yuzu-era files GPL-2.0-or-later) |
| sirit | SPIR-V code emitter used by the shader recompiler | github.com/ReinUsesLisp/sirit @ `7b9a59b7c803df9f861b6d5619a75ef3e46bbed5` | BSD-3-Clause |
| fmt | formatting | github.com/fmtlib/fmt @ 12.1.0 | MIT |
| Boost (headers only) | containers, intrusive lists, interval sets | archives.boost.io boost_1_91_0 | Boost Software License 1.0 |
| SPIRV-Headers | SPIR-V definitions | github.com/KhronosGroup/SPIRV-Headers @ vulkan-sdk-1.4.341.0 | MIT-style (Khronos) |
| Dear ImGui (core, Win32 and Vulkan backends) | the settings menu and overlays (runtime/gpu/ui.cpp) | github.com/ocornut/imgui @ v1.91.9 (`third_party/imgui`, licence in `third_party/imgui/LICENSE.txt`) | MIT |
| Marcellus (font) | menu headings | github.com/google/fonts `ofl/marcellus` (`res/fonts`, licence `res/fonts/OFL-Marcellus.txt`) | SIL Open Font Licence 1.1 |
| Lato (font) | menu text | github.com/google/fonts `ofl/lato` (`res/fonts`, licence `res/fonts/OFL-Lato.txt`) | SIL Open Font Licence 1.1 |

Notes:

- We use upstream sirit (BSD-3), not Eden's fork: Eden's fork adds an AGPLv3 licence file
  ("install target, static/shared split, agplv3"), and its source files still say BSD-3. Upstream keeps Zonaite
  cleanly GPLv3.
- `third_party/eden-src/` and `third_party/deps/` are local checkouts, git-ignored. The files Zonaite builds are
  copied from them into `third_party/eden/` with their licence headers intact, and `tools/vendor_eden.py` records
  which ones.
- None of these components contain game data. The guard script still runs on everything committed.
