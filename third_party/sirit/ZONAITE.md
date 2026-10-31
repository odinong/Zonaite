# sirit (vendored)

Upstream: https://github.com/ReinUsesLisp/sirit @ 7b9a59b7c803df9f861b6d5619a75ef3e46bbed5, BSD-3-Clause
(`LICENSE.txt`). Only `include/` and `src/` are vendored.

Zonaite additions (marked "Zonaite addition" in the source), needed by Eden's shader recompiler:

- `Module::OpGroupNonUniformQuadBroadcast`: one more SPIR-V group instruction.
- `Module::PatchDeferredPhi` overload whose callback returns `{value, parent block}` for each OpPhi operand,
  instead of the value only.

These were written for Zonaite from the SPIR-V instruction layout. They are not copied from Eden's sirit fork,
whose added AGPLv3 licence file we avoid (see THIRD_PARTY.md).
