# Developing Zonaite

Zonaite recompiles Tears of the Kingdom's Switch code (AArch64) into C++ and builds it on your PC, from your own
dump. The GPU, audio and controller side is built on Eden's code.

## Rules for this repo

- No game code, game files or keys, ever. The C++ generated from a dump stays on the machine that made it
  (`generated/` and `build/` are ignored by git).
- Zonaite never decrypts anything. It only reads folders you already extracted (`Exefs/` and `RomFs/`).
- `scripts/guard_no_game_data.py` runs before every commit and on every release package. If it complains, fix the
  cause instead of skipping it. Set `ZONAITE_DUMP` to your dump folder so it also catches exact copies of your
  files.
- Tests use a copy of a save in `scratch/`, never your real one.

## What's where

| Path | What |
|---|---|
| `recomp/` | The recompiler |
| `runtime/` | Everything the game runs on: system services, threads, files, graphics, audio, input, frame interpolation, the settings menu |
| `launcher/` | The launcher (`Zonaite.exe`) |
| `config/targets/` | One file per supported game version |
| `tools/` | Build, test and packaging scripts |
| `scripts/` | The game-data guard and its git hook installer |

## Setup

You need Python 3.10+, LLVM/clang 21, the Vulkan SDK and the Visual Studio Build Tools. After cloning:

```
python scripts/install_hooks.py
```

## Building

Make a dump as the launcher's guide describes (Ryujinx: pick the update, then Extract Data for ExeFS and RomFS into
one folder). Then:

```
python tools/build_game.py <dump folder>
python tools/build_runtime.py
python tools/build_launcher.py
```

`build_game.py` works out the version from the dump and builds it into `build\game-<version>`. Each version builds
next to the others. The first build takes about 12 minutes. `build_runtime.py` links `build\zonaite-<version>.exe`
for every version you've built.

Run the launcher (`build\Zonaite.exe`), or a version directly:

```
build\zonaite-1.2.1.exe build\game-1.2.1 <dump folder>
```

A release package (`python tools/package.py`) has no game code in it: the launcher builds the game on the player's
PC the first time they play.

## Licence

GPL v3 or later (`LICENSE`). Eden's code is GPL v3 too; everything else third-party is listed in
`THIRD_PARTY.md`. The game, its files and anything generated from a dump belong to Nintendo and are not part of
this repo.
