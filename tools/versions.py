"""Where each version of the game is built: one recompiled game and one executable per version, side by side
(build/game-<version>, generated/game-<version>, build/zonaite-<version>.exe), so switching versions in the
launcher never rebuilds."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def game_dir(version):
    return os.path.join(ROOT, "build", f"game-{version}")


def generated_dir(version):
    return os.path.join(ROOT, "generated", f"game-{version}")


def exe(version):
    return os.path.join(ROOT, "build", f"zonaite-{version}.exe")


def built():
    """Versions whose game has been recompiled here, newest first"""
    out = []
    build = os.path.join(ROOT, "build")
    for name in os.listdir(build) if os.path.isdir(build) else []:
        if name.startswith("game-") and os.path.isfile(os.path.join(build, name, "image.json")):
            out.append(name[5:])
    return sorted(out, key=lambda v: [int(x) for x in v.split(".")], reverse=True)
