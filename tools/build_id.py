#!/usr/bin/env python3
"""The build ID: a hash of everything that decides the generated game code (the recompiler, the game build
script, the hook table, the target facts and the headers generated code includes). A player rebuilds the game
when it changes (tools/play.py); tools/package.py checks that a package's copy hashes like the repository.

usage:
  build_id.py [ROOT]     print the build ID of a repository or package folder
"""
import glob
import hashlib
import os
import sys

INPUTS = ["recomp/*.py", "tools/build_game.py", "tools/romfs.py", "config/hooks.txt", "config/targets/*.json",
          "runtime/include/zonaite/guest*.h"]


def compute(root):
    h = hashlib.sha256()
    for pattern in INPUTS:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            data = open(path, "rb").read().replace(b"\r\n", b"\n")
            h.update(rel.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    print(compute(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
