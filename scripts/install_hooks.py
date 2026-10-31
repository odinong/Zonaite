#!/usr/bin/env python3
"""Install the game-data guard as this repository's git pre-commit hook."""
import os
import subprocess
import sys

root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
hook = os.path.join(root, ".git", "hooks", "pre-commit")
python = sys.executable.replace("\\", "/")
with open(hook, "w", newline="\n") as f:
    f.write("#!/bin/sh\n"
            "# Installed by scripts/install_hooks.py: block game data from commits.\n"
            f'exec "{python}" "$(git rev-parse --show-toplevel)/scripts/guard_no_game_data.py"\n')
os.chmod(hook, 0o755)
print(f"installed {hook}")
