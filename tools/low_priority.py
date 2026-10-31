"""Builds run at idle CPU priority (Windows), so a game or anything else in use keeps the CPU first and the build
takes what is left; the compilers and linkers it starts inherit it. ZN_BUILD_PRIORITY=normal turns this off.
Imported for its effect by the build scripts."""
import os
import sys

if sys.platform == "win32" and os.environ.get("ZN_BUILD_PRIORITY", "").lower() != "normal":
    import ctypes

    IDLE_PRIORITY_CLASS = 0x40
    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), IDLE_PRIORITY_CLASS)
