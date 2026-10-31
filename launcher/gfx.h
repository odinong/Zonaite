// The launcher's window and drawing: a Win32 window rendered with Dear ImGui on Vulkan (vulkan-1.dll is loaded
// at run time, so a PC without a Vulkan driver gets a message instead of a missing-DLL error).
#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>

namespace launcher {

// Opens the window and Vulkan; on failure, error says why
bool gfx_open(const wchar_t* title, int width, int height, std::string* error);
// Runs one frame: false once the window was closed. Between gfx_begin and gfx_end, ImGui calls draw the frame.
bool gfx_begin();
void gfx_end();
void gfx_close();
HWND gfx_window();

}  // namespace launcher
