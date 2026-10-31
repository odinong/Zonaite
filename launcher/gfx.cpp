// The launcher's window and drawing (gfx.h): Win32 + Dear ImGui on Vulkan, after Dear ImGui's
// example_win32_vulkan, with Vulkan's functions loaded from vulkan-1.dll at run time.
#include "gfx.h"

#define VK_USE_PLATFORM_WIN32_KHR
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_win32.h"
#include "imgui.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace launcher {

namespace {

// The Vulkan functions the launcher calls itself (the ImGui backend loads its own)
#define LAUNCHER_VK_INSTANCE(X)                                                                                    \
    X(vkDestroyInstance) X(vkCreateWin32SurfaceKHR) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define LAUNCHER_VK_DEVICE(X)                                                                                      \
    X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkDestroyDevice) X(vkAcquireNextImageKHR) X(vkWaitForFences)          \
    X(vkResetFences) X(vkResetCommandPool) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBeginRenderPass)    \
    X(vkCmdEndRenderPass) X(vkQueueSubmit) X(vkQueuePresentKHR)
#define DECLARE(name) PFN_##name name = nullptr;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
LAUNCHER_VK_INSTANCE(DECLARE)
LAUNCHER_VK_DEVICE(DECLARE)
#undef DECLARE

HWND g_hwnd = nullptr;
VkInstance g_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_gpu = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
uint32_t g_queue_family = 0;
VkQueue g_queue = VK_NULL_HANDLE;
ImGui_ImplVulkanH_Window g_wd;
constexpr uint32_t kMinImages = 2;
bool g_rebuild = false;
bool g_closed = false;
int g_width = 0, g_height = 0;

PFN_vkVoidFunction load(const char* name, void*) {
    return vkGetInstanceProcAddr(g_instance, name);
}

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp)) return 1;
    switch (msg) {
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED) {
            g_width = LOWORD(lp);
            g_height = HIWORD(lp);
            g_rebuild = true;
        }
        return 0;
    case WM_GETMINMAXINFO:
        reinterpret_cast<MINMAXINFO*>(lp)->ptMinTrackSize = {900, 600};
        return 0;
    case WM_SYSCOMMAND:
        if ((wp & 0xfff0) == SC_KEYMENU) return 0;  // Alt doesn't open a window menu
        break;
    case WM_DESTROY:
        g_closed = true;
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool fail(std::string* error, const char* why) {
    *error = why;
    return false;
}

void resize() {
    if (g_width <= 0 || g_height <= 0) return;
    ImGui_ImplVulkan_SetMinImageCount(kMinImages);
    ImGui_ImplVulkanH_CreateOrResizeWindow(g_instance, g_gpu, g_device, &g_wd, g_queue_family, nullptr, g_width,
                                           g_height, kMinImages);
    g_wd.FrameIndex = 0;
    g_rebuild = false;
}

}  // namespace

HWND gfx_window() { return g_hwnd; }

bool gfx_open(const wchar_t* title, int width, int height, std::string* error) {
    HMODULE vk = LoadLibraryW(L"vulkan-1.dll");
    if (!vk) return fail(error, "No Vulkan driver was found. Update your graphics driver (NVIDIA, AMD or Intel).");
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vk, "vkGetInstanceProcAddr"));
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
    if (!vkCreateInstance) return fail(error, "The Vulkan driver is incomplete. Update your graphics driver.");

    const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "Zonaite launcher", 1, "Zonaite", 1,
                                VK_API_VERSION_1_0};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = extensions;
    if (vkCreateInstance(&ici, nullptr, &g_instance) != VK_SUCCESS)
        return fail(error, "Vulkan could not start. Update your graphics driver.");
#define LOAD(name) name = reinterpret_cast<PFN_##name>(vkGetInstanceProcAddr(g_instance, #name));
    LAUNCHER_VK_INSTANCE(LOAD)
    if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_0, load))
        return fail(error, "The Vulkan driver is missing functions. Update your graphics driver.");

    g_gpu = ImGui_ImplVulkanH_SelectPhysicalDevice(g_instance);
    if (!g_gpu) return fail(error, "No graphics card with Vulkan support was found.");
    g_queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(g_gpu);
    if (g_queue_family == static_cast<uint32_t>(-1)) return fail(error, "The graphics card has no usable queue.");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = device_extensions;
    if (vkCreateDevice(g_gpu, &dci, nullptr, &g_device) != VK_SUCCESS)
        return fail(error, "The graphics card could not be opened with Vulkan.");
#define LOAD_DEVICE(name) name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(g_device, #name));
    LAUNCHER_VK_DEVICE(LOAD_DEVICE)
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    // the window, centred on the primary monitor, sized for its DPI
    const HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{sizeof wc, CS_CLASSDC, wndproc, 0, 0, inst, LoadIconW(inst, MAKEINTRESOURCEW(1)),
                   LoadCursorW(nullptr, IDC_ARROW), nullptr, nullptr, L"ZonaiteLauncher", nullptr};
    RegisterClassExW(&wc);
    const float dpi = static_cast<float>(GetDpiForSystem()) / 96.0f;
    const int w = static_cast<int>(width * dpi), h = static_cast<int>(height * dpi);
    RECT r{0, 0, w, h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int ww = r.right - r.left, wh = r.bottom - r.top;
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, (GetSystemMetrics(SM_CXSCREEN) - ww) / 2,
                             (GetSystemMetrics(SM_CYSCREEN) - wh) / 2, ww, wh, nullptr, nullptr, inst, nullptr);
    BOOL dark = TRUE;  // a dark title bar to match (DWMWA_USE_IMMERSIVE_DARK_MODE)
    using SetAttr = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
    if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll"))
        if (auto set = reinterpret_cast<SetAttr>(GetProcAddress(dwm, "DwmSetWindowAttribute")))
            set(g_hwnd, 20, &dark, sizeof dark);

    VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = inst;
    sci.hwnd = g_hwnd;
    VkSurfaceKHR surface;
    if (vkCreateWin32SurfaceKHR(g_instance, &sci, nullptr, &surface) != VK_SUCCESS)
        return fail(error, "Vulkan could not draw to the window.");
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_gpu, g_queue_family, surface, &supported);
    if (!supported) return fail(error, "The graphics card can't present to this window.");
    g_wd.Surface = surface;
    const VkFormat formats[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    g_wd.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(g_gpu, surface, formats, 2,
                                                               VK_COLORSPACE_SRGB_NONLINEAR_KHR);
    const VkPresentModeKHR modes[] = {VK_PRESENT_MODE_FIFO_KHR};
    g_wd.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_gpu, surface, modes, 1);
    RECT client;
    GetClientRect(g_hwnd, &client);
    g_width = client.right;
    g_height = client.bottom;
    ImGui_ImplVulkanH_CreateOrResizeWindow(g_instance, g_gpu, g_device, &g_wd, g_queue_family, nullptr, g_width,
                                           g_height, kMinImages);
    ShowWindow(g_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_0;
    info.Instance = g_instance;
    info.PhysicalDevice = g_gpu;
    info.Device = g_device;
    info.QueueFamily = g_queue_family;
    info.Queue = g_queue;
    info.DescriptorPoolSize = 16;
    info.RenderPass = g_wd.RenderPass;
    info.MinImageCount = kMinImages;
    info.ImageCount = g_wd.ImageCount;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    if (!ImGui_ImplVulkan_Init(&info)) return fail(error, "The drawing could not start.");
    return true;
}

bool gfx_begin() {
    for (MSG msg; PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE);) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (g_closed) return false;
    if (IsIconic(g_hwnd)) {
        Sleep(30);
        return gfx_begin();
    }
    if (g_rebuild) resize();
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    return true;
}

void gfx_end() {
    ImGui::Render();
    ImDrawData* draw = ImGui::GetDrawData();
    if (draw->DisplaySize.x <= 0 || draw->DisplaySize.y <= 0) return;
    ImGui_ImplVulkanH_Window* wd = &g_wd;
    VkSemaphore acquired = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore done = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkResult err = vkAcquireNextImageKHR(g_device, wd->Swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE,
                                         &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) g_rebuild = true;
    if (err == VK_ERROR_OUT_OF_DATE_KHR) return;
    ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
    vkWaitForFences(g_device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_device, 1, &fd->Fence);
    vkResetCommandPool(g_device, fd->CommandPool, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd->CommandBuffer, &begin);
    VkClearValue clear{};
    clear.color = {{0.027f, 0.043f, 0.051f, 1.0f}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = wd->RenderPass;
    rp.framebuffer = fd->Framebuffer;
    rp.renderArea.extent = {static_cast<uint32_t>(wd->Width), static_cast<uint32_t>(wd->Height)};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(fd->CommandBuffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw, fd->CommandBuffer);
    vkCmdEndRenderPass(fd->CommandBuffer);
    vkEndCommandBuffer(fd->CommandBuffer);
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &fd->CommandBuffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &done;
    vkQueueSubmit(g_queue, 1, &submit, fd->Fence);
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &done;
    present.swapchainCount = 1;
    present.pSwapchains = &wd->Swapchain;
    present.pImageIndices = &wd->FrameIndex;
    err = vkQueuePresentKHR(g_queue, &present);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) g_rebuild = true;
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount;
}

void gfx_close() {
    if (g_device) vkDeviceWaitIdle(g_device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    ImGui_ImplVulkanH_DestroyWindow(g_instance, g_device, &g_wd, nullptr);
    vkDestroyDevice(g_device, nullptr);
    vkDestroyInstance(g_instance, nullptr);
    DestroyWindow(g_hwnd);
}

}  // namespace launcher
