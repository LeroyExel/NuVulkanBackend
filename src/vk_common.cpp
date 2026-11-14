#include "vk_common.h"

#include <cstdarg>

#define VK_DEF(name) PFN_##name name = nullptr;
VK_FUNC_LIST(VK_DEF)
#undef VK_DEF

PFN_vkGetInstanceProcAddr vkGetInstanceProcAddrReal = nullptr;

// ---------------------------------------------------------------------------
static FILE* g_log = nullptr;
static CRITICAL_SECTION g_logLock;
static bool g_logLockInit = false;

void LogOpen(const char* path) {
    if (!g_logLockInit) {
        InitializeCriticalSection(&g_logLock);
        g_logLockInit = true;
    }
    if (g_log) return;
    g_log = fopen(path, "w");
}

void LogClose() {
    if (g_log) {
        fflush(g_log);
        fclose(g_log);
        g_log = nullptr;
    }
}

void Logf(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g_logLockInit) EnterCriticalSection(&g_logLock);
    if (g_log) {
        fputs(buf, g_log);
        fputc('\n', g_log);
        fflush(g_log);
    }
    if (IsDebuggerPresent()) {
        OutputDebugStringA(buf);
        OutputDebugStringA("\n");
    }
    if (g_logLockInit) LeaveCriticalSection(&g_logLock);
}

bool vkLoadLoader() {
    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    if (!loader) {
        LOG("[vk] vulkan-1.dll not found (32-bit loader missing)");
        return false;
    }
    vkGetInstanceProcAddrReal = (PFN_vkGetInstanceProcAddr)GetProcAddress(
        loader, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddrReal) {
        LOG("[vk] vkGetInstanceProcAddr missing");
        return false;
    }
    vkGetInstanceProcAddr = vkGetInstanceProcAddrReal;
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)GetProcAddress(
        loader, "vkGetDeviceProcAddr");

    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(
        VK_NULL_HANDLE, "vkCreateInstance");
    if (!vkCreateInstance) {
        LOG("[vk] vkCreateInstance missing");
        return false;
    }
    vkEnumerateInstanceExtensionProperties =
        (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(
            VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    return true;
}

bool vkResolveInstance(VkInstance instance) {
#define VK_LOAD(name)                                                          \
    name = (PFN_##name)vkGetInstanceProcAddrReal(instance, #name);
    VK_FUNC_LIST(VK_LOAD)
#undef VK_LOAD
    vkGetInstanceProcAddr = vkGetInstanceProcAddrReal;
    return vkCreateDevice && vkGetPhysicalDeviceProperties &&
           vkCreateWin32SurfaceKHR;
}

bool vkResolveDevice(VkDevice device) {
    if (!vkGetDeviceProcAddr) return false;
#define VK_DEVLOAD(name)                                                       \
    {                                                                          \
        PFN_vkVoidFunction p = vkGetDeviceProcAddr(device, #name);              \
        if (p) name = (PFN_##name)p;                                            \
    }
    VK_FUNC_LIST(VK_DEVLOAD)
#undef VK_DEVLOAD
    return vkCmdBeginRendering != nullptr;
}
