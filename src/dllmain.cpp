#include "shim.h"
#include "vk_common.h"

// Verified in docs/stage1_recon.md: d3d9!Direct3DCreate9 IAT slot.
static const uint32_t kDirect3DCreate9IatRva = 0x00C49360;

typedef void*(__stdcall* PFN_Direct3DCreate9)(uint32_t);

static PFN_Direct3DCreate9 g_realCreate9 = nullptr;
static HMODULE g_self = nullptr;

static void* __stdcall Hook_Direct3DCreate9(uint32_t sdkVersion) {
    LOG("[hook] Direct3DCreate9(%u) -> NuVulkan shim", sdkVersion);
    (void)g_realCreate9;
    return ShimDirect3DCreate9(sdkVersion);
}

// Fallback for a build whose import table moved: walk the descriptors instead
// of trusting the recorded RVA.
static void** FindIatSlot(HMODULE module, const char* dllName,
                          const char* funcName) {
    uint8_t* base = (uint8_t*)module;
    auto dos = (IMAGE_DOS_HEADER*)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
    auto nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return nullptr;

    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return nullptr;

    auto imp = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress);
    for (; imp->Name; ++imp) {
        const char* mod = (const char*)(base + imp->Name);
        if (_stricmp(mod, dllName) != 0) continue;
        auto thunk = (IMAGE_THUNK_DATA*)(base + imp->OriginalFirstThunk);
        auto iat = (IMAGE_THUNK_DATA*)(base + imp->FirstThunk);
        if (!imp->OriginalFirstThunk) thunk = iat;
        for (; thunk->u1.AddressOfData; ++thunk, ++iat) {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            auto ibn = (IMAGE_IMPORT_BY_NAME*)(base + thunk->u1.AddressOfData);
            if (strcmp((const char*)ibn->Name, funcName) == 0)
                return (void**)&iat->u1.Function;
        }
    }
    return nullptr;
}

static bool InstallIatHook() {
    HMODULE game = GetModuleHandleW(nullptr);
    if (!game) return false;

    HMODULE d3d9 = GetModuleHandleW(L"d3d9.dll");
    if (!d3d9) {
        LOG("[hook] d3d9.dll not loaded yet");
        return false;
    }
    FARPROC real = GetProcAddress(d3d9, "Direct3DCreate9");
    g_realCreate9 = (PFN_Direct3DCreate9)real;

    void** slot = (void**)((uint8_t*)game + kDirect3DCreate9IatRva);
    if (*slot != (void*)real) {
        LOG("[hook] IAT rva mismatch, walking imports");
        slot = FindIatSlot(game, "d3d9.dll", "Direct3DCreate9");
    }
    if (!slot) {
        LOG("[hook] Direct3DCreate9 IAT slot not found");
        return false;
    }

    DWORD oldProt = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &oldProt)) {
        LOG("[hook] VirtualProtect failed (%lu)", GetLastError());
        return false;
    }
    *slot = (void*)&Hook_Direct3DCreate9;
    VirtualProtect(slot, sizeof(void*), oldProt, &oldProt);
    LOG("[hook] Direct3DCreate9 IAT patched at %p (rva 0x%X)", slot,
        (uint32_t)((uint8_t*)slot - (uint8_t*)game));
    return true;
}

static DWORD WINAPI InitThread(LPVOID) {
    char logPath[MAX_PATH];
    GetModuleFileNameA(g_self, logPath, MAX_PATH);
    char* dot = strrchr(logPath, '.');
    if (dot) strcpy_s(dot, MAX_PATH - (dot - logPath), ".log");
    LogOpen(logPath);

    LOG("=== NuVulkanBackend ===");
    LOG("[init] module %p game base %p", (void*)g_self,
        (void*)GetModuleHandleW(nullptr));

    ShimInit();
    InstallIatHook();
    return 0;
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID lpReserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = module;
        DisableThreadLibraryCalls(module);
        HANDLE t = CreateThread(nullptr, 0, InitThread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
    } else if (reason == DLL_PROCESS_DETACH) {
        if (!lpReserved) ShimShutdown();
        LogClose();
    }
    return TRUE;
}

extern "C" __declspec(dllexport) void NuVkInstall() { InstallIatHook(); }

// Native exports so the DLL can be dropped in as d3d9.dll. The .def file keeps
// these undecorated; without it __stdcall would export _Direct3DCreate9@4.
extern "C" void* __stdcall Direct3DCreate9(uint32_t sdkVersion) {
    LOG("[proxy] Direct3DCreate9(%u)", sdkVersion);
    return ShimDirect3DCreate9(sdkVersion);
}

extern "C" long __stdcall Direct3DCreate9Ex(uint32_t sdkVersion, void** out) {
    LOG("[proxy] Direct3DCreate9Ex(%u)", sdkVersion);
    if (out) *out = ShimDirect3DCreate9(sdkVersion);
    return 0;
}
