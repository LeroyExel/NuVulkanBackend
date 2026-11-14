#include "d3d9_internal.h"

#include "shim.h"

using namespace d3d9;

FakeObj g_d3d9{};

PresentParams ReadPresentParams(const void* pp) {
    PresentParams r;
    if (!pp) return r;
    const uint32_t* p = (const uint32_t*)pp;
    if (p[0]) r.w = p[0];
    if (p[1]) r.h = p[1];
    r.windowed = p[8] != 0;
    r.refreshRate = p[12] ? p[12] : 60;
    r.presentationInterval = p[13];
    if (r.w == 0 || r.h == 0) {
        RECT rc{};
        if (g_deviceWnd && GetClientRect(g_deviceWnd, &rc)) {
            if (rc.right > rc.left) r.w = rc.right - rc.left;
            if (rc.bottom > rc.top) r.h = rc.bottom - rc.top;
        }
    }
    return r;
}

void ApplyWindowMode(HWND hwnd, const PresentParams& p) {
    if (!hwnd || !IsWindow(hwnd)) return;
    LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    if (!p.windowed) {
        style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX |
                   WS_SYSMENU);
        style |= WS_POPUP;
        SetWindowLongPtrW(hwnd, GWL_STYLE, style);
        SetWindowPos(hwnd, HWND_TOP, 0, 0, (int)p.w, (int)p.h,
                     SWP_FRAMECHANGED);
    } else {
        style &= ~WS_POPUP;
        style |= WS_OVERLAPPEDWINDOW;
        SetWindowLongPtrW(hwnd, GWL_STYLE, style);
        RECT rc = {0, 0, (LONG)p.w, (LONG)p.h};
        AdjustWindowRect(&rc, (DWORD)style, FALSE);
        SetWindowPos(hwnd, HWND_NOTOPMOST, 50, 50, rc.right - rc.left,
                     rc.bottom - rc.top, SWP_FRAMECHANGED);
    }
    // SW_SHOWDEFAULT honours the STARTUPINFO: a hidden launch stays hidden,
    // an ordinary launch shows the window without stealing focus.
    ShowWindow(hwnd, SW_SHOWDEFAULT);
}

// ---------------------------------------------------------------------------
// IDirect3D9
// ---------------------------------------------------------------------------
static uint32_t __stdcall D3D9_QueryInterface(void*, const void*, void** out) {
    if (out) *out = &g_d3d9;
    return 0;
}
static uint32_t __stdcall D3D9_AddRef(void*) { return ++g_d3d9.refs; }
static uint32_t __stdcall D3D9_Release(void*) { return --g_d3d9.refs; }
static uint32_t __stdcall D3D9_GetAdapterCount(void*) { return 1; }

static uint32_t __stdcall D3D9_GetAdapterIdentifier(void*, uint32_t, uint32_t,
                                                    void* out) {
    if (out) FillAdapterIdentifier(out);
    return 0;
}

struct DisplayMode {
    uint32_t width;
    uint32_t height;
    uint32_t refreshRate;
    uint32_t format;
};

static std::vector<DisplayMode> g_displayModes;
static bool g_displayModesInit = false;

static void EnsureDisplayModes() {
    if (g_displayModesInit) return;
    g_displayModesInit = true;
    g_displayModes.clear();

    DEVMODEW dm{};
    dm.dmSize = sizeof(dm);
    for (DWORD i = 0; EnumDisplaySettingsW(nullptr, i, &dm); ++i) {
        if (dm.dmBitsPerPel != 32) continue;
        if (dm.dmPelsWidth < 640 || dm.dmPelsHeight < 480) continue;
        DisplayMode m{dm.dmPelsWidth, dm.dmPelsHeight,
                      dm.dmDisplayFrequency ? dm.dmDisplayFrequency : 60u, 22u};
        bool duplicate = false;
        for (const auto& e : g_displayModes) {
            if (e.width == m.width && e.height == m.height &&
                e.refreshRate == m.refreshRate) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) g_displayModes.push_back(m);
    }

    // The game exposes common refresh rates even when the desktop mode list
    // only reports one of them.
    std::vector<DisplayMode> extra;
    for (const auto& m : g_displayModes) {
        for (uint32_t r : {60u, 120u, 144u, 240u}) {
            bool found = false;
            for (const auto& e : g_displayModes) {
                if (e.width == m.width && e.height == m.height &&
                    e.refreshRate == r) {
                    found = true;
                    break;
                }
            }
            for (const auto& e : extra) {
                if (e.width == m.width && e.height == m.height &&
                    e.refreshRate == r)
                    found = true;
            }
            if (!found) extra.push_back({m.width, m.height, r, 22});
        }
    }
    g_displayModes.insert(g_displayModes.end(), extra.begin(), extra.end());
    if (g_displayModes.empty()) g_displayModes.push_back({1024, 768, 60, 22});

    std::sort(g_displayModes.begin(), g_displayModes.end(),
              [](const DisplayMode& a, const DisplayMode& b) {
                  if (a.width != b.width) return a.width < b.width;
                  if (a.height != b.height) return a.height < b.height;
                  return a.refreshRate < b.refreshRate;
              });
}

static uint32_t __stdcall D3D9_GetAdapterModeCount(void*, uint32_t, uint32_t) {
    EnsureDisplayModes();
    return (uint32_t)g_displayModes.size();
}

static uint32_t __stdcall D3D9_EnumAdapterModes(void*, uint32_t, uint32_t format,
                                                uint32_t modeIndex, void* out) {
    EnsureDisplayModes();
    if (modeIndex >= g_displayModes.size()) return 0x8876086C;
    if (out) {
        const auto& m = g_displayModes[modeIndex];
        uint32_t* p = (uint32_t*)out;
        p[0] = m.width;
        p[1] = m.height;
        p[2] = m.refreshRate;
        p[3] = format ? format : m.format;
    }
    return 0;
}

static uint32_t __stdcall D3D9_GetAdapterDisplayMode(void*, uint32_t,
                                                     void* out) {
    if (out) {
        DEVMODEW dm{};
        dm.dmSize = sizeof(dm);
        EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &dm);
        uint32_t* m = (uint32_t*)out;
        m[0] = dm.dmPelsWidth ? dm.dmPelsWidth : 1024;
        m[1] = dm.dmPelsHeight ? dm.dmPelsHeight : 768;
        m[2] = g_currentRefreshRate
                   ? g_currentRefreshRate
                   : (dm.dmDisplayFrequency ? dm.dmDisplayFrequency : 60);
        m[3] = 22;
    }
    return 0;
}

static uint32_t __stdcall D3D9_GetDeviceCaps(void*, uint32_t, uint32_t,
                                             void* caps) {
    if (caps) FillCaps(caps);
    return 0;
}

static uint32_t __stdcall D3D9_GetAdapterMonitor(void*, uint32_t) {
    return (uint32_t)(uintptr_t)MonitorFromWindow(GetDesktopWindow(),
                                                  MONITOR_DEFAULTTOPRIMARY);
}
static uint32_t __stdcall D3D9_CheckDeviceType(void*, uint32_t, uint32_t,
                                               uint32_t, uint32_t, uint32_t) {
    return 0;
}
static uint32_t __stdcall D3D9_CheckDeviceFormat(void*, uint32_t, uint32_t,
                                                 uint32_t, uint32_t, uint32_t,
                                                 uint32_t) {
    return 0;
}
static uint32_t __stdcall D3D9_CheckDeviceMultiSampleType(void*, uint32_t,
                                                          uint32_t, uint32_t,
                                                          uint32_t, uint32_t,
                                                          void* q) {
    if (q) *(uint32_t*)q = 0;
    return 0;
}
static uint32_t __stdcall D3D9_CheckDeviceFormatConversion(void*, uint32_t,
                                                           uint32_t, uint32_t,
                                                           uint32_t) {
    return 0;
}

static uint32_t __stdcall D3D9_CreateDevice(void*, uint32_t adapter,
                                            uint32_t devType, HWND focus,
                                            uint32_t flags, void* pp,
                                            FakeObj** out) {
    PresentParams params = ReadPresentParams(pp);
    const uint32_t w = params.w, h = params.h;
    g_currentRefreshRate = params.refreshRate ? params.refreshRate : 60;
    g_currentInterval = params.presentationInterval;
    g_backend.SetVSync(params.presentationInterval != 0x80000000);
    if (pp) {
        uint32_t* p = (uint32_t*)pp;
        p[0] = w;
        p[1] = h;
        p[12] = g_currentRefreshRate;
        p[13] = g_currentInterval;
    }

    LOG("[d3d9] CreateDevice adapter=%u type=%u hwnd=%p flags=0x%X %ux%u "
        "windowed=%d",
        adapter, devType, (void*)focus, flags, w, h, (int)params.windowed);

    g_deviceWnd = focus;
    ApplyWindowMode(focus, params);

    if (!g_backendReady) {
        if (!g_backend.Init(focus, w, h)) {
            LOG("[d3d9] Vulkan backend init failed");
            return 0x8876086C;
        }
        if (!g_constRing.Init(64u * 1024 * 1024)) {
            LOG("[d3d9] constant ring init failed");
            return 0x8876086C;
        }
        g_backend.BindUniformRing(g_constRing.Buffer(), ConstantRing::kBlockBytes);
        g_shaders.Init("shaders");
        g_resources.Init();
        g_psos.Init();
        g_backendReady = true;
    }
    if (!g_frameActive) StartNewFrame();
    InitDefaultRenderStates();
    if (out) *out = &g_device;
    return 0;
}

// ---------------------------------------------------------------------------
// IDirect3DSwapChain9
// ---------------------------------------------------------------------------
struct FakeSwapChain {
    void** vt;
    uint32_t refs;
};

static uint32_t __stdcall Sc_QueryInterface(FakeSwapChain*, const void*,
                                            void** out) {
    if (out) *out = nullptr;
    return 0x80004002u;
}
static uint32_t __stdcall Sc_AddRef(FakeSwapChain* s) { return ++s->refs; }
static uint32_t __stdcall Sc_Release(FakeSwapChain* s) { return --s->refs; }
static uint32_t __stdcall Sc_Present(FakeSwapChain*, const RECT* a,
                                     const RECT* b, void* c, const void* d) {
    return Dev_Present(nullptr, a, b, c, d);
}
static uint32_t __stdcall Sc_GetFrontBufferData(FakeSwapChain*, uint32_t,
                                                FakeSurface* surf) {
    return surf ? 0u : 0x8876086Cu;
}
static uint32_t __stdcall Sc_GetBackBuffer(FakeSwapChain*, uint32_t, uint32_t,
                                           void** out) {
    if (out) *out = g_resources.Backbuffer();
    return (out && *out) ? 0u : 0x8876086Cu;
}
static uint32_t __stdcall Sc_GetRasterStatus(FakeSwapChain*, uint32_t,
                                             void* out) {
    if (out) memset(out, 0, 16);
    return 0;
}
static uint32_t __stdcall Sc_GetDisplayMode(FakeSwapChain*, void* out) {
    return Dev_GetDisplayMode(nullptr, 0, out);
}
static uint32_t __stdcall Sc_GetDevice(FakeSwapChain*, void** out) {
    if (out) *out = &g_device;
    return 0;
}
static uint32_t __stdcall Sc_GetPresentParameters(FakeSwapChain*, void* out) {
    if (out) {
        uint32_t* p = (uint32_t*)out;
        memset(p, 0, 56);
        FakeSurface* bb = g_resources.Backbuffer();
        p[0] = bb ? bb->width : 1024;
        p[1] = bb ? bb->height : 768;
        p[2] = bb ? bb->format : 22;
        p[3] = 1;
        p[6] = 1;
        p[7] = (uint32_t)(uintptr_t)g_deviceWnd;
        p[8] = 0;
        p[12] = g_currentRefreshRate;
        p[13] = g_currentInterval;
    }
    return 0;
}

static void* g_scVt[10] = {
    (void*)Sc_QueryInterface,     (void*)Sc_AddRef,
    (void*)Sc_Release,            (void*)Sc_Present,
    (void*)Sc_GetFrontBufferData, (void*)Sc_GetBackBuffer,
    (void*)Sc_GetRasterStatus,    (void*)Sc_GetDisplayMode,
    (void*)Sc_GetDevice,          (void*)Sc_GetPresentParameters};
static FakeSwapChain g_swapchain{g_scVt, 1};

static uint32_t __stdcall Dev_GetSwapChain(void*, uint32_t, void** out) {
    if (out) *out = &g_swapchain;
    return 0;
}

// ---------------------------------------------------------------------------
// IDirect3DQuery9
// ---------------------------------------------------------------------------
struct FakeQuery {
    void** vt;
    uint32_t refs;
};
static uint32_t __stdcall Q_QueryInterface(FakeQuery*, const void*, void** o) {
    if (o) *o = nullptr;
    return 0x80004002u;
}
static uint32_t __stdcall Q_AddRef(FakeQuery* q) { return ++q->refs; }
static uint32_t __stdcall Q_Release(FakeQuery* q) { return --q->refs; }
static uint32_t __stdcall Q_GetDevice(FakeQuery*, void** o) {
    if (o) *o = &g_device;
    return 0;
}
static uint32_t __stdcall Q_GetType(FakeQuery*) { return 1; }
static uint32_t __stdcall Q_GetDataSize(FakeQuery*) { return 4; }
static uint32_t __stdcall Q_Issue(FakeQuery*, uint32_t) { return 0; }
static uint32_t __stdcall Q_GetData(FakeQuery*, void* data, uint32_t size,
                                    uint32_t) {
    if (data && size >= 4) *(uint32_t*)data = 1;  // never leave the engine spinning
    return 0;
}
static void* g_queryVt[8] = {(void*)Q_QueryInterface, (void*)Q_AddRef,
                             (void*)Q_Release,        (void*)Q_GetDevice,
                             (void*)Q_GetType,        (void*)Q_GetDataSize,
                             (void*)Q_Issue,          (void*)Q_GetData};
static FakeQuery g_query{g_queryVt, 1};

static uint32_t __stdcall Dev_CreateQuery(void*, uint32_t, void** out) {
    if (out) *out = &g_query;
    return 0;
}

void BuildD3D9Vtable() {
    static void* vt[32];
    constexpr int kFactoryCount =
        (int)(sizeof(kFactoryArgs) / sizeof(kFactoryArgs[0]));
    for (int i = 0; i < 32; ++i)
        vt[i] = StubForArityExport(i < kFactoryCount ? kFactoryArgs[i] : 0);

    vt[kD3D9QueryInterface] = (void*)D3D9_QueryInterface;
    vt[kD3D9AddRef] = (void*)D3D9_AddRef;
    vt[kD3D9Release] = (void*)D3D9_Release;
    vt[kD3D9GetAdapterCount] = (void*)D3D9_GetAdapterCount;
    vt[kD3D9GetAdapterIdentifier] = (void*)D3D9_GetAdapterIdentifier;
    vt[kD3D9GetAdapterModeCount] = (void*)D3D9_GetAdapterModeCount;
    vt[kD3D9EnumAdapterModes] = (void*)D3D9_EnumAdapterModes;
    vt[kD3D9GetAdapterDisplayMode] = (void*)D3D9_GetAdapterDisplayMode;
    vt[kD3D9CheckDeviceType] = (void*)D3D9_CheckDeviceType;
    vt[kD3D9CheckDeviceFormat] = (void*)D3D9_CheckDeviceFormat;
    vt[kD3D9CheckDeviceMultiSampleType] = (void*)D3D9_CheckDeviceMultiSampleType;
    vt[kD3D9CheckDeviceFormatConversion] =
        (void*)D3D9_CheckDeviceFormatConversion;
    vt[kD3D9GetAdapterMonitor] = (void*)D3D9_GetAdapterMonitor;
    vt[kD3D9CheckDepthStencilMatch] = (void*)D3D9_CheckDeviceType;
    vt[kD3D9GetDeviceCaps] = (void*)D3D9_GetDeviceCaps;
    vt[kD3D9CreateDevice] = (void*)D3D9_CreateDevice;

    // Device vtable entries that create auxilliary objects live here with the
    // objects they hand out.
    g_device.vt[kGetSwapChain] = (void*)Dev_GetSwapChain;
    g_device.vt[kCreateQuery] = (void*)Dev_CreateQuery;

    g_d3d9.vt = vt;
    g_d3d9.refs = 1;
}

bool ShimInit() {
    BuildDeviceVtable();
    BuildD3D9Vtable();
    LOG("[shim] vtables built");
    return true;
}

void ShimShutdown() {
    if (g_backendReady) {
        g_backend.WaitIdle();
        g_psos.Shutdown();
        g_resources.Shutdown();
        g_constRing.Shutdown();
        g_backend.Shutdown();
        g_backendReady = false;
    }
}

void* __stdcall ShimDirect3DCreate9(uint32_t sdkVersion) {
    LOG("[shim] Direct3DCreate9(%u) intercepted", sdkVersion);
    return &g_d3d9;
}
