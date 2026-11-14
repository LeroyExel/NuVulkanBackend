#include "d3d9_internal.h"
#include "stud_shadows.h"

using namespace d3d9;

DeviceState g_st;
FakeObj g_device{};
HWND g_deviceWnd = nullptr;
uint32_t g_currentRefreshRate = 60;
uint32_t g_currentInterval = 0;
bool g_backendReady = false;
bool g_frameActive = false;
uint32_t g_frameNumber = 0;
uint32_t g_drawCalls = 0;
std::unordered_map<uint32_t, VkSampler> g_samplerCache;

namespace {

constexpr uint32_t kNotImpl = 0x80004001u;  // E_NOTIMPL

// Every unimplemented slot must pop exactly the bytes its caller pushed, so
// stubs exist per arity and always consume `this`.
uint32_t g_stubHits[9];

uint32_t __stdcall Stub0(void*) { ++g_stubHits[0]; return kNotImpl; }
uint32_t __stdcall Stub1(void*, uint32_t) { ++g_stubHits[1]; return kNotImpl; }
uint32_t __stdcall Stub2(void*, uint32_t, uint32_t) { ++g_stubHits[2]; return kNotImpl; }
uint32_t __stdcall Stub3(void*, uint32_t, uint32_t, uint32_t) { ++g_stubHits[3]; return kNotImpl; }
uint32_t __stdcall Stub4(void*, uint32_t, uint32_t, uint32_t, uint32_t) {
    ++g_stubHits[4];
    return kNotImpl;
}
uint32_t __stdcall Stub5(void*, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t) {
    ++g_stubHits[5];
    return kNotImpl;
}
uint32_t __stdcall Stub6(void*, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t) {
    ++g_stubHits[6];
    return kNotImpl;
}
uint32_t __stdcall Stub7(void*, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t) {
    ++g_stubHits[7];
    return kNotImpl;
}
uint32_t __stdcall Stub8(void*, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t) {
    ++g_stubHits[8];
    return kNotImpl;
}

void* StubForArity(int n) {
    switch (n) {
        case 0: return (void*)Stub0;
        case 1: return (void*)Stub1;
        case 2: return (void*)Stub2;
        case 3: return (void*)Stub3;
        case 4: return (void*)Stub4;
        case 5: return (void*)Stub5;
        case 6: return (void*)Stub6;
        case 7: return (void*)Stub7;
        default: return (void*)Stub8;
    }
}

constexpr int kDevSlotCount = (int)(sizeof(kDevArgs) / sizeof(kDevArgs[0]));

}  // namespace

// The factory's own vtable needs arity-correct stubs too.
void* StubForArityExport(int n) { return StubForArity(n); }

// ---------------------------------------------------------------------------
// Shader objects
// ---------------------------------------------------------------------------
static uint32_t __stdcall Sh_QueryInterface(FakeShaderObj*, const void*,
                                            void** out) {
    if (out) *out = nullptr;
    return 0x80004002u;
}
static uint32_t __stdcall Sh_AddRef(FakeShaderObj* s) { return ++s->refs; }
static uint32_t __stdcall Sh_Release(FakeShaderObj* s) {
    uint32_t n = --s->refs;
    if (n == 0) delete s;
    return n;
}
static uint32_t __stdcall Sh_GetDevice(FakeShaderObj*, void** out) {
    if (out) *out = &g_device;
    return 0;
}
static uint32_t __stdcall Sh_GetFunction(FakeShaderObj*, void* data,
                                         uint32_t* size) {
    if (size) *size = 0;
    (void)data;
    return 0;
}

void* g_shaderVt[5] = {(void*)Sh_QueryInterface, (void*)Sh_AddRef,
                       (void*)Sh_Release,        (void*)Sh_GetDevice,
                       (void*)Sh_GetFunction};

FakeShaderObj* MakeShaderObj(uint64_t hash, bool pixel) {
    return new FakeShaderObj{g_shaderVt, 1, 0x5A4D4001u, pixel ? 1u : 0u, hash};
}

uint64_t ShaderHashOf(FakeResource* sh) {
    return (sh && sh->type == 0x5A4D4001u) ? ((FakeShaderObj*)sh)->hash : 0;
}

// Walks the token stream instead of scanning for a 0x0000FFFF word: a COMMENT
// payload can contain that value.
uint32_t ShaderDwordSize(const uint32_t* code) {
    if (!code) return 0;
    uint32_t i = 1;  // skip the version token
    while (i < 16384) {
        const uint32_t head = code[i];
        if ((head & 0xFFFF) == 0xFFFF) return (i + 1) * 4;
        i += Sm3InstructionDwords(head);
    }
    return (i + 1) * 4;
}

// ---------------------------------------------------------------------------
// Capability report
// ---------------------------------------------------------------------------
void FillAdapterIdentifier(void* out) {
    uint8_t* p = (uint8_t*)out;
    memset(p, 0, 1100);
    strcpy_s((char*)p, 512, "NuVulkan 1.3 backend");
    strcpy_s((char*)p + 512, 512, "NuVulkan 1.3");
    strcpy_s((char*)p + 1024, 32, "\\\\.\\DISPLAY1");
    *(uint32_t*)(p + 1064) = 0x8086;  // vendor: Intel, skips AMD/NVIDIA paths
    *(uint32_t*)(p + 1068) = 0x1234;
    *(uint32_t*)(p + 1096) = 0;  // WHQLLevel
}

void FillCaps(void* out) {
    uint32_t* c = (uint32_t*)out;
    memset(c, 0, 304);
    c[0] = 1;
    c[1] = 0;
    c[2] = 0x00000001;
    c[3] = 0x00020000;
    c[5] = 0x80000000;
    c[7] = 0x00090400;
    c[8] = 0x00000001;
    c[9] = 0x00000001 | 0x00000040 | 0x00000100 | 0x00000200 | 0x00010000;
    c[14] = 0x00000200;
    c[15] = 0x00000028 | 0x00000001 | 0x00000004 | 0x00000008;
    c[16] = 0x02000000;
    c[19] = 0x00000100 | 0x00000200;
    c[22] = 4096;
    c[23] = 4096;
    c[24] = 2048;
    c[25] = 8192;
    c[26] = 8192;
    c[27] = 16;
    c[34] = 0xFF;
    c[35] = 0x00000100;
    c[37] = 8;
    c[38] = 8;
    c[39] = 0x000001DB;
    c[40] = 8;
    c[45] = 0xFFFF;
    c[46] = 0xFFFF;
    c[47] = 16;
    c[48] = 255;
    c[49] = 0xFFFF0300;
    c[50] = 256;
    c[51] = 0xFFFF0300;
    c[52] = 0x3F800000;
    c[53] = 0;
    c[58] = 1;
    c[59] = 0x000003FF;
    c[60] = 4;
    c[61] = 0x03000300;
    c[62] = 0x00000001;
    c[63] = 24;
    c[64] = 32;
    c[65] = 4;
    c[66] = 0x0000001F;
    c[67] = 24;
    c[68] = 32;
    c[69] = 4;
    c[70] = 512;
    c[72] = 65535;
    c[73] = 65535;
    c[74] = 512;
    c[75] = 512;
}

// ---------------------------------------------------------------------------
// Device identity
// ---------------------------------------------------------------------------
static const uint8_t kIID_IUnknown[16] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0xC0, 0x00, 0x00, 0x00,
                                          0x00, 0x00, 0x00, 0x46};
static const uint8_t kIID_Device9[16] = {0x96, 0x3B, 0x22, 0xD0, 0x7A, 0xBF,
                                         0xFD, 0x43, 0x92, 0xBD, 0xA4, 0x3B,
                                         0x0D, 0x82, 0xB9, 0xEB};

bool SameIID(const void* riid, const uint8_t* iid) {
    return riid && memcmp(riid, iid, 16) == 0;
}

// Answering every IID would hand the engine an IDirect3DDevice9Ex it does not
// have: the Ex methods live past slot 118 and have no implementation here.
static uint32_t __stdcall Dev_QueryInterface(void*, const void* riid,
                                             void** out) {
    if (!out) return 0x80004003u;
    if (SameIID(riid, kIID_IUnknown) || SameIID(riid, kIID_Device9)) {
        *out = &g_device;
        ++g_device.refs;
        return 0;
    }
    *out = nullptr;
    return 0x80004002u;
}
static uint32_t __stdcall Dev_AddRef(void*) { return ++g_device.refs; }
static uint32_t __stdcall Dev_Release(void*) { return --g_device.refs; }
static uint32_t __stdcall Dev_TestCooperativeLevel(void*) { return 0; }
static uint32_t __stdcall Dev_EvictManagedResources(void*) { return 0; }
static uint32_t __stdcall Dev_GetDeviceCaps(void*, void* caps) {
    if (caps) FillCaps(caps);
    return 0;
}
static uint32_t __stdcall Dev_GetDirect3D(void*, void** out) {
    if (out) *out = &g_d3d9;
    return 0;
}

uint32_t __stdcall Dev_GetDisplayMode(void*, uint32_t, void* out) {
    if (out) {
        uint32_t* m = (uint32_t*)out;
        FakeSurface* bb = g_resources.Backbuffer();
        m[0] = bb ? bb->width : 1024;
        m[1] = bb ? bb->height : 768;
        m[2] = g_currentRefreshRate;
        m[3] = bb ? bb->format : 22;
    }
    return 0;
}

static uint32_t __stdcall Dev_GetCreationParameters(void*, void* out) {
    if (out) {
        uint32_t* p = (uint32_t*)out;
        p[0] = 0;
        p[1] = 1;
        p[2] = (uint32_t)(uintptr_t)g_deviceWnd;
        p[3] = 0x50;
    }
    return 0;
}

static uint32_t __stdcall Dev_Reset(void*, void* pp) {
    if (!g_backendReady) return 0;
    PresentParams params = ReadPresentParams(pp);
    const uint32_t w = params.w, h = params.h;
    g_currentRefreshRate = params.refreshRate ? params.refreshRate : 60;
    g_currentInterval = params.presentationInterval;
    g_backend.SetVSync(params.presentationInterval != 0x80000000);
    if (g_frameActive) {
        g_backend.EndFrame(false);
        g_frameActive = false;
    }
    ApplyWindowMode(g_deviceWnd, params);
    g_backend.RecreateSwapchain(w, h);
    g_resources.UpdateBackbuffer(w, h, 22);
    g_constRing.Reset(0);
    g_st.scissor = {0, 0, (LONG)w, (LONG)h};
    g_st.scissorValid = true;
    g_st.viewport = {0, 0, w, h, 0.0f, 1.0f};
    g_st.viewportValid = true;
    g_st.targetsDirty = false;
    ApplyTargets();
    ResetDrawCache();
    if (pp) {
        uint32_t* p = (uint32_t*)pp;
        p[0] = w;
        p[1] = h;
        p[12] = g_currentRefreshRate;
        p[13] = g_currentInterval;
    }
    return 0;
}

bool StartNewFrame() {
    if (!g_backendReady) return false;
    if (!g_backend.BeginFrame()) return false;
    g_resources.OnFrameBegin(g_backend.FrameSlot());
    g_constRing.Reset(g_backend.FrameSlot());
    ResResetDynamicRing(g_backend.FrameSlot());
    g_resources.FlushPendingUploads(true);
    g_st.vsConstDirty = true;
    g_st.psConstDirty = true;
    g_st.targetsDirty = true;
    ResetDrawCache();
    g_studShadows.BeginFrame();
    g_frameActive = true;
    return true;
}

uint32_t __stdcall Dev_Present(void*, const RECT*, const RECT*, const void*,
                               const void*) {
    if (g_backendReady && g_frameActive) {
        g_studShadows.FlushEndOfFrame();
        g_backend.EndFrame(true);
        g_frameActive = false;
        g_psos.FlushCache();
    }
    ++g_frameNumber;
    return 0;
}

static uint32_t __stdcall Dev_BeginScene(void*) {
    if (g_backendReady && !g_frameActive) {
        StartNewFrame();
    }
    g_st.sceneOpen = true;
    return 0;
}
static uint32_t __stdcall Dev_EndScene(void*) {
    g_st.sceneOpen = false;
    return 0;
}

static uint32_t __stdcall Dev_Clear(void*, uint32_t count, const RECT* rects,
                                    uint32_t flags, uint32_t color, float z,
                                    uint32_t stencil) {
    if (!g_backendReady) return 0;
    if (!g_frameActive) g_frameActive = StartNewFrame();
    if ((flags & 2u) && g_studShadows.FlushBeforeDepthClear(g_st.depthTarget)) {
        ResetDrawCache();
    }
    if (g_st.targetsDirty) {
        g_st.targetsDirty = false;
        ApplyTargets();
    }
    const VkExtent2D ext = g_backend.TargetExtent();
    int32_t clipL = 0;
    int32_t clipT = 0;
    int32_t clipR = (int32_t)ext.width;
    int32_t clipB = (int32_t)ext.height;
    if (g_st.viewportValid && g_st.viewport.width && g_st.viewport.height) {
        const int32_t vpL = (int32_t)g_st.viewport.x;
        const int32_t vpT = (int32_t)g_st.viewport.y;
        const int32_t vpR = vpL + (int32_t)g_st.viewport.width;
        const int32_t vpB = vpT + (int32_t)g_st.viewport.height;
        if (vpL > clipL) clipL = vpL;
        if (vpT > clipT) clipT = vpT;
        if (vpR < clipR) clipR = vpR;
        if (vpB < clipB) clipB = vpB;
    }
    if (g_st.rs[RS_SCISSORTESTENABLE] && g_st.scissorValid) {
        if (g_st.scissor.left > clipL) clipL = g_st.scissor.left;
        if (g_st.scissor.top > clipT) clipT = g_st.scissor.top;
        if (g_st.scissor.right < clipR) clipR = g_st.scissor.right;
        if (g_st.scissor.bottom < clipB) clipB = g_st.scissor.bottom;
    }
    if (clipL >= clipR || clipT >= clipB) return 0;

    if (!count || !rects) {
        if (clipL == 0 && clipT == 0 && clipR == (int32_t)ext.width &&
            clipB == (int32_t)ext.height) {
            g_backend.Clear(flags, color, z, stencil, nullptr, 0);
        } else {
            RECT r{clipL, clipT, clipR, clipB};
            g_backend.Clear(flags, color, z, stencil, &r, 1);
        }
        return 0;
    }

    RECT localRects[16];
    std::vector<RECT> dynRects;
    RECT* clipped = count <= 16 ? localRects : nullptr;
    if (!clipped) {
        dynRects.resize(count);
        clipped = dynRects.data();
    }
    uint32_t outCount = 0;
    for (uint32_t i = 0; i < count; ++i) {
        int32_t l = rects[i].left > clipL ? rects[i].left : clipL;
        int32_t t = rects[i].top > clipT ? rects[i].top : clipT;
        int32_t r = rects[i].right < clipR ? rects[i].right : clipR;
        int32_t b = rects[i].bottom < clipB ? rects[i].bottom : clipB;
        if (l < r && t < b) {
            clipped[outCount++] = {l, t, r, b};
        }
    }
    if (outCount == 0) return 0;
    if (outCount == 1 && clipped[0].left == 0 && clipped[0].top == 0 &&
        clipped[0].right == (int32_t)ext.width &&
        clipped[0].bottom == (int32_t)ext.height) {
        g_backend.Clear(flags, color, z, stencil, nullptr, 0);
    } else {
        g_backend.Clear(flags, color, z, stencil, clipped, outCount);
    }
    return 0;
}

static uint32_t __stdcall Dev_SetRenderState(void*, uint32_t state,
                                             uint32_t value) {
    if (state < 256) {
        if (g_st.rs[state] == value &&
            !(state == RS_COLORWRITEENABLE && !g_st.colorWriteSet)) {
            return 0;
        }
        g_st.rs[state] = value;
        switch (state) {
            case RS_FILLMODE:
            case RS_ALPHATESTENABLE:
            case RS_SRCBLEND:
            case RS_DESTBLEND:
            case RS_ALPHAREF:
            case RS_ALPHAFUNC:
            case RS_ALPHABLENDENABLE:
            case RS_COLORWRITEENABLE:
            case RS_BLENDOP:
            case RS_SEPARATEALPHABLENDENABLE:
            case RS_SRCBLENDALPHA:
            case RS_DESTBLENDALPHA:
            case RS_BLENDOPALPHA:
                g_pipeDirty = true;
                break;
            case RS_ZENABLE:
            case RS_ZWRITEENABLE:
            case RS_CULLMODE:
            case RS_ZFUNC:
            case RS_STENCILENABLE:
                g_dynRsDirty = true;
                break;
            case RS_STENCILFAIL:
            case RS_STENCILZFAIL:
            case RS_STENCILPASS:
            case RS_STENCILFUNC:
            case RS_STENCILREF:
            case RS_STENCILMASK:
            case RS_STENCILWRITEMASK:
            case RS_TWOSIDEDSTENCILMODE:
            case RS_CCW_STENCILFAIL:
            case RS_CCW_STENCILZFAIL:
            case RS_CCW_STENCILPASS:
            case RS_CCW_STENCILFUNC:
                g_stencilDirty = true;
                break;
            case RS_SCISSORTESTENABLE:
                g_scDirty = true;
                break;
            default:
                break;
        }
    }
    if (state == RS_COLORWRITEENABLE) g_st.colorWriteSet = true;
    return 0;
}

void InitDefaultRenderStates() {
    g_st.rs[RS_ZENABLE] = 1;
    g_st.rs[RS_FILLMODE] = 3;
    g_st.rs[RS_SHADEMODE] = 2;
    g_st.rs[RS_ZWRITEENABLE] = 1;
    g_st.rs[RS_ALPHATESTENABLE] = 0;
    g_st.rs[RS_SRCBLEND] = 2;
    g_st.rs[RS_DESTBLEND] = 1;
    g_st.rs[RS_CULLMODE] = 3;
    g_st.rs[RS_ZFUNC] = 4;
    g_st.rs[RS_ALPHAREF] = 0;
    g_st.rs[RS_ALPHAFUNC] = 8;
    g_st.rs[RS_ALPHABLENDENABLE] = 0;
    g_st.rs[RS_STENCILENABLE] = 0;
    g_st.rs[RS_STENCILFAIL] = 1;
    g_st.rs[RS_STENCILZFAIL] = 1;
    g_st.rs[RS_STENCILPASS] = 1;
    g_st.rs[RS_STENCILFUNC] = 8;
    g_st.rs[RS_STENCILREF] = 0;
    g_st.rs[RS_STENCILMASK] = 0xFFFFFFFFu;
    g_st.rs[RS_STENCILWRITEMASK] = 0xFFFFFFFFu;
    g_st.rs[RS_COLORWRITEENABLE] = 0xFu;
    g_st.rs[RS_BLENDOP] = 1;
    g_st.rs[RS_TWOSIDEDSTENCILMODE] = 0;
    g_st.rs[RS_CCW_STENCILFAIL] = 1;
    g_st.rs[RS_CCW_STENCILZFAIL] = 1;
    g_st.rs[RS_CCW_STENCILPASS] = 1;
    g_st.rs[RS_CCW_STENCILFUNC] = 8;
    g_st.rs[RS_SRCBLENDALPHA] = 2;
    g_st.rs[RS_DESTBLENDALPHA] = 1;
    g_st.rs[RS_BLENDOPALPHA] = 1;
}
static uint32_t __stdcall Dev_GetRenderState(void*, uint32_t state, void* out) {
    if (out) *(uint32_t*)out = state < 256 ? g_st.rs[state] : 0;
    return 0;
}
static uint32_t __stdcall Dev_SetTransform(void*, uint32_t, const void*) {
    return 0;
}
static uint32_t __stdcall Dev_MultiplyTransform(void*, uint32_t, const void*) {
    return 0;
}

static uint32_t __stdcall Dev_SetViewport(void*, const void* vp) {
    if (!vp) return 0;
    memcpy(&g_st.viewport, vp, sizeof(D3DViewport));
    g_st.viewportValid = true;
    g_vpDirty = true;
    return 0;
}

static uint32_t __stdcall Dev_SetScissorRect(void*, const RECT* r) {
    if (!r || !g_backendReady) return 0;
    if (g_st.scissorValid && memcmp(&g_st.scissor, r, sizeof(RECT)) == 0)
        return 0;
    g_st.scissor = *r;
    g_st.scissorValid = true;
    g_scDirty = true;
    return 0;
}

static uint32_t __stdcall Dev_SetStreamSource(void*, uint32_t stream,
                                              FakeResource* vb, uint32_t offset,
                                              uint32_t stride) {
    if (stream < 16) {
        g_st.streams[stream].vb = vb;
        g_st.streams[stream].offset = offset;
        g_st.streams[stream].stride = stride;
        const uint16_t bit = (uint16_t)(1u << stream);
        const uint16_t prev = g_st.streamMask;
        if (vb && stride > 0)
            g_st.streamMask |= bit;
        else
            g_st.streamMask &= (uint16_t)~bit;
        if (g_st.streamMask != prev) g_pipeDirty = true;
    }
    return 0;
}

static uint32_t __stdcall Dev_SetStreamSourceFreq(void*, uint32_t stream,
                                                  uint32_t setting) {
    if (stream < 16 && g_st.streams[stream].freq != setting) {
        g_st.streams[stream].freq = setting;
        uint16_t instMask = 0;
        uint32_t instCount = 1;
        for (uint32_t s = 0; s < 16; ++s) {
            const uint32_t f = g_st.streams[s].freq;
            if (f & 0x80000000u) {
                instMask |= (uint16_t)(1u << s);
            } else if (f & 0x40000000u) {
                const uint32_t n = f & 0x3FFFFFFFu;
                if (n > instCount) instCount = n;
            }
        }
        if (g_st.instancedMask != instMask) g_pipeDirty = true;
        g_st.instancedMask = instMask;
        g_st.instanceCount = instCount;
    }
    return 0;
}

static uint32_t __stdcall Dev_GetStreamSourceFreq(void*, uint32_t stream,
                                                  uint32_t* out) {
    if (out) *out = stream < 16 ? g_st.streams[stream].freq : 1;
    return 0;
}

static uint32_t __stdcall Dev_SetIndices(void*, FakeResource* ib) {
    g_st.indices = ib;
    if (ib) g_st.indexFormat = ib->p0;
    return 0;
}

static uint32_t __stdcall Dev_SetVertexDeclaration(void*, FakeResource* d) {
    if (g_st.decl != d) {
        g_st.decl = d;
        g_pipeDirty = true;
    }
    return 0;
}

// FVF synthesis is not implemented. The engine pairs shader draws with an
// explicit declaration; keeping the declaration unchanged matches D3D9 for
// these draws.
static uint32_t __stdcall Dev_SetFVF(void*, uint32_t) { return 0; }

static uint32_t __stdcall Dev_CreateVertexDeclaration(void*,
                                                      const uint32_t* tokens,
                                                      FakeResource** out) {
    FakeResource* d = g_resources.CreateVertexDeclaration(tokens);
    if (out) *out = d;
    return d ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_CreateVertexShader(void*, const uint32_t* code,
                                                 FakeResource** out) {
    const uint32_t size = ShaderDwordSize(code);
    const uint64_t h = g_shaders.Capture(false, code, size);
    if (out) *out = (FakeResource*)MakeShaderObj(h, false);
    return 0;
}

static uint32_t __stdcall Dev_CreatePixelShader(void*, const uint32_t* code,
                                                FakeResource** out) {
    const uint32_t size = ShaderDwordSize(code);
    const uint64_t h = g_shaders.Capture(true, code, size);
    if (out) *out = (FakeResource*)MakeShaderObj(h, true);
    return 0;
}

static uint32_t __stdcall Dev_SetVertexShader(void*, FakeResource* sh) {
    const uint64_t h = ShaderHashOf(sh);
    if (g_st.vsHash != h) {
        g_st.vsHash = h;
        g_st.vsConstInfo = g_shaders.GetConstInfo(false, h);
        g_pipeDirty = true;
    }
    return 0;
}
static uint32_t __stdcall Dev_SetPixelShader(void*, FakeResource* sh) {
    const uint64_t h = ShaderHashOf(sh);
    if (g_st.psHash != h) {
        g_st.psHash = h;
        g_st.psConstInfo = g_shaders.GetConstInfo(true, h);
        g_pipeDirty = true;
    }
    return 0;
}
static uint32_t __stdcall Dev_GetVertexShader(void*, FakeResource** out) {
    if (out) *out = nullptr;
    return 0;
}
static uint32_t __stdcall Dev_GetPixelShader(void*, FakeResource** out) {
    if (out) *out = nullptr;
    return 0;
}

// Constants are copied verbatim, as D3D9 does. Non-finite values are the
// engine's data; the shader emitter handles division and power edge cases.
static uint32_t __stdcall Dev_SetVertexShaderConstantF(void*, uint32_t reg,
                                                       const float* data,
                                                       uint32_t count) {
    if (reg >= 256 || !data || !count) return 0;
    uint32_t n = count;
    if (n > 256 - reg) n = 256 - reg;
    if (!g_st.vsConstDirty &&
        memcmp(&g_st.vsConst[reg * 4], data, n * 4 * sizeof(float)) == 0)
        return 0;
    memcpy(&g_st.vsConst[reg * 4], data, n * 4 * sizeof(float));
    if (reg + n > g_st.vsHighReg) g_st.vsHighReg = reg + n;
    g_st.vsConstDirty = true;
    return 0;
}

static uint32_t __stdcall Dev_SetPixelShaderConstantF(void*, uint32_t reg,
                                                      const float* data,
                                                      uint32_t count) {
    if (reg >= 224 || !data || !count) return 0;
    uint32_t n = count;
    if (n > 224 - reg) n = 224 - reg;
    if (!g_st.psConstDirty &&
        memcmp(&g_st.psConst[reg * 4], data, n * 4 * sizeof(float)) == 0)
        return 0;
    memcpy(&g_st.psConst[reg * 4], data, n * 4 * sizeof(float));
    if (reg + n > g_st.psHighReg) g_st.psHighReg = reg + n;
    g_st.psConstDirty = true;
    return 0;
}

static uint32_t __stdcall Dev_CreateVertexBuffer(void*, uint32_t length,
                                                 uint32_t usage, uint32_t fvf,
                                                 uint32_t pool,
                                                 FakeResource** out,
                                                 void** handle) {
    FakeResource* r = g_resources.CreateVertexBuffer(length, usage, fvf);
    if (out) *out = r;
    if (handle) *handle = nullptr;
    return r ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_CreateIndexBuffer(void*, uint32_t length,
                                                uint32_t usage, uint32_t format,
                                                uint32_t pool,
                                                FakeResource** out,
                                                void** handle) {
    FakeResource* r = g_resources.CreateIndexBuffer(length, usage, format);
    if (out) *out = r;
    if (handle) *handle = nullptr;
    return r ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_CreateTexture(void*, uint32_t w, uint32_t h,
                                            uint32_t levels, uint32_t usage,
                                            uint32_t format, uint32_t pool,
                                            FakeResource** out, void** handle) {
    FakeResource* r = g_resources.CreateTexture(w, h, levels, usage, format);
    if (out) *out = r;
    if (handle) *handle = nullptr;
    return r ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_SetTexture(void*, uint32_t stage,
                                         FakeResource* tex) {
    if (stage >= 16) return 0;
    if (g_st.textures[stage] == tex) return 0;
    g_st.textures[stage] = tex;
    const uint16_t bit = (uint16_t)(1u << stage);
    if (tex)
        g_st.texMask |= bit;
    else
        g_st.texMask &= (uint16_t)~bit;
    GpuTexture* t = tex ? g_resources.TextureOf(tex) : nullptr;
    if (t && (t->renderTarget || t->depth))
        g_st.rtStageMask |= bit;
    else
        g_st.rtStageMask &= (uint16_t)~bit;
    const bool depthCmp = t && !t->cube && !t->volume &&
                          IsShadowCompareD3DFormat(t->d3dFormat);
    if (depthCmp)
        g_st.shadowCompareMask |= (1u << stage);
    else
        g_st.shadowCompareMask &= ~(1u << stage);
    VkSampler samp =
        t ? (t->volume ? g_backend.VolumeSampler()
                       : GetSamplerForStage(stage, depthCmp))
          : g_backend.DefaultSampler();
    g_backend.SetTextureSlot(stage, t ? t->view : VK_NULL_HANDLE, samp,
                             t ? t->cube : false, t ? t->volume : false);
    return 0;
}

static uint32_t __stdcall Dev_DrawPrimitive(void*, uint32_t topo,
                                            uint32_t startVertex,
                                            uint32_t primCount) {
    TranslateDraw(topo, 0, startVertex, primCount, false);
    return 0;
}

static uint32_t __stdcall Dev_DrawIndexedPrimitive(void*, uint32_t topo,
                                                   int32_t baseVertex,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t startIndex,
                                                   uint32_t primCount) {
    TranslateDraw(topo, baseVertex, startIndex, primCount, true);
    return 0;
}

static uint32_t __stdcall Dev_GetBackBuffer(void*, uint32_t, uint32_t, uint32_t,
                                            FakeSurface** out) {
    if (out) *out = g_resources.Backbuffer();
    return 0;
}

static uint32_t __stdcall Dev_GetRenderTarget(void*, uint32_t index,
                                              FakeSurface** out) {
    if (out) *out = index < 4 ? g_st.colorTargets[index] : nullptr;
    return (index < 4 && g_st.colorTargets[index]) ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_SetRenderTarget(void*, uint32_t index,
                                              FakeSurface* surf) {
    if (index >= 4) return 0x8876086C;
    if (g_st.colorTargets[index] != surf) {
        g_st.colorTargets[index] = surf;
        g_st.targetsDirty = true;
    }
    if (index == 0 && surf) {
        g_st.scissor = {0, 0, (LONG)surf->width, (LONG)surf->height};
        g_st.scissorValid = true;
        g_st.viewport = {0, 0, surf->width, surf->height, 0.0f, 1.0f};
        g_st.viewportValid = true;
        g_vpDirty = true;
        g_scDirty = true;
    }
    return 0;
}

static uint32_t __stdcall Dev_GetDepthStencilSurface(void*, FakeSurface** out) {
    if (out) *out = g_st.depthTarget;
    return g_st.depthTarget ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_SetDepthStencilSurface(void*, FakeSurface* surf) {
    if (g_st.depthTarget == surf) return 0;
    g_st.depthTarget = surf;
    g_st.targetsDirty = true;
    return 0;
}

static uint32_t __stdcall Dev_CreateRenderTarget(void*, uint32_t w, uint32_t h,
                                                 uint32_t format, uint32_t,
                                                 uint32_t, uint32_t,
                                                 FakeSurface** out,
                                                 void** shared) {
    FakeSurface* s = g_resources.CreateStandaloneSurface(w, h, format, false, 1);
    if (out) *out = s;
    if (shared) *shared = nullptr;
    return s ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_CreateDepthStencilSurface(
    void*, uint32_t w, uint32_t h, uint32_t format, uint32_t, uint32_t, uint32_t,
    FakeSurface** out, void** shared) {
    FakeSurface* s = g_resources.CreateStandaloneSurface(w, h, format, true, 0);
    if (out) *out = s;
    if (shared) *shared = nullptr;
    return s ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_StretchRect(void*, FakeSurface* src,
                                          const RECT* srcRect,
                                          FakeSurface* dst, const RECT* dstRect,
                                          uint32_t) {
    if (!g_backendReady || !src || !dst) return 0x8876086C;
    if (g_st.targetsDirty) {
        g_st.targetsDirty = false;
        ApplyTargets();
    }
    if (g_studShadows.FlushIfReading(src)) {
        ResetDrawCache();
    }
    g_backend.BlitImage(MakeImageRef(src), srcRect, MakeImageRef(dst), dstRect);
    return 0;
}

static uint32_t __stdcall Dev_CreateCubeTexture(void*, uint32_t edge,
                                                uint32_t levels, uint32_t usage,
                                                uint32_t format, uint32_t,
                                                FakeResource** out,
                                                void** shared) {
    FakeResource* r = g_resources.CreateCubeTexture(edge, levels, usage, format);
    if (out) *out = r;
    if (shared) *shared = nullptr;
    return r ? 0 : 0x8876086C;
}

static uint32_t __stdcall Dev_SetSamplerState(void*, uint32_t stage,
                                              uint32_t type, uint32_t value) {
    if (stage >= 16) return 0;
    SamplerStageState& s = g_st.samplers[stage];
    bool changed = false;
    switch (type) {
        case 1: changed = (s.addressU != value); s.addressU = value; break;
        case 2: changed = (s.addressV != value); s.addressV = value; break;
        case 3: changed = (s.addressW != value); s.addressW = value; break;
        case 4: changed = (s.borderColor != value); s.borderColor = value; break;
        case 5: changed = (s.magFilter != value); s.magFilter = value; break;
        case 6: changed = (s.minFilter != value); s.minFilter = value; break;
        case 7: changed = (s.mipFilter != value); s.mipFilter = value; break;
        case 8: {
            float f = *(float*)&value;
            changed = (s.mipLodBias != f);
            s.mipLodBias = f;
            break;
        }
        case 9: changed = (s.maxMipLevel != value); s.maxMipLevel = value; break;
        case 10: changed = (s.maxAnisotropy != value); s.maxAnisotropy = value; break;
        case 11: changed = (s.srgbTexture != value); s.srgbTexture = value; break;
        default: break;
    }
    if (!changed) return 0;
    s.samplerDirty = true;
    if (!g_st.textures[stage]) return 0;
    GpuTexture* t = g_resources.TextureOf(g_st.textures[stage]);
    if (!t) return 0;
    const bool depthCmp =
        !t->cube && !t->volume && IsShadowCompareD3DFormat(t->d3dFormat);
    VkSampler samp =
        t->volume ? g_backend.VolumeSampler()
                  : GetSamplerForStage(stage, depthCmp);
    g_backend.SetTextureSlot(stage, t->view, samp, t->cube, t->volume);
    return 0;
}

static uint32_t __stdcall Dev_SetTextureStageState(void*, uint32_t, uint32_t,
                                                   uint32_t) {
    return 0;
}

// ---------------------------------------------------------------------------
// State blocks. Recording applies nothing until Apply restores the snapshot,
// which is what the engine's fullscreen composite depends on.
// ---------------------------------------------------------------------------
struct FakeStateBlock {
    void** vt;
    uint32_t refs;
    uint32_t captured;
    DeviceState saved;
};

static void* g_sbVt[6];
static FakeStateBlock* g_recordingBlock = nullptr;

static FakeStateBlock* NewStateBlock() {
    FakeStateBlock* b = new FakeStateBlock();
    b->vt = g_sbVt;
    b->refs = 1;
    b->captured = 0;
    return b;
}

static uint32_t __stdcall S_QueryInterface(FakeStateBlock*, const void*,
                                           void** o) {
    if (o) *o = nullptr;
    return 0x80004002u;
}
static uint32_t __stdcall S_AddRef(FakeStateBlock* b) { return ++b->refs; }
static uint32_t __stdcall S_Release(FakeStateBlock* b) {
    if (!b || !b->refs) return 0;
    if (--b->refs == 0) {
        delete b;
        return 0;
    }
    return b->refs;
}
static uint32_t __stdcall S_GetDevice(FakeStateBlock*, void** o) {
    if (o) *o = &g_device;
    return 0;
}
static uint32_t __stdcall S_Capture(FakeStateBlock* b) {
    if (b) {
        b->saved = g_st;
        b->captured = 1;
    }
    return 0;
}
static uint32_t __stdcall S_Apply(FakeStateBlock* b) {
    if (!b || !b->captured) return 0;
    g_st = b->saved;
    g_st.vsConstDirty = true;
    g_st.psConstDirty = true;
    g_st.targetsDirty = true;
    g_st.vsHighReg = 257;
    g_st.psHighReg = 257;
    g_st.vsConstInfo = g_shaders.GetConstInfo(false, g_st.vsHash);
    g_st.psConstInfo = g_shaders.GetConstInfo(true, g_st.psHash);
    g_pipeDirty = true;
    g_dynRsDirty = true;
    g_stencilDirty = true;
    g_vpDirty = true;
    g_scDirty = true;
    uint16_t rtMask = 0;
    uint32_t cmpMask = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        GpuTexture* t = g_st.textures[i] ? g_resources.TextureOf(g_st.textures[i])
                                         : nullptr;
        if (t && (t->renderTarget || t->depth)) rtMask |= (uint16_t)(1u << i);
        const bool depthCmp = t && !t->cube && !t->volume &&
                              IsShadowCompareD3DFormat(t->d3dFormat);
        if (depthCmp) cmpMask |= (1u << i);
        VkSampler samp =
            t ? (t->volume ? g_backend.VolumeSampler()
                           : GetSamplerForStage(i, depthCmp))
              : g_backend.DefaultSampler();
        g_backend.SetTextureSlot(i, t ? t->view : VK_NULL_HANDLE, samp,
                                 t ? t->cube : false, t ? t->volume : false);
    }
    g_st.rtStageMask = rtMask;
    g_st.shadowCompareMask = cmpMask;
    return 0;
}

static uint32_t __stdcall Dev_CreateStateBlock(void*, uint32_t type, void** out) {
    (void)type;
    FakeStateBlock* b = NewStateBlock();
    if (b) {
        b->saved = g_st;
        b->captured = 1;
    }
    if (out) *out = b;
    return b ? 0 : 0x8876086C;
}
static uint32_t __stdcall Dev_BeginStateBlock(void*) {
    g_recordingBlock = NewStateBlock();
    return 0;
}
static uint32_t __stdcall Dev_EndStateBlock(void*, void** out) {
    FakeStateBlock* b = g_recordingBlock;
    g_recordingBlock = nullptr;
    if (b) {
        b->saved = g_st;
        b->captured = 1;
    }
    if (out) *out = b;
    return b ? 0 : 0x8876086C;
}

// ---------------------------------------------------------------------------
// Palette, getters and cosmetic setters. Every getter must always write its
// out-parameter: the engine treats an untouched one as a live object.
// ---------------------------------------------------------------------------
static uint32_t __stdcall Dev_SetPaletteEntries(void*, uint32_t, const void*) {
    return 0;
}
static uint32_t __stdcall Dev_GetPaletteEntries(void*, uint32_t, void* entries) {
    if (entries) memset(entries, 0, 256 * 4);
    return 0;
}
static uint32_t __stdcall Dev_SetCurrentTexturePalette(void*, uint32_t) {
    return 0;
}
static uint32_t __stdcall Dev_GetCurrentTexturePalette(void*, uint32_t* out) {
    if (out) *out = 0;
    return 0;
}

static uint32_t __stdcall Dev_GetTextureStageState(void*, uint32_t, uint32_t type,
                                                   uint32_t* out) {
    if (out) *out = (type == 1) ? 1u : 0u;  // COLOROP: SELECTARG1
    return 0;
}
static uint32_t __stdcall Dev_GetSamplerState(void*, uint32_t, uint32_t,
                                              uint32_t* out) {
    if (out) *out = 0;
    return 0;
}
static uint32_t __stdcall Dev_GetTexture(void*, uint32_t stage,
                                         FakeResource** out) {
    if (out) *out = (stage < 16) ? g_st.textures[stage] : nullptr;
    return 0;
}
static uint32_t __stdcall Dev_GetViewport(void*, void* out) {
    if (out) {
        if (g_st.viewportValid) {
            memcpy(out, &g_st.viewport, sizeof(D3DViewport));
        } else {
            const VkExtent2D ext =
                g_backendReady ? g_backend.TargetExtent() : VkExtent2D{1024, 768};
            D3DViewport def{0, 0, ext.width, ext.height, 0.0f, 1.0f};
            memcpy(out, &def, sizeof(D3DViewport));
        }
    }
    return 0;
}
static uint32_t __stdcall Dev_GetFVF(void*, uint32_t* out) {
    if (out) *out = 0;
    return 0;
}

static uint32_t __stdcall Dev_SetGammaRamp(void*, uint32_t, uint32_t,
                                           const void*) {
    return 0;
}
static uint32_t __stdcall Dev_SetClipPlane(void*, uint32_t, const float*) {
    return 0;
}
static uint32_t __stdcall Dev_SetNPatchMode(void*, float) { return 0; }
static uint32_t __stdcall Dev_ShowCursor(void*, uint32_t) { return 0; }

// IDirect3DDevice9Ex::CreateVolumeTexture takes nine arguments: the trailing
// pSharedHandle must be popped or the caller's retn reads garbage.
static uint32_t __stdcall Dev_CreateVolumeTexture(void*, uint32_t w, uint32_t h,
                                                  uint32_t depth,
                                                  uint32_t levels,
                                                  uint32_t usage,
                                                  uint32_t format, uint32_t,
                                                  void** out, void** shared) {
    FakeVolume* v =
        g_resources.CreateVolumeTexture(w, h, depth, levels, usage, format);
    if (out) *out = v;
    if (shared) *shared = nullptr;
    return v ? 0 : 0x8876086C;
}

static void* g_devVt[128];

void BuildDeviceVtable() {
    void** vt = g_devVt;
    for (int i = 0; i < 128; ++i)
        vt[i] = StubForArity(i < kDevSlotCount ? kDevArgs[i] : 0);

    vt[kQueryInterface] = (void*)Dev_QueryInterface;
    vt[kAddRef] = (void*)Dev_AddRef;
    vt[kRelease] = (void*)Dev_Release;
    vt[kTestCooperativeLevel] = (void*)Dev_TestCooperativeLevel;
    vt[kEvictManagedResources] = (void*)Dev_EvictManagedResources;
    vt[kGetDirect3D] = (void*)Dev_GetDirect3D;
    vt[kGetDeviceCaps] = (void*)Dev_GetDeviceCaps;
    vt[kGetDisplayMode] = (void*)Dev_GetDisplayMode;
    vt[kGetCreationParameters] = (void*)Dev_GetCreationParameters;
    vt[kReset] = (void*)Dev_Reset;
    vt[kPresent] = (void*)Dev_Present;
    vt[kBeginScene] = (void*)Dev_BeginScene;
    vt[kEndScene] = (void*)Dev_EndScene;
    vt[kClear] = (void*)Dev_Clear;
    vt[kSetTransform] = (void*)Dev_SetTransform;
    vt[kMultiplyTransform] = (void*)Dev_MultiplyTransform;
    vt[kSetViewport] = (void*)Dev_SetViewport;
    vt[kSetRenderState] = (void*)Dev_SetRenderState;
    vt[kGetRenderState] = (void*)Dev_GetRenderState;
    vt[kSetScissorRect] = (void*)Dev_SetScissorRect;
    vt[kDrawPrimitive] = (void*)Dev_DrawPrimitive;
    vt[kDrawIndexedPrimitive] = (void*)Dev_DrawIndexedPrimitive;
    vt[kCreateVertexDeclaration] = (void*)Dev_CreateVertexDeclaration;
    vt[kSetVertexDeclaration] = (void*)Dev_SetVertexDeclaration;
    vt[kSetFVF] = (void*)Dev_SetFVF;
    vt[kCreateVertexShader] = (void*)Dev_CreateVertexShader;
    vt[kSetVertexShader] = (void*)Dev_SetVertexShader;
    vt[kSetVertexShaderConstantF] = (void*)Dev_SetVertexShaderConstantF;
    vt[kSetStreamSource] = (void*)Dev_SetStreamSource;
    vt[kSetStreamSourceFreq] = (void*)Dev_SetStreamSourceFreq;
    vt[kGetStreamSourceFreq] = (void*)Dev_GetStreamSourceFreq;
    vt[kSetIndices] = (void*)Dev_SetIndices;
    vt[kCreatePixelShader] = (void*)Dev_CreatePixelShader;
    vt[kSetPixelShader] = (void*)Dev_SetPixelShader;
    vt[kSetPixelShaderConstantF] = (void*)Dev_SetPixelShaderConstantF;
    vt[kGetBackBuffer] = (void*)Dev_GetBackBuffer;
    vt[kCreateStateBlock] = (void*)Dev_CreateStateBlock;
    vt[kBeginStateBlock] = (void*)Dev_BeginStateBlock;
    vt[kEndStateBlock] = (void*)Dev_EndStateBlock;
    vt[kSetPaletteEntries] = (void*)Dev_SetPaletteEntries;
    vt[kGetPaletteEntries] = (void*)Dev_GetPaletteEntries;
    vt[kSetCurrentTexturePalette] = (void*)Dev_SetCurrentTexturePalette;
    vt[kGetCurrentTexturePalette] = (void*)Dev_GetCurrentTexturePalette;
    vt[kStretchRect] = (void*)Dev_StretchRect;
    vt[kCreateCubeTexture] = (void*)Dev_CreateCubeTexture;
    vt[kSetRenderTarget] = (void*)Dev_SetRenderTarget;
    vt[kGetRenderTarget] = (void*)Dev_GetRenderTarget;
    vt[kGetDepthStencilSurface] = (void*)Dev_GetDepthStencilSurface;
    vt[kSetDepthStencilSurface] = (void*)Dev_SetDepthStencilSurface;
    vt[kCreateRenderTarget] = (void*)Dev_CreateRenderTarget;
    vt[kCreateDepthStencilSurface] = (void*)Dev_CreateDepthStencilSurface;
    vt[kSetSamplerState] = (void*)Dev_SetSamplerState;
    vt[kGetTextureStageState] = (void*)Dev_GetTextureStageState;
    vt[kGetSamplerState] = (void*)Dev_GetSamplerState;
    vt[kGetTexture] = (void*)Dev_GetTexture;
    vt[kGetViewport] = (void*)Dev_GetViewport;
    vt[kGetFVF] = (void*)Dev_GetFVF;
    vt[21] = (void*)Dev_SetGammaRamp;   // SetGammaRamp(iSwapChain, Flags, Ramp)
    vt[55] = (void*)Dev_SetClipPlane;
    vt[79] = (void*)Dev_SetNPatchMode;
    vt[12] = (void*)Dev_ShowCursor;
    vt[24] = (void*)Dev_CreateVolumeTexture;
    vt[kSetTextureStageState] = (void*)Dev_SetTextureStageState;
    vt[kCreateVertexBuffer] = (void*)Dev_CreateVertexBuffer;
    vt[kCreateIndexBuffer] = (void*)Dev_CreateIndexBuffer;
    vt[kCreateTexture] = (void*)Dev_CreateTexture;
    vt[kSetTexture] = (void*)Dev_SetTexture;

    g_sbVt[0] = (void*)S_QueryInterface;
    g_sbVt[1] = (void*)S_AddRef;
    g_sbVt[2] = (void*)S_Release;
    g_sbVt[3] = (void*)S_GetDevice;
    g_sbVt[4] = (void*)S_Capture;
    g_sbVt[5] = (void*)S_Apply;
    g_recordingBlock = nullptr;

    g_device.vt = vt;
    g_device.refs = 1;
    (void)g_stubHits;
}
