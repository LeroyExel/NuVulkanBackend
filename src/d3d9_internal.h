#pragma once

#include "constants.h"
#include "d3d9_vtable.h"
#include "pso_cache.h"
#include "resources.h"
#include "shaders.h"
#include "sm3_to_spirv.h"
#include "vk_context.h"

#include <algorithm>

// D3DPRESENT_PARAMETERS, as DWORD indices: 0 width, 1 height, 2 format,
// 3 backbuffer count, 4 msaa type, 5 msaa quality, 6 swap effect, 7 device
// window, 8 windowed, 9 auto depth, 10 depth format, 11 flags, 12 refresh rate,
// 13 presentation interval.
struct PresentParams {
    uint32_t w = 1024, h = 768;
    bool windowed = true;
    uint32_t refreshRate = 60;
    uint32_t presentationInterval = 0;
};

PresentParams ReadPresentParams(const void* pp);
void ApplyWindowMode(HWND hwnd, const PresentParams& p);

struct StreamState {
    FakeResource* vb = nullptr;
    uint32_t offset = 0;
    uint32_t stride = 0;
    // D3D9 instance frequency: 0x80000000 marks instanced data, 0x40000000
    // carries the instance count. The engine reads it back.
    uint32_t freq = 1;
};

struct D3DViewport {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    float minZ;
    float maxZ;
};

struct SamplerStageState {
    uint32_t addressU = 1;
    uint32_t addressV = 1;
    uint32_t addressW = 1;
    uint32_t borderColor = 0;
    uint32_t magFilter = 2;
    uint32_t minFilter = 2;
    uint32_t mipFilter = 2;
    float mipLodBias = 0.0f;
    uint32_t maxMipLevel = 0;
    uint32_t maxAnisotropy = 1;
    uint32_t srgbTexture = 0;
    VkSampler cachedSampler = VK_NULL_HANDLE;
    uint32_t lastKey = 0xFFFFFFFFu;
    bool samplerDirty = true;
};

inline bool IsShadowCompareD3DFormat(uint32_t fmt) {
    return fmt == 70 || fmt == 71 || fmt == 75 || fmt == 77 || fmt == 79 ||
           fmt == 80 || fmt == 82;
}

// Everything a draw depends on. Constants are stored as 257 registers: c0..c255
// plus the zeroed padding register c[256] used by relative addressing.
struct DeviceState {
    uint32_t rs[256]{};
    float vsConst[ConstantRing::kRegisters * 4]{};
    float psConst[ConstantRing::kRegisters * 4]{};
    StreamState streams[16]{};
    FakeResource* indices = nullptr;
    uint32_t indexFormat = 101;
    FakeResource* decl = nullptr;
    FakeResource* textures[16]{};
    SamplerStageState samplers[16]{};
    uint32_t shadowCompareMask = 0;
    FakeSurface* colorTargets[4]{};
    FakeSurface* depthTarget = nullptr;
    uint64_t vsHash = 0;
    uint64_t psHash = 0;
    ShaderConstInfo vsConstInfo{};
    ShaderConstInfo psConstInfo{};
    bool colorWriteSet = false;
    uint32_t topology = 4;
    RECT scissor{0, 0, 0, 0};
    bool scissorValid = false;
    D3DViewport viewport{0, 0, 0, 0, 0.0f, 1.0f};
    bool viewportValid = false;
    uint32_t clearColor = 0;
    float clearDepth = 1.0f;
    uint32_t clearStencil = 0;
    bool sceneOpen = false;
    bool vsConstDirty = true;
    bool psConstDirty = true;
    uint16_t texMask = 0;
    uint16_t rtStageMask = 0;
    uint16_t streamMask = 0;
    uint16_t instancedMask = 0;
    uint32_t instanceCount = 1;
    uint32_t vsHighReg = 0;
    uint32_t psHighReg = 0;
    bool targetsDirty = true;
};

extern DeviceState g_st;
extern HWND g_deviceWnd;
extern uint32_t g_currentRefreshRate;
extern uint32_t g_currentInterval;
extern bool g_backendReady;
extern bool g_frameActive;
extern uint32_t g_frameNumber;
extern uint32_t g_drawCalls;
extern std::unordered_map<uint32_t, VkSampler> g_samplerCache;

// Minimal COM identity: a vtable pointer and a reference count.
struct FakeObj {
    void** vt;
    uint32_t refs;
};

extern FakeObj g_device;
extern FakeObj g_d3d9;
extern void* g_shaderVt[5];

// IDirect3DVertexShader9 / IDirect3DPixelShader9 are real objects: the engine
// calls AddRef/Release on them, and a hash encoded in the pointer value faults
// on release.
struct FakeShaderObj {
    void** vtbl;
    uint32_t refs;
    uint32_t type;  // 0x5A4D4001 sentinel
    uint32_t isPixel;
    uint64_t hash;
};

FakeShaderObj* MakeShaderObj(uint64_t hash, bool pixel);
uint64_t ShaderHashOf(FakeResource* sh);

// Total DWORDs of an SM3 token stream, walked with the shared instruction
// length decoder.
uint32_t ShaderDwordSize(const uint32_t* code);

VkSampler GetSamplerForStage(uint32_t stage, bool depthCompare = false);
ImageRef MakeImageRef(FakeSurface* s);
void ApplyTargets();
void InitDefaultRenderStates();
void TranslateDraw(uint32_t topo, int32_t baseVertex, uint32_t startIndex,
                   uint32_t primCount, bool indexed);
void ResetDrawCache();
extern bool g_pipeDirty;
extern bool g_dynRsDirty;
extern bool g_stencilDirty;
extern bool g_vpDirty;
extern bool g_scDirty;

void InvalidateDescriptorBinding();
void InvalidateViewportScissor();
bool StartNewFrame();

// Filled by the device and factory capability queries.
void FillAdapterIdentifier(void* out);
void FillCaps(void* out);

void BuildDeviceVtable();
void BuildD3D9Vtable();
void* StubForArityExport(int n);

// Shared by the device and the swap-chain object.
uint32_t __stdcall Dev_Present(void*, const RECT*, const RECT*, const void*,
                               const void*);
uint32_t __stdcall Dev_GetDisplayMode(void*, uint32_t, void* out);

bool SameIID(const void* riid, const uint8_t* iid);
