#include "resources.h"

#include <cmath>
#define XXH_INLINE_ALL
#include "xxhash.h"

#include "sm3_to_spirv.h"
#include "vk_context.h"

ResourceTable g_resources;

// ---------------------------------------------------------------------------
// Fake COM vtable. Slot order follows IDirect3DResource9:
//   0 QI, 1 AddRef, 2 Release, 3 GetDevice, 4 SetPrivateData, 5 GetPrivateData,
//   6 FreePrivateData, 7 SetPriority, 8 GetPriority, 9 PreLoad, 10 GetType
// then type-specific entries.
// ---------------------------------------------------------------------------
static bool IsBlockCompressed(VkFormat f);
static uint32_t BytesPerPixel(VkFormat f);
static uint32_t LevelBytes(VkFormat f, uint32_t w, uint32_t h, uint32_t* pitchOut);

static uint32_t __stdcall Res_AddRef(FakeResource* r) { return ++r->refs; }

static uint32_t __stdcall Res_ReleaseThunk(FakeResource* r) {
    if (!r) return 0;
    if (r->refs) --r->refs;
    return r->refs;
}

static uint32_t __stdcall Res_QueryInterface(FakeResource* r, const void*, void** out) {
    if (out) *out = r;
    return 0;
}
static uint32_t __stdcall Res_SetPrivateData(FakeResource*, const void*, const void*, uint32_t) { return 0; }
static uint32_t __stdcall Res_GetPrivateData(FakeResource*, const void*, void*, uint32_t*) { return 0; }
static uint32_t __stdcall Res_FreePrivateData(FakeResource*, const void*) { return 0; }
static uint32_t __stdcall Res_SetPriority(FakeResource*, uint32_t) { return 0; }
static uint32_t __stdcall Res_GetPriority(FakeResource*) { return 0; }
static uint32_t __stdcall Res_PreLoad(FakeResource*) { return 0; }
static uint32_t __stdcall Res_GetType(FakeResource* r) { return r->type; }
static uint32_t __stdcall Res_GetDevice(FakeResource*, void** dev) {
    if (dev) *dev = nullptr;
    return 0;
}

static uint32_t __stdcall VB_GetDesc(FakeResource* r, void* desc) {
    if (desc) {
        uint32_t* d = (uint32_t*)desc;  // D3DVERTEXBUFFER_DESC
        memset(d, 0, 32);
        d[0] = 1;  // Format = D3DFMT_VERTEXDATA
        d[1] = 1;  // Type  = D3DRTYPE_VERTEXBUFFER
        d[2] = 0;  // Usage
        d[3] = 0;  // Pool
        d[4] = r->p1;
        d[5] = r->p0;  // FVF
    }
    return 0;
}
static uint32_t __stdcall IB_GetDesc(FakeResource* r, void* desc) {
    if (desc) {
        uint32_t* d = (uint32_t*)desc;
        memset(d, 0, 32);
        d[0] = 101;  // D3DFMT_INDEX16 / INDEX32
        d[1] = 2;    // D3DRTYPE_INDEXBUFFER
        d[4] = r->p1;
        d[5] = r->p0;
    }
    return 0;
}
static uint32_t __stdcall Buf_Lock(FakeResource* r, uint32_t offset, uint32_t size,
                                   void** out, uint32_t flags) {
    return g_resources.Lock(r, 0, offset, size, out, 0, flags) ? 0 : 0x8876086C;
}
static uint32_t __stdcall Buf_Unlock(FakeResource* r) {
    g_resources.Unlock(r, 0, 0);
    return 0;
}

static uint32_t __stdcall Tex_GetLevelCount(FakeResource* r) {
    GpuTexture* t = g_resources.TextureOf(r);
    return t ? t->levels : 1;
}
static uint32_t __stdcall Tex_GetLevelDesc(FakeResource* r, uint32_t level, void* desc) {
    GpuTexture* t = g_resources.TextureOf(r);
    if (desc && t && level < t->levels) {
        uint32_t* d = (uint32_t*)desc;
        memset(d, 0, 32);
        d[0] = r->p0;
        d[1] = 3;
        d[2] = t->renderTarget ? 1u : 0u;
        d[3] = 1;
        d[4] = 1;
        d[5] = 0;
        d[6] = t->levelW[level];
        d[7] = t->levelH[level];
    }
    return 0;
}
// IDirect3DTexture9::LockRect(This, Level, D3DLOCKED_RECT*, const RECT*, Flags).
// The out structure is parameter 3, not 4: pRect is usually NULL, and reading it
// as the out-pointer wrote through 0x00000000 (d3d9.dll+0x14893).
static uint32_t __stdcall Tex_LockRect(FakeResource* r, uint32_t level,
                                       void* lockedRect, const void* rect,
                                       uint32_t flags) {
    void* p = nullptr;
    if (!g_resources.Lock(r, level, 0, 0, &p, 0, flags)) return 0x8876086C;
    if (lockedRect) {
        uint32_t* lr = (uint32_t*)lockedRect;
        GpuTexture* t = g_resources.TextureOf(r);
        uint32_t pitch = 0;
        if (t) {
            uint32_t lw = level < t->levels ? t->levelW[level] : t->levelW[0];
            uint32_t lh = level < t->levels ? t->levelH[level] : t->levelH[0];
            if (t->expandLum) pitch = lw * t->srcBytes;
            else LevelBytes(t->format, lw, lh, &pitch);
            if (rect) {
                const RECT* rc = (const RECT*)rect;
                if (IsBlockCompressed(t->format)) {
                    uint32_t blockBytes = (t->format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) ? 8 : 16;
                    p = (uint8_t*)p + (rc->top / 4) * pitch + (rc->left / 4) * blockBytes;
                } else {
                    uint32_t bpp = t->expandLum ? t->srcBytes : BytesPerPixel(t->format);
                    p = (uint8_t*)p + rc->top * pitch + rc->left * bpp;
                }
            }
        }
        lr[0] = pitch;
        lr[1] = (uint32_t)(uintptr_t)p;
    }
    return 0;
}
static uint32_t __stdcall Tex_UnlockRect(FakeResource* r, uint32_t level) {
    g_resources.Unlock(r, level, 0);
    return 0;
}static uint32_t __stdcall Tex_AddDirtyRect(FakeResource*, const void*) { return 0; }
static uint32_t __stdcall Tex_GetSurfaceLevel(FakeResource* r, uint32_t level, void** s) {
    // Called once per bound surface per frame; logging every one of them wrote
    // hundreds of megabytes during a single play session.
    static int levelLogs = 0;
    if (levelLogs++ < 8)
        LOG("[res] GetSurfaceLevel tex=%u level=%u", r ? r->index : 0, level);
    FakeSurface* surf = g_resources.SurfaceLevel(r, level);
    if (s) *s = surf;
    return surf ? 0 : 0x8876086C;
}

// Cube faces are not individually lockable yet; report failure rather than
// handing back a surface for the wrong subresource.
static uint32_t __stdcall Cube_GetCubeMapSurface(FakeResource*, uint32_t, uint32_t,
                                                 void** s) {
    if (s) *s = nullptr;
    return 0x8876086C;
}
static uint32_t __stdcall Tex_SetLOD(FakeResource*, uint32_t) { return 0; }
static uint32_t __stdcall Tex_GetLOD(FakeResource*) { return 0; }
static uint32_t __stdcall Tex_SetAutoGenFilterType(FakeResource*, uint32_t) { return 0; }
static uint32_t __stdcall Tex_GetAutoGenFilterType(FakeResource*, uint32_t* out) {
    if (out) *out = 2;  // D3DTEXF_LINEAR
    return 0;
}
static uint32_t __stdcall Tex_GenerateMipSubLevels(FakeResource*) { return 0; }

// ---------------------------------------------------------------------------
// IDirect3DSurface9
// ---------------------------------------------------------------------------
static uint32_t __stdcall Surf_QueryInterface(FakeSurface* s, const void*, void** out) {
    if (out) *out = s;
    return 0;
}
static uint32_t __stdcall Surf_AddRef(FakeSurface* s) { return ++s->refs; }
static uint32_t __stdcall Surf_Release(FakeSurface* s) {
    if (!s) return 0;
    if (s->refs) --s->refs;
    return s->refs;
}
static uint32_t __stdcall Surf_GetDevice(FakeSurface*, void** dev) {
    if (dev) *dev = nullptr;
    return 0;
}
static uint32_t __stdcall Surf_SetPrivateData(FakeSurface*, const void*, const void*, uint32_t) { return 0; }
static uint32_t __stdcall Surf_GetPrivateData(FakeSurface*, const void*, void*, uint32_t*) { return 0; }
static uint32_t __stdcall Surf_FreePrivateData(FakeSurface*, const void*) { return 0; }
static uint32_t __stdcall Surf_SetPriority(FakeSurface*, uint32_t) { return 0; }
static uint32_t __stdcall Surf_GetPriority(FakeSurface*) { return 0; }
static uint32_t __stdcall Surf_PreLoad(FakeSurface*) { return 0; }
static uint32_t __stdcall Surf_GetType(FakeSurface*) { return 3; }  // D3DRTYPE_SURFACE

static uint32_t __stdcall Surf_GetDesc(FakeSurface* s, void* desc) {
    if (!desc || !s) return 0x8876086C;
    uint32_t* d = (uint32_t*)desc;  // D3DSURFACE_DESC: 8 DWORDs
    memset(d, 0, 32);
    d[0] = s->format;                       // Format
    d[1] = 3;                               // Type = SURFACE
    d[2] = s->isDepth ? 1u : 0u;            // Usage
    d[3] = 1;                               // Pool = DEFAULT
    d[4] = 1;                               // MultiSampleType
    d[5] = 0;                               // MultiSampleQuality
    d[6] = s->width;
    d[7] = s->height;
    return 0;
}

// IDirect3DSurface9::LockRect has no Level parameter and takes the
// D3DLOCKED_RECT* as its first argument after This.
static uint32_t __stdcall Surf_LockRect(FakeSurface* s, void* lockedRect,
                                        const void* rect, uint32_t flags) {
    (void)flags;
    static int surfLockLogs = 0;
    if (surfLockLogs++ < 4)
        LOG("[res] surf lock surf=%p tex=%p level=%u backbuffer=%u", (void*)s,
            (void*)(s ? s->tex : nullptr), s ? s->level : 0u,
            s ? s->isBackbuffer : 0u);
    if (!s) return 0x8876086C;
    void* p = nullptr;
    uint32_t pitch = 0;
    if (s->tex) {
        if (!g_resources.Lock(s->tex, s->level, 0, 0, &p)) return 0x8876086C;
        GpuTexture* t = g_resources.TextureOf(s->tex);
        if (t) {
            uint32_t lw = s->level < t->levels ? t->levelW[s->level] : t->levelW[0];
            uint32_t lh = s->level < t->levels ? t->levelH[s->level] : t->levelH[0];
            if (t->expandLum) pitch = lw * t->srcBytes;
            else LevelBytes(t->format, lw, lh, &pitch);
            if (rect) {
                const RECT* rc = (const RECT*)rect;
                if (IsBlockCompressed(t->format)) {
                    uint32_t blockBytes = (t->format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) ? 8 : 16;
                    p = (uint8_t*)p + (rc->top / 4) * pitch + (rc->left / 4) * blockBytes;
                } else {
                    uint32_t bpp = t->expandLum ? t->srcBytes : BytesPerPixel(t->format);
                    p = (uint8_t*)p + rc->top * pitch + rc->left * bpp;
                }
            }
        }
    } else {
        pitch = s->width * 4;
        uint32_t bytes = pitch * s->height;
        if (!s->cpu || s->cpuBytes < bytes) {
            free(s->cpu);
            s->cpu = malloc(bytes ? bytes : 4);
            s->cpuBytes = s->cpu ? (bytes ? bytes : 4) : 0;
        }
        p = s->cpu;
        if (!p) return 0x8876086C;
        if (rect) {
            const RECT* rc = (const RECT*)rect;
            p = (uint8_t*)p + rc->top * pitch + rc->left * 4;
        }
    }
    if (lockedRect) {
        uint32_t* lr = (uint32_t*)lockedRect;
        lr[0] = pitch;
        lr[1] = (uint32_t)(uintptr_t)p;
    }
    return 0;
}
static uint32_t __stdcall Surf_UnlockRect(FakeSurface* s) {
    if (s && s->tex) g_resources.Unlock(s->tex, s->level, 0);
    return 0;
}
// A render target surface has no CPU-visible DC; reporting failure is what the
// engine expects for D3DPOOL_DEFAULT memory.
static uint32_t __stdcall Surf_GetDC(FakeSurface*, void**) { return 0x8876086C; }
static uint32_t __stdcall Surf_ReleaseDC(FakeSurface*, void*) { return 0x8876086C; }
static uint32_t __stdcall Surf_GetContainer(FakeSurface* s, const void*, void** out) {
    if (out) *out = s ? s->tex : nullptr;
    return (s && s->tex) ? 0 : 0x8876086C;
}

// ---------------------------------------------------------------------------
// Volume texture / volume. Arity per slot follows d3d9.h; a wrong arity
// unbalances the caller's stack exactly like the generic stubs did.
// ---------------------------------------------------------------------------
static void** g_volumeVtbl = nullptr;
static uint32_t __stdcall Vol_SetLOD(FakeVolume*, uint32_t) { return 0; }
static uint32_t __stdcall Vol_GetLOD(FakeVolume*) { return 0; }
static uint32_t __stdcall Vol_GetLevelCount(FakeVolume* v) { return v ? v->levels : 1; }
static uint32_t __stdcall Vol_SetAutoGen(FakeVolume*, uint32_t) { return 0; }
static uint32_t __stdcall Vol_GetAutoGen(FakeVolume*, uint32_t* out) {
    if (out) *out = 2;  // D3DTEXF_LINEAR
    return 0;
}
static uint32_t __stdcall Vol_Generate(FakeVolume*) { return 0; }
// IDirect3DVolumeTexture9 takes the level first on all three of these:
// LockBox(Level, pLockedBox, pBox, Flags), UnlockBox(Level),
// AddDirtyBox(Level, pDirtyBox). Without it, Level 0 landed in the out-pointer
// argument, the box was never written, and the engine wrote through pBits=0.
static uint32_t __stdcall Vol_AddDirtyBox(FakeVolume*, uint32_t, const void*) {
    return 0;
}
static uint32_t __stdcall Vol_UnlockBox(FakeVolume* v, uint32_t level) {
    if (!v) return 0x8876086C;
    g_resources.UnlockVolume(v->owner ? v->owner : v, level);
    return 0;
}
static uint32_t __stdcall Vol_GetDesc(FakeVolume* v, void* desc) {
    if (!v || !desc) return 0x8876086C;
    uint32_t* d = (uint32_t*)desc;  // D3DVOLUME_DESC
    memset(d, 0, 28);
    d[0] = v->format;
    d[1] = 5;  // D3DRTYPE_VOLUME
    d[3] = 1;  // D3DPOOL_DEFAULT
    d[4] = v->width;
    d[5] = v->height;
    d[6] = v->depth;
    return 0;
}
static uint32_t __stdcall Vol_LockBox(FakeVolume* v, uint32_t level, void* box,
                                      const void* rect, uint32_t flags) {
    (void)rect; (void)flags;
    if (!v || !box) return 0x8876086C;
    uint32_t* b = (uint32_t*)box;
    if (!v->rowPitch) v->rowPitch = v->width * 4;
    if (!v->slicePitch) v->slicePitch = v->rowPitch * v->height;
    void* ptr = v->stagingPtr ? v->stagingPtr : (v->owner ? v->owner->stagingPtr : nullptr);
    b[0] = v->rowPitch;
    b[1] = v->slicePitch;
    b[2] = (uint32_t)(uintptr_t)ptr;
    static int boxLogs = 0;
    if (boxLogs++ < 6)
        LOG("[res] vol lock level=%u %ux%ux%u row=%u slice=%u pBits=%p", level,
            v->width, v->height, v->depth, b[0], b[1], (void*)(uintptr_t)b[2]);
    return 0;
}
static uint32_t __stdcall Vol_GetLevelDesc(FakeVolume* v, uint32_t level, void* desc) {
    if (!v || !desc) return 0x8876086C;
    if (level >= v->levels) return 0x8876086C;
    uint32_t* d = (uint32_t*)desc;
    memset(d, 0, 28);
    d[0] = v->format;
    d[1] = 6;
    d[3] = 1;
    d[4] = (v->width >> level) ? (v->width >> level) : 1;
    d[5] = (v->height >> level) ? (v->height >> level) : 1;
    d[6] = (v->depth >> level) ? (v->depth >> level) : 1;
    return 0;
}
static uint32_t __stdcall Vol_GetVolumeLevel(FakeVolume* v, uint32_t level, void** out) {
    if (!v || !out) return 0x8876086C;
    if (level >= v->levels) return 0x8876086C;
    FakeVolume* lv = new FakeVolume();
    lv->vtbl = g_volumeVtbl;
    lv->refs = 1;
    lv->type = 5;
    lv->index = v->index;
    lv->p0 = v->p0;
    lv->p1 = 0;
    lv->ptr = v->ptr;
    lv->width = (v->width >> level) ? (v->width >> level) : 1;
    lv->height = (v->height >> level) ? (v->height >> level) : 1;
    lv->depth = (v->depth >> level) ? (v->depth >> level) : 1;
    LOG("[res] volume level %u -> %p %ux%ux%u", level, (void*)lv, lv->width,
        lv->height, lv->depth);
    lv->levels = 1;
    lv->format = v->format;
    lv->rowPitch = v->rowPitch;
    lv->slicePitch = lv->rowPitch * lv->height;
    lv->stagingPtr = v->stagingPtr;
    lv->stagingSize = v->stagingSize;
    lv->owner = v;
    *out = lv;
    return 0;
}

static void* g_resVtbl[32];
static void* g_vbVtbl[32];
static void* g_ibVtbl[32];
static void* g_texVtbl[32];
static void* g_cubeVtbl[32];
static void* g_surfVtbl[24];
static void* g_volTexVtbl[32];
static void* g_volVtbl[24];

// IDirect3DCubeTexture9 has different arities than IDirect3DTexture9:
// LockRect(Level, FaceType, pLockedRect, pRect, Flags) = 5 arguments,
// UnlockRect(Level, FaceType) = 2, AddDirtyRect(Level, FaceType, pRect) = 3.
// Reusing the 2D thunks left 4 bytes on the stack per call.
static uint32_t __stdcall Cube_LockRect(FakeResource* r, uint32_t level,
                                        uint32_t face, void* lockedRect,
                                        const void* rect, uint32_t flags) {
    void* p = nullptr;
    if (!g_resources.Lock(r, level, 0, 0, &p, face, flags)) return 0x8876086C;
    if (lockedRect) {
        uint32_t* lr = (uint32_t*)lockedRect;
        GpuTexture* t = g_resources.TextureOf(r);
        uint32_t pitch = 0;
        if (t) {
            uint32_t lw = level < t->levels ? t->levelW[level] : t->levelW[0];
            uint32_t lh = level < t->levels ? t->levelH[level] : t->levelH[0];
            LevelBytes(t->format, lw, lh, &pitch);
            if (rect) {
                const RECT* rc = (const RECT*)rect;
                if (IsBlockCompressed(t->format)) {
                    uint32_t blockBytes = (t->format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) ? 8 : 16;
                    p = (uint8_t*)p + (rc->top / 4) * pitch + (rc->left / 4) * blockBytes;
                } else {
                    uint32_t bpp = BytesPerPixel(t->format);
                    p = (uint8_t*)p + rc->top * pitch + rc->left * bpp;
                }
            }
        }
        lr[0] = pitch;
        lr[1] = (uint32_t)(uintptr_t)p;
    }
    return 0;
}
static uint32_t __stdcall Cube_UnlockRect(FakeResource* r, uint32_t level,
                                          uint32_t face) {
    g_resources.Unlock(r, level, face);
    return 0;
}
static uint32_t __stdcall Cube_AddDirtyRect(FakeResource*, uint32_t, uint32_t,
                                            const void*) {
    return 0;
}

static void BuildVtables() {
    for (int i = 0; i < 32; ++i) {
        g_resVtbl[i] = (void*)Res_PreLoad;
        g_vbVtbl[i] = (void*)Res_PreLoad;
        g_ibVtbl[i] = (void*)Res_PreLoad;
        g_texVtbl[i] = (void*)Res_PreLoad;
        g_cubeVtbl[i] = (void*)Res_PreLoad;
    }
    for (int i = 0; i < 24; ++i) g_surfVtbl[i] = (void*)Surf_PreLoad;
    g_surfVtbl[0] = (void*)Surf_QueryInterface;
    g_surfVtbl[1] = (void*)Surf_AddRef;
    g_surfVtbl[2] = (void*)Surf_Release;
    g_surfVtbl[3] = (void*)Surf_GetDevice;
    g_surfVtbl[4] = (void*)Surf_SetPrivateData;
    g_surfVtbl[5] = (void*)Surf_GetPrivateData;
    g_surfVtbl[6] = (void*)Surf_FreePrivateData;
    g_surfVtbl[7] = (void*)Surf_SetPriority;
    g_surfVtbl[8] = (void*)Surf_GetPriority;
    g_surfVtbl[9] = (void*)Surf_PreLoad;
    g_surfVtbl[10] = (void*)Surf_GetType;
    // IDirect3DSurface9: 11 GetContainer, 12 GetDesc, 13 LockRect,
    // 14 UnlockRect, 15 GetDC, 16 ReleaseDC.
    g_surfVtbl[11] = (void*)Surf_GetContainer;
    g_surfVtbl[12] = (void*)Surf_GetDesc;
    g_surfVtbl[13] = (void*)Surf_LockRect;
    g_surfVtbl[14] = (void*)Surf_UnlockRect;
    g_surfVtbl[15] = (void*)Surf_GetDC;
    g_surfVtbl[16] = (void*)Surf_ReleaseDC;
    for (int k = 0; k < 4; ++k) {
        void** v = k == 0 ? g_resVtbl : k == 1 ? g_vbVtbl : k == 2 ? g_ibVtbl : g_texVtbl;
        v[0] = (void*)Res_QueryInterface;
        v[1] = (void*)Res_AddRef;
        v[2] = (void*)Res_ReleaseThunk;
        v[3] = (void*)Res_GetDevice;
        v[4] = (void*)Res_SetPrivateData;
        v[5] = (void*)Res_GetPrivateData;
        v[6] = (void*)Res_FreePrivateData;
        v[7] = (void*)Res_SetPriority;
        v[8] = (void*)Res_GetPriority;
        v[9] = (void*)Res_PreLoad;
        v[10] = (void*)Res_GetType;
    }
    // IDirect3DVertexBuffer9/IndexBuffer9: 11 Lock, 12 Unlock, 13 GetDesc.
    g_vbVtbl[11] = (void*)Buf_Lock;
    g_vbVtbl[12] = (void*)Buf_Unlock;
    g_vbVtbl[13] = (void*)VB_GetDesc;
    g_ibVtbl[11] = (void*)Buf_Lock;
    g_ibVtbl[12] = (void*)Buf_Unlock;
    g_ibVtbl[13] = (void*)IB_GetDesc;
    // IDirect3DBaseTexture9: 11 SetLOD, 12 GetLOD, 13 GetLevelCount,
    // 14 SetAutoGenFilterType, 15 GetAutoGenFilterType, 16 GenerateMipSubLevels.
    // IDirect3DTexture9: 17 GetLevelDesc, 18 GetSurfaceLevel, 19 LockRect,
    // 20 UnlockRect, 21 AddDirtyRect. The old table was shifted: the engine's
    // GetSurfaceLevel call landed on GetLOD and returned no surface.
    // Every slot must be non-null: the engine dispatches through this table and
    // a zero entry jumps to address 0 (observed fault: eip=00000000).
    for (int i = 0; i < 32; ++i) g_texVtbl[i] = (void*)Res_PreLoad;
    g_texVtbl[11] = (void*)Tex_SetLOD;
    g_texVtbl[12] = (void*)Tex_GetLOD;
    g_texVtbl[13] = (void*)Tex_GetLevelCount;
    g_texVtbl[14] = (void*)Tex_SetAutoGenFilterType;
    g_texVtbl[15] = (void*)Tex_GetAutoGenFilterType;
    g_texVtbl[16] = (void*)Tex_GenerateMipSubLevels;
    g_texVtbl[17] = (void*)Tex_GetLevelDesc;
    g_texVtbl[18] = (void*)Tex_GetSurfaceLevel;
    g_texVtbl[19] = (void*)Tex_LockRect;
    g_texVtbl[20] = (void*)Tex_UnlockRect;
    g_texVtbl[21] = (void*)Tex_AddDirtyRect;

    // IDirect3DCubeTexture9: GetLevelCount, GetLevelDesc, GetCubeMapSurface,
    // LockRect, UnlockRect, AddDirtyRect.
    for (int i = 0; i < 32; ++i) g_cubeVtbl[i] = (void*)Res_PreLoad;
    // Slots 0..10 are the IDirect3DResource9 block. Filling them with PreLoad
    // left GetDevice/GetType writing no out-parameter at all, so the engine read
    // stack garbage straight after CreateCubeTexture.
    for (int i = 0; i < 11; ++i) g_cubeVtbl[i] = g_resVtbl[i];
    g_cubeVtbl[11] = (void*)Tex_SetLOD;
    g_cubeVtbl[12] = (void*)Tex_GetLOD;
    g_cubeVtbl[13] = (void*)Tex_GetLevelCount;
    g_cubeVtbl[14] = (void*)Tex_SetAutoGenFilterType;
    g_cubeVtbl[15] = (void*)Tex_GetAutoGenFilterType;
    g_cubeVtbl[16] = (void*)Tex_GenerateMipSubLevels;
    g_cubeVtbl[17] = (void*)Tex_GetLevelDesc;
    g_cubeVtbl[18] = (void*)Cube_GetCubeMapSurface;
    g_cubeVtbl[19] = (void*)Cube_LockRect;
    g_cubeVtbl[20] = (void*)Cube_UnlockRect;
    g_cubeVtbl[21] = (void*)Cube_AddDirtyRect;

    // IDirect3DVolumeTexture9 and IDirect3DVolume9. Slots 0..10 are the
    // IDirect3DResource9 block and reuse the resource thunks (FakeVolume's
    // first four members match FakeResource's layout).
    for (int i = 0; i < 32; ++i) g_volTexVtbl[i] = (void*)Res_PreLoad;
    for (int i = 0; i < 24; ++i) g_volVtbl[i] = (void*)Res_PreLoad;
    for (int i = 0; i < 11; ++i) {
        g_volTexVtbl[i] = g_resVtbl[i];
        g_volVtbl[i] = g_resVtbl[i];
    }
    g_volTexVtbl[11] = (void*)Vol_SetLOD;
    g_volTexVtbl[12] = (void*)Vol_GetLOD;
    g_volTexVtbl[13] = (void*)Vol_GetLevelCount;
    g_volTexVtbl[14] = (void*)Vol_SetAutoGen;
    g_volTexVtbl[15] = (void*)Vol_GetAutoGen;
    g_volTexVtbl[16] = (void*)Vol_Generate;
    g_volTexVtbl[17] = (void*)Vol_GetLevelDesc;
    g_volTexVtbl[18] = (void*)Vol_GetVolumeLevel;
    g_volTexVtbl[19] = (void*)Vol_LockBox;
    g_volTexVtbl[20] = (void*)Vol_UnlockBox;
    g_volTexVtbl[21] = (void*)Vol_AddDirtyBox;
    g_volVtbl[8] = (void*)Vol_GetDesc;
    g_volVtbl[9] = (void*)Vol_LockBox;
    g_volVtbl[10] = (void*)Vol_UnlockBox;
    g_volumeVtbl = g_volVtbl;
    LOG("[shim] vtbl res=%p vb=%p ib=%p tex=%p cube=%p surf=%p volTex=%p vol=%p",
        (void*)g_resVtbl, (void*)g_vbVtbl, (void*)g_ibVtbl, (void*)g_texVtbl,
        (void*)g_cubeVtbl, (void*)g_surfVtbl, (void*)g_volTexVtbl,
        (void*)g_volVtbl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dynamic vertex/index renaming ring.
//
// D3D9 lets an application lock a buffer with D3DLOCK_DISCARD and rewrite it
// while the GPU still reads the old contents, and it commonly does so several
// times inside one frame. Vulkan records draws and only executes them at submit,
// so handing back the same mapped memory for a second lock silently rewrites the
// vertices of every draw already recorded from that buffer in the same command
// buffer. The visible result is a draw fetching another draw's data: screen-space
// overlay vertices read as a giant skinned character whose clip coordinates run
// into the thousands. Renaming each DISCARD lock into its own region, as DXVK
// does, keeps every recorded draw's data intact until its frame completes.
// ---------------------------------------------------------------------------
struct CsGuard {
    CRITICAL_SECTION* cs;
    explicit CsGuard(CRITICAL_SECTION* c) : cs(c) { EnterCriticalSection(cs); }
    ~CsGuard() { LeaveCriticalSection(cs); }
};

static VkBuffer g_dynBuf = VK_NULL_HANDLE;
static VmaAllocation g_dynAlloc = VK_NULL_HANDLE;
static uint8_t* g_dynMapped = nullptr;
static uint32_t g_dynCapacity = 0;
static uint32_t g_dynHead = 0;
static uint32_t g_dynSlice = 0;
static uint32_t g_dynResetFrame = 0xFFFFFFFFu;
static constexpr uint32_t kDynSlices = 4u;
static CRITICAL_SECTION g_dynLock;
static bool g_dynLockInit = false;
static constexpr uint32_t kDynNoSpace = 0xFFFFFFFFu;

static bool DynRingCreate() {
    if (g_dynMapped) return true;
    const uint32_t caps[2] = {256u * 1024u * 1024u, 128u * 1024u * 1024u};
    for (uint32_t cap : caps) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = cap;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci = g_backend.HostAlloc();
        aci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(g_backend.Allocator(), &bci, &aci, &g_dynBuf,
                            &g_dynAlloc, &info) == VK_SUCCESS) {
            g_dynMapped = (uint8_t*)info.pMappedData;
            g_dynCapacity = cap;
            LOG("[res] dynamic vertex ring %u MB", cap >> 20);
            return true;
        }
    }
    LOG("[res] dynamic ring allocation failed");
    return false;
}

void ResResetDynamicRing(uint32_t frameSlot) {
    (void)frameSlot;
    if (!g_dynCapacity) return;
    const uint32_t curFrame = g_backend.FrameNumber();
    if (g_dynLockInit) EnterCriticalSection(&g_dynLock);
    if (g_dynResetFrame != curFrame) {
        g_dynResetFrame = curFrame;
        g_dynSlice = curFrame & (kDynSlices - 1u);
        g_dynHead = g_dynSlice * (g_dynCapacity / kDynSlices);
    }
    if (g_dynLockInit) LeaveCriticalSection(&g_dynLock);
}

static uint32_t DynAllocate(uint32_t bytes) {
    if (g_dynLockInit) EnterCriticalSection(&g_dynLock);
    if (!g_dynMapped) DynRingCreate();
    uint32_t res = kDynNoSpace;
    if (g_dynMapped && bytes) {
        const uint32_t curFrame = g_backend.FrameNumber();
        const uint32_t sliceBytes = g_dynCapacity / kDynSlices;
        if (g_dynResetFrame != curFrame) {
            g_dynResetFrame = curFrame;
            g_dynSlice = curFrame & (kDynSlices - 1u);
            g_dynHead = g_dynSlice * sliceBytes;
        }
        const uint32_t halfStart = g_dynSlice * sliceBytes;
        const uint32_t halfEnd = halfStart + sliceBytes;
        uint32_t at = (g_dynHead + 255u) & ~255u;
        if (at >= g_dynHead && at + bytes <= halfEnd) {
            g_dynHead = at + bytes;
            res = at;
        }
    }
    if (g_dynLockInit) LeaveCriticalSection(&g_dynLock);
    return res;
}

bool ResourceTable::Init() {
    if (!locksInit_) {
        InitializeCriticalSection(&resLock_);
        InitializeCriticalSection(&uploadLock_);
        locksInit_ = true;
    }
    if (!g_dynLockInit) {
        InitializeCriticalSection(&g_dynLock);
        g_dynLockInit = true;
    }
    BuildVtables();
    return true;
}

void ResourceTable::Shutdown() {
    if (!locksInit_) return;
    {
        CsGuard ul(&uploadLock_);
        if (curSlab_) {
            vmaDestroyBuffer(g_backend.Allocator(), curSlab_->buf, curSlab_->alloc);
            delete curSlab_;
            curSlab_ = nullptr;
        }
        for (auto* s : fullSlabs_) {
            vmaDestroyBuffer(g_backend.Allocator(), s->buf, s->alloc);
            delete s;
        }
        fullSlabs_.clear();
        for (auto* s : freeSlabs_) {
            vmaDestroyBuffer(g_backend.Allocator(), s->buf, s->alloc);
            delete s;
        }
        freeSlabs_.clear();
        for (int i = 0; i < 2; ++i) {
            for (auto* s : inFlightSlabs_[i]) {
                vmaDestroyBuffer(g_backend.Allocator(), s->buf, s->alloc);
                delete s;
            }
            inFlightSlabs_[i].clear();
            for (auto& rb : retiredBuffers_[i]) {
                if (rb.buf && rb.alloc) {
                    vmaDestroyBuffer(g_backend.Allocator(), rb.buf, rb.alloc);
                }
            }
            retiredBuffers_[i].clear();
        }
        pendingBufs_.clear();
        pendingImgs_.clear();
        pendingUploadCount_.store(0, std::memory_order_release);
    }
    {
        CsGuard rl(&resLock_);
        for (auto& b : buffers_) {
            free(b.cpuShadow);
            b.cpuShadow = nullptr;
        }
    }
    if (g_dynBuf && g_dynAlloc) {
        vmaDestroyBuffer(g_backend.Allocator(), g_dynBuf, g_dynAlloc);
        g_dynBuf = VK_NULL_HANDLE;
        g_dynAlloc = VK_NULL_HANDLE;
        g_dynMapped = nullptr;
        g_dynCapacity = 0;
        g_dynHead = 0;
        g_dynSlice = 0;
        g_dynResetFrame = 0xFFFFFFFFu;
    }
}

void ResourceTable::OnFrameBegin(uint32_t frameSlot) {
    const uint32_t slot = frameSlot & 1u;
    std::vector<RetiredBuffer> toDestroy;
    {
        CsGuard ul(&uploadLock_);
        activeSlot_ = slot;
        for (auto* s : inFlightSlabs_[slot]) {
            s->used = 0;
            if (freeSlabs_.size() < 16) {
                freeSlabs_.push_back(s);
            } else {
                toDestroy.push_back({s->buf, s->alloc});
                delete s;
            }
        }
        inFlightSlabs_[slot].clear();
        toDestroy.insert(toDestroy.end(), retiredBuffers_[slot].begin(),
                         retiredBuffers_[slot].end());
        retiredBuffers_[slot].clear();
    }
    for (const auto& rb : toDestroy) {
        if (rb.buf && rb.alloc) {
            vmaDestroyBuffer(g_backend.Allocator(), rb.buf, rb.alloc);
        }
    }
}

bool ResourceTable::AllocStaging(uint32_t bytes, VkBuffer& outBuf,
                                 VkDeviceSize& outOff, uint8_t*& outPtr) {
    const uint32_t aligned = AlignUp(bytes ? bytes : 16u, 16u);
    if (!curSlab_ || curSlab_->used + aligned > curSlab_->cap) {
        if (curSlab_) {
            fullSlabs_.push_back(curSlab_);
            curSlab_ = nullptr;
        }
        for (size_t i = 0; i < freeSlabs_.size(); ++i) {
            if (freeSlabs_[i]->cap >= aligned) {
                curSlab_ = freeSlabs_[i];
                freeSlabs_[i] = freeSlabs_.back();
                freeSlabs_.pop_back();
                curSlab_->used = 0;
                break;
            }
        }
        if (!curSlab_) {
            const uint32_t defaultCap = 4u * 1024u * 1024u;
            uint32_t cap = aligned > defaultCap ? AlignUp(aligned, 4096u) : defaultCap;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = cap;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo aci = g_backend.HostAlloc();
            VmaAllocationInfo info{};
            StagingSlab* slab = new StagingSlab();
            if (vmaCreateBuffer(g_backend.Allocator(), &bci, &aci, &slab->buf,
                                &slab->alloc, &info) != VK_SUCCESS) {
                delete slab;
                return false;
            }
            slab->mapped = (uint8_t*)info.pMappedData;
            slab->cap = cap;
            slab->used = 0;
            curSlab_ = slab;
        }
    }
    outBuf = curSlab_->buf;
    outOff = curSlab_->used;
    outPtr = curSlab_->mapped + curSlab_->used;
    curSlab_->used += aligned;
    return true;
}

bool ResourceTable::EnsureTextureStaging(GpuTexture* t) {
    if (!t) return false;
    if (!t->stagingBuf || !t->staging) {
        uint32_t bytes = t->stagingBytes;
        if (!bytes) {
            for (uint32_t i = 0; i < t->levels; ++i) {
                bytes += t->expandLum ? (t->levelW[i] * t->levelH[i] * t->srcBytes)
                                      : LevelBytes(t->format, t->levelW[i], t->levelH[i], nullptr);
            }
            if (t->cube) bytes *= 6;
            bytes = bytes + bytes / 2 + (t->cube ? 65536u : 4096u);
            t->stagingBytes = bytes;
        }
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo saci = g_backend.HostAlloc();
        VmaAllocationInfo sinfo{};
        if (vmaCreateBuffer(g_backend.Allocator(), &bci, &saci, &t->stagingBuf,
                            &t->stagingAlloc, &sinfo) != VK_SUCCESS) {
            return false;
        }
        t->staging = sinfo.pMappedData;
        if (t->staging) memset(t->staging, 0, bytes);
    }
    if (t->expandLum && (!t->expandBuf || !t->expandMapped)) {
        uint64_t need = t->expandBytes;
        if (!need) {
            for (uint32_t i = 0; i < t->levels; ++i)
                need += (uint64_t)t->levelW[i] * t->levelH[i] * 4;
            t->expandBytes = (uint32_t)need;
        }
        VkBufferCreateInfo eci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        eci.size = need;
        eci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo eaci = g_backend.HostAlloc();
        VmaAllocationInfo einfo{};
        if (vmaCreateBuffer(g_backend.Allocator(), &eci, &eaci, &t->expandBuf,
                            &t->expandAlloc, &einfo) != VK_SUCCESS) {
            return false;
        }
        t->expandMapped = einfo.pMappedData;
        if (t->expandMapped) memset(t->expandMapped, 0, (size_t)need);
    }
    return t->staging != nullptr;
}

VkFormat ResourceTable::ToVkFormat(uint32_t f) {
    switch (f) {
        case 21:  // A8R8G8B8
        case 22:  // X8R8G8B8
            return VK_FORMAT_B8G8R8A8_UNORM;
        case 23:  // R5G6B5
            return VK_FORMAT_B5G6R5_UNORM_PACK16;
        case 25:  // A4R4G4B4
            return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
        case 28:  // A8
            return VK_FORMAT_R8_UNORM;
        case 50:  // L8
            return VK_FORMAT_R8_UNORM;
        case 51:  // A8L8
            return VK_FORMAT_R8G8_UNORM;
        case 0x31545844:  // DXT1
            return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case 0x33545844:  // DXT3
            return VK_FORMAT_BC2_UNORM_BLOCK;
        case 0x35545844:  // DXT5
            return VK_FORMAT_BC3_UNORM_BLOCK;
        case 111:  // R16F
            return VK_FORMAT_R16_SFLOAT;
        case 112:  // G16R16F
            return VK_FORMAT_R16G16_SFLOAT;
        case 113:  // A16B16G16R16F
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case 114:  // R32F
            return VK_FORMAT_R32_SFLOAT;
        case 116:  // A32B32G32R32F
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case 32:  // A8B8G8R8 (byte order already RGBA)
            return VK_FORMAT_R8G8B8A8_UNORM;
        case 36:  // A16B16G16R16
            return VK_FORMAT_R16G16B16A16_UNORM;
        default:
            return VK_FORMAT_B8G8R8A8_UNORM;
    }
}

static void ApplyFormatSwizzle(uint32_t d3dFormat, VkComponentMapping& c) {
    if (d3dFormat == 22) {
        c.a = VK_COMPONENT_SWIZZLE_ONE;
    } else if (d3dFormat == 28) {
        c = {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
             VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_R};
    } else if (d3dFormat == 50) {
        c = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
             VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
    } else if (d3dFormat == 51) {
        c = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
             VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G};
    }
}

static bool IsBlockCompressed(VkFormat f) {
    return f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK ||
           f == VK_FORMAT_BC2_UNORM_BLOCK ||
           f == VK_FORMAT_BC3_UNORM_BLOCK;
}

static uint32_t BytesPerPixel(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM:
            return 1;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_B5G6R5_UNORM_PACK16:
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
            return 2;
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SFLOAT:
            return 4;
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16;
        default:
            return 4;
    }
}

// Bytes in one mip level. Block-compressed levels are 8/16 bytes per 4x4
// block, so a linear width*bpp formula would overrun the staging buffer.
static uint32_t LevelBytes(VkFormat f, uint32_t w, uint32_t h,
                           uint32_t* pitchOut) {
    if (IsBlockCompressed(f)) {
        uint32_t bx = (w + 3) / 4, by = (h + 3) / 4;
        uint32_t blockBytes = (f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK) ? 8 : 16;
        if (pitchOut) *pitchOut = bx * blockBytes;
        return bx * by * blockBytes;
    }
    uint32_t pitch = w * BytesPerPixel(f);
    if (pitchOut) *pitchOut = pitch;
    return pitch * h;
}

FakeResource* ResourceTable::CreateVertexBuffer(uint32_t length, uint32_t usage,
                                                uint32_t fvf) {
    GpuBuffer b{};
    b.size = length;
    b.usage = usage;
    b.isIndex = false;
    const bool isDynamic = (usage & 0x200u) != 0u;
    b.deviceLocal = !isDynamic;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = length ? length : 16;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    VmaAllocationInfo info{};
    if (isDynamic) {
        aci = g_backend.HostAlloc();
    } else {
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    if (vmaCreateBuffer(g_backend.Allocator(), &bci, &aci, &b.buffer, &b.alloc,
                        &info) != VK_SUCCESS) {
        LOG("[res] VB allocation failed (%u bytes)", length);
        return nullptr;
    }
    if (isDynamic) b.mapped = info.pMappedData;

    CsGuard rl(&resLock_);
    buffers_.push_back(b);
    FakeResource* r = new FakeResource();
    r->vtbl = g_vbVtbl;
    r->refs = 1;
    r->type = kResVB;
    r->index = (uint32_t)buffers_.size() - 1;
    r->p0 = fvf;
    r->p1 = length;
    r->ptr = &buffers_.back();
    live_.push_back(r);
    return r;
}

FakeResource* ResourceTable::CreateIndexBuffer(uint32_t length, uint32_t usage,
                                               uint32_t format) {
    GpuBuffer b{};
    b.size = length;
    b.usage = usage;
    b.isIndex = true;
    const bool isDynamic = (usage & 0x200u) != 0u;
    b.deviceLocal = !isDynamic;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = length ? length : 16;
    bci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    VmaAllocationInfo info{};
    if (isDynamic) {
        aci = g_backend.HostAlloc();
    } else {
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    if (vmaCreateBuffer(g_backend.Allocator(), &bci, &aci, &b.buffer, &b.alloc,
                        &info) != VK_SUCCESS) {
        LOG("[res] IB allocation failed (%u bytes)", length);
        return nullptr;
    }
    if (isDynamic) b.mapped = info.pMappedData;

    CsGuard rl(&resLock_);
    buffers_.push_back(b);
    FakeResource* r = new FakeResource();
    r->vtbl = g_ibVtbl;
    r->refs = 1;
    r->type = kResIB;
    r->index = (uint32_t)buffers_.size() - 1;
    r->p0 = format;
    r->p1 = length;
    r->ptr = &buffers_.back();
    live_.push_back(r);
    return r;
}

FakeResource* ResourceTable::CreateTexture(uint32_t w, uint32_t h,
                                           uint32_t levels, uint32_t usage,
                                           uint32_t format) {
    GpuTexture t{};
    t.width = w;
    t.height = h;
    // Depth-stencil textures arrive through CreateTexture (INTZ/DF24 and the
    // numeric depth formats). They need a depth format and a depth-stencil
    // attachment usage, otherwise binding one as a depth target leaves a colour
    // format in the pipeline's depthAttachmentFormat and faults the driver.
    const bool depthTex = IsDepthFormatD3D(format);
    t.depth = depthTex;
    t.format = depthTex ? ToVkDepthFormat(format, nullptr) : ToVkFormat(format);
    t.d3dFormat = format;
    // D3DFMT_L8 = 50, A8 = 28, A8L8 = 51, A4L4 = 52. Sampling these as R8 gives
    // (L,0,0,1) instead of D3D9's (L,L,L,1), so a luminance video frame loses
    // two thirds of its colour and an alpha font atlas becomes opaque. Expand
    // them to RGBA8 on upload instead.
    if (!depthTex && format == 52) {
        t.expandLum = true;
        t.srcBytes = 1u;
        t.format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    t.renderTarget = (usage & 1) != 0;  // D3DUSAGE_RENDERTARGET
    // aspectMask 0 is not a legal VkImageAspectFlags value; the first texture
    // the engine creates (1024x768 A16B16G16R16F) died building its view.
    t.aspect = depthTex ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    // D3D9 passes Levels explicitly; 0 means the full chain.
    uint32_t lv = levels;
    if (!lv) {
        uint32_t m = w > h ? w : h;
        lv = 1;
        while (m > 1) {
            m >>= 1;
            ++lv;
        }
    }
    if (lv > 16) lv = 16;
    t.levels = lv;

    uint32_t total = 0;
    for (uint32_t i = 0; i < lv; ++i) {
        uint32_t lw = w >> i;
        uint32_t lh = h >> i;
        if (!lw) lw = 1;
        if (!lh) lh = 1;
        t.levelW[i] = lw;
        t.levelH[i] = lh;
        t.levelOffset[i] = total;
        // The staging buffer keeps the engine's own layout, which for an
        // expanded luminance texture is one or two bytes per texel.
        total += t.expandLum ? (lw * lh * t.srcBytes)
                             : LevelBytes(t.format, lw, lh, nullptr);
    }
    {
        uint32_t p = 0;
        if (t.expandLum)
            p = w * t.srcBytes;
        else
            LevelBytes(t.format, w, h, &p);
        t.rowPitch = p;
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = t.format;
    ici.extent = {w, h, 1};
    ici.mipLevels = t.levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (t.renderTarget)
        ici.usage |= t.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                             : (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (t.depth) ici.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &t.image, &t.alloc,
                       nullptr) != VK_SUCCESS) {
        LOG("[res] texture allocation failed %ux%u", w, h);
        return nullptr;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = t.format;
    vci.subresourceRange = {t.aspect, 0, t.levels, 0, 1};
    ApplyFormatSwizzle(format, vci.components);
    vkCreateImageView(g_backend.Device(), &vci, nullptr, &t.view);
    t.keepStaging = (usage & 0x200u) != 0;
    t.stagingBytes = total + total / 2 + 4096;
    if (t.expandLum) {
        uint64_t need = 0;
        for (uint32_t i = 0; i < t.levels; ++i)
            need += (uint64_t)t.levelW[i] * t.levelH[i] * 4;
        t.expandBytes = (uint32_t)need;
    }
    if (!t.renderTarget && !t.depth) {
        EnsureTextureStaging(&t);
    }

    CsGuard rl(&resLock_);
    textures_.push_back(t);

    FakeResource* r = new FakeResource();
    r->vtbl = g_texVtbl;
    r->refs = 1;
    r->type = kResTex;
    r->index = (uint32_t)textures_.size() - 1;
    r->p0 = format;
    r->p1 = 0;
    r->ptr = &textures_.back();
    live_.push_back(r);
    return r;
}

FakeResource* ResourceTable::CreateVertexDeclaration(const uint32_t* tokens) {
    VertexDecl d{};
    const D3DVertexElement* e = (const D3DVertexElement*)tokens;
    uint64_t key = 1469598103934665603ull;
    // D3DDECL_END() is {0xFF, 0, D3DDECLTYPE_UNUSED, 0, 0, 0}: stream 0xFF, not
    // 0xFFFF. Stopping on the wrong sentinel walked off the array into stack
    // garbage and produced attributes at bogus, often duplicated locations.
    for (int i = 0; i < 64 && e[i].stream != 0xFF; ++i) {
        if (e[i].type == 17) continue;
        VertexAttr a{};
        a.stream = e[i].stream;
        a.offset = e[i].offset;
        a.location = Sm3LocationFor(e[i].usage, e[i].usageIndex);
        switch (e[i].type) {
            case 0: a.format = VK_FORMAT_R32_SFLOAT; break;
            case 1: a.format = VK_FORMAT_R32G32_SFLOAT; break;
            case 2: a.format = VK_FORMAT_R32G32B32_SFLOAT; break;
            case 3: a.format = VK_FORMAT_R32G32B32A32_SFLOAT; break;
            case 4: a.format = VK_FORMAT_B8G8R8A8_UNORM; break;
            case 5: a.format = VK_FORMAT_R8G8B8A8_USCALED; break;
            case 6: a.format = VK_FORMAT_R16G16_SSCALED; break;
            case 7: a.format = VK_FORMAT_R16G16B16A16_SSCALED; break;
            case 8: a.format = VK_FORMAT_R8G8B8A8_UNORM; break;
            case 9: a.format = VK_FORMAT_R16G16_SNORM; break;
            case 10: a.format = VK_FORMAT_R16G16B16A16_SNORM; break;
            case 11: a.format = VK_FORMAT_R16G16_UNORM; break;
            case 12: a.format = VK_FORMAT_R16G16B16A16_UNORM; break;
            case 13: a.format = VK_FORMAT_A2B10G10R10_USCALED_PACK32; break;
            case 14: a.format = VK_FORMAT_A2B10G10R10_SNORM_PACK32; break;
            case 15: a.format = VK_FORMAT_R16G16_SFLOAT; break;
            case 16: a.format = VK_FORMAT_R16G16B16A16_SFLOAT; break;
            default: a.format = VK_FORMAT_R32G32B32_SFLOAT; break;
        }
        // Decoding depends on D3DDECLTYPE alone. The hardware vertex fetcher
        // never changes conversion because of the semantic: BLENDINDICES stored
        // as UBYTE4N or D3DCOLOR still arrives in [0,1] and the shader scales it
        // itself before mova. Overriding those two types to USCALED here made
        // the shader scale a second time (indexes to 65k) and swapped R/B for
        // D3DCOLOR, so the override is deliberately absent.
        key = (key ^ ((uint64_t)a.location << 8) ^ ((uint64_t)e[i].type << 16) ^
               ((uint64_t)e[i].usageIndex << 24) ^ ((uint64_t)a.offset << 32) ^
               ((uint64_t)a.stream << 40))
              * 1099511628211ull;
        if (a.stream < 16) d.streamMask |= (uint16_t)(1u << a.stream);
        d.attrs.push_back(a);
    }
    d.key = key;
    CsGuard rl(&resLock_);
    decls_.push_back(d);

    FakeResource* r = new FakeResource();
    r->vtbl = g_resVtbl;
    r->refs = 1;
    r->type = kResDecl;
    r->index = (uint32_t)decls_.size() - 1;
    r->p0 = 0;
    r->p1 = 0;
    r->ptr = &decls_.back();
    live_.push_back(r);
    return r;
}

GpuBuffer* ResourceTable::Buffer(const FakeResource* r) {
    if (!r || (r->type != kResVB && r->type != kResIB)) return nullptr;
    return (GpuBuffer*)r->ptr;
}

GpuTexture* ResourceTable::TextureOf(const FakeResource* r) {
    if (!r || (r->type != kResTex && r->type != kResCube && r->type != kResVolume))
        return nullptr;
    return (GpuTexture*)r->ptr;
}

VertexDecl* ResourceTable::Decl(const FakeResource* r) {
    if (!r || r->type != kResDecl) return nullptr;
    return (VertexDecl*)r->ptr;
}

void ResourceTable::AddRef(FakeResource* r) {
    if (r) ++r->refs;
}

void ResourceTable::Release(FakeResource* r) { (void)r; }

uint32_t ResourceTable::BufferSize(const FakeResource* r) const {
    if (!r || r->index >= buffers_.size()) return 0;
    return buffers_[r->index].size;
}

bool ResourceTable::Lock(FakeResource* r, uint32_t level, uint32_t offset,
                         uint32_t size, void** out, uint32_t face,
                         uint32_t flags) {
    if (!r || !out) return false;
    if (r->type == kResTex || r->type == kResCube) {
        CsGuard rl(&resLock_);
        GpuTexture* t = r->index < textures_.size() ? &textures_[r->index] : nullptr;
        if (!t || level >= t->levels) return false;
        ++t->lockCount;
        t->lastLockLevel = level;
        t->lastLockFace = face;
        if (!(flags & 0x0010u) && (!t->stagingBuf || !t->staging)) {
            if (t->uploadGen > 0) t->keepStaging = true;
            if (!EnsureTextureStaging(t)) {
                --t->lockCount;
                return false;
            }
        }
        if (!t->staging) {
            if (!EnsureTextureStaging(t)) {
                --t->lockCount;
                return false;
            }
        }
        uint32_t faceOff = (r->type == kResCube && face < 6) ? (face * t->faceBytes) : 0;
        *out = (uint8_t*)t->staging + faceOff + t->levelOffset[level];
        return true;
    }

    CsGuard rl(&resLock_);
    GpuBuffer* b = r->index < buffers_.size() ? &buffers_[r->index] : nullptr;
    if (!b) return false;
    uint32_t curFrame = g_backend.FrameNumber();
    if (b->lastLockFrame != curFrame) {
        b->lastLockFrame = curFrame;
        b->locksInFrame = 1;
    } else {
        ++b->locksInFrame;
    }
    b->lastLockOffset = offset;
    b->lastLockSize = size;
    b->lastLockFlags = flags;

    const bool discard = (flags & 0x2000u) != 0u;
    const bool nooverwrite = (flags & 0x1000u) != 0u;
    const bool readonly = (flags & 0x0010u) != 0u;
    const uint32_t need = size ? (offset + size) : b->size;

    if (b->deviceLocal) {
        if (discard) {
            const uint32_t allocSize = need ? need : b->size;
            const uint32_t region = DynAllocate(allocSize);
            if (region != kDynNoSpace) {
                if (b->buffer && b->alloc) {
                    CsGuard ul(&uploadLock_);
                    retiredBuffers_[activeSlot_].push_back({b->buffer, b->alloc});
                }
                b->buffer = g_dynBuf;
                b->alloc = g_dynAlloc;
                b->baseOffset = region;
                b->baseCapacity = allocSize;
                b->mapped = g_dynMapped + region;
                b->ringFrame = curFrame;
                b->deviceLocal = false;
                b->uploadPending = false;
                b->drawnSinceLock = false;
                b->dirtyStart = 0;
                b->dirtyEnd = 0;
                free(b->cpuShadow);
                b->cpuShadow = nullptr;
                *out = (uint8_t*)b->mapped + offset;
                return true;
            }
        }
        if (!b->cpuShadow) {
            uint32_t allocBytes = b->size ? b->size : 4u;
            b->cpuShadow = (uint8_t*)malloc(allocBytes);
            if (!b->cpuShadow) return false;
            memset(b->cpuShadow, 0, allocBytes);
        }
        if (!readonly) {
            uint32_t lockOff = offset < b->size ? offset : 0;
            uint32_t lockEnd = size ? (lockOff + size) : b->size;
            if (lockEnd > b->size) lockEnd = b->size;
            if (b->dirtyEnd == 0 || lockOff < b->dirtyStart) b->dirtyStart = lockOff;
            if (lockEnd > b->dirtyEnd) b->dirtyEnd = lockEnd;
            if (b->drawnSinceLock) {
                ++b->reuploadCount;
                b->drawnSinceLock = false;
            }
        }
        *out = b->cpuShadow + offset;
        return true;
    }

    if (!b->mapped) return false;
    const bool inRing = g_dynBuf != VK_NULL_HANDLE && b->buffer == g_dynBuf;
    const bool fits = !inRing || need <= b->baseCapacity;
    const bool moveForDraw =
        !nooverwrite && b->drawnSinceLock && (b->lastDrawFrame + 1u >= curFrame);
    const bool moveForRing = inRing && (b->ringFrame != curFrame);
    const bool needRelocate =
        b->size && (discard || (!readonly && (!fits || moveForDraw || moveForRing)));
    if (needRelocate) {
        const uint32_t allocSize = discard ? need : b->size;
        const uint32_t region = DynAllocate(allocSize);
        if (region != kDynNoSpace) {
            if (!discard) {
                uint32_t copyBytes =
                    (inRing && b->baseCapacity < b->size) ? b->baseCapacity : b->size;
                if (copyBytes > allocSize) copyBytes = allocSize;
                if (!b->cpuShadow && b->size) {
                    b->cpuShadow = (uint8_t*)malloc(b->size);
                    if (b->cpuShadow) {
                        memcpy(b->cpuShadow, b->mapped, copyBytes);
                        if (b->size > copyBytes)
                            memset(b->cpuShadow + copyBytes, 0, b->size - copyBytes);
                    }
                }
                const void* src = b->cpuShadow ? (const void*)b->cpuShadow : b->mapped;
                memcpy(g_dynMapped + region, src, copyBytes);
            } else if (b->cpuShadow) {
                free(b->cpuShadow);
                b->cpuShadow = nullptr;
            }
            b->buffer = g_dynBuf;
            b->alloc = g_dynAlloc;
            b->baseOffset = region;
            b->baseCapacity = allocSize;
            b->mapped = g_dynMapped + region;
            b->ringFrame = curFrame;
            b->drawnSinceLock = false;
        }
    } else if (!readonly) {
        b->drawnSinceLock = false;
    }
    if (!discard && !readonly && b->cpuShadow) {
        uint32_t lockOff = offset < b->size ? offset : 0;
        uint32_t lockEnd = size ? (lockOff + size) : b->size;
        if (lockEnd > b->size) lockEnd = b->size;
        if (b->dirtyEnd == 0 || lockOff < b->dirtyStart) b->dirtyStart = lockOff;
        if (lockEnd > b->dirtyEnd) b->dirtyEnd = lockEnd;
        *out = b->cpuShadow + offset;
        return true;
    }
    if (readonly && b->cpuShadow) {
        *out = b->cpuShadow + offset;
        return true;
    }
    *out = (uint8_t*)b->mapped + offset;
    return true;
}

struct DdsPixelFormat {
    uint32_t size;
    uint32_t flags;
    uint32_t fourCC;
    uint32_t rgbBitCount;
    uint32_t rBitMask;
    uint32_t gBitMask;
    uint32_t bBitMask;
    uint32_t aBitMask;
};

struct DdsHeader {
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitchOrLinearSize;
    uint32_t depth;
    uint32_t mipMapCount;
    uint32_t reserved1[11];
    DdsPixelFormat ddspf;
    uint32_t caps;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
    uint32_t reserved2;
};

static bool LoadDdsTexture(const char* path, uint32_t& outW, uint32_t& outH,
                           VkFormat& outFmt, std::vector<uint8_t>& outPixels) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) return false;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x20534444u) {
        fclose(f);
        return false;
    }
    DdsHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || hdr.size != 124u) {
        fclose(f);
        return false;
    }
    outW = hdr.width;
    outH = hdr.height;
    if (hdr.ddspf.flags & 0x40u) {
        if (hdr.ddspf.rgbBitCount == 32u) {
            outFmt = (hdr.ddspf.rBitMask == 0x00FF0000u)
                         ? VK_FORMAT_B8G8R8A8_UNORM
                         : VK_FORMAT_R8G8B8A8_UNORM;
            const uint32_t bytes = outW * outH * 4u;
            outPixels.resize(bytes);
            const size_t readBytes = fread(outPixels.data(), 1, bytes, f);
            fclose(f);
            return readBytes == bytes;
        }
    } else if (hdr.ddspf.flags & 0x4u) {
        if (hdr.ddspf.fourCC == 0x35545844u) {
            outFmt = VK_FORMAT_BC3_UNORM_BLOCK;
            const uint32_t bytes = ((outW + 3u) / 4u) * ((outH + 3u) / 4u) * 16u;
            outPixels.resize(bytes);
            const size_t readBytes = fread(outPixels.data(), 1, bytes, f);
            fclose(f);
            return readBytes == bytes;
        } else if (hdr.ddspf.fourCC == 0x31545844u) {
            outFmt = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            const uint32_t bytes = ((outW + 3u) / 4u) * ((outH + 3u) / 4u) * 8u;
            outPixels.resize(bytes);
            const size_t readBytes = fread(outPixels.data(), 1, bytes, f);
            fclose(f);
            return readBytes == bytes;
        }
    }
    fclose(f);
    return false;
}

static void DecompressDxt5Block(const uint8_t* block, uint32_t* out4x4) {
    uint8_t a0 = block[0], a1 = block[1];
    uint8_t alphas[8];
    alphas[0] = a0;
    alphas[1] = a1;
    if (a0 > a1) {
        for (int i = 1; i <= 6; ++i)
            alphas[1 + i] = (uint8_t)(((7 - i) * a0 + i * a1 + 3) / 7);
    } else {
        for (int i = 1; i <= 4; ++i)
            alphas[1 + i] = (uint8_t)(((5 - i) * a0 + i * a1 + 2) / 5);
        alphas[6] = 0;
        alphas[7] = 255;
    }
    uint64_t aBits = 0;
    for (int i = 0; i < 6; ++i) {
        aBits |= ((uint64_t)block[2 + i]) << (i * 8);
    }

    uint16_t c0 = (uint16_t)(block[8] | (block[9] << 8));
    uint16_t c1 = (uint16_t)(block[10] | (block[11] << 8));
    uint32_t codeTable = (uint32_t)(block[12] | (block[13] << 8) |
                                   (block[14] << 16) | (block[15] << 24));

    uint8_t r0 = (uint8_t)(((c0 >> 11) & 0x1F) * 255 / 31);
    uint8_t g0 = (uint8_t)(((c0 >> 5) & 0x3F) * 255 / 63);
    uint8_t b0 = (uint8_t)((c0 & 0x1F) * 255 / 31);

    uint8_t r1 = (uint8_t)(((c1 >> 11) & 0x1F) * 255 / 31);
    uint8_t g1 = (uint8_t)(((c1 >> 5) & 0x3F) * 255 / 63);
    uint8_t b1 = (uint8_t)((c1 & 0x1F) * 255 / 31);

    uint8_t colors[4][3];
    colors[0][0] = r0; colors[0][1] = g0; colors[0][2] = b0;
    colors[1][0] = r1; colors[1][1] = g1; colors[1][2] = b1;

    if (c0 > c1) {
        colors[2][0] = (uint8_t)((2 * r0 + r1 + 1) / 3);
        colors[2][1] = (uint8_t)((2 * g0 + g1 + 1) / 3);
        colors[2][2] = (uint8_t)((2 * b0 + b1 + 1) / 3);

        colors[3][0] = (uint8_t)((r0 + 2 * r1 + 1) / 3);
        colors[3][1] = (uint8_t)((g0 + 2 * g1 + 1) / 3);
        colors[3][2] = (uint8_t)((b0 + 2 * b1 + 1) / 3);
    } else {
        colors[2][0] = (uint8_t)((r0 + r1) / 2);
        colors[2][1] = (uint8_t)((g0 + g1) / 2);
        colors[2][2] = (uint8_t)((b0 + b1) / 2);

        colors[3][0] = 0;
        colors[3][1] = 0;
        colors[3][2] = 0;
    }

    for (uint32_t py = 0; py < 4; ++py) {
        for (uint32_t px = 0; px < 4; ++px) {
            uint32_t k = py * 4 + px;
            uint32_t aIdx = (uint32_t)((aBits >> (k * 3)) & 0x7);
            uint32_t cIdx = (codeTable >> (k * 2)) & 0x3;
            uint8_t a = alphas[aIdx];
            uint8_t r = colors[cIdx][0];
            uint8_t g = colors[cIdx][1];
            uint8_t b = colors[cIdx][2];
            out4x4[py * 4 + px] = ((uint32_t)a << 24) | ((uint32_t)r << 16) |
                                  ((uint32_t)g << 8) | (uint32_t)b;
        }
    }
}

static void DecompressDxt5(const uint8_t* src, uint32_t w, uint32_t h,
                            std::vector<uint32_t>& dst) {
    dst.resize(w * h);
    uint32_t blocksX = (w + 3u) / 4u;
    uint32_t blocksY = (h + 3u) / 4u;
    for (uint32_t by = 0; by < blocksY; ++by) {
        for (uint32_t bx = 0; bx < blocksX; ++bx) {
            const uint8_t* block = src + (by * blocksX + bx) * 16u;
            uint32_t out4x4[16];
            DecompressDxt5Block(block, out4x4);
            for (uint32_t py = 0; py < 4; ++py) {
                uint32_t y = by * 4 + py;
                if (y >= h) continue;
                for (uint32_t px = 0; px < 4; ++px) {
                    uint32_t x = bx * 4 + px;
                    if (x >= w) continue;
                    dst[y * w + x] = out4x4[py * 4 + px];
                }
            }
        }
    }
}

static float CatmullRom(float p0, float p1, float p2, float p3, float t) {
    float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
    float a2 = -0.5f * p0 + 0.5f * p2;
    float a3 = p1;
    float v = ((a0 * t + a1) * t + a2) * t + a3;
    return v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
}

static void UpscaleAndSharpen2x(const std::vector<uint32_t>& src, uint32_t sw,
                               uint32_t sh, std::vector<uint8_t>& dstBytes) {
    const uint32_t dw = sw * 2u;
    const uint32_t dh = sh * 2u;
    std::vector<uint32_t> temp(dw * dh);

    auto getSample = [&](int x, int y, int ch) -> float {
        if (x < 0) x = 0;
        if (x >= (int)sw) x = (int)sw - 1;
        if (y < 0) y = 0;
        if (y >= (int)sh) y = (int)sh - 1;
        uint32_t p = src[(uint32_t)y * sw + (uint32_t)x];
        return (float)((p >> (ch * 8)) & 0xFFu);
    };

    for (uint32_t dy = 0; dy < dh; ++dy) {
        float sy = ((float)dy + 0.5f) * 0.5f - 0.5f;
        int iy = (int)floorf(sy);
        float ty = sy - (float)iy;
        for (uint32_t dx = 0; dx < dw; ++dx) {
            float sx = ((float)dx + 0.5f) * 0.5f - 0.5f;
            int ix = (int)floorf(sx);
            float tx = sx - (float)ix;

            uint32_t pix = 0;
            for (int ch = 0; ch < 4; ++ch) {
                float col[4];
                for (int m = 0; m < 4; ++m) {
                    float s0 = getSample(ix - 1, iy - 1 + m, ch);
                    float s1 = getSample(ix + 0, iy - 1 + m, ch);
                    float s2 = getSample(ix + 1, iy - 1 + m, ch);
                    float s3 = getSample(ix + 2, iy - 1 + m, ch);
                    col[m] = CatmullRom(s0, s1, s2, s3, tx);
                }
                float val = CatmullRom(col[0], col[1], col[2], col[3], ty);
                uint32_t uval = (uint32_t)(val + 0.5f);
                if (uval > 255u) uval = 255u;
                pix |= (uval << (ch * 8));
            }
            temp[dy * dw + dx] = pix;
        }
    }

    dstBytes.resize(dw * dh * 4u);
    uint32_t* outWords = (uint32_t*)dstBytes.data();

    for (uint32_t y = 0; y < dh; ++y) {
        for (uint32_t x = 0; x < dw; ++x) {
            uint32_t c = temp[y * dw + x];
            uint32_t n = temp[(y > 0 ? y - 1 : 0) * dw + x];
            uint32_t s = temp[(y + 1 < dh ? y + 1 : dh - 1) * dw + x];
            uint32_t w = temp[y * dw + (x > 0 ? x - 1 : 0)];
            uint32_t e = temp[y * dw + (x + 1 < dw ? x + 1 : dw - 1)];

            uint8_t a = (uint8_t)((c >> 24) & 0xFFu);
            uint8_t r = (uint8_t)((c >> 16) & 0xFFu);
            uint8_t g = (uint8_t)((c >> 8) & 0xFFu);
            uint8_t b = (uint8_t)(c & 0xFFu);

            if (r < 22 && g < 22 && b < 22) {
                r = 0; g = 0; b = 0;
            } else {
                float br = (float)(((n >> 16) & 0xFF) + ((s >> 16) & 0xFF) + ((w >> 16) & 0xFF) + ((e >> 16) & 0xFF)) * 0.25f;
                float bg = (float)(((n >> 8) & 0xFF) + ((s >> 8) & 0xFF) + ((w >> 8) & 0xFF) + ((e >> 8) & 0xFF)) * 0.25f;
                float bb = (float)((n & 0xFF) + (s & 0xFF) + (w & 0xFF) + (e & 0xFF)) * 0.25f;
                float sr = (float)r + 0.6f * ((float)r - br);
                float sg = (float)g + 0.6f * ((float)g - bg);
                float sb = (float)b + 0.6f * ((float)b - bb);
                r = (uint8_t)(sr < 0.0f ? 0 : (sr > 255.0f ? 255 : (int)(sr + 0.5f)));
                g = (uint8_t)(sg < 0.0f ? 0 : (sg > 255.0f ? 255 : (int)(sg + 0.5f)));
                b = (uint8_t)(sb < 0.0f ? 0 : (sb > 255.0f ? 255 : (int)(sb + 0.5f)));
            }
            outWords[y * dw + x] = ((uint32_t)a << 24) | ((uint32_t)r << 16) |
                                  ((uint32_t)g << 8) | (uint32_t)b;
        }
    }
}

bool ResourceTable::TryEnhanceTexture(GpuTexture* t, uint32_t resIndex) {
    (void)resIndex;
    if (!t || t->renderTarget || t->depth || t->cube || t->volume) return false;
    if (!t->staging) return false;
    if (t->width < 16 || t->height < 16) return false;

    uint32_t level0Bytes = t->levels > 1 ? t->levelOffset[1]
                                         : LevelBytes(t->format, t->width, t->height, nullptr);
    if (!level0Bytes) return false;
    const uint64_t hash = XXH64((const uint8_t*)t->staging, level0Bytes, 0);

    uint32_t enhW = 0, enhH = 0;
    VkFormat enhFmt = VK_FORMAT_UNDEFINED;
    std::vector<uint8_t> pixelData;

    char pathBuf[260];
    snprintf(pathBuf, sizeof(pathBuf), "textures/enhanced/%016llx.dds", hash);
    bool loaded = LoadDdsTexture(pathBuf, enhW, enhH, enhFmt, pixelData);
    if (!loaded && hash == 0xfe5227ebba7c0e6full) {
        loaded = LoadDdsTexture("textures/enhanced/emmet_legal_english.dds", enhW, enhH,
                                enhFmt, pixelData);
    }
    if (!loaded) {
        snprintf(pathBuf, sizeof(pathBuf), "textures/replacement/%016llx.dds", hash);
        loaded = LoadDdsTexture(pathBuf, enhW, enhH, enhFmt, pixelData);
    }

    if (!loaded && hash == 0xfe5227ebba7c0e6full && t->d3dFormat == 0x35545844u) {
        std::vector<uint32_t> decoded;
        DecompressDxt5((const uint8_t*)t->staging, t->width, t->height, decoded);
        UpscaleAndSharpen2x(decoded, t->width, t->height, pixelData);
        enhW = t->width * 2u;
        enhH = t->height * 2u;
        enhFmt = VK_FORMAT_B8G8R8A8_UNORM;
        loaded = true;
    }

    if (!loaded || pixelData.empty() || enhW == 0 || enhH == 0 || enhFmt == VK_FORMAT_UNDEFINED) {
        return false;
    }

    VkBufferCreateInfo stagingBci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingBci.size = pixelData.size();
    stagingBci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingBci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo stagingAci{};
    stagingAci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    stagingAci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo stagingInfo{};
    if (vmaCreateBuffer(g_backend.Allocator(), &stagingBci, &stagingAci, &stagingBuf,
                        &stagingAlloc, &stagingInfo) != VK_SUCCESS) {
        return false;
    }
    memcpy(stagingInfo.pMappedData, pixelData.data(), pixelData.size());

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = enhFmt;
    ici.extent = {enhW, enhH, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkImage enhImg = VK_NULL_HANDLE;
    VmaAllocation enhAlloc = VK_NULL_HANDLE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &enhImg, &enhAlloc,
                       nullptr) != VK_SUCCESS) {
        vmaDestroyBuffer(g_backend.Allocator(), stagingBuf, stagingAlloc);
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = enhImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = enhFmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView enhView = VK_NULL_HANDLE;
    if (vkCreateImageView(g_backend.Device(), &vci, nullptr, &enhView) != VK_SUCCESS) {
        vmaDestroyImage(g_backend.Allocator(), enhImg, enhAlloc);
        vmaDestroyBuffer(g_backend.Allocator(), stagingBuf, stagingAlloc);
        return false;
    }

    struct UploadData {
        VkImage img;
        VkBuffer buf;
        uint32_t w;
        uint32_t h;
    } ud{enhImg, stagingBuf, enhW, enhH};

    g_backend.ImmediateSubmit(
        [](VkCommandBuffer cmd, void* user) {
            UploadData* d = (UploadData*)user;
            VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b1.image = d->img;
            b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b1);

            VkBufferImageCopy bic{};
            bic.bufferOffset = 0;
            bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            bic.imageExtent = {d->w, d->h, 1};
            vkCmdCopyBufferToImage(cmd, d->buf, d->img,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);

            VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.image = d->img;
            b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b2);
        },
        &ud);

    vmaDestroyBuffer(g_backend.Allocator(), stagingBuf, stagingAlloc);
    g_backend.SetImageLayout(enhImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (t->view) vkDestroyImageView(g_backend.Device(), t->view, nullptr);
    if (t->image) vmaDestroyImage(g_backend.Allocator(), t->image, t->alloc);

    t->image = enhImg;
    t->alloc = enhAlloc;
    t->view = enhView;
    t->levelView[0] = enhView;
    t->isEnhanced = true;
    t->format = enhFmt;
    t->width = enhW;
    t->height = enhH;
    t->levels = 1;
    t->levelW[0] = enhW;
    t->levelH[0] = enhH;
    t->levelOffset[0] = 0;

    LOG("[res] texture enhanced: %ux%u -> %ux%u (hash=%016llx, fmt=%u)",
        enhW / 2u, enhH / 2u, enhW, enhH, hash, (uint32_t)enhFmt);
    return true;
}

void ResourceTable::Unlock(FakeResource* r, uint32_t level, uint32_t face) {
    if (!r) return;
    if (r->type == kResVB || r->type == kResIB) {
        CsGuard rl(&resLock_);
        GpuBuffer* b = r->index < buffers_.size() ? &buffers_[r->index] : nullptr;
        if (!b) return;
        if (b->lastLockFlags & 0x0010u) return;
        if (b->deviceLocal) {
            if (b->dirtyEnd > b->dirtyStart && b->dirtyEnd <= b->size &&
                b->cpuShadow && b->buffer) {
                const uint32_t bytes = b->dirtyEnd - b->dirtyStart;
                VkBuffer srcBuf = VK_NULL_HANDLE;
                VkDeviceSize srcOff = 0;
                uint8_t* dstPtr = nullptr;
                CsGuard ul(&uploadLock_);
                if (AllocStaging(bytes, srcBuf, srcOff, dstPtr)) {
                    memcpy(dstPtr, b->cpuShadow + b->dirtyStart, bytes);
                    PendingBufCopy c{};
                    c.srcBuf = srcBuf;
                    c.srcOff = srcOff;
                    c.dstBuf = b->buffer;
                    c.dstOff = b->dirtyStart;
                    c.size = bytes;
                    c.resIndex = r->index;
                    c.gen = ++b->uploadGen;
                    pendingBufs_.push_back(c);
                    b->uploadPending = true;
                    pendingUploadCount_.store((uint32_t)(pendingBufs_.size() + pendingImgs_.size()), std::memory_order_release);
                }
            }
            b->dirtyStart = 0;
            b->dirtyEnd = 0;
            return;
        }
        if (b->cpuShadow && b->dirtyEnd > b->dirtyStart &&
            b->dirtyEnd <= b->size && b->mapped) {
            memcpy((uint8_t*)b->mapped + b->dirtyStart,
                   b->cpuShadow + b->dirtyStart, b->dirtyEnd - b->dirtyStart);
            b->dirtyStart = 0;
            b->dirtyEnd = 0;
        }
        return;
    }
    if (r->type != kResTex && r->type != kResCube) return;
    CsGuard rl(&resLock_);
    GpuTexture* t = r->index < textures_.size() ? &textures_[r->index] : nullptr;
    if (!t) return;
    if (t->lockCount > 0) --t->lockCount;
    if (!t->staging || !t->stagingBuf) return;

    if (level >= t->levels) level = t->lastLockLevel < t->levels ? t->lastLockLevel : 0;
    const uint32_t f = (r->type == kResCube && face < 6) ? face : 0;
    if (level < 32) t->uploadedLevelsMask |= (1u << level);
    if (level < 16) t->uploadedFacesMask[level] |= (uint8_t)(1u << f);

    if (level == 0 && !t->isEnhanced && !t->renderTarget && !t->depth && r->type == kResTex) {
        if (TryEnhanceTexture(t, r->index)) {
            t->uploadPending = false;
            return;
        }
    }

    if (t->expandLum && t->expandBuf && t->expandMapped) {
        uint32_t expOff = 0;
        for (uint32_t i = 0; i < level && i < t->levels; ++i)
            expOff += t->levelW[i] * t->levelH[i] * 4u;
        const uint32_t lw = t->levelW[level], lh = t->levelH[level];
        const uint32_t count = lw * lh;
        const uint8_t* src = (const uint8_t*)t->staging + t->levelOffset[level];
        uint32_t* dst32 = (uint32_t*)((uint8_t*)t->expandMapped + expOff);
        for (uint32_t p = 0; p < count; ++p) {
            uint8_t v = src[p];
            uint32_t l = (v & 0x0Fu) * 17u;
            uint32_t a = (v >> 4) * 17u;
            dst32[p] = (a << 24) | (l * 0x010101u);
        }
    }

    CsGuard ul(&uploadLock_);
    const uint32_t gen = ++t->uploadGen;
    t->uploadPending = true;
    VkBuffer uploadBuf =
        (t->expandLum && t->expandBuf) ? t->expandBuf : t->stagingBuf;

    for (auto& existing : pendingImgs_) {
        if (existing.image == t->image) {
            if (level < 32) existing.dirtyLevelMask |= (1u << level);
            if (level < 16) existing.dirtyFaceMask[level] |= (uint8_t)(1u << f);
            existing.buf = uploadBuf;
            existing.origStagingBuf = t->stagingBuf;
            existing.gen = gen;
            return;
        }
    }

    PendingImgCopy up{};
    up.image = t->image;
    up.buf = uploadBuf;
    up.origStagingBuf = t->stagingBuf;
    up.texIndex = r->index;
    up.levels = t->levels;
    up.copyLevels = t->levels;
    up.layers = t->cube ? 6u : 1u;
    up.depthDim = 1;
    up.faceBytes = t->faceBytes;
    if (level < 32) up.dirtyLevelMask = (1u << level);
    if (level < 16) up.dirtyFaceMask[level] = (uint8_t)(1u << f);
    uint32_t expAcc = 0;
    for (uint32_t i = 0; i < t->levels && i < 16; ++i) {
        up.w[i] = t->levelW[i];
        up.h[i] = t->levelH[i];
        if (t->expandLum && t->expandBuf) {
            up.off[i] = expAcc;
            expAcc += t->levelW[i] * t->levelH[i] * 4u;
        } else {
            up.off[i] = t->levelOffset[i];
        }
    }
    up.gen = gen;
    pendingImgs_.push_back(up);
    pendingUploadCount_.store((uint32_t)(pendingBufs_.size() + pendingImgs_.size()), std::memory_order_release);
}

void ResourceTable::FlushPendingUploads(bool allowEndRendering) {
    (void)allowEndRendering;
    if (!g_backend.InRecording()) return;
    if (pendingUploadCount_.load(std::memory_order_acquire) == 0) return;

    std::vector<PendingBufCopy> localBufs;
    std::vector<PendingImgCopy> localImgs;
    uint32_t slot = 0;
    {
        CsGuard ul(&uploadLock_);
        if (pendingBufs_.empty() && pendingImgs_.empty()) {
            pendingUploadCount_.store(0, std::memory_order_release);
            return;
        }
        localBufs.swap(pendingBufs_);
        localImgs.swap(pendingImgs_);
        pendingUploadCount_.store(0, std::memory_order_release);
        slot = activeSlot_;
        for (auto* s : fullSlabs_) inFlightSlabs_[slot].push_back(s);
        fullSlabs_.clear();
        if (curSlab_) {
            inFlightSlabs_[slot].push_back(curSlab_);
            curSlab_ = nullptr;
        }
    }

    if (g_backend.InRendering()) g_backend.EndRendering();
    VkCommandBuffer cmd = g_backend.Cmd();

    if (!localBufs.empty()) {
        VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        pre.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                            VK_ACCESS_INDEX_READ_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
        pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0,
                             nullptr, 0, nullptr);
        for (const auto& c : localBufs) {
            VkBufferCopy bc{c.srcOff, c.dstOff, c.size};
            vkCmdCopyBuffer(cmd, c.srcBuf, c.dstBuf, 1, &bc);
        }
        VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        post.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                             VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &post,
                             0, nullptr, 0, nullptr);
    }

    for (const auto& u : localImgs) {
        VkImageLayout oldLayout = g_backend.GetImageLayout(u.image);
        VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b1.oldLayout = oldLayout;
        b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b1.image = u.image;
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, u.levels, 0,
                               u.layers};
        b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
            b1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        vkCmdPipelineBarrier(cmd, srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &b1);

        VkBufferImageCopy regions[96];
        uint32_t regCount = 0;
        const bool copyAll = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        for (uint32_t layer = 0; layer < u.layers && regCount < 96; ++layer) {
            for (uint32_t i = 0; i < u.copyLevels && regCount < 96; ++i) {
                if (!copyAll && i < 16 && !(u.dirtyFaceMask[i] & (1u << layer)))
                    continue;
                VkBufferImageCopy& c = regions[regCount++];
                c = VkBufferImageCopy{};
                c.bufferOffset = (VkDeviceSize)layer * u.faceBytes + u.off[i];
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, layer, 1};
                c.imageExtent = {u.w[i], u.h[i], u.depthDim};
            }
        }
        if (regCount > 0) {
            vkCmdCopyBufferToImage(cmd, u.buf, u.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   regCount, regions);
        }

        VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b2.image = u.image;
        b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, u.levels, 0,
                               u.layers};
        b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b2);
        g_backend.SetImageLayout(u.image,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    CsGuard rl(&resLock_);
    for (const auto& c : localBufs) {
        if (c.resIndex < buffers_.size()) {
            GpuBuffer& b = buffers_[c.resIndex];
            if (b.uploadGen == c.gen) b.uploadPending = false;
        }
    }
    CsGuard ul(&uploadLock_);
    for (const auto& u : localImgs) {
        if (u.texIndex < textures_.size()) {
            GpuTexture& t = textures_[u.texIndex];
            if (t.uploadGen == u.gen) {
                t.uploadPending = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Surfaces, cube textures and standalone render targets
// ---------------------------------------------------------------------------
VkImageView ResourceTable::LevelView(GpuTexture* t, uint32_t level) {
    if (!t || level >= t->levels) return VK_NULL_HANDLE;
    if (t->levelView[level]) return t->levelView[level];
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t->image;
    vci.viewType = t->cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = t->format;
    vci.subresourceRange = {t->aspect, level, 1, 0, t->cube ? 6u : 1u};
    ApplyFormatSwizzle(t->d3dFormat, vci.components);
    if (vkCreateImageView(g_backend.Device(), &vci, nullptr,
                          &t->levelView[level]) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return t->levelView[level];
}

FakeSurface* ResourceTable::SurfaceLevel(FakeResource* tex, uint32_t level) {
    CsGuard rl(&resLock_);
    GpuTexture* t = (tex && (tex->type == kResTex || tex->type == kResCube || tex->type == kResVolume) &&
                     tex->index < textures_.size())
                        ? &textures_[tex->index]
                        : nullptr;
    if (!t || level >= t->levels) return nullptr;
    for (auto* s : surfaces_)
        if (s->tex == tex && s->level == level) return s;

    FakeSurface* s = new FakeSurface();
    s->vtbl = g_surfVtbl;
    s->refs = 1;
    s->kind = t->depth ? 2u : 0u;
    s->tex = tex;
    s->level = level;
    s->width = t->levelW[level];
    s->height = t->levelH[level];
    s->format = tex->p0;
    s->image = t->image;
    s->view = LevelView(t, level);
    if (t->depth && (t->format == VK_FORMAT_D24_UNORM_S8_UINT ||
                     t->format == VK_FORMAT_D32_SFLOAT_S8_UINT)) {
        VkImageViewCreateInfo svci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        svci.image = t->image;
        svci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        svci.format = t->format;
        svci.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, level, 1, 0, 1};
        vkCreateImageView(g_backend.Device(), &svci, nullptr, &s->stencilView);
    }
    s->isDepth = t->depth ? 1u : 0u;
    s->isBackbuffer = 0;
    surfaces_.push_back(s);
    return s;
}

FakeSurface* ResourceTable::Backbuffer() {
    if (!backbuffer_) {
        VkExtent2D e = g_backend.SwapchainExtent();
        backbuffer_ = new FakeSurface();
        backbuffer_->vtbl = g_surfVtbl;
        backbuffer_->refs = 1;
        backbuffer_->kind = 1;
        backbuffer_->tex = nullptr;
        backbuffer_->level = 0;
        backbuffer_->width = e.width;
        backbuffer_->height = e.height;
        backbuffer_->format = 22;  // D3DFMT_X8R8G8B8
        backbuffer_->image = VK_NULL_HANDLE;
        backbuffer_->view = VK_NULL_HANDLE;
        backbuffer_->isDepth = 0;
        backbuffer_->isBackbuffer = 1;
        surfaces_.push_back(backbuffer_);
    }
    return backbuffer_;
}

void ResourceTable::UpdateBackbuffer(uint32_t w, uint32_t h, uint32_t d3dfmt) {
    FakeSurface* s = Backbuffer();
    if (w) s->width = w;
    if (h) s->height = h;
    if (d3dfmt) s->format = d3dfmt;
}

bool ResourceTable::IsDepthFormatD3D(uint32_t f) {
    switch (f) {
        case 71: case 73: case 75: case 77: case 79: case 80:
        case 118: case 119:
        case 0x5A544E49u:  // MAKEFOURCC('I','N','T','Z')
        case 0x34324644u:  // MAKEFOURCC('D','F','2','4')
        case 0x5A574152u:  // MAKEFOURCC('R','A','W','Z')
            return true;
        default:
            return false;
    }
}

VkFormat ResourceTable::ToVkDepthFormat(uint32_t f, VkImageAspectFlags* aspect) {
    if (aspect) *aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    switch (f) {
        case 75:   // D24S8
        case 79:   // D24X4S4
        case 73:   // D15S1
        case 119:  // DF24
        case 0x5A544E49u:  // INTZ
        case 0x34324644u:  // DF24 fourcc
        case 0x5A574152u:  // RAWZ
            if (aspect)
                *aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case 71:  // D32
            return VK_FORMAT_D32_SFLOAT;
        case 77:  // D24X8
            return VK_FORMAT_X8_D24_UNORM_PACK32;
        case 80:   // D16
        case 118:  // DF16
            return VK_FORMAT_D16_UNORM;
        default:
            if (aspect)
                *aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            return VK_FORMAT_D24_UNORM_S8_UINT;
    }
}

FakeSurface* ResourceTable::CreateStandaloneSurface(uint32_t w, uint32_t h,
                                                    uint32_t format,
                                                    bool depth,
                                                    uint32_t usage) {
    (void)usage;
    if (!w) w = 1;
    if (!h) h = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkFormat vk = depth ? ToVkDepthFormat(format, &aspect) : ToVkFormat(format);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = vk;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                       : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &img, &alloc,
                       nullptr) != VK_SUCCESS) {
        LOG("[res] %s %ux%u fmt=%d allocation failed", depth ? "depth" : "target",
            w, h, (int)vk);
        return nullptr;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = vk;
    vci.subresourceRange = {depth ? (VkImageAspectFlags)VK_IMAGE_ASPECT_DEPTH_BIT : aspect, 0, 1, 0, 1};
    ApplyFormatSwizzle(format, vci.components);
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(g_backend.Device(), &vci, nullptr, &view);
    VkImageView stencilView = VK_NULL_HANDLE;
    if (depth && (vk == VK_FORMAT_D24_UNORM_S8_UINT || vk == VK_FORMAT_D32_SFLOAT_S8_UINT)) {
        vci.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        vkCreateImageView(g_backend.Device(), &vci, nullptr, &stencilView);
    }

    FakeSurface* s = new FakeSurface();
    s->vtbl = g_surfVtbl;
    s->refs = 1;
    s->kind = depth ? 2u : 0u;
    s->tex = nullptr;
    s->level = 0;
    s->width = w;
    s->height = h;
    s->format = format;
    s->image = img;
    s->view = view;
    s->stencilView = stencilView;
    s->isDepth = depth ? 1u : 0u;
    s->isBackbuffer = 0;
    CsGuard rl(&resLock_);
    surfaces_.push_back(s);
    return s;
}

FakeResource* ResourceTable::CreateCubeTexture(uint32_t edge, uint32_t levels,
                                               uint32_t usage, uint32_t format) {
    (void)usage;
    if (!edge) edge = 1;
    uint32_t lv = levels;
    if (!lv) {
        uint32_t m = edge;
        lv = 1;
        while (m > 1) {
            m >>= 1;
            ++lv;
        }
    }
    if (lv > 16) lv = 16;

    GpuTexture t{};
    t.width = t.height = edge;
    t.levels = lv;
    t.format = ToVkFormat(format);
    t.cube = true;
    t.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t faceBytes = 0;
    for (uint32_t i = 0; i < lv; ++i) {
        uint32_t e = edge >> i;
        if (!e) e = 1;
        t.levelW[i] = t.levelH[i] = e;
        t.levelOffset[i] = faceBytes;
        faceBytes += LevelBytes(t.format, e, e, nullptr);
    }
    t.faceBytes = faceBytes;
    {
        uint32_t p = 0;
        LevelBytes(t.format, edge, edge, &p);
        t.rowPitch = p;
    }
    uint32_t total = faceBytes * 6;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = t.format;
    ici.extent = {edge, edge, 1};
    ici.mipLevels = lv;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &t.image, &t.alloc,
                       nullptr) != VK_SUCCESS) {
        LOG("[res] cube %u fmt=%d allocation failed", edge, (int)t.format);
        return nullptr;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    vci.format = t.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, lv, 0, 6};
    ApplyFormatSwizzle(format, vci.components);
    vkCreateImageView(g_backend.Device(), &vci, nullptr, &t.view);
    t.stagingBytes = total + total / 2 + 65536;
    EnsureTextureStaging(&t);

    CsGuard rl(&resLock_);
    textures_.push_back(t);

    FakeResource* r = new FakeResource();
    r->vtbl = g_cubeVtbl;
    r->refs = 1;
    r->type = kResCube;
    r->index = (uint32_t)textures_.size() - 1;
    r->p0 = format;
    r->p1 = 0;
    r->ptr = &textures_.back();
    live_.push_back(r);
    return r;
}

FakeVolume* ResourceTable::CreateVolumeTexture(uint32_t w, uint32_t h,
                                              uint32_t depth, uint32_t levels,
                                              uint32_t usage, uint32_t format) {
    (void)usage;
    if (!w) w = 1;
    if (!h) h = 1;
    if (!depth) depth = 1;
    uint32_t lv = levels ? levels : 1;
    if (lv > 16) lv = 16;

    GpuTexture t{};
    t.volume = true;
    t.width = w;
    t.height = h;
    t.depthDim = depth;
    t.levels = lv;
    t.format = ToVkFormat(format);
    t.rowPitch = w * BytesPerPixel(t.format);

    uint32_t total = t.rowPitch * h * depth;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = t.format;
    ici.extent = {w, h, depth};
    ici.mipLevels = lv;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &t.image, &t.alloc,
                       nullptr) != VK_SUCCESS) {
        LOG("[res] volume %ux%ux%u fmt=%d allocation failed", w, h, depth, (int)t.format);
        return nullptr;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vci.format = t.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, lv, 0, 1};
    ApplyFormatSwizzle(format, vci.components);
    vkCreateImageView(g_backend.Device(), &vci, nullptr, &t.view);

    VkBufferCreateInfo sbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbi.size = total + 4096;
    sbi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo saci = g_backend.HostAlloc();
    VmaAllocationInfo sinfo{};
    if (vmaCreateBuffer(g_backend.Allocator(), &sbi, &saci, &t.stagingBuf,
                        &t.stagingAlloc, &sinfo) == VK_SUCCESS)
        t.staging = sinfo.pMappedData;

    CsGuard rl(&resLock_);
    textures_.push_back(t);
    uint32_t texIdx = (uint32_t)textures_.size() - 1;

    FakeVolume* v = new FakeVolume();
    v->vtbl = g_volTexVtbl;
    v->refs = 1;
    v->type = kResVolume;
    v->index = texIdx;
    v->p0 = format;
    v->p1 = 0;
    v->ptr = &textures_[texIdx];
    v->width = w;
    v->height = h;
    v->depth = depth;
    v->levels = lv;
    v->format = format;
    v->rowPitch = t.rowPitch;
    v->slicePitch = v->rowPitch * h;
    v->stagingPtr = t.staging;
    v->stagingSize = total;
    v->owner = nullptr;
    live_.push_back((FakeResource*)v);
    return v;
}

void ResourceTable::UnlockVolume(FakeVolume* v, uint32_t level) {
    if (!v || level != 0) return;
    CsGuard rl(&resLock_);
    if (v->index >= textures_.size()) return;
    GpuTexture* t = &textures_[v->index];
    if (!t->stagingBuf || !t->image) return;

    CsGuard ul(&uploadLock_);
    const uint32_t gen = ++t->uploadGen;
    t->uploadPending = true;
    t->uploadedLevelsMask |= 1u;

    for (auto& existing : pendingImgs_) {
        if (existing.image == t->image) {
            existing.dirtyLevelMask |= 1u;
            existing.dirtyFaceMask[0] |= 1u;
            existing.buf = t->stagingBuf;
            existing.origStagingBuf = t->stagingBuf;
            existing.gen = gen;
            return;
        }
    }

    PendingImgCopy up{};
    up.image = t->image;
    up.buf = t->stagingBuf;
    up.origStagingBuf = t->stagingBuf;
    up.texIndex = v->index;
    up.levels = t->levels;
    up.copyLevels = 1;
    up.layers = 1;
    up.depthDim = t->depthDim;
    up.dirtyLevelMask = 1u;
    up.dirtyFaceMask[0] = 1u;
    up.w[0] = t->width;
    up.h[0] = t->height;
    up.off[0] = 0;
    up.gen = gen;
    pendingImgs_.push_back(up);
    pendingUploadCount_.store((uint32_t)(pendingBufs_.size() + pendingImgs_.size()), std::memory_order_release);
}
