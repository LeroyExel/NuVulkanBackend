#pragma once
#include "vk_common.h"
#include <vk_mem_alloc.h>
#include <atomic>
#include <deque>

enum ResType : uint32_t {
    kResVB = 0,
    kResIB = 1,
    kResTex = 2,
    kResDecl = 3,
    kResCube = 4,
    kResVolume = 6
};

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    uint32_t size = 0;
    uint32_t stride = 0;
    uint32_t lastLockFrame = 0;
    uint32_t lastLockOffset = 0;
    uint32_t lastLockSize = 0;
    uint32_t lastLockFlags = 0;
    uint32_t locksInFrame = 0;
    // Byte offset of this buffer's live copy inside the dynamic vertex ring.
    // A re-lock with D3DLOCK_DISCARD renames the buffer into a fresh region so
    // draws already recorded in the command buffer keep their own vertices.
    uint32_t baseOffset = 0;
    // Bytes reserved at baseOffset, so an append lock that grows past the
    // region is moved instead of writing into the neighbouring allocation.
    uint32_t baseCapacity = 0;
    uint32_t lastDrawFrame = 0;
    uint32_t ringFrame = 0xFFFFFFFFu;
    bool drawnSinceLock = false;
    uint8_t* cpuShadow = nullptr;
    uint32_t usage = 0;
    uint32_t dirtyStart = 0;
    uint32_t dirtyEnd = 0;
    uint32_t reuploadCount = 0;
    uint32_t uploadGen = 0;
    bool deviceLocal = false;
    bool isIndex = false;
    bool uploadPending = false;
};

// Rewinds the renaming ring for the frame slot that is about to start.
void ResResetDynamicRing(uint32_t frameSlot);

struct GpuTexture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    // Per-mip views, created on demand by GetSurfaceLevel. A render target
    // surface must be a single level, so it cannot reuse the whole-chain view.
    VkImageView levelView[16]{};
    void* staging = nullptr;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0, levels = 1;
    uint32_t rowPitch = 0;
    // Per-level geometry and offset into the staging buffer. Block-compressed
    // levels are 8/16 bytes per 4x4 block, so this cannot be derived by
    // multiplying width by a constant pitch.
    uint32_t levelOffset[16]{};
    uint32_t levelW[16]{};
    uint32_t levelH[16]{};
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    // D3D9 luminance/alpha formats do not sample like a single-channel Vulkan
    // format: L8 returns (L,L,L,1) and A8 returns (0,0,0,A), where R8_UNORM
    // returns (R,0,0,1). They are therefore uploaded expanded to RGBA8, from a
    // staging buffer that keeps the engine's original one/two byte layout.
    bool expandLum = false;
    uint32_t srcBytes = 1;
    uint32_t d3dFormat = 0;
    VkBuffer expandBuf = VK_NULL_HANDLE;
    VmaAllocation expandAlloc = VK_NULL_HANDLE;
    void* expandMapped = nullptr;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    bool renderTarget = false;
    bool depth = false;
    bool cube = false;
    bool volume = false;
    uint32_t depthDim = 1;
    uint32_t faceBytes = 0;
    bool uploadPending = false;
    uint32_t lockCount = 0;
    uint32_t lastLockLevel = 0;
    uint32_t lastLockFace = 0;
    uint32_t uploadedLevelsMask = 0;
    uint8_t uploadedFacesMask[16]{};
    uint32_t stagingBytes = 0;
    uint32_t expandBytes = 0;
    uint32_t uploadGen = 0;
    bool keepStaging = false;
    bool isEnhanced = false;
};

struct VertexAttr {
    uint32_t location = 0;
    VkFormat format = VK_FORMAT_R32G32B32_SFLOAT;
    uint32_t offset = 0;
    uint32_t stream = 0;
};

struct VertexDecl {
    std::vector<VertexAttr> attrs;
    uint64_t key = 0;
    uint16_t streamMask = 0;
};

struct FakeResource {
    void** vtbl;
    uint32_t refs;
    uint32_t type;
    uint32_t index;
    uint32_t p0;
    uint32_t p1;
    void* ptr;
};

// IDirect3DSurface9. Slot order:
//   0 QI, 1 AddRef, 2 Release, 3 GetDevice, 4 SetPrivateData, 5 GetPrivateData,
//   6 FreePrivateData, 7 SetPriority, 8 GetPriority, 9 PreLoad, 10 GetType,
//   11 GetDesc, 12 LockRect, 13 UnlockRect, 14 GetDC, 15 ReleaseDC,
//   16 GetContainer
struct FakeSurface {
    void** vtbl;
    uint32_t refs;
    uint32_t kind;       // 0 = texture/surface level, 1 = backbuffer, 2 = depth
    FakeResource* tex;   // owning resource, null for the backbuffer
    uint32_t level;
    uint32_t width, height, format;  // D3DFORMAT for GetDesc
    VkImage image;
    VkImageView view;
    VkImageView stencilView = VK_NULL_HANDLE;
    uint32_t isDepth;
    uint32_t isBackbuffer;
    // CPU shadow for a standalone surface (render target, depth target,
    // backbuffer), whose LockRect has no texture staging to hand out. The
    // engine ignores the HRESULT and writes through D3DLOCKED_RECT, so a lock
    // must always come back with a real pointer.
    void* cpu = nullptr;
    uint32_t cpuBytes = 0;
};

// IDirect3DVolumeTexture9 / IDirect3DVolume9. The engine creates a volume
// texture through slot 24; both objects are CPU-backed because nothing samples
// them, but every slot needs its correct arity so the caller's stack stays
// balanced.
struct FakeVolume {
    void** vtbl;
    uint32_t refs;
    uint32_t type;
    uint32_t index;
    uint32_t p0;
    uint32_t p1;
    void* ptr;
    uint32_t width, height, depth, levels, format;
    uint32_t rowPitch, slicePitch;
    void* stagingPtr;
    uint32_t stagingSize;
    FakeVolume* owner;
};

class ResourceTable {
public:
    bool Init();
    void Shutdown();
    void OnFrameBegin(uint32_t frameSlot);
    void FlushPendingUploads(bool allowEndRendering);
    bool HasPendingUploads() const {
        return pendingUploadCount_.load(std::memory_order_acquire) != 0;
    }

    FakeResource* CreateVertexBuffer(uint32_t length, uint32_t usage, uint32_t fvf);
    FakeResource* CreateIndexBuffer(uint32_t length, uint32_t usage, uint32_t format);
    FakeResource* CreateTexture(uint32_t w, uint32_t h, uint32_t levels,
                                uint32_t usage, uint32_t format);
    FakeResource* CreateCubeTexture(uint32_t edge, uint32_t levels,
                                    uint32_t usage, uint32_t format);
    FakeSurface* CreateStandaloneSurface(uint32_t w, uint32_t h, uint32_t format,
                                         bool depth, uint32_t usage);
    FakeResource* CreateVertexDeclaration(const uint32_t* tokens);
    FakeVolume* CreateVolumeTexture(uint32_t w, uint32_t h, uint32_t depth,
                                    uint32_t levels, uint32_t usage,
                                    uint32_t format);
    void UnlockVolume(FakeVolume* v, uint32_t level);

    FakeSurface* SurfaceLevel(FakeResource* tex, uint32_t level);
    FakeSurface* Backbuffer();

    GpuBuffer* Buffer(const FakeResource* r);
    GpuTexture* TextureOf(const FakeResource* r);
    VertexDecl* Decl(const FakeResource* r);

    void AddRef(FakeResource* r);
    void Release(FakeResource* r);
    bool Lock(FakeResource* r, uint32_t level, uint32_t offset, uint32_t size,
              void** out, uint32_t face = 0, uint32_t flags = 0);
    void Unlock(FakeResource* r, uint32_t level = 0, uint32_t face = 0);

    uint32_t BufferSize(const FakeResource* r) const;
    static VkFormat ToVkFormat(uint32_t d3dfmt);
    static bool IsDepthFormatD3D(uint32_t d3dfmt);
    static VkFormat ToVkDepthFormat(uint32_t d3dfmt, VkImageAspectFlags* aspect);

    void UpdateBackbuffer(uint32_t w, uint32_t h, uint32_t d3dfmt);
    bool TryEnhanceTexture(GpuTexture* t, uint32_t resIndex);

private:
    struct StagingSlab {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        uint8_t* mapped = nullptr;
        uint32_t cap = 0;
        uint32_t used = 0;
    };
    struct RetiredBuffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
    };
    struct PendingBufCopy {
        VkBuffer srcBuf = VK_NULL_HANDLE;
        VkDeviceSize srcOff = 0;
        VkBuffer dstBuf = VK_NULL_HANDLE;
        VkDeviceSize dstOff = 0;
        VkDeviceSize size = 0;
        uint32_t resIndex = 0;
        uint32_t gen = 0;
    };
    struct PendingImgCopy {
        VkImage image = VK_NULL_HANDLE;
        VkBuffer buf = VK_NULL_HANDLE;
        VkBuffer origStagingBuf = VK_NULL_HANDLE;
        uint32_t texIndex = 0;
        uint32_t levels = 1;
        uint32_t copyLevels = 1;
        uint32_t layers = 1;
        uint32_t depthDim = 1;
        uint32_t faceBytes = 0;
        uint32_t dirtyLevelMask = 0;
        uint8_t dirtyFaceMask[16]{};
        uint32_t w[16]{};
        uint32_t h[16]{};
        uint32_t off[16]{};
        uint32_t gen = 0;
    };

    bool AllocStaging(uint32_t bytes, VkBuffer& outBuf, VkDeviceSize& outOff,
                      uint8_t*& outPtr);
    bool EnsureTextureStaging(GpuTexture* t);
    VkImageView LevelView(GpuTexture* t, uint32_t level);
    std::deque<GpuBuffer> buffers_;
    std::deque<GpuTexture> textures_;
    std::deque<VertexDecl> decls_;
    std::vector<FakeResource*> live_;
    std::vector<FakeSurface*> surfaces_;
    FakeSurface* backbuffer_ = nullptr;

    CRITICAL_SECTION resLock_{};
    CRITICAL_SECTION uploadLock_{};
    bool locksInit_ = false;
    uint32_t activeSlot_ = 0;
    StagingSlab* curSlab_ = nullptr;
    std::vector<StagingSlab*> fullSlabs_;
    std::vector<StagingSlab*> freeSlabs_;
    std::vector<StagingSlab*> inFlightSlabs_[2];
    std::vector<RetiredBuffer> retiredBuffers_[2];
    std::vector<PendingBufCopy> pendingBufs_;
    std::vector<PendingImgCopy> pendingImgs_;
    std::atomic<uint32_t> pendingUploadCount_{0};
};

extern ResourceTable g_resources;

// D3DVERTEXELEMENT9 is 8 bytes packed: WORD Stream, WORD Offset, BYTE Type,
// BYTE Method, BYTE Usage, BYTE UsageIndex. The terminator has Stream 0xFFFF.
struct D3DVertexElement {
    uint16_t stream;
    uint16_t offset;
    uint8_t type;
    uint8_t method;
    uint8_t usage;
    uint8_t usageIndex;
};
static_assert(sizeof(D3DVertexElement) == 8, "D3DVERTEXELEMENT9 is 8 bytes");
