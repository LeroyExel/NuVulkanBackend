#pragma once
#include <vk_mem_alloc.h>

#include "vk_common.h"

struct VkFrame {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
};

struct ClearRequest {
    bool valid = false;
    bool clearColor = false;
    bool clearDepth = false;
    bool clearStencil = false;
    VkClearColorValue color{};
    float depth = 1.0f;
    uint32_t stencil = 0;
};

// A colour or depth attachment described by the D3D9 layer. When `swapchain`
// is set the context substitutes its current swapchain image.
struct ImageRef {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView stencilView = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    bool swapchain = false;
    bool depth = false;
};

// Vulkan 1.3 core context: instance, device, swapchain, descriptor ABI and the
// two-frame command/semaphore ring. Dynamic rendering is the only render-pass
// mechanism; there are no render pass objects and no offscreen readbacks.
class VkBackend {
public:
    bool Init(HWND hwnd, uint32_t width, uint32_t height);
    void Shutdown();
    bool RecreateSwapchain(uint32_t width, uint32_t height);

    bool BeginFrame();
    void EndFrame(bool present);
    void WaitIdle();

    bool InRecording() const { return recording; }
    void SetImageLayout(VkImage img, VkImageLayout layout) {
        if (img) layouts_[img] = layout;
    }
    VkImageLayout GetImageLayout(VkImage img) const {
        auto it = layouts_.find(img);
        return it != layouts_.end() ? it->second : VK_IMAGE_LAYOUT_UNDEFINED;
    }
    VkCommandBuffer Cmd() const { return frames[frameIndex].cmd; }
    bool InRendering() const { return renderingActive; }

    void BeginRendering(const ClearRequest* clear);
    void EndRendering();
    void EnsureRendering();
    void Clear(uint32_t flags, uint32_t color, float depth, uint32_t stencil,
               const RECT* rects, uint32_t rectCount);
    void SetPendingClear(uint32_t flags, uint32_t color, float depth,
                         uint32_t stencil);

    // Ends the current dynamic-rendering scope, returns outgoing textures to
    // SHADER_READ_ONLY, moves incoming attachments to their layouts and
    // re-establishes viewport/scissor.
    bool SetRenderTargets(const ImageRef* colors, uint32_t count,
                          const ImageRef* depth);
    bool BindSwapchainTargets();
    // Depth attachment matching an offscreen colour target's size. D3D9 keeps a
    // single device-wide depth surface at swapchain size; Vulkan requires every
    // attachment of a rendering scope to share dimensions.
    ImageRef DepthFor(uint32_t w, uint32_t h, VkFormat format);

    uint32_t ColorTargetCount() const { return targetCount; }
    VkFormat ColorFormatAt(uint32_t i) const {
        return i < targetCount ? targetFormat[i] : VK_FORMAT_UNDEFINED;
    }

    void BlitImage(const ImageRef& src, const RECT* srcRect,
                   const ImageRef& dst, const RECT* dstRect);

    // Bindings 0/1 are the dynamic uniform blocks every shader reads its c[]
    // registers from.
    void BindUniformRing(VkBuffer buffer, uint32_t range);

    void SetTextureSlot(uint32_t slot, VkImageView view, VkSampler sampler,
                        bool cube, bool volume = false);
    void FlushTextures();

    VkDevice Device() const { return device; }
    VkPhysicalDevice Physical() const { return phys; }
    VkInstance Instance() const { return instance; }
    VkExtent2D Extent() const {
        return {targetW ? targetW : extent.width,
                targetH ? targetH : extent.height};
    }
    VkExtent2D TargetExtent() const { return Extent(); }
    VkExtent2D SwapchainExtent() const { return extent; }
    VkFormat ColorFormat() const {
        return targetCount ? targetFormat[0] : colorFormat;
    }
    VkFormat DepthFormat() const { return targetDepthFormat; }
    VkPipelineLayout PipeLayout() const { return pipelineLayout; }
    VkDescriptorSet DescriptorSet() const { return currentDescSet; }
    VmaAllocator Allocator() const { return allocator; }
    bool Eds3Enabled() const { return eds3; }
    bool HasDepthClip() const { return hasDepthClip_; }
    uint32_t MinUboAlign() const { return minUboAlign; }
    uint32_t FrameSlot() const { return frameIndex; }
    uint32_t FrameNumber() const { return frameCounter; }
    VkSampler DefaultSampler() const { return defaultSampler; }
    VkSampler VolumeSampler() const {
        return defaultVolumeSampler ? defaultVolumeSampler : defaultSampler;
    }
    VkImageView DefaultView() const { return defaultView; }
    VkImageView DefaultCubeView() const { return defaultCubeView; }
    VkBuffer DummyVertexBuffer() const { return dummyVb; }
    void SetVSync(bool enable) { vsyncEnabled = enable; }
    bool VSync() const { return vsyncEnabled; }
    VmaAllocationCreateInfo HostAlloc();

    VkResult ImmediateSubmit(void (*fn)(VkCommandBuffer, void*), void* user);
    void TransitionImage(VkImage image, VkImageAspectFlags aspect,
                         VkImageLayout newLayout, uint32_t mipLevels = 1);

private:
    bool CreateSwapchainObjects();
    void DestroySwapchainObjects();
    bool CreateDefaultTexture();
    bool CreateDefaultCube();
    bool CreateDefaultVolume();
    bool PickPhysicalDevice();
    bool CreateDescriptorPool();

    VkImage ResolveImage(const ImageRef& r) const;
    VkImageView ResolveView(const ImageRef& r) const;

    HWND hwnd = nullptr;
    uint32_t reqWidth = 0, reqHeight = 0;

    // Vulkan requires queue submission and the upload command buffer to be
    // externally synchronized. The engine uploads textures from loader threads
    // while the render thread presents.
    CRITICAL_SECTION queueLock{};

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue gfxQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    uint32_t gfxFamily = 0, presentFamily = 0;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthAlloc = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;

    struct DepthTarget {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
    };
    std::vector<DepthTarget> depthTargets;

    // Active attachments. targetCount == 0 means "the swapchain".
    ImageRef targetColor[4]{};
    uint32_t targetCount = 0;
    ImageRef targetDepth{};
    VkFormat targetFormat[4]{};
    VkFormat targetDepthFormat = VK_FORMAT_UNDEFINED;
    uint32_t targetW = 0, targetH = 0;

    struct TextureState {
        VkImageView views[16];
        VkSampler samplers[16];
        bool operator==(const TextureState& o) const {
            return memcmp(views, o.views, sizeof(views)) == 0 &&
                   memcmp(samplers, o.samplers, sizeof(samplers)) == 0;
        }
    };
    struct TextureStateHash {
        size_t operator()(const TextureState& s) const;
    };
    struct TexMruEntry {
        uint64_t hash = 0;
        TextureState state{};
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    VmaAllocator allocator = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descPools;
    VkDescriptorPool currentDescPool = VK_NULL_HANDLE;
    std::unordered_map<TextureState, VkDescriptorSet, TextureStateHash> descCache;
    TexMruEntry texMru_[8]{};
    uint32_t texMruNext_ = 0;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    VkDescriptorSet currentDescSet = VK_NULL_HANDLE;
    VkBuffer ringBuffer_ = VK_NULL_HANDLE;
    uint32_t ringRange_ = 0;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    bool hasDepthClip_ = false;

    VkSampler defaultSampler = VK_NULL_HANDLE;
    VkImage defaultImage = VK_NULL_HANDLE;
    VmaAllocation defaultAlloc = VK_NULL_HANDLE;
    VkImageView defaultView = VK_NULL_HANDLE;
    VkImage defaultCubeImage = VK_NULL_HANDLE;
    VmaAllocation defaultCubeAlloc = VK_NULL_HANDLE;
    VkImageView defaultCubeView = VK_NULL_HANDLE;
    VkImage defaultVolumeImage = VK_NULL_HANDLE;
    VmaAllocation defaultVolumeAlloc = VK_NULL_HANDLE;
    VkImageView defaultVolumeView = VK_NULL_HANDLE;
    VkSampler defaultVolumeSampler = VK_NULL_HANDLE;
    VkBuffer dummyVb = VK_NULL_HANDLE;
    VmaAllocation dummyVbAlloc = VK_NULL_HANDLE;
    bool vsyncEnabled = true;

    VkFrame frames[2];
    uint32_t frameIndex = 0;
    uint32_t imageIndex = 0;
    uint32_t frameCounter = 0;
    bool renderingActive = false;
    bool recording = false;
    ClearRequest pendingClear;
    bool eds3 = false;
    uint32_t minUboAlign = 256;
    VkImageView boundViews[16]{};
    VkSampler boundSamplers[16]{};
    bool boundCube[16]{};
    bool boundVolume[16]{};
    bool texturesDirty = false;
    VkFence immediateFence = VK_NULL_HANDLE;
    VkCommandBuffer immediateCmd = VK_NULL_HANDLE;

    std::unordered_map<VkImage, VkImageLayout> layouts_;
};

extern VkBackend g_backend;
