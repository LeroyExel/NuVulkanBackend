#pragma once

#include "vk_common.h"

#include <vk_mem_alloc.h>

class FpsOverlay {
public:
    bool Init();
    void Shutdown();
    void TickAndDraw(VkCommandBuffer cmd, VkImage targetImage, VkImageView targetView,
                     VkFormat format, VkExtent2D extent, uint32_t frameSlot);

private:
    bool EnsurePipeline(VkFormat format);
    void BuildFontAtlasData(uint8_t* outPixels, uint32_t w, uint32_t h);
    uint32_t BuildQuads(uint32_t frameSlot, VkExtent2D extent);

    VkShaderModule vs_ = VK_NULL_HANDLE;
    VkShaderModule fs_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkFormat pipeFormat_ = VK_FORMAT_UNDEFINED;

    VkImage fontImage_ = VK_NULL_HANDLE;
    VmaAllocation fontAlloc_ = VK_NULL_HANDLE;
    VkImageView fontView_ = VK_NULL_HANDLE;
    VkSampler fontSampler_ = VK_NULL_HANDLE;

    VkBuffer vb_ = VK_NULL_HANDLE;
    VmaAllocation vbAlloc_ = VK_NULL_HANDLE;
    void* vbMapped_ = nullptr;

    LARGE_INTEGER freq_{};
    LARGE_INTEGER lastTick_{};
    uint32_t accumFrames_ = 0;
    uint32_t displayFps_ = 0;
    uint32_t cachedFps_[2]{0xFFFFFFFFu, 0xFFFFFFFFu};
    uint32_t cachedW_[2]{0, 0};
    uint32_t cachedH_[2]{0, 0};
    uint32_t cachedVerts_[2]{0, 0};
    bool initialized_ = false;
};

extern FpsOverlay g_fpsOverlay;
