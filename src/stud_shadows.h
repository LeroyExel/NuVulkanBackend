#pragma once

#include "vk_common.h"

#include <vk_mem_alloc.h>

struct FakeSurface;

class StudShadows {
public:
    bool Init();
    void Shutdown();
    void BeginFrame();
    void CollectStud(const float* vsConst, FakeSurface* gbufColor,
                     FakeSurface* gbufDepth);
    void TrackTargetDraw(FakeSurface* color0, FakeSurface* depth);
    bool NeedsFlush() const {
        return initialized_ && studCount_ > 0 && !flushedThisFrame_ &&
               sceneColor_ != nullptr && gbufDepth_ != nullptr;
    }
    FakeSurface* SceneColor() const { return sceneColor_; }
    FakeSurface* DepthSurface() const { return gbufDepth_; }
    bool FlushScene();
    bool FlushBeforeDepthClear(FakeSurface* depth);
    bool FlushIfReading(FakeSurface* texSurf);
    bool FlushEndOfFrame();

private:
    struct StudItem {
        float px, py, pz;
        float viewProj[16];
    };

    bool EnsurePipeline(VkFormat colorFormat);
    bool EnsureDepthDescriptor(VkImageView depthView, uint32_t frameSlot);
    uint32_t BuildQuads(uint32_t frameSlot, uint32_t width, uint32_t height);
    bool Render();

    static constexpr uint32_t kMaxStuds = 512;
    static constexpr uint32_t kVertsPerStud = 6;
    static constexpr uint32_t kMaxVerts = kMaxStuds * kVertsPerStud;

    StudItem studs_[kMaxStuds]{};
    uint32_t studCount_ = 0;
    FakeSurface* gbufColor_ = nullptr;
    FakeSurface* gbufDepth_ = nullptr;
    FakeSurface* sceneColor_ = nullptr;
    bool flushedThisFrame_ = false;
    bool initialized_ = false;

    VkShaderModule vs_ = VK_NULL_HANDLE;
    VkShaderModule fs_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSets_[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkImageView boundDepthViews_[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkSampler depthSampler_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkFormat pipeFormat_ = VK_FORMAT_UNDEFINED;

    VkBuffer vb_ = VK_NULL_HANDLE;
    VmaAllocation vbAlloc_ = VK_NULL_HANDLE;
    void* vbMapped_ = nullptr;
};

extern StudShadows g_studShadows;
