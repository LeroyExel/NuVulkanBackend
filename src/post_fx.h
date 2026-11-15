#pragma once

#include "vk_common.h"

#include <vk_mem_alloc.h>

struct FakeSurface;

class PostFx {
public:
    bool Init();
    void Shutdown();
    void BeginFrame();
    void OnSwapchainRecreated();

    void TrackDraw(FakeSurface* color0, FakeSurface* color1, FakeSurface* depth,
                   bool zEnable, bool zWrite, uint32_t colorWrite,
                   bool readsRt, uint32_t topo, uint32_t primCount);
    void MarkBackbufferComposite();
    void MarkBackbufferStretch();

    bool NeedsSceneFlush() const {
        return initialized_ && hasScene3D_ && !scenePostDone_ &&
               sceneColor_ != nullptr && sceneDepth_ != nullptr;
    }
    FakeSurface* SceneColor() const { return sceneColor_; }
    bool HasSceneThisFrame() const { return hasScene3D_; }
    bool IsBackbufferPostDone() const { return bbPostDone_; }

    bool FlushScenePass(FakeSurface* colorOverride = nullptr,
                        FakeSurface* depthOverride = nullptr);
    bool FlushBeforeDepthClear(FakeSurface* depth);
    bool FlushIfReading(FakeSurface* texSurf);
    bool FlushBeforeHud();
    bool FlushEndOfFrame();

private:
    struct RenderTarget {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    bool CreateTarget(RenderTarget& rt, uint32_t w, uint32_t h, VkFormat fmt,
                      VkImageUsageFlags extraUsage = 0);
    void DestroyTarget(RenderTarget& rt);
    bool EnsureTargets(uint32_t fullW, uint32_t fullH, VkFormat sceneFmt,
                       VkFormat bbFmt);
    void DestroyTargets();
    bool EnsureGraphicsPipeline(VkShaderModule fs, VkFormat colorFmt,
                                VkPipeline& pipe, VkFormat& pipeFmt);
    void UpdateSet(uint32_t idx, VkImageView v0, VkSampler s0, VkImageView v1,
                   VkSampler s1, VkImageView v2, VkSampler s2);
    void DrawPass(VkCommandBuffer cmd, VkPipeline pipe, uint32_t setIdx,
                  VkImage dstImg, VkImageView dstView, uint32_t w, uint32_t h,
                  const void* pushData, uint32_t pushBytes);

    bool initialized_ = false;
    bool hasScene3D_ = false;
    bool gbufSeen_ = false;
    bool scenePostDone_ = false;
    bool bbPostDone_ = false;
    bool bbCompositeSeen_ = false;
    bool bbDirect3D_ = false;

    FakeSurface* gbufDepth_ = nullptr;
    FakeSurface* sceneColor_ = nullptr;
    FakeSurface* sceneDepth_ = nullptr;
    FakeSurface* forwardCandColor_ = nullptr;
    FakeSurface* forwardCandDepth_ = nullptr;
    uint32_t forwardDrawCount_ = 0;
    uint32_t bb3dDrawCount_ = 0;
    uint32_t bbDraws_ = 0;

    uint32_t allocW_ = 0;
    uint32_t allocH_ = 0;
    VkFormat allocSceneFmt_ = VK_FORMAT_UNDEFINED;
    VkFormat allocBbFmt_ = VK_FORMAT_UNDEFINED;

    RenderTarget gtaoRaw_{};
    RenderTarget gtaoTmp_{};
    RenderTarget gtaoFinal_{};
    RenderTarget sceneCopy_{};
    RenderTarget bbCopy_{};
    RenderTarget smaaEdges_{};
    RenderTarget smaaWeights_{};

    VkSampler nearestSampler_ = VK_NULL_HANDLE;
    VkSampler linearSampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSets_[16]{};

    VkShaderModule vsFull_ = VK_NULL_HANDLE;
    VkShaderModule fsGtao_ = VK_NULL_HANDLE;
    VkShaderModule fsBlurH_ = VK_NULL_HANDLE;
    VkShaderModule fsBlurV_ = VK_NULL_HANDLE;
    VkShaderModule fsSceneEnh_ = VK_NULL_HANDLE;
    VkShaderModule fsSmaaEdge_ = VK_NULL_HANDLE;
    VkShaderModule fsSmaaWeight_ = VK_NULL_HANDLE;
    VkShaderModule fsSmaaBlend_ = VK_NULL_HANDLE;

    VkPipeline pipeGtao_ = VK_NULL_HANDLE;
    VkFormat fmtGtao_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeBlurH_ = VK_NULL_HANDLE;
    VkFormat fmtBlurH_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeBlurV_ = VK_NULL_HANDLE;
    VkFormat fmtBlurV_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeSceneEnh_ = VK_NULL_HANDLE;
    VkFormat fmtSceneEnh_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeSmaaEdge_ = VK_NULL_HANDLE;
    VkFormat fmtSmaaEdge_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeSmaaWeight_ = VK_NULL_HANDLE;
    VkFormat fmtSmaaWeight_ = VK_FORMAT_UNDEFINED;
    VkPipeline pipeSmaaBlend_ = VK_NULL_HANDLE;
    VkFormat fmtSmaaBlend_ = VK_FORMAT_UNDEFINED;
};

extern PostFx g_postFx;
