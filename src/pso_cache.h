#pragma once

#include "resources.h"
#include "vk_common.h"

struct PipelineKey {
    uint64_t vsHash;
    uint64_t psHash;
    uint64_t declKey;
    uint32_t blendEnable;
    uint32_t srcBlend;
    uint32_t dstBlend;
    uint32_t blendOp;
    uint32_t srcBlendAlpha;
    uint32_t dstBlendAlpha;
    uint32_t blendOpAlpha;
    uint32_t alphaTest;
    uint32_t alphaRef;
    uint32_t alphaFunc;
    uint32_t fillMode;
    uint32_t colorWrite;
    uint32_t topology;
    VkFormat colorFormats[4];
    VkFormat depthFormat;
    uint32_t colorCount;
    uint16_t activeStreams;
    uint16_t instancedMask;
    uint32_t psDrefMask;
    uint32_t padDref_;
};

class PsoCache {
public:
    void Init();
    void Shutdown();
    void FlushCache();
    VkPipeline Get(const PipelineKey& key, const VertexDecl* decl);
    uint32_t Count() const { return (uint32_t)map_.size(); }

private:
    void SaveCache();
    VkPipeline Build(const PipelineKey& key, const VertexDecl* decl);
    std::unordered_map<uint64_t, VkPipeline> map_;
    uint64_t mruHash_[4]{};
    PipelineKey mruKey_[4]{};
    VkPipeline mruPipe_[4]{};
    uint32_t mruNext_ = 0;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    uint32_t builtSinceSave_ = 0;
    uint32_t lastSaveTick_ = 0;
};

extern PsoCache g_psos;

uint64_t HashPipelineKey(const PipelineKey& k);
