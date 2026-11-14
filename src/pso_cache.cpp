#include "pso_cache.h"

#include "shaders.h"
#include "vk_context.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

PsoCache g_psos;

uint64_t HashPipelineKey(const PipelineKey& k) {
    return XXH64(&k, sizeof(k), 0x50534F43ull);
}

void PsoCache::Init() {
    map_.clear();
    map_.reserve(2048);
    memset(mruHash_, 0, sizeof(mruHash_));
    memset(mruPipe_, 0, sizeof(mruPipe_));
    mruNext_ = 0;
    builtSinceSave_ = 0;
    lastSaveTick_ = GetTickCount();

    std::vector<uint8_t> initData;
    FILE* f = fopen("shaders\\vk_pso.cache", "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0) {
            initData.resize((size_t)sz);
            if (fread(initData.data(), 1, (size_t)sz, f) != (size_t)sz) {
                initData.clear();
            }
        }
        fclose(f);
    }

    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = initData.size();
    ci.pInitialData = initData.empty() ? nullptr : initData.data();
    if (vkCreatePipelineCache(g_backend.Device(), &ci, nullptr, &cache_) != VK_SUCCESS) {
        cache_ = VK_NULL_HANDLE;
    }
}

void PsoCache::SaveCache() {
    if (!cache_) return;
    size_t sz = 0;
    if (vkGetPipelineCacheData(g_backend.Device(), cache_, &sz, nullptr) != VK_SUCCESS || !sz) {
        return;
    }
    std::vector<uint8_t> data(sz);
    if (vkGetPipelineCacheData(g_backend.Device(), cache_, &sz, data.data()) != VK_SUCCESS) {
        return;
    }
    CreateDirectoryA("shaders", nullptr);
    FILE* f = fopen("shaders\\vk_pso.cache.tmp", "wb");
    if (f) {
        size_t written = fwrite(data.data(), 1, sz, f);
        fclose(f);
        if (written == sz) {
            MoveFileExA("shaders\\vk_pso.cache.tmp", "shaders\\vk_pso.cache",
                        MOVEFILE_REPLACE_EXISTING);
        } else {
            DeleteFileA("shaders\\vk_pso.cache.tmp");
        }
    }
    builtSinceSave_ = 0;
    lastSaveTick_ = GetTickCount();
}

void PsoCache::FlushCache() {}

void PsoCache::Shutdown() {
    SaveCache();
    for (auto& kv : map_) {
        if (kv.second) vkDestroyPipeline(g_backend.Device(), kv.second, nullptr);
    }
    map_.clear();
    if (cache_) {
        vkDestroyPipelineCache(g_backend.Device(), cache_, nullptr);
        cache_ = VK_NULL_HANDLE;
    }
    memset(mruHash_, 0, sizeof(mruHash_));
    memset(mruPipe_, 0, sizeof(mruPipe_));
    mruNext_ = 0;
}

static VkBlendFactor ToVkBlend(uint32_t b) {
    switch (b) {
        case 1: return VK_BLEND_FACTOR_ZERO;
        case 2: return VK_BLEND_FACTOR_ONE;
        case 3: return VK_BLEND_FACTOR_SRC_COLOR;
        case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 7: return VK_BLEND_FACTOR_DST_ALPHA;
        case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR;
        case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

static VkBlendOp ToVkBlendOp(uint32_t o) {
    switch (o) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

static VkPolygonMode ToVkFill(uint32_t f) {
    return f == 2 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
}

static VkPrimitiveTopology ToVkTopo(uint32_t t) {
    switch (t) {
        case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

VkPipeline PsoCache::Get(const PipelineKey& key, const VertexDecl* decl) {
    const uint32_t last = (mruNext_ - 1u) & 3u;
    if (mruPipe_[last] && memcmp(&mruKey_[last], &key, sizeof(PipelineKey)) == 0) {
        return mruPipe_[last];
    }

    const uint64_t h = HashPipelineKey(key);
    for (uint32_t i = 0; i < 4; ++i) {
        if (i != last && mruPipe_[i] && mruHash_[i] == h &&
            memcmp(&mruKey_[i], &key, sizeof(PipelineKey)) == 0) {
            return mruPipe_[i];
        }
    }

    auto it = map_.find(h);
    VkPipeline p = VK_NULL_HANDLE;
    if (it != map_.end()) {
        p = it->second;
    } else {
        p = Build(key, decl);
        map_.emplace(h, p);
    }
    if (p) {
        const uint32_t slot = mruNext_ & 3u;
        mruNext_ = slot + 1u;
        mruHash_[slot] = h;
        mruKey_[slot] = key;
        mruPipe_[slot] = p;
    }
    return p;
}

VkPipeline PsoCache::Build(const PipelineKey& key, const VertexDecl* decl) {
    VkShaderModule vs = g_shaders.Get(key.vsHash, false);
    VkShaderModule ps = g_shaders.GetPixel(key.psHash, key.psDrefMask);
    if (!vs || !ps) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = ps;
    stages[1].pName = "main";

    // Alpha test is evaluated in the pixel shader from specialization
    // constants, because Vulkan has no fixed-function alpha test.
    VkSpecializationMapEntry specEntries[3]{
        {0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
        {1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
        {2, 2 * sizeof(uint32_t), sizeof(float)},
    };
    struct {
        uint32_t alphaTest;
        uint32_t alphaFunc;
        float alphaRef;
    } specData{
        key.alphaTest,
        key.alphaFunc,
        (float)key.alphaRef / 255.0f,
    };
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 3;
    specInfo.pMapEntries = specEntries;
    specInfo.dataSize = sizeof(specData);
    specInfo.pData = &specData;
    stages[1].pSpecializationInfo = &specInfo;

    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attrs;
    uint32_t usedStreams = 0;
    for (uint32_t s = 0; s < 16; ++s) {
        if (!(key.activeStreams & (1u << s))) continue;
        bool used = false;
        if (decl) {
            for (const auto& a : decl->attrs) {
                if (a.stream == s) {
                    used = true;
                    break;
                }
            }
        }
        if (!used) continue;
        usedStreams |= (1u << s);
        VkVertexInputBindingDescription b{};
        b.binding = s;
        b.stride = 0;
        b.inputRate = (key.instancedMask & (1u << s))
                          ? VK_VERTEX_INPUT_RATE_INSTANCE
                          : VK_VERTEX_INPUT_RATE_VERTEX;
        bindings.push_back(b);
    }

    uint32_t providedMask = 0;
    if (decl) {
        for (const auto& a : decl->attrs) {
            if (a.stream >= 16 || !(key.activeStreams & (1u << a.stream))) continue;
            VkVertexInputAttributeDescription d{};
            d.location = a.location;
            d.binding = a.stream;
            d.format = a.format;
            d.offset = a.offset;
            attrs.push_back(d);
            if (a.location < 32) providedMask |= (1u << a.location);
        }
    }

    // A shader may read an input the current declaration does not provide.
    // D3D9 supplies a per-semantic default for unbound attributes; here the
    // dummy buffer holds those defaults, indexed by location, with stride 0 so
    // every vertex reads the same 16-byte slot.
    uint32_t inMask = g_shaders.GetVsInputMask(key.vsHash);
    uint32_t missing = inMask & ~providedMask;
    if (missing) {
        uint32_t dummyStream = 15;
        while (dummyStream > 0 && (usedStreams & (1u << dummyStream)))
            --dummyStream;
        VkVertexInputBindingDescription dummyB{};
        dummyB.binding = dummyStream;
        dummyB.stride = 0;
        dummyB.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        bindings.push_back(dummyB);
        for (uint32_t loc = 0; loc < 32; ++loc) {
            if (!(missing & (1u << loc))) continue;
            VkVertexInputAttributeDescription d{};
            d.location = loc;
            d.binding = dummyStream;
            d.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            d.offset = loc * 16;
            attrs.push_back(d);
        }
    }

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = (uint32_t)bindings.size();
    vi.pVertexBindingDescriptions = bindings.data();
    vi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = ToVkTopo(key.topology);
    ia.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = ToVkFill(key.fillMode);
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineRasterizationDepthClipStateCreateInfoEXT depthClipInfo{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
    if (g_backend.HasDepthClip()) {
        depthClipInfo.depthClipEnable = VK_TRUE;
        rs.pNext = &depthClipInfo;
    }

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState cb[4]{};
    uint32_t atts = key.colorCount ? key.colorCount : 1;
    if (atts > 4) atts = 4;
    for (uint32_t i = 0; i < atts; ++i) {
        cb[i].blendEnable = key.blendEnable ? VK_TRUE : VK_FALSE;
        cb[i].srcColorBlendFactor = ToVkBlend(key.srcBlend);
        cb[i].dstColorBlendFactor = ToVkBlend(key.dstBlend);
        cb[i].colorBlendOp = ToVkBlendOp(key.blendOp);
        cb[i].srcAlphaBlendFactor = ToVkBlend(key.srcBlendAlpha);
        cb[i].dstAlphaBlendFactor = ToVkBlend(key.dstBlendAlpha);
        cb[i].alphaBlendOp = ToVkBlendOp(key.blendOpAlpha);
        cb[i].colorWriteMask = key.colorWrite & 0xF;
    }

    VkPipelineColorBlendStateCreateInfo cbs{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbs.attachmentCount = atts;
    cbs.pAttachments = cb;

    VkDynamicState dyn[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
    };
    VkPipelineDynamicStateCreateInfo dsi{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dsi.dynamicStateCount = (uint32_t)(sizeof(dyn) / sizeof(dyn[0]));
    dsi.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo prc{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prc.colorAttachmentCount = atts;
    prc.pColorAttachmentFormats = key.colorFormats;
    prc.depthAttachmentFormat = key.depthFormat;
    prc.stencilAttachmentFormat =
        (key.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
         key.depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT)
            ? key.depthFormat
            : VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo gp{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &prc;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cbs;
    gp.pDynamicState = &dsi;
    gp.layout = g_backend.PipeLayout();

    VkPipeline out = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(g_backend.Device(), cache_, 1,
                                           &gp, nullptr, &out);
    if (r != VK_SUCCESS) {
        LOG("[pso] pipeline creation failed (%d) vs=%016llx ps=%016llx",
            (int)r, (unsigned long long)key.vsHash,
            (unsigned long long)key.psHash);
        return VK_NULL_HANDLE;
    }
    ++builtSinceSave_;
    return out;
}
