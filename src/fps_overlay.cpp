#include "fps_overlay.h"

#include "vk_context.h"

FpsOverlay g_fpsOverlay;

namespace {

struct OverlayVert {
    float x, y, u, v, r, g, b, a;
};

constexpr uint32_t kMaxQuads = 16;
constexpr uint32_t kVertsPerSlot = kMaxQuads * 6;
constexpr uint32_t kSlotBytes = kVertsPerSlot * sizeof(OverlayVert);
constexpr uint32_t kAtlasCells = 16;
constexpr uint32_t kCellW = 8;
constexpr uint32_t kCellH = 9;
constexpr uint32_t kAtlasW = kAtlasCells * kCellW;
constexpr uint32_t kAtlasH = kCellH;

const uint8_t kGlyphs[16][7] = {
    {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
    {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
    {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
    {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E},
    {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
    {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
    {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
    {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
    {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
    {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
    {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
    {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
    {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},
    {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00},
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F},
};

struct SpvBuilder {
    std::vector<uint32_t> header;
    std::vector<uint32_t> caps;
    std::vector<uint32_t> entry;
    std::vector<uint32_t> decs;
    std::vector<uint32_t> types;
    std::vector<uint32_t> fn;
    uint32_t nextId = 1;

    uint32_t id() { return nextId++; }

    static void emit(std::vector<uint32_t>& dst, uint32_t op,
                     std::initializer_list<uint32_t> args) {
        dst.push_back(((uint32_t)(args.size() + 1) << 16) | op);
        for (uint32_t a : args) dst.push_back(a);
    }

    std::vector<uint32_t> finish() {
        std::vector<uint32_t> out;
        out.reserve(caps.size() + entry.size() + decs.size() + types.size() + fn.size() + 8);
        out.push_back(0x07230203u);
        out.push_back(0x00010300u);
        out.push_back(0u);
        out.push_back(nextId);
        out.push_back(0u);
        out.insert(out.end(), caps.begin(), caps.end());
        out.insert(out.end(), entry.begin(), entry.end());
        out.insert(out.end(), decs.begin(), decs.end());
        out.insert(out.end(), types.begin(), types.end());
        out.insert(out.end(), fn.begin(), fn.end());
        return out;
    }
};

std::vector<uint32_t> BuildOverlayVs() {
    SpvBuilder b;
    b.emit(b.caps, 17, {1});
    b.emit(b.caps, 14, {0, 1});

    uint32_t tVoid = b.id();
    uint32_t tFn = b.id();
    uint32_t tFloat = b.id();
    uint32_t tVec2 = b.id();
    uint32_t tVec4 = b.id();
    uint32_t tInV2 = b.id();
    uint32_t tInV4 = b.id();
    uint32_t tOutV2 = b.id();
    uint32_t tOutV4 = b.id();

    uint32_t vInPos = b.id();
    uint32_t vInUv = b.id();
    uint32_t vInCol = b.id();
    uint32_t vOutPos = b.id();
    uint32_t vOutUv = b.id();
    uint32_t vOutCol = b.id();

    uint32_t c0 = b.id();
    uint32_t c1 = b.id();
    uint32_t fnMain = b.id();
    uint32_t lbl = b.id();

    b.emit(b.entry, 15, {0, fnMain, 0x6E69616Du, 0u, vInPos, vInUv, vInCol, vOutPos, vOutUv, vOutCol});

    b.emit(b.decs, 71, {vInPos, 30, 0});
    b.emit(b.decs, 71, {vInUv, 30, 1});
    b.emit(b.decs, 71, {vInCol, 30, 2});
    b.emit(b.decs, 71, {vOutPos, 11, 0});
    b.emit(b.decs, 71, {vOutUv, 30, 0});
    b.emit(b.decs, 71, {vOutCol, 30, 1});

    b.emit(b.types, 19, {tVoid});
    b.emit(b.types, 33, {tFn, tVoid});
    b.emit(b.types, 22, {tFloat, 32});
    b.emit(b.types, 23, {tVec2, tFloat, 2});
    b.emit(b.types, 23, {tVec4, tFloat, 4});
    b.emit(b.types, 32, {tInV2, 1, tVec2});
    b.emit(b.types, 32, {tInV4, 1, tVec4});
    b.emit(b.types, 32, {tOutV2, 3, tVec2});
    b.emit(b.types, 32, {tOutV4, 3, tVec4});
    b.emit(b.types, 43, {tFloat, c0, 0x00000000u});
    b.emit(b.types, 43, {tFloat, c1, 0x3F800000u});
    b.emit(b.types, 59, {tInV2, vInPos, 1});
    b.emit(b.types, 59, {tInV2, vInUv, 1});
    b.emit(b.types, 59, {tInV4, vInCol, 1});
    b.emit(b.types, 59, {tOutV4, vOutPos, 3});
    b.emit(b.types, 59, {tOutV2, vOutUv, 3});
    b.emit(b.types, 59, {tOutV4, vOutCol, 3});

    uint32_t lPos = b.id();
    uint32_t px = b.id();
    uint32_t py = b.id();
    uint32_t outPosVal = b.id();
    uint32_t lUv = b.id();
    uint32_t lCol = b.id();

    b.emit(b.fn, 54, {tVoid, fnMain, 0, tFn});
    b.emit(b.fn, 248, {lbl});
    b.emit(b.fn, 61, {tVec2, lPos, vInPos});
    b.emit(b.fn, 81, {tFloat, px, lPos, 0});
    b.emit(b.fn, 81, {tFloat, py, lPos, 1});
    b.emit(b.fn, 80, {tVec4, outPosVal, px, py, c0, c1});
    b.emit(b.fn, 62, {vOutPos, outPosVal});
    b.emit(b.fn, 61, {tVec2, lUv, vInUv});
    b.emit(b.fn, 62, {vOutUv, lUv});
    b.emit(b.fn, 61, {tVec4, lCol, vInCol});
    b.emit(b.fn, 62, {vOutCol, lCol});
    b.emit(b.fn, 253, {});
    b.emit(b.fn, 56, {});
    return b.finish();
}

std::vector<uint32_t> BuildOverlayFs() {
    SpvBuilder b;
    b.emit(b.caps, 17, {1});
    b.emit(b.caps, 14, {0, 1});

    uint32_t tVoid = b.id();
    uint32_t tFn = b.id();
    uint32_t tFloat = b.id();
    uint32_t tVec2 = b.id();
    uint32_t tVec4 = b.id();
    uint32_t tImg = b.id();
    uint32_t tSmpImg = b.id();
    uint32_t tUniPtr = b.id();
    uint32_t tInV2 = b.id();
    uint32_t tInV4 = b.id();
    uint32_t tOutV4 = b.id();

    uint32_t vInUv = b.id();
    uint32_t vInCol = b.id();
    uint32_t vOutCol = b.id();
    uint32_t vAtlas = b.id();

    uint32_t fnMain = b.id();
    uint32_t lbl = b.id();

    b.emit(b.entry, 15, {4, fnMain, 0x6E69616Du, 0u, vInUv, vInCol, vOutCol});
    b.emit(b.entry, 16, {fnMain, 7});

    b.emit(b.decs, 71, {vInUv, 30, 0});
    b.emit(b.decs, 71, {vInCol, 30, 1});
    b.emit(b.decs, 71, {vOutCol, 30, 0});
    b.emit(b.decs, 71, {vAtlas, 34, 0});
    b.emit(b.decs, 71, {vAtlas, 33, 0});

    b.emit(b.types, 19, {tVoid});
    b.emit(b.types, 33, {tFn, tVoid});
    b.emit(b.types, 22, {tFloat, 32});
    b.emit(b.types, 23, {tVec2, tFloat, 2});
    b.emit(b.types, 23, {tVec4, tFloat, 4});
    b.emit(b.types, 25, {tImg, tFloat, 1, 0, 0, 0, 1, 0});
    b.emit(b.types, 27, {tSmpImg, tImg});
    b.emit(b.types, 32, {tUniPtr, 0, tSmpImg});
    b.emit(b.types, 32, {tInV2, 1, tVec2});
    b.emit(b.types, 32, {tInV4, 1, tVec4});
    b.emit(b.types, 32, {tOutV4, 3, tVec4});
    b.emit(b.types, 59, {tInV2, vInUv, 1});
    b.emit(b.types, 59, {tInV4, vInCol, 1});
    b.emit(b.types, 59, {tOutV4, vOutCol, 3});
    b.emit(b.types, 59, {tUniPtr, vAtlas, 0});

    uint32_t lUv = b.id();
    uint32_t lCol = b.id();
    uint32_t lAtlas = b.id();
    uint32_t tex = b.id();
    uint32_t mul = b.id();

    b.emit(b.fn, 54, {tVoid, fnMain, 0, tFn});
    b.emit(b.fn, 248, {lbl});
    b.emit(b.fn, 61, {tVec2, lUv, vInUv});
    b.emit(b.fn, 61, {tVec4, lCol, vInCol});
    b.emit(b.fn, 61, {tSmpImg, lAtlas, vAtlas});
    b.emit(b.fn, 87, {tVec4, tex, lAtlas, lUv});
    b.emit(b.fn, 133, {tVec4, mul, lCol, tex});
    b.emit(b.fn, 62, {vOutCol, mul});
    b.emit(b.fn, 253, {});
    b.emit(b.fn, 56, {});
    return b.finish();
}

int GlyphSlotFor(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c == 'F') return 10;
    if (c == 'P') return 11;
    if (c == 'S') return 12;
    if (c == ':') return 13;
    return 14;
}

void PushQuad(OverlayVert*& dst, float x0, float y0, float x1, float y1,
              float u0, float v0, float u1, float v1, float invW, float invH,
              float r, float g, float b, float a) {
    const float nx0 = x0 * invW * 2.0f - 1.0f;
    const float ny0 = y0 * invH * 2.0f - 1.0f;
    const float nx1 = x1 * invW * 2.0f - 1.0f;
    const float ny1 = y1 * invH * 2.0f - 1.0f;
    *dst++ = {nx0, ny0, u0, v0, r, g, b, a};
    *dst++ = {nx1, ny0, u1, v0, r, g, b, a};
    *dst++ = {nx1, ny1, u1, v1, r, g, b, a};
    *dst++ = {nx0, ny0, u0, v0, r, g, b, a};
    *dst++ = {nx1, ny1, u1, v1, r, g, b, a};
    *dst++ = {nx0, ny1, u0, v1, r, g, b, a};
}

}

void FpsOverlay::BuildFontAtlasData(uint8_t* outPixels, uint32_t w, uint32_t h) {
    memset(outPixels, 0, (size_t)w * h);
    for (uint32_t slot = 0; slot < kAtlasCells; ++slot) {
        const uint32_t ox = slot * kCellW + 1;
        for (uint32_t row = 0; row < 7; ++row) {
            const uint8_t bits = kGlyphs[slot][row];
            uint8_t* dstRow = outPixels + (size_t)(row + 1) * w + ox;
            for (uint32_t col = 0; col < 5; ++col) {
                if (bits & (1u << (4u - col))) dstRow[col] = 0xFFu;
            }
        }
    }
}

bool FpsOverlay::Init() {
    if (initialized_) return true;
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (!dev || !alloc) return false;

    QueryPerformanceFrequency(&freq_);
    QueryPerformanceCounter(&lastTick_);
    accumFrames_ = 0;
    displayFps_ = 0;
    cachedFps_[0] = cachedFps_[1] = 0xFFFFFFFFu;
    cachedW_[0] = cachedW_[1] = 0;
    cachedH_[0] = cachedH_[1] = 0;
    cachedVerts_[0] = cachedVerts_[1] = 0;

    std::vector<uint32_t> vsWords = BuildOverlayVs();
    std::vector<uint32_t> fsWords = BuildOverlayFs();
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = vsWords.size() * sizeof(uint32_t);
    smci.pCode = vsWords.data();
    if (vkCreateShaderModule(dev, &smci, nullptr, &vs_) != VK_SUCCESS) return false;
    smci.codeSize = fsWords.size() * sizeof(uint32_t);
    smci.pCode = fsWords.data();
    if (vkCreateShaderModule(dev, &smci, nullptr, &fs_) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 1;
    slci.pBindings = &bind;
    if (vkCreateDescriptorSetLayout(dev, &slci, nullptr, &setLayout_) != VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipeLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize poolSz{};
    poolSz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSz.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &descPool_) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &setLayout_;
    if (vkAllocateDescriptorSets(dev, &dsai, &descSet_) != VK_SUCCESS) return false;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8_UNORM;
    ici.extent = {kAtlasW, kAtlasH, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo iaci{};
    iaci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(alloc, &ici, &iaci, &fontImage_, &fontAlloc_, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = fontImage_;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8_UNORM;
    ivci.components = {VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_ONE,
                       VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_R};
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev, &ivci, nullptr, &fontView_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &sci, nullptr, &fontSampler_) != VK_SUCCESS) return false;

    VkBufferCreateInfo sbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbci.size = kAtlasW * kAtlasH;
    sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo saci = g_backend.HostAlloc();
    VmaAllocationInfo sinfo{};
    VkBuffer stgBuf = VK_NULL_HANDLE;
    VmaAllocation stgAlloc = VK_NULL_HANDLE;
    if (vmaCreateBuffer(alloc, &sbci, &saci, &stgBuf, &stgAlloc, &sinfo) != VK_SUCCESS)
        return false;
    BuildFontAtlasData((uint8_t*)sinfo.pMappedData, kAtlasW, kAtlasH);

    struct UploadCtx {
        VkImage img;
        VkBuffer buf;
    } ctx{fontImage_, stgBuf};
    g_backend.ImmediateSubmit(
        [](VkCommandBuffer cmd, void* user) {
            auto* c = (UploadCtx*)user;
            VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b1.image = c->img;
            b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                 nullptr, 1, &b1);
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {kAtlasW, kAtlasH, 1};
            vkCmdCopyBufferToImage(cmd, c->buf, c->img,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.image = c->img;
            b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b2);
        },
        &ctx);
    vmaDestroyBuffer(alloc, stgBuf, stgAlloc);

    VkDescriptorImageInfo dii{fontSampler_, fontView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = descSet_;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wds.pImageInfo = &dii;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    VkBufferCreateInfo vbci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    vbci.size = kSlotBytes * 2;
    vbci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo vaci = g_backend.HostAlloc();
    VmaAllocationInfo vinfo{};
    if (vmaCreateBuffer(alloc, &vbci, &vaci, &vb_, &vbAlloc_, &vinfo) != VK_SUCCESS)
        return false;
    vbMapped_ = vinfo.pMappedData;

    initialized_ = true;
    return true;
}

void FpsOverlay::Shutdown() {
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (!dev) return;
    if (pipeline_) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    pipeFormat_ = VK_FORMAT_UNDEFINED;
    if (vb_ && vbAlloc_) {
        vmaDestroyBuffer(alloc, vb_, vbAlloc_);
        vb_ = VK_NULL_HANDLE;
        vbAlloc_ = VK_NULL_HANDLE;
        vbMapped_ = nullptr;
    }
    if (fontSampler_) {
        vkDestroySampler(dev, fontSampler_, nullptr);
        fontSampler_ = VK_NULL_HANDLE;
    }
    if (fontView_) {
        vkDestroyImageView(dev, fontView_, nullptr);
        fontView_ = VK_NULL_HANDLE;
    }
    if (fontImage_ && fontAlloc_) {
        vmaDestroyImage(alloc, fontImage_, fontAlloc_);
        fontImage_ = VK_NULL_HANDLE;
        fontAlloc_ = VK_NULL_HANDLE;
    }
    if (descPool_) {
        vkDestroyDescriptorPool(dev, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        descSet_ = VK_NULL_HANDLE;
    }
    if (pipeLayout_) {
        vkDestroyPipelineLayout(dev, pipeLayout_, nullptr);
        pipeLayout_ = VK_NULL_HANDLE;
    }
    if (setLayout_) {
        vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
        setLayout_ = VK_NULL_HANDLE;
    }
    if (fs_) {
        vkDestroyShaderModule(dev, fs_, nullptr);
        fs_ = VK_NULL_HANDLE;
    }
    if (vs_) {
        vkDestroyShaderModule(dev, vs_, nullptr);
        vs_ = VK_NULL_HANDLE;
    }
    initialized_ = false;
}

bool FpsOverlay::EnsurePipeline(VkFormat format) {
    if (pipeline_ && pipeFormat_ == format) return true;
    VkDevice dev = g_backend.Device();
    if (!dev || !vs_ || !fs_) return false;
    if (pipeline_) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vib{0, sizeof(OverlayVert), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[3]{
        {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32_SFLOAT, 8},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions = via;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = 2;
    dynInfo.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &format;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rci;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dynInfo;
    pci.layout = pipeLayout_;

    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_) != VK_SUCCESS) {
        pipeline_ = VK_NULL_HANDLE;
        return false;
    }
    pipeFormat_ = format;
    return true;
}

uint32_t FpsOverlay::BuildQuads(uint32_t frameSlot, VkExtent2D extent) {
    const uint32_t s = frameSlot & 1u;
    if (cachedFps_[s] == displayFps_ && cachedW_[s] == extent.width &&
        cachedH_[s] == extent.height && cachedVerts_[s] > 0) {
        return cachedVerts_[s];
    }

    char text[16];
    const int len = snprintf(text, sizeof(text), "FPS: %u", displayFps_);
    if (len <= 0 || len >= (int)kMaxQuads) return 0;

    const float scale = (extent.height >= 1200) ? 3.0f : 2.0f;
    const float charW = 5.0f * scale;
    const float charH = 7.0f * scale;
    const float adv = 6.0f * scale;
    const float padX = 3.0f * scale;
    const float padY = 2.0f * scale;
    const float textW = (float)len * adv - scale;
    const float boxW = textW + padX * 2.0f;
    const float boxH = charH + padY * 2.0f;
    const float bx = (float)extent.width - boxW - 12.0f;
    const float by = 10.0f;
    const float invW = 1.0f / (float)extent.width;
    const float invH = 1.0f / (float)extent.height;

    OverlayVert* dst = (OverlayVert*)((uint8_t*)vbMapped_ + (size_t)s * kSlotBytes);
    OverlayVert* begin = dst;

    const float solidU = (15.0f * (float)kCellW + 3.5f) / (float)kAtlasW;
    const float solidV = 3.5f / (float)kAtlasH;
    PushQuad(dst, bx, by, bx + boxW, by + boxH, solidU, solidV, solidU, solidV,
             invW, invH, 0.04f, 0.04f, 0.06f, 0.72f);

    float cr = 0.25f, cg = 1.0f, cb = 0.35f;
    if (displayFps_ < 30) {
        cr = 1.0f;
        cg = 0.25f;
        cb = 0.25f;
    } else if (displayFps_ < 60) {
        cr = 1.0f;
        cg = 0.85f;
        cb = 0.25f;
    }

    const float v0 = 1.0f / (float)kAtlasH;
    const float v1 = 8.0f / (float)kAtlasH;
    float cx = bx + padX;
    const float cy = by + padY;
    for (int i = 0; i < len; ++i) {
        const int slot = GlyphSlotFor(text[i]);
        if (slot != 14) {
            const float u0 = (float)(slot * kCellW + 1) / (float)kAtlasW;
            const float u1 = (float)(slot * kCellW + 6) / (float)kAtlasW;
            PushQuad(dst, cx, cy, cx + charW, cy + charH, u0, v0, u1, v1,
                     invW, invH, cr, cg, cb, 1.0f);
        }
        cx += adv;
    }

    cachedFps_[s] = displayFps_;
    cachedW_[s] = extent.width;
    cachedH_[s] = extent.height;
    cachedVerts_[s] = (uint32_t)(dst - begin);
    return cachedVerts_[s];
}

void FpsOverlay::TickAndDraw(VkCommandBuffer cmd, VkImage targetImage,
                             VkImageView targetView, VkFormat format,
                             VkExtent2D extent, uint32_t frameSlot) {
    if (!initialized_ || !vbMapped_ || !targetView || !extent.width || !extent.height)
        return;

    ++accumFrames_;
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    const double elapsedMs =
        (double)(now.QuadPart - lastTick_.QuadPart) * 1000.0 / (double)freq_.QuadPart;
    if (elapsedMs >= 250.0) {
        displayFps_ = (uint32_t)((double)accumFrames_ * 1000.0 / elapsedMs + 0.5);
        accumFrames_ = 0;
        lastTick_ = now;
    }

    if (!EnsurePipeline(format)) return;
    const uint32_t vertCount = BuildQuads(frameSlot, extent);
    if (!vertCount) return;

    g_backend.TransitionImage(targetImage, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = targetView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, extent};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout_, 0, 1,
                            &descSet_, 0, nullptr);
    const VkDeviceSize vbOff = (VkDeviceSize)(frameSlot & 1u) * kSlotBytes;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb_, &vbOff);
    VkViewport vp{0.0f, 0.0f, (float)extent.width, (float)extent.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, vertCount, 1, 0, 0);
    vkCmdEndRendering(cmd);
}
