#include "d3d9_internal.h"
#include "stud_shadows.h"
#include <intrin.h>

using namespace d3d9;

VkSampler GetSamplerForStage(uint32_t stage, bool depthCompare) {
    if (stage >= 16) return g_backend.DefaultSampler();
    SamplerStageState& s = g_st.samplers[stage];

    uint32_t borderClass = 0;
    if (s.borderColor != 0) {
        const uint32_t r = (s.borderColor >> 16) & 0xFFu;
        const uint32_t g = (s.borderColor >> 8) & 0xFFu;
        const uint32_t b = s.borderColor & 0xFFu;
        borderClass = (r + g + b >= 384u) ? 1u : 2u;
    }

    uint32_t key = (s.addressU & 7);
    key |= (s.addressV & 7) << 3;
    key |= (s.addressW & 7) << 6;
    key |= (s.magFilter & 3) << 9;
    key |= (s.minFilter & 3) << 11;
    key |= (s.mipFilter & 3) << 13;
    key |= (s.maxAnisotropy & 0x1F) << 15;
    key |= borderClass << 20;
    key |= (depthCompare ? 1u : 0u) << 22;

    if (!s.samplerDirty && s.cachedSampler && s.lastKey == key)
        return s.cachedSampler;

    auto it = g_samplerCache.find(key);
    if (it != g_samplerCache.end()) {
        s.cachedSampler = it->second;
        s.lastKey = key;
        s.samplerDirty = false;
        return it->second;
    }

    auto addressMode = [](uint32_t mode) {
        switch (mode) {
            case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case 3: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    };

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = (s.magFilter == 1) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.minFilter = (s.minFilter == 1) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    sci.mipmapMode = (s.mipFilter == 1) ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                        : VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = addressMode(s.addressU);
    sci.addressModeV = addressMode(s.addressV);
    sci.addressModeW = addressMode(s.addressW);
    sci.mipLodBias = s.mipLodBias;
    sci.anisotropyEnable =
        (s.minFilter == 3 || s.magFilter == 3 || s.maxAnisotropy > 1) ? VK_TRUE
                                                                     : VK_FALSE;
    sci.maxAnisotropy =
        sci.anisotropyEnable ? (float)(s.maxAnisotropy ? s.maxAnisotropy : 1)
                             : 1.0f;
    sci.compareEnable = depthCompare ? VK_TRUE : VK_FALSE;
    sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    sci.minLod = 0.0f;
    sci.maxLod = (s.mipFilter == 0) ? 0.25f : 1000.0f;
    sci.borderColor = (borderClass == 0)
                          ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                          : (borderClass == 1
                                 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                 : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);

    VkSampler samp = VK_NULL_HANDLE;
    if (vkCreateSampler(g_backend.Device(), &sci, nullptr, &samp) == VK_SUCCESS) {
        g_samplerCache[key] = samp;
        s.cachedSampler = samp;
        s.lastKey = key;
        s.samplerDirty = false;
        return samp;
    }
    return g_backend.DefaultSampler();
}

bool g_pipeDirty = true;
bool g_dynRsDirty = true;
bool g_stencilDirty = true;
bool g_vpDirty = true;
bool g_scDirty = true;

namespace {

PipelineKey g_curPipeKey{};
uint32_t g_lastVsOff = 0xFFFFFFFFu;
uint32_t g_lastPsOff = 0xFFFFFFFFu;
uint32_t g_lastVsRegs = 0;
uint32_t g_lastPsRegs = 0;
VkPipeline g_lastPipe = VK_NULL_HANDLE;
VkBuffer g_lastBoundVb[16] = {};
VkDeviceSize g_lastBoundOffset[16] = {};
VkDeviceSize g_lastBoundStride[16] = {};
uint16_t g_lastBoundStreamMask = 0;
bool g_vbInitializedThisFrame = false;
VkBuffer g_lastBoundIb = VK_NULL_HANDLE;
uint32_t g_lastBoundIbOffset = 0xFFFFFFFFu;
VkIndexType g_lastBoundIndexType = VK_INDEX_TYPE_UINT16;
VkDescriptorSet g_lastBoundSet = VK_NULL_HANDLE;
uint32_t g_lastDyn[2] = {0xFFFFFFFFu, 0xFFFFFFFFu};
VkRect2D g_lastScissor{};
VkViewport g_lastViewport{};

struct DynStateCache {
    bool valid = false;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkBool32 depthTest = VK_FALSE;
    VkBool32 depthWrite = VK_FALSE;
    VkCompareOp depthOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkBool32 stencilTest = VK_FALSE;
    VkStencilOp fFail = VK_STENCIL_OP_KEEP;
    VkStencilOp fPass = VK_STENCIL_OP_KEEP;
    VkStencilOp fZFail = VK_STENCIL_OP_KEEP;
    VkCompareOp fFunc = VK_COMPARE_OP_ALWAYS;
    VkStencilOp bFail = VK_STENCIL_OP_KEEP;
    VkStencilOp bPass = VK_STENCIL_OP_KEEP;
    VkStencilOp bZFail = VK_STENCIL_OP_KEEP;
    VkCompareOp bFunc = VK_COMPARE_OP_ALWAYS;
    uint32_t stencilMask = 0;
    uint32_t stencilWrite = 0;
    uint32_t stencilRef = 0;
};
DynStateCache g_dynState;

VkCullModeFlags ToVkCull(uint32_t c) {
    switch (c) {
        case 2: return VK_CULL_MODE_FRONT_BIT;
        case 3: return VK_CULL_MODE_BACK_BIT;
        default: return VK_CULL_MODE_NONE;
    }
}

VkStencilOp ToVkStencilOp(uint32_t op) {
    if (op >= 1 && op <= 8) return (VkStencilOp)(op - 1);
    return VK_STENCIL_OP_KEEP;
}

VkCompareOp ToVkCompareOp(uint32_t fn, VkCompareOp fallback) {
    if (fn >= 1 && fn <= 8) return (VkCompareOp)(fn - 1);
    return fallback;
}

uint32_t PrimitiveIndexCount(uint32_t topo, uint32_t prims) {
    switch (topo) {
        case 1: return prims;
        case 2: return prims * 2;
        case 3: return prims + 1;
        case 5: return prims + 2;
        case 6: return prims + 2;
        default: return prims * 3;
    }
}

}

void InvalidateDescriptorBinding() {
    g_lastBoundSet = VK_NULL_HANDLE;
    g_lastDyn[0] = 0xFFFFFFFFu;
    g_lastDyn[1] = 0xFFFFFFFFu;
}

void InvalidateViewportScissor() {
    g_lastScissor = VkRect2D{};
    g_lastViewport = VkViewport{};
    g_vpDirty = true;
    g_scDirty = true;
}

void ResetDrawCache() {
    g_lastPipe = VK_NULL_HANDLE;
    memset(g_lastBoundVb, 0, sizeof(g_lastBoundVb));
    memset(g_lastBoundOffset, 0, sizeof(g_lastBoundOffset));
    for (int i = 0; i < 16; ++i) g_lastBoundStride[i] = (VkDeviceSize)0xFFFFFFFFu;
    g_lastBoundStreamMask = 0;
    g_vbInitializedThisFrame = false;
    g_lastBoundIb = VK_NULL_HANDLE;
    g_lastBoundIbOffset = 0xFFFFFFFFu;
    g_lastBoundIndexType = VK_INDEX_TYPE_UINT16;
    g_lastVsOff = 0xFFFFFFFFu;
    g_lastPsOff = 0xFFFFFFFFu;
    g_lastVsRegs = 0;
    g_lastPsRegs = 0;
    g_dynState.valid = false;
    g_pipeDirty = true;
    g_dynRsDirty = true;
    g_stencilDirty = true;
    InvalidateViewportScissor();
    InvalidateDescriptorBinding();
}

void TranslateDraw(uint32_t topo, int32_t baseVertex, uint32_t startIndex,
                   uint32_t primCount, bool indexed) {
    if (!g_backendReady) return;
    if (!g_frameActive && !StartNewFrame()) return;
    if (!g_st.vsHash || !g_st.psHash) return;

    if (g_st.targetsDirty) {
        g_st.targetsDirty = false;
        ApplyTargets();
    }

    VertexDecl* decl = g_resources.Decl(g_st.decl);
    const uint16_t activeMask = decl ? (uint16_t)(decl->streamMask & g_st.streamMask) : 0;
    const uint32_t instanceCount = g_st.instanceCount;

    const uint32_t psDrefMask =
        g_st.shadowCompareMask & g_shaders.GetPsTexldpMask(g_st.psHash);

    VkPipeline pipe = g_lastPipe;
    if (g_pipeDirty || pipe == VK_NULL_HANDLE || g_curPipeKey.topology != topo ||
        g_curPipeKey.activeStreams != activeMask ||
        g_curPipeKey.psDrefMask != psDrefMask) {
        PipelineKey k{};
        k.vsHash = g_st.vsHash;
        k.psHash = g_st.psHash;
        k.declKey = g_st.decl ? g_st.decl->index : 0;
        k.blendEnable = g_st.rs[RS_ALPHABLENDENABLE] ? 1 : 0;
        k.srcBlend = g_st.rs[RS_SRCBLEND] ? g_st.rs[RS_SRCBLEND] : 2;
        k.dstBlend = g_st.rs[RS_DESTBLEND] ? g_st.rs[RS_DESTBLEND] : 1;
        k.blendOp = g_st.rs[RS_BLENDOP] ? g_st.rs[RS_BLENDOP] : 1;
        if (g_st.rs[RS_SEPARATEALPHABLENDENABLE]) {
            k.srcBlendAlpha = g_st.rs[RS_SRCBLENDALPHA] ? g_st.rs[RS_SRCBLENDALPHA] : 2;
            k.dstBlendAlpha = g_st.rs[RS_DESTBLENDALPHA] ? g_st.rs[RS_DESTBLENDALPHA] : 1;
            k.blendOpAlpha = g_st.rs[RS_BLENDOPALPHA] ? g_st.rs[RS_BLENDOPALPHA] : 1;
        } else {
            k.srcBlendAlpha = k.srcBlend;
            k.dstBlendAlpha = k.dstBlend;
            k.blendOpAlpha = k.blendOp;
        }
        k.alphaTest = g_st.rs[RS_ALPHATESTENABLE] ? 1 : 0;
        if (k.alphaTest) {
            k.alphaRef = g_st.rs[RS_ALPHAREF] & 0xFF;
            k.alphaFunc = g_st.rs[RS_ALPHAFUNC] ? g_st.rs[RS_ALPHAFUNC] : 8;
        } else {
            k.alphaRef = 0;
            k.alphaFunc = 8;
        }
        k.fillMode = g_st.rs[RS_FILLMODE] ? g_st.rs[RS_FILLMODE] : 3;
        k.colorWrite = g_st.colorWriteSet ? g_st.rs[RS_COLORWRITEENABLE] : 0xF;
        k.topology = topo;
        k.colorCount = g_backend.ColorTargetCount();
        for (uint32_t i = 0; i < 4; ++i) k.colorFormats[i] = g_backend.ColorFormatAt(i);
        k.depthFormat = g_backend.DepthFormat();
        k.activeStreams = activeMask;
        k.instancedMask = g_st.instancedMask;
        k.psDrefMask = psDrefMask;
        g_curPipeKey = k;
        pipe = g_psos.Get(k, decl);
        g_pipeDirty = false;
    }
    if (!pipe) return;
    ++g_drawCalls;
    if (g_studShadows.NeedsFlush()) {
        FakeSurface* sc = g_studShadows.SceneColor();
        const uint16_t checkMask = g_st.rtStageMask & g_st.texMask;
        for (uint32_t m = checkMask; m; ) {
            unsigned long s = 0;
            _BitScanForward(&s, m);
            m &= m - 1u;
            GpuTexture* t = g_resources.TextureOf(g_st.textures[s]);
            if (t && t->image == sc->image) {
                if (g_studShadows.FlushScene()) {
                    ResetDrawCache();
                }
                break;
            }
        }
    }
    g_studShadows.TrackTargetDraw(g_st.colorTargets[0], g_st.depthTarget);
    if (!indexed && topo == 4 && primCount == 2 && activeMask == 0x1 &&
        g_st.streams[0].stride == 28 &&
        g_st.vsHash == 0xda91b812e150759fULL &&
        g_st.psHash == 0xa2655343d57e2da7ULL &&
        g_curPipeKey.colorWrite != 0) {
        GpuTexture* gt5 = g_resources.TextureOf(g_st.textures[5]);
        if (gt5 && gt5->width == 64 && gt5->height == 64) {
            g_studShadows.CollectStud(g_st.vsConst, g_st.colorTargets[0],
                                      g_st.depthTarget);
        }
    }
    const uint32_t curFrame = g_backend.FrameNumber();
    bool needUploadFlush = false;

    const uint16_t rtBound = g_st.rtStageMask & g_st.texMask;
    for (uint32_t m = rtBound; m; ) {
        unsigned long s = 0;
        _BitScanForward(&s, m);
        m &= m - 1u;
        GpuTexture* t = g_resources.TextureOf(g_st.textures[s]);
        if (t && t->image) {
            bool isCurrentTarget =
                (g_st.depthTarget && g_st.depthTarget->image == t->image);
            for (uint32_t rt = 0; rt < 4 && !isCurrentTarget; ++rt) {
                if (g_st.colorTargets[rt] &&
                    g_st.colorTargets[rt]->image == t->image) {
                    isCurrentTarget = true;
                }
            }
            if (!isCurrentTarget) {
                VkImageAspectFlags aspect =
                    t->depth ? t->aspect
                             : (VkImageAspectFlags)VK_IMAGE_ASPECT_COLOR_BIT;
                if (t->depth &&
                    (t->format == VK_FORMAT_D24_UNORM_S8_UINT ||
                     t->format == VK_FORMAT_D32_SFLOAT_S8_UINT)) {
                    aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
                }
                g_backend.TransitionImage(
                    t->image, aspect, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }
    }

    const bool hasPending = g_resources.HasPendingUploads();
    if (hasPending) {
        for (uint32_t m = g_st.texMask; m; ) {
            unsigned long s = 0;
            _BitScanForward(&s, m);
            m &= m - 1u;
            GpuTexture* t = g_resources.TextureOf(g_st.textures[s]);
            if (t && t->uploadPending) {
                needUploadFlush = true;
                break;
            }
        }
    }

    GpuBuffer* streamBufs[16]{};
    for (uint32_t m = activeMask; m; ) {
        unsigned long s = 0;
        _BitScanForward(&s, m);
        m &= m - 1u;
        GpuBuffer* b = g_resources.Buffer(g_st.streams[s].vb);
        streamBufs[s] = b;
        if (hasPending && b && b->uploadPending) needUploadFlush = true;
    }
    GpuBuffer* ibBuf = indexed ? g_resources.Buffer(g_st.indices) : nullptr;
    if (indexed && !ibBuf) return;
    if (hasPending && ibBuf && ibBuf->uploadPending) needUploadFlush = true;

    if (needUploadFlush) {
        g_resources.FlushPendingUploads(true);
    }

    g_backend.FlushTextures();
    g_backend.EnsureRendering();
    VkCommandBuffer cmd = g_backend.Cmd();

    const ShaderConstInfo vsCi = g_st.vsConstInfo;
    uint32_t needVsRegs = vsCi.usesRelAddr
                              ? (vsCi.regCount > g_st.vsHighReg ? vsCi.regCount : g_st.vsHighReg)
                              : vsCi.regCount;
    if (needVsRegs > ConstantRing::kRegisters) needVsRegs = ConstantRing::kRegisters;
    if (g_st.vsConstDirty || g_lastVsOff == 0xFFFFFFFFu || needVsRegs > g_lastVsRegs) {
        const uint32_t bytes = needVsRegs * 16u;
        uint32_t off = g_constRing.AllocateBytes(bytes);
        if (off != 0xFFFFFFFFu) {
            g_constRing.Write(off, g_st.vsConst, bytes);
            g_lastVsOff = off;
            g_lastVsRegs = needVsRegs;
        } else if (g_lastVsOff == 0xFFFFFFFFu) {
            g_lastVsOff = 0;
            g_lastVsRegs = 0;
        }
        g_st.vsConstDirty = false;
    }
    const ShaderConstInfo psCi = g_st.psConstInfo;
    uint32_t needPsRegs = psCi.usesRelAddr
                              ? (psCi.regCount > g_st.psHighReg ? psCi.regCount : g_st.psHighReg)
                              : psCi.regCount;
    if (needPsRegs > ConstantRing::kRegisters) needPsRegs = ConstantRing::kRegisters;
    if (g_st.psConstDirty || g_lastPsOff == 0xFFFFFFFFu || needPsRegs > g_lastPsRegs) {
        const uint32_t bytes = needPsRegs * 16u;
        uint32_t off = g_constRing.AllocateBytes(bytes);
        if (off != 0xFFFFFFFFu) {
            g_constRing.Write(off, g_st.psConst, bytes);
            g_lastPsOff = off;
            g_lastPsRegs = needPsRegs;
        } else if (g_lastPsOff == 0xFFFFFFFFu) {
            g_lastPsOff = 0;
            g_lastPsRegs = 0;
        }
        g_st.psConstDirty = false;
    }
    uint32_t dyn[2] = {g_lastVsOff, g_lastPsOff};
    VkDescriptorSet set = g_backend.DescriptorSet();

    if (g_lastPipe != pipe) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        g_lastPipe = pipe;
    }

    if (!g_vbInitializedThisFrame) {
        const VkBuffer dummy = g_backend.DummyVertexBuffer();
        VkBuffer vbs[16];
        VkDeviceSize offs[16];
        VkDeviceSize strides[16];
        for (uint32_t s = 0; s < 16; ++s) {
            GpuBuffer* b = streamBufs[s];
            if (b) {
                b->drawnSinceLock = true;
                b->lastDrawFrame = curFrame;
            }
            vbs[s] = b ? b->buffer : dummy;
            offs[s] = b ? (VkDeviceSize)(b->baseOffset + g_st.streams[s].offset) : 0;
            strides[s] = b ? (VkDeviceSize)g_st.streams[s].stride : 0;
            g_lastBoundVb[s] = vbs[s];
            g_lastBoundOffset[s] = offs[s];
            g_lastBoundStride[s] = strides[s];
        }
        vkCmdBindVertexBuffers2(cmd, 0, 16, vbs, offs, nullptr, strides);
        g_vbInitializedThisFrame = true;
        g_lastBoundStreamMask = activeMask;
    } else {
        const VkBuffer dummy = g_backend.DummyVertexBuffer();
        for (uint32_t m = (uint32_t)(activeMask | g_lastBoundStreamMask); m; ) {
            unsigned long s = 0;
            _BitScanForward(&s, m);
            m &= m - 1u;
            GpuBuffer* b = streamBufs[s];
            if (b) {
                b->drawnSinceLock = true;
                b->lastDrawFrame = curFrame;
            }
            VkBuffer vb = b ? b->buffer : dummy;
            VkDeviceSize off =
                b ? (VkDeviceSize)(b->baseOffset + g_st.streams[s].offset) : 0;
            VkDeviceSize stride = b ? (VkDeviceSize)g_st.streams[s].stride : 0;
            if (!vb) {
                g_lastBoundVb[s] = VK_NULL_HANDLE;
                continue;
            }
            if (g_lastBoundVb[s] != vb || g_lastBoundOffset[s] != off ||
                g_lastBoundStride[s] != stride) {
                vkCmdBindVertexBuffers2(cmd, s, 1, &vb, &off, nullptr, &stride);
                g_lastBoundVb[s] = vb;
                g_lastBoundOffset[s] = off;
                g_lastBoundStride[s] = stride;
            }
        }
        g_lastBoundStreamMask = activeMask;
    }

    if (indexed) {
        ibBuf->drawnSinceLock = true;
        ibBuf->lastDrawFrame = curFrame;
        VkIndexType itype = (g_st.indexFormat == 102 ? VK_INDEX_TYPE_UINT32
                                                     : VK_INDEX_TYPE_UINT16);
        if (g_lastBoundIb != ibBuf->buffer || g_lastBoundIndexType != itype ||
            g_lastBoundIbOffset != ibBuf->baseOffset) {
            vkCmdBindIndexBuffer(cmd, ibBuf->buffer, ibBuf->baseOffset, itype);
            g_lastBoundIbOffset = ibBuf->baseOffset;
            g_lastBoundIb = ibBuf->buffer;
            g_lastBoundIndexType = itype;
        }
    }

    if (!g_dynState.valid || g_dynRsDirty) {
        const VkCullModeFlags cullMode =
            ToVkCull(g_st.rs[RS_CULLMODE] ? g_st.rs[RS_CULLMODE] : 3);
        const VkBool32 depthTest = g_st.rs[RS_ZENABLE] ? VK_TRUE : VK_FALSE;
        const VkBool32 depthWrite = g_st.rs[RS_ZWRITEENABLE] ? VK_TRUE : VK_FALSE;
        const VkCompareOp depthOp =
            ToVkCompareOp(g_st.rs[RS_ZFUNC] ? g_st.rs[RS_ZFUNC] : 4,
                          VK_COMPARE_OP_LESS_OR_EQUAL);
        const VkBool32 stencilTest = g_st.rs[RS_STENCILENABLE] ? VK_TRUE : VK_FALSE;

        if (!g_dynState.valid || g_dynState.cullMode != cullMode) {
            vkCmdSetCullMode(cmd, cullMode);
            g_dynState.cullMode = cullMode;
        }
        if (!g_dynState.valid || g_dynState.depthTest != depthTest) {
            vkCmdSetDepthTestEnable(cmd, depthTest);
            g_dynState.depthTest = depthTest;
        }
        if (!g_dynState.valid || g_dynState.depthWrite != depthWrite) {
            vkCmdSetDepthWriteEnable(cmd, depthWrite);
            g_dynState.depthWrite = depthWrite;
        }
        if (!g_dynState.valid || g_dynState.depthOp != depthOp) {
            vkCmdSetDepthCompareOp(cmd, depthOp);
            g_dynState.depthOp = depthOp;
        }
        if (!g_dynState.valid || g_dynState.stencilTest != stencilTest) {
            vkCmdSetStencilTestEnable(cmd, stencilTest);
            g_dynState.stencilTest = stencilTest;
        }
        g_dynRsDirty = false;
    }
    if (!g_dynState.valid || (g_dynState.stencilTest && g_stencilDirty)) {
        const VkStencilOp fFail =
            ToVkStencilOp(g_st.rs[RS_STENCILFAIL] ? g_st.rs[RS_STENCILFAIL] : 1);
        const VkStencilOp fPass =
            ToVkStencilOp(g_st.rs[RS_STENCILPASS] ? g_st.rs[RS_STENCILPASS] : 1);
        const VkStencilOp fZFail =
            ToVkStencilOp(g_st.rs[RS_STENCILZFAIL] ? g_st.rs[RS_STENCILZFAIL] : 1);
        const VkCompareOp fFunc =
            ToVkCompareOp(g_st.rs[RS_STENCILFUNC] ? g_st.rs[RS_STENCILFUNC] : 8,
                          VK_COMPARE_OP_ALWAYS);
        VkStencilOp bFail = fFail, bPass = fPass, bZFail = fZFail;
        VkCompareOp bFunc = fFunc;
        if (g_st.rs[RS_TWOSIDEDSTENCILMODE]) {
            bFail = ToVkStencilOp(
                g_st.rs[RS_CCW_STENCILFAIL] ? g_st.rs[RS_CCW_STENCILFAIL] : 1);
            bPass = ToVkStencilOp(
                g_st.rs[RS_CCW_STENCILPASS] ? g_st.rs[RS_CCW_STENCILPASS] : 1);
            bZFail = ToVkStencilOp(
                g_st.rs[RS_CCW_STENCILZFAIL] ? g_st.rs[RS_CCW_STENCILZFAIL] : 1);
            bFunc = ToVkCompareOp(
                g_st.rs[RS_CCW_STENCILFUNC] ? g_st.rs[RS_CCW_STENCILFUNC] : 8,
                VK_COMPARE_OP_ALWAYS);
        }
        const bool fDiff = !g_dynState.valid || g_dynState.fFail != fFail ||
                           g_dynState.fPass != fPass || g_dynState.fZFail != fZFail ||
                           g_dynState.fFunc != fFunc;
        const bool bDiff = !g_dynState.valid || g_dynState.bFail != bFail ||
                           g_dynState.bPass != bPass || g_dynState.bZFail != bZFail ||
                           g_dynState.bFunc != bFunc;
        if (fDiff || bDiff) {
            if (fFail == bFail && fPass == bPass && fZFail == bZFail &&
                fFunc == bFunc) {
                vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, fFail,
                                  fPass, fZFail, fFunc);
            } else {
                if (fDiff)
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, fFail,
                                      fPass, fZFail, fFunc);
                if (bDiff)
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, bFail,
                                      bPass, bZFail, bFunc);
            }
            g_dynState.fFail = fFail;
            g_dynState.fPass = fPass;
            g_dynState.fZFail = fZFail;
            g_dynState.fFunc = fFunc;
            g_dynState.bFail = bFail;
            g_dynState.bPass = bPass;
            g_dynState.bZFail = bZFail;
            g_dynState.bFunc = bFunc;
        }
        const uint32_t stMask = g_st.rs[RS_STENCILMASK];
        const uint32_t stWrite = g_st.rs[RS_STENCILWRITEMASK];
        const uint32_t stRef = g_st.rs[RS_STENCILREF];
        if (!g_dynState.valid || g_dynState.stencilMask != stMask) {
            vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                       stMask);
            g_dynState.stencilMask = stMask;
        }
        if (!g_dynState.valid || g_dynState.stencilWrite != stWrite) {
            vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                     stWrite);
            g_dynState.stencilWrite = stWrite;
        }
        if (!g_dynState.valid || g_dynState.stencilRef != stRef) {
            vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                     stRef);
            g_dynState.stencilRef = stRef;
        }
        g_stencilDirty = false;
    }
    g_dynState.valid = true;

    if (g_scDirty) {
        VkRect2D sc;
        if (g_st.rs[RS_SCISSORTESTENABLE] && g_st.scissorValid) {
            int32_t l = (int32_t)g_st.scissor.left;
            int32_t t = (int32_t)g_st.scissor.top;
            int32_t r = (int32_t)g_st.scissor.right;
            int32_t b = (int32_t)g_st.scissor.bottom;
            const VkExtent2D ext = g_backend.TargetExtent();
            if (l < 0) l = 0;
            if (t < 0) t = 0;
            if (r > (int32_t)ext.width) r = (int32_t)ext.width;
            if (b > (int32_t)ext.height) b = (int32_t)ext.height;
            sc.offset = {l, t};
            sc.extent = {(uint32_t)(r > l ? r - l : 0),
                         (uint32_t)(b > t ? b - t : 0)};
        } else {
            sc.offset = {0, 0};
            sc.extent = g_backend.TargetExtent();
        }
        if (memcmp(&g_lastScissor, &sc, sizeof(VkRect2D)) != 0) {
            vkCmdSetScissor(cmd, 0, 1, &sc);
            g_lastScissor = sc;
        }
        g_scDirty = false;
    }

    if (g_vpDirty) {
        VkViewport curVp;
        if (g_st.viewportValid && g_st.viewport.width && g_st.viewport.height) {
            curVp = {(float)g_st.viewport.x,
                     (float)g_st.viewport.y + (float)g_st.viewport.height,
                     (float)g_st.viewport.width,
                     -(float)g_st.viewport.height,
                     g_st.viewport.minZ,
                     g_st.viewport.maxZ};
        } else {
            const VkExtent2D ext = g_backend.TargetExtent();
            curVp = {0.0f, (float)ext.height, (float)ext.width,
                     -(float)ext.height, 0.0f, 1.0f};
        }
        if (memcmp(&g_lastViewport, &curVp, sizeof(VkViewport)) != 0) {
            vkCmdSetViewport(cmd, 0, 1, &curVp);
            g_lastViewport = curVp;
        }
        g_vpDirty = false;
    }

    if (g_lastBoundSet != set || g_lastDyn[0] != dyn[0] || g_lastDyn[1] != dyn[1]) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                g_backend.PipeLayout(), 0, 1, &set, 2, dyn);
        g_lastBoundSet = set;
        g_lastDyn[0] = dyn[0];
        g_lastDyn[1] = dyn[1];
    }

    if (indexed) {
        vkCmdDrawIndexed(cmd, PrimitiveIndexCount(topo, primCount), instanceCount,
                         startIndex, baseVertex, 0);
    } else {
        vkCmdDraw(cmd, PrimitiveIndexCount(topo, primCount), instanceCount,
                  startIndex, 0);
    }
}

ImageRef MakeImageRef(FakeSurface* s) {
    ImageRef r;
    if (!s) return r;
    if (s->isBackbuffer) {
        const VkExtent2D e = g_backend.SwapchainExtent();
        r.swapchain = true;
        r.depth = s->isDepth != 0;
        r.width = e.width;
        r.height = e.height;
        r.format = r.depth ? g_backend.DepthFormat() : g_backend.ColorFormat();
        return r;
    }
    r.image = s->image;
    r.view = s->view;
    r.stencilView = s->stencilView;
    r.width = s->width;
    r.height = s->height;
    // A depth-format texture surface reports isDepth 0 because it was created
    // as a texture, but D3D9 still binds it through SetDepthStencilSurface.
    r.depth = s->isDepth != 0 || ResourceTable::IsDepthFormatD3D(s->format);
    r.format = r.depth ? ResourceTable::ToVkDepthFormat(s->format, nullptr)
                       : ResourceTable::ToVkFormat(s->format);
    return r;
}

void ApplyTargets() {
    if (!g_backendReady) return;
    ImageRef colors[4];
    uint32_t n = 0;
    for (uint32_t i = 0; i < 4; ++i)
        if (g_st.colorTargets[i]) colors[n++] = MakeImageRef(g_st.colorTargets[i]);
    if (n == 0) {
        if (g_backend.BindSwapchainTargets()) {
            g_pipeDirty = true;
            InvalidateViewportScissor();
        }
        return;
    }

    ImageRef depth;
    bool haveDepth = true;
    if (g_st.depthTarget) {
        depth = MakeImageRef(g_st.depthTarget);
    } else {
        const VkExtent2D e = g_backend.SwapchainExtent();
        depth.swapchain = true;
        depth.depth = true;
        depth.format = g_backend.DepthFormat();
        depth.width = e.width;
        depth.height = e.height;
    }

    if (depth.width != colors[0].width || depth.height != colors[0].height) {
        ImageRef sized =
            g_backend.DepthFor(colors[0].width, colors[0].height, depth.format);
        if (sized.image)
            depth = sized;
        else
            haveDepth = false;
    }

    if (g_backend.SetRenderTargets(colors, n, haveDepth ? &depth : nullptr)) {
        g_pipeDirty = true;
        InvalidateViewportScissor();
    }
}
