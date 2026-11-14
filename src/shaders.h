#pragma once

#include "vk_common.h"

// Captures every D3D9 SM3 token stream the engine creates, keys it by XXH64,
// JIT-compiles it to SPIR-V in-process and caches the result on disk so the
// next launch starts instantly.
struct ShaderConstInfo {
    uint16_t regCount = 257;
    bool usesRelAddr = false;
};

class ShaderDB {
public:
    void Init(const char* directory);

    uint64_t Capture(bool pixel, const void* code, size_t size);

    VkShaderModule Get(uint64_t hash, bool pixel);
    VkShaderModule GetPixel(uint64_t hash, uint32_t drefMask);
    uint32_t GetVsInputMask(uint64_t hash) const;
    uint32_t GetPsTexldpMask(uint64_t hash) const;
    ShaderConstInfo GetConstInfo(bool pixel, uint64_t hash) const;

    bool Ready(uint64_t hash, bool pixel);

    uint32_t JitCount() const { return jit_; }
    uint32_t FailCount() const { return failed_; }

private:
    VkShaderModule LoadSpirv(uint64_t hash, bool pixel);
    VkShaderModule LoadSpirvPath(uint64_t hash, const std::string& path);
    VkShaderModule Compile(uint64_t hash, bool pixel, const uint32_t* tokens,
                           uint32_t dwords);
    VkShaderModule CreateModule(const std::vector<uint32_t>& words,
                                uint64_t hash);
    std::string PathFor(uint64_t hash, bool pixel, const char* ext) const;
    std::string VariantPathFor(uint64_t hash, uint32_t drefMask) const;

    std::string dir_;
    std::unordered_map<uint64_t, VkShaderModule> vs_;
    std::unordered_map<uint64_t, VkShaderModule> ps_;
    std::unordered_map<uint64_t, std::unordered_map<uint32_t, VkShaderModule>> psVariants_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> psTokens_;
    mutable std::unordered_map<uint64_t, uint32_t> vsInMasks_;
    mutable std::unordered_map<uint64_t, uint32_t> psTexldpMasks_;
    mutable std::unordered_map<uint64_t, ShaderConstInfo> vsConstInfo_;
    mutable std::unordered_map<uint64_t, ShaderConstInfo> psConstInfo_;
    std::unordered_map<uint64_t, bool> logged_;
    uint32_t dumps_ = 0;
    uint32_t jit_ = 0;
    uint32_t failed_ = 0;
};

extern ShaderDB g_shaders;
