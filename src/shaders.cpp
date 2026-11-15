#include "shaders.h"

#include "sm3_to_spirv.h"
#include "vk_context.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

ShaderDB g_shaders;

void ShaderDB::Init(const char* directory) {
    dir_ = directory ? directory : "shaders";
    CreateDirectoryA(dir_.c_str(), nullptr);
    LOG("[shader] cache directory: %s", dir_.c_str());
}

std::string ShaderDB::PathFor(uint64_t hash, bool pixel, const char* ext) const {
    // The cache is keyed by SM3 bytecode hash, so a translator change would
    // otherwise keep loading modules produced by the old emitter. Bump the tag
    // whenever sm3_to_spirv.cpp changes.
    char buf[MAX_PATH];
    snprintf(buf, sizeof(buf), "%s\\v35_%016llx_%s%s", dir_.c_str(),
             (unsigned long long)hash, pixel ? "ps" : "vs", ext);
    return buf;
}

std::string ShaderDB::VariantPathFor(uint64_t hash, uint32_t drefMask) const {
    char buf[MAX_PATH];
    snprintf(buf, sizeof(buf), "%s\\v35_%016llx_d%x_ps.spv", dir_.c_str(),
             (unsigned long long)hash, drefMask);
    return buf;
}

VkShaderModule ShaderDB::CreateModule(const std::vector<uint32_t>& words,
                                      uint64_t hash) {
    if (words.empty() || words[0] != 0x07230203u) {
        LOG("[shader] %016llx: not a SPIR-V module", (unsigned long long)hash);
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words.size() * sizeof(uint32_t);
    ci.pCode = words.data();
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g_backend.Device(), &ci, nullptr, &m) != VK_SUCCESS) {
        LOG("[shader] %016llx: vkCreateShaderModule rejected the module",
            (unsigned long long)hash);
        return VK_NULL_HANDLE;
    }
    return m;
}

VkShaderModule ShaderDB::LoadSpirvPath(uint64_t hash, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return VK_NULL_HANDLE;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || (n & 3)) {
        fclose(f);
        return VK_NULL_HANDLE;
    }
    std::vector<uint32_t> words((size_t)n / 4);
    size_t got = fread(words.data(), 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) return VK_NULL_HANDLE;
    return CreateModule(words, hash);
}

VkShaderModule ShaderDB::LoadSpirv(uint64_t hash, bool pixel) {
    return LoadSpirvPath(hash, PathFor(hash, pixel, ".spv"));
}

VkShaderModule ShaderDB::Compile(uint64_t hash, bool pixel,
                                 const uint32_t* tokens, uint32_t dwords) {
    std::vector<uint32_t> spv;
    std::string error;
    if (!Sm3CompileToSpirv(pixel, tokens, dwords, spv, error)) {
        ++failed_;
        if (!logged_[hash]) {
            logged_[hash] = true;
            LOG("[shader] JIT failed %016llx (%s): %s", (unsigned long long)hash,
                pixel ? "ps" : "vs", error.c_str());
        }
        return VK_NULL_HANDLE;
    }

    std::string path = PathFor(hash, pixel, ".spv");
    FILE* f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(spv.data(), 4, spv.size(), f);
        fclose(f);
    }
    ++jit_;
    LOG("[shader] JIT %016llx %s: %u tokens -> %u SPIR-V words",
        (unsigned long long)hash, pixel ? "ps" : "vs", dwords,
        (uint32_t)spv.size());
    return CreateModule(spv, hash);
}

uint64_t ShaderDB::Capture(bool pixel, const void* code, size_t size) {
    if (!code || size < 4) return 0;
    const uint64_t h = XXH64(code, size, 0x4E75566Bull);

    // Vertex input mask: every dcl_* destination register of type INPUT names
    // an attribute the shader expects the pipeline to provide.
    if (!pixel && vsInMasks_.find(h) == vsInMasks_.end()) {
        uint32_t mask = 0;
        const uint32_t* tok = (const uint32_t*)code;
        const uint32_t count = (uint32_t)(size / 4);
        uint32_t i = 1;
        while (i < count) {
            uint32_t w = tok[i];
            if ((w & 0xFFFFu) == 0xFFFEu) {
                i += 1u + ((w >> 16) & 0x7FFFu);
                continue;
            }
            if (w == 0x0000FFFFu) break;
            const uint32_t op = w & 0xFFFFu;
            uint32_t len = 1u + ((w >> 24) & 0x0Fu);
            if (w & 0x10000000u) ++len;
            if (op == 31 && len >= 3 && i + 2 < count) {
                const uint32_t dToken = tok[i + 2];
                const uint32_t rt =
                    ((dToken & 0x70000000u) >> 28) |
                    ((dToken & 0x00001800u) >> 8);
                if (rt == 1) {
                    const uint32_t decl = tok[i + 1];
                    const uint32_t loc = Sm3LocationFor(decl & 0x1Fu,
                                                        (decl >> 16) & 0x0Fu);
                    if (loc < 32) mask |= (1u << loc);
                }
            }
            i += len;
        }
        vsInMasks_[h] = mask;
    }

    auto& infoMap = pixel ? psConstInfo_ : vsConstInfo_;
    if (infoMap.find(h) == infoMap.end()) {
        ShaderConstInfo ci{};
        ci.regCount = 1;
        ci.usesRelAddr = false;
        const uint32_t* tok = (const uint32_t*)code;
        const uint32_t count = (uint32_t)(size / 4);
        uint32_t i = 1;
        while (i < count) {
            const uint32_t w = tok[i];
            if ((w & 0xFFFFu) == 0xFFFEu) {
                i += 1u + ((w >> 16) & 0x7FFFu);
                continue;
            }
            if (w == 0x0000FFFFu) break;
            const uint32_t op = w & 0xFFFFu;
            uint32_t len = 1u + ((w >> 24) & 0x0Fu);
            if (w & 0x10000000u) ++len;
            if (op != 31 && op != 47 && op != 48 && op != 81) {
                uint32_t span = 1;
                if (op == 20 || op == 22) span = 4;
                else if (op == 21 || op == 23) span = 3;
                else if (op == 24) span = 2;
                const uint32_t endTok = (i + len < count) ? (i + len) : count;
                for (uint32_t j = i + 1; j < endTok; ++j) {
                    const uint32_t pt = tok[j];
                    const uint32_t rt =
                        ((pt & 0x70000000u) >> 28) | ((pt & 0x00001800u) >> 8);
                    const bool rel = (pt & 0x00002000u) != 0;
                    if (rt == 2 || rt == 11 || rt == 12 || rt == 13) {
                        if (rel) ci.usesRelAddr = true;
                        uint32_t reg = pt & 0x7FFu;
                        uint32_t hi = reg + span;
                        if (hi > 257u) hi = 257u;
                        if (hi > ci.regCount) ci.regCount = (uint16_t)hi;
                    }
                    if (rel && j + 1 < endTok) ++j;
                }
            }
            i += len;
        }
        infoMap[h] = ci;
    }

    if (pixel && psTexldpMasks_.find(h) == psTexldpMasks_.end()) {
        const uint32_t dwords = (uint32_t)(size / 4);
        const uint32_t* tok = (const uint32_t*)code;
        const uint32_t mask = Sm3TexldpSamplerMask(tok, dwords);
        psTexldpMasks_[h] = mask;
        if (mask != 0) psTokens_[h].assign(tok, tok + dwords);
    }

    if (vs_.count(h) || ps_.count(h)) return h;

    if (dumps_ < 4000) {
        std::string raw = PathFor(h, pixel, ".sm3");
        if (GetFileAttributesA(raw.c_str()) == INVALID_FILE_ATTRIBUTES) {
            FILE* f = fopen(raw.c_str(), "wb");
            if (f) {
                fwrite(code, 1, size, f);
                fclose(f);
            }
        }
        ++dumps_;
    }

    VkShaderModule m = LoadSpirv(h, pixel);
    if (!m) m = Compile(h, pixel, (const uint32_t*)code, (uint32_t)(size / 4));

    // Key by hash even when compilation failed so a broken shader is not
    // recompiled on every draw.
    if (pixel)
        ps_[h] = m;
    else
        vs_[h] = m;
    return h;
}

VkShaderModule ShaderDB::Get(uint64_t hash, bool pixel) {
    if (pixel) {
        auto it = ps_.find(hash);
        return it == ps_.end() ? VK_NULL_HANDLE : it->second;
    }
    auto it = vs_.find(hash);
    return it == vs_.end() ? VK_NULL_HANDLE : it->second;
}

VkShaderModule ShaderDB::GetPixel(uint64_t hash, uint32_t drefMask) {
    if (!drefMask) return Get(hash, true);
    auto& byMask = psVariants_[hash];
    auto it = byMask.find(drefMask);
    if (it != byMask.end()) return it->second;

    std::string path = VariantPathFor(hash, drefMask);
    VkShaderModule m = LoadSpirvPath(hash, path);
    if (!m) {
        auto tit = psTokens_.find(hash);
        if (tit != psTokens_.end()) {
            std::vector<uint32_t> spv;
            std::string error;
            if (Sm3CompileToSpirv(true, tit->second.data(),
                                  (uint32_t)tit->second.size(), spv, error,
                                  drefMask)) {
                FILE* f = fopen(path.c_str(), "wb");
                if (f) {
                    fwrite(spv.data(), 4, spv.size(), f);
                    fclose(f);
                }
                ++jit_;
                LOG("[shader] JIT %016llx ps (dref=0x%x): %u SPIR-V words",
                    (unsigned long long)hash, drefMask, (uint32_t)spv.size());
                m = CreateModule(spv, hash);
            } else {
                ++failed_;
                LOG("[shader] JIT failed %016llx (ps dref=0x%x): %s",
                    (unsigned long long)hash, drefMask, error.c_str());
            }
        }
    }
    if (!m) m = Get(hash, true);
    byMask[drefMask] = m;
    return m;
}

uint32_t ShaderDB::GetVsInputMask(uint64_t hash) const {
    auto it = vsInMasks_.find(hash);
    return it != vsInMasks_.end() ? it->second : 0;
}

uint32_t ShaderDB::GetPsTexldpMask(uint64_t hash) const {
    auto it = psTexldpMasks_.find(hash);
    return it != psTexldpMasks_.end() ? it->second : 0;
}

ShaderConstInfo ShaderDB::GetConstInfo(bool pixel, uint64_t hash) const {
    const auto& infoMap = pixel ? psConstInfo_ : vsConstInfo_;
    auto it = infoMap.find(hash);
    return it != infoMap.end() ? it->second : ShaderConstInfo{};
}

bool ShaderDB::Ready(uint64_t hash, bool pixel) {
    return Get(hash, pixel) != VK_NULL_HANDLE;
}
