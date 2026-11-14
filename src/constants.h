#pragma once
#include <vk_mem_alloc.h>

#include "vk_common.h"

// D3D9 exposes 256 float4 vertex constant registers (c0..c255). The pixel
// constant file is architecturally smaller but uses the same storage, so one
// layout serves both stages.
//
// A 257th register, c[256], is appended and kept zeroed. Relative addressing
// (c[a0.x + n]) can name one register past the end; without that slot the
// access chain would read past the uniform block. The extra 16 bytes are part
// of the block, not a special "guard" uniform: no instruction can write c[256].
class ConstantRing {
public:
    static constexpr uint32_t kRegisters = 257;  // c0..c255 + zero padding
    static constexpr uint32_t kRegisterBytes = 16;
    static constexpr uint32_t kBlockBytes = kRegisters * kRegisterBytes;

    struct Block {
        uint32_t vsOffset;
        uint32_t psOffset;
        bool valid;
    };

    bool Init(uint32_t capacityBytes);
    void Shutdown();
    void Reset(uint32_t frameIndex);

    Block Allocate();
    uint32_t AllocateSlot();
    uint32_t AllocateBytes(uint32_t bytes);
    void Write(uint32_t offset, const void* src, uint32_t bytes);

    VkBuffer Buffer() const { return buffer_; }
    uint32_t Stride() const { return stride_; }

private:
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation alloc_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    uint32_t capacity_ = 0;
    uint32_t head_ = 0;
    uint32_t frameStart_ = 0;
    uint32_t frameEnd_ = 0;
    uint32_t stride_ = kBlockBytes;
    uint32_t align_ = 64;
};

extern ConstantRing g_constRing;
