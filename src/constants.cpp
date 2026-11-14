#include "constants.h"

#include "vk_context.h"

ConstantRing g_constRing;

bool ConstantRing::Init(uint32_t capacityBytes) {
    const uint32_t align = g_backend.MinUboAlign();
    align_ = align > 64u ? align : 64u;
    stride_ = AlignUp(kBlockBytes, align_);

    // Two equal halves, one per frame in flight, each holding whole draw
    // blocks (a VS block followed by a PS block).
    const uint32_t frameBytes = stride_ * 2u * 4096u;
    capacity_ = capacityBytes < frameBytes ? frameBytes : capacityBytes;
    capacity_ = (capacity_ / (stride_ * 2u)) * (stride_ * 2u);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = capacity_;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(g_backend.Allocator(), &bci, &aci, &buffer_, &alloc_,
                        &info) != VK_SUCCESS) {
        LOG("[const] ring allocation failed (%u bytes)", capacity_);
        return false;
    }
    mapped_ = info.pMappedData;
    if (!mapped_) {
        LOG("[const] ring is not host visible");
        return false;
    }
    memset(mapped_, 0, capacity_);
    frameStart_ = 0;
    frameEnd_ = capacity_ / 2u;
    head_ = frameStart_;
    LOG("[const] ring %u bytes, stride %u (%u registers)", capacity_, stride_,
        kRegisters);
    return true;
}

void ConstantRing::Shutdown() {
    if (buffer_ && alloc_) vmaDestroyBuffer(g_backend.Allocator(), buffer_, alloc_);
    buffer_ = VK_NULL_HANDLE;
    alloc_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    capacity_ = 0;
    head_ = 0;
}

void ConstantRing::Reset(uint32_t frameIndex) {
    const uint32_t half = capacity_ / 2u;
    frameStart_ = (frameIndex % 2u) * half;
    frameEnd_ = frameStart_ + half;
    head_ = frameStart_;
}

void ConstantRing::Write(uint32_t offset, const void* src, uint32_t bytes) {
    if (!mapped_ || !src || offset + bytes > capacity_) return;
    memcpy((uint8_t*)mapped_ + offset, src, bytes);
}

uint32_t ConstantRing::AllocateSlot() {
    return AllocateBytes(kBlockBytes);
}

uint32_t ConstantRing::AllocateBytes(uint32_t bytes) {
    if (!mapped_) return 0xFFFFFFFFu;
    const uint32_t step = AlignUp(bytes ? bytes : 16u, align_);
    const uint32_t needEnd = step > kBlockBytes ? step : kBlockBytes;
    if (head_ + needEnd > frameEnd_) {
        head_ = frameStart_;
    }
    uint32_t off = head_;
    head_ += step;
    return off;
}

ConstantRing::Block ConstantRing::Allocate() {
    Block b{};
    if (!mapped_) return b;
    if (head_ + stride_ * 2u > frameEnd_) head_ = frameStart_;
    b.vsOffset = head_;
    b.psOffset = head_ + stride_;
    b.valid = true;
    head_ += stride_ * 2u;
    return b;
}
