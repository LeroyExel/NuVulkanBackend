#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#define VK_USE_PLATFORM_WIN32_KHR 1
#define VK_NO_PROTOTYPES 1
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Vulkan entry points. Resolved from vulkan-1.dll at Init time so the DLL has
// no import dependency on the loader (and keeps working on machines that only
// have the runtime, not the SDK).
// ---------------------------------------------------------------------------
#define VK_FUNC_LIST(X)                                                        \
    X(vkGetInstanceProcAddr) X(vkGetDeviceProcAddr)                             \
    X(vkCreateInstance) X(vkDestroyInstance)                                   \
    X(vkEnumerateInstanceExtensionProperties)                                   \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)              \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                \
    X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceMemoryProperties)      \
    X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue)                    \
    X(vkCreateWin32SurfaceKHR) X(vkDestroySurfaceKHR)                           \
    X(vkGetPhysicalDeviceSurfaceSupportKHR)                                     \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)                                \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR)                                     \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)                                \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR)  \
    X(vkCreateImageView) X(vkDestroyImageView)                                  \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements)        \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)           \
    X(vkCreateShaderModule) X(vkDestroyShaderModule)                            \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                        \
    X(vkCreatePipelineCache) X(vkDestroyPipelineCache) X(vkGetPipelineCacheData) \
    X(vkCreateGraphicsPipelines) X(vkDestroyPipeline)                           \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout)               \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)                       \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers)  \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer)        \
    X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle)                     \
    X(vkCreateSemaphore) X(vkDestroySemaphore)                                  \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)       \
    X(vkAcquireNextImageKHR) X(vkQueuePresentKHR)                               \
    X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdPipelineBarrier)          \
    X(vkCmdBindPipeline) X(vkCmdBindVertexBuffers) X(vkCmdBindVertexBuffers2)   \
    X(vkCmdBindIndexBuffer)                                                     \
    X(vkCmdBindDescriptorSets) X(vkCmdDrawIndexed) X(vkCmdDraw)                  \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdClearAttachments)              \
    X(vkCmdPushConstants) X(vkCmdSetCullMode) X(vkCmdSetFrontFace)               \
    X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthWriteEnable)                       \
    X(vkCmdSetDepthCompareOp) X(vkCmdSetPrimitiveTopology)                       \
    X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilOp)                            \
    X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask)                    \
    X(vkCmdSetStencilReference)                                                  \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyBuffer)                                \
    X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdClearColorImage)                 \
    X(vkCreateSampler) X(vkDestroySampler)

#define VK_DECL(name) extern PFN_##name name;
VK_FUNC_LIST(VK_DECL)
#undef VK_DECL

extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddrReal;

bool vkLoadLoader();
bool vkResolveInstance(VkInstance instance);
bool vkResolveDevice(VkDevice device);

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
void LogOpen(const char* path);
void LogClose();
void Logf(const char* fmt, ...);

#define LOG(...) Logf(__VA_ARGS__)

#define VK_CHECK(expr)                                                         \
    do {                                                                       \
        VkResult _r = (expr);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            LOG("VK_CHECK failed %s -> %d (%s:%d)", #expr, (int)_r, __FILE__,   \
                __LINE__);                                                      \
        }                                                                       \
    } while (0)

inline uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
