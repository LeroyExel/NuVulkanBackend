#include "vk_context.h"
#include "fps_overlay.h"
#include "stud_shadows.h"

#define XXH_INLINE_ALL
#include "xxhash.h"

#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

VkBackend g_backend;

static const char* kValidationLayers[] = {"VK_LAYER_KHRONOS_validation"};

// Vulkan requires queue submission, fence resets and the upload command buffer
// to be externally synchronized across threads; the engine uses several.
struct VkLockGuard {
    CRITICAL_SECTION* cs;
    explicit VkLockGuard(CRITICAL_SECTION* c) : cs(c) {
        EnterCriticalSection(cs);
    }
    ~VkLockGuard() { LeaveCriticalSection(cs); }
};

bool VkBackend::Init(HWND hwndIn, uint32_t width, uint32_t height) {
    InitializeCriticalSection(&queueLock);
    hwnd = hwndIn;
    reqWidth = width ? width : 1280;
    reqHeight = height ? height : 720;

    if (!vkLoadLoader()) return false;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "NuVulkanBackend";
    app.apiVersion = VK_API_VERSION_1_3;

    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                          VK_KHR_WIN32_SURFACE_EXTENSION_NAME};

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = exts;
    (void)kValidationLayers;

    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        LOG("[vk] vkCreateInstance failed");
        return false;
    }
    if (!vkResolveInstance(instance)) {
        LOG("[vk] instance function resolution failed");
        return false;
    }

    VkWin32SurfaceCreateInfoKHR sci{
        VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd = hwnd;
    if (vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface) !=
        VK_SUCCESS) {
        LOG("[vk] surface creation failed");
        return false;
    }
    if (!PickPhysicalDevice()) return false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci[2]{};
    qci[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci[0].queueFamilyIndex = gfxFamily;
    qci[0].queueCount = 1;
    qci[0].pQueuePriorities = &prio;
    uint32_t qcount = 1;
    if (presentFamily != gfxFamily) {
        qci[1] = qci[0];
        qci[1].queueFamilyIndex = presentFamily;
        qcount = 2;
    }

    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3f{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    f13.pNext = &eds3f;
    memset(&eds3f, 0, sizeof(eds3f));
    eds3f.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    f13.pNext = &eds3f;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    eds3 = eds3f.extendedDynamicState3ColorBlendEnable == VK_TRUE;

    if (!f13.dynamicRendering) {
        LOG("[vk] dynamicRendering unavailable - Vulkan 1.3 required");
        return false;
    }

    VkPhysicalDeviceVulkan13Features enable13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enable13.dynamicRendering = VK_TRUE;
    enable13.synchronization2 = VK_TRUE;

    // Enable VK_EXT_device_fault when the driver has it: a lost device then
    // reports the faulting address and what kind of access it was, instead of
    // only VK_ERROR_DEVICE_LOST.
    auto enumDevExts =
        (PFN_vkEnumerateDeviceExtensionProperties)vkGetInstanceProcAddr(
            instance, "vkEnumerateDeviceExtensionProperties");
    bool hasFaultExt = false;
    if (enumDevExts) {
        uint32_t extCount = 0;
        enumDevExts(phys, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        if (extCount) enumDevExts(phys, nullptr, &extCount, exts.data());
        for (uint32_t i = 0; i < extCount; ++i) {
            if (strcmp(exts[i].extensionName, "VK_EXT_device_fault") == 0)
                hasFaultExt = true;
            if (strcmp(exts[i].extensionName, VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME) == 0)
                hasDepthClip_ = true;
        }
    }

    const char* devExts[4] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    uint32_t devExtCount = 1;
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    VkPhysicalDeviceFaultFeaturesEXT faultFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    if (hasFaultExt) {
        faultFeatures.deviceFault = VK_TRUE;
        faultFeatures.pNext = enable13.pNext;
        enable13.pNext = &faultFeatures;
        devExts[devExtCount++] = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
        LOG("[vk] VK_EXT_device_fault enabled");
    }
#else
    (void)hasFaultExt;
#endif

    VkPhysicalDeviceDepthClipEnableFeaturesEXT depthClipFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
    if (hasDepthClip_) {
        depthClipFeatures.depthClipEnable = VK_TRUE;
        depthClipFeatures.pNext = enable13.pNext;
        enable13.pNext = &depthClipFeatures;
        devExts[devExtCount++] = VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME;
        LOG("[vk] VK_EXT_depth_clip_enable enabled");
    }

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enable13;
    dci.queueCreateInfoCount = qcount;
    dci.pQueueCreateInfos = qci;
    dci.enabledExtensionCount = devExtCount;
    dci.ppEnabledExtensionNames = devExts;

    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        LOG("[vk] vkCreateDevice failed");
        return false;
    }
    if (!vkResolveDevice(device)) {
        LOG("[vk] device function resolution failed");
        return false;
    }
    vkGetDeviceQueue(device, gfxFamily, 0, &gfxQueue);
    vkGetDeviceQueue(device, presentFamily, 0, &presentQueue);

    VmaVulkanFunctions vf{};
    vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = phys;
    aci.device = device;
    aci.instance = instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &vf;
    if (vmaCreateAllocator(&aci, &allocator) != VK_SUCCESS) {
        LOG("[vk] vmaCreateAllocator failed");
        return false;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    minUboAlign = (uint32_t)props.limits.minUniformBufferOffsetAlignment;
    if (minUboAlign < 64) minUboAlign = 64;
    LOG("[vk] %s | api %u.%u.%u | minUboAlign=%u | eds3=%d", props.deviceName,
        VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
        VK_VERSION_PATCH(props.apiVersion), minUboAlign, (int)eds3);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = gfxFamily;
    vkCreateCommandPool(device, &pci, nullptr, &cmdPool);

    // One-shot upload path must exist before any texture/default-image setup.
    {
        VkCommandBufferAllocateInfo cbai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cbai, &immediateCmd);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(device, &fci, nullptr, &immediateFence);
    }

    // 16 individual sampler bindings (2..17) rather than one descriptor array:
    // an array is homogeneous in SPIR-V, so it cannot mix sampler2D and
    // samplerCube, and the engine needs both.
    VkDescriptorSetLayoutBinding binds[18]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 0; i < 16; ++i) {
        binds[2 + i].binding = 2 + i;
        binds[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[2 + i].descriptorCount = 1;
        binds[2 + i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo slci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 18;
    slci.pBindings = binds;
    vkCreateDescriptorSetLayout(device, &slci, nullptr, &setLayout);

    if (!CreateDescriptorPool()) return false;

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);

    if (!CreateDefaultTexture()) return false;
    if (!CreateDefaultCube()) return false;
    if (!CreateDefaultVolume()) return false;

    VkBufferCreateInfo dummyBci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    dummyBci.size = 4096;
    dummyBci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    dummyBci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo dummyAci{};
    dummyAci.usage = VMA_MEMORY_USAGE_AUTO;
    dummyAci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    dummyAci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VmaAllocationInfo dummyInfo{};
    if (vmaCreateBuffer(allocator, &dummyBci, &dummyAci, &dummyVb, &dummyVbAlloc, &dummyInfo) == VK_SUCCESS) {
        if (dummyInfo.pMappedData) {
            uint8_t* p = (uint8_t*)dummyInfo.pMappedData;
            for (uint32_t i = 0; i < 4096u / 16u; ++i) {
                uint32_t loc = i % 32u;
                float v[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                if (loc == 19 || loc == 20) {
                    v[0] = v[1] = v[2] = v[3] = 1.0f;
                } else if (loc == 1 || loc == 28 || loc == 29) {
                    v[0] = v[1] = v[2] = v[3] = 0.0f;
                }
                memcpy(p + (size_t)i * 16u, v, 16u);
            }
        }
    }

    for (int i = 0; i < 2; ++i) {
        VkCommandBufferAllocateInfo cbai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &cbai, &frames[i].cmd);

        VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device, &semci, nullptr, &frames[i].imageAvailable);
        vkCreateSemaphore(device, &semci, nullptr, &frames[i].renderFinished);

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fci, nullptr, &frames[i].inFlight);
    }

    if (!CreateSwapchainObjects()) return false;
    descCache.reserve(1024);
    g_fpsOverlay.Init();
    g_studShadows.Init();
    targetCount = 0;
    targetDepthFormat = depthFormat;
    targetW = extent.width;
    targetH = extent.height;
    recording = false;
    LOG("[vk] init complete %ux%u fmt=%d", extent.width, extent.height,
        (int)colorFormat);
    return true;
}

#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
static bool deviceFaultLogged = false;

// The vendored Vulkan-Headers alias the EXT fault structs to the newer KHR
// ones, whose layout is not what vkGetDeviceFaultInfoEXT writes. Pass the
// spec's EXT layout explicitly so the report parses correctly.
namespace {
struct FaultCounts {
    VkStructureType sType;
    void* pNext;
    uint32_t addressInfoCount;
    uint32_t vendorInfoCount;
    VkDeviceSize vendorBinarySize;
};
struct FaultAddressInfo {
    int32_t addressType;
    uint64_t reportedAddress;
    uint64_t addressPrecision;
};
struct FaultVendorInfo {
    char description[VK_MAX_DESCRIPTION_SIZE];
    uint64_t vendorFaultCode;
    uint64_t vendorFaultAddress;
};
struct FaultInfoExt {
    VkStructureType sType;
    void* pNext;
    char description[VK_MAX_DESCRIPTION_SIZE];
    FaultAddressInfo* pAddressInfos;
    FaultVendorInfo* pVendorInfos;
    void* pVendorBinaryData;
};
typedef VkResult(VKAPI_PTR* PFN_GetDeviceFaultInfo)(VkDevice, FaultCounts*,
                                                   FaultInfoExt*);
}  // namespace
#endif

// Logs a failed submit together with the driver's own fault record. Without
// this the only evidence is VK_ERROR_DEVICE_LOST (-4).
static void LogDeviceFault(VkDevice dev, const char* where, VkResult r) {
    LOG("[vk] %s failed (%d)", where, (int)r);
#ifdef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
    if (deviceFaultLogged) return;
    deviceFaultLogged = true;
    auto pfn = (PFN_GetDeviceFaultInfo)vkGetDeviceProcAddr(
        dev, "vkGetDeviceFaultInfoEXT");
    if (!pfn) {
        LOG("[vk] no vkGetDeviceFaultInfoEXT");
        return;
    }
    FaultAddressInfo addresses[8]{};
    FaultVendorInfo vendors[8]{};
    FaultInfoExt info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
    info.pAddressInfos = addresses;
    info.pVendorInfos = vendors;
    FaultCounts counts{};
    counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
    counts.addressInfoCount = 8;
    counts.vendorInfoCount = 8;
    VkResult fr = pfn(dev, &counts, &info);
    // VK_INCOMPLETE only means there were more entries than capacity; what the
    // driver did write is still the useful part.
    if (fr != VK_SUCCESS && fr != VK_INCOMPLETE) {
        LOG("[vk] vkGetDeviceFaultInfoEXT returned %d", (int)fr);
        return;
    }
    LOG("[vk] fault desc='%s' addresses=%u vendors=%u binary=%llu",
        info.description, counts.addressInfoCount, counts.vendorInfoCount,
        (unsigned long long)counts.vendorBinarySize);
    for (uint32_t i = 0; i < counts.addressInfoCount && i < 8; ++i)
        LOG("[vk] fault address[%u] type=%d reported=0x%llx precision=%llu", i,
            (int)addresses[i].addressType,
            (unsigned long long)addresses[i].reportedAddress,
            (unsigned long long)addresses[i].addressPrecision);
    for (uint32_t i = 0; i < counts.vendorInfoCount && i < 8; ++i)
        LOG("[vk] fault vendor[%u] '%s' code=%llu address=0x%llx", i,
            vendors[i].description,
            (unsigned long long)vendors[i].vendorFaultCode,
            (unsigned long long)vendors[i].vendorFaultAddress);
#endif
}

bool VkBackend::PickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count) return false;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    for (VkPhysicalDevice d : devs) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qs.data());
        int g = -1, p = -1;
        for (uint32_t i = 0; i < qn; ++i) {
            if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && g < 0) g = (int)i;
            VkBool32 sup = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface, &sup);
            if (sup && p < 0) p = (int)i;
        }
        if (g < 0 || p < 0) continue;
        phys = d;
        gfxFamily = (uint32_t)g;
        presentFamily = (uint32_t)p;
        return true;
    }
    LOG("[vk] no suitable physical device");
    return false;
}

bool VkBackend::CreateDefaultTexture() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(allocator, &ici, &aci, &defaultImage, &defaultAlloc,
                       nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = defaultImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &vci, nullptr, &defaultView);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkCreateSampler(device, &sci, nullptr, &defaultSampler);

    for (uint32_t i = 0; i < 16; ++i) {
        boundViews[i] = defaultView;
        boundSamplers[i] = defaultSampler;
        boundCube[i] = false;
        boundVolume[i] = false;
    }

    // The fallback must be white, not undefined memory: an unbound texture
    // stage multiplies the shader result by what it samples.
    ImmediateSubmit(
        [](VkCommandBuffer cmd, void* user) {
            VkImage img = (VkImage)user;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = img;
            b.subresourceRange = range;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b);
            VkClearColorValue white{};
            for (int i = 0; i < 4; ++i) white.float32[i] = 1.0f;
            vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &white, 1, &range);
            VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.image = img;
            b2.subresourceRange = range;
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b2);
        },
        (void*)defaultImage);
    return true;
}

bool VkBackend::CreateDescriptorPool() {
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    sizes[0].descriptorCount = 16384 * 2;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = 16384 * 16;
    VkDescriptorPoolCreateInfo dpci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 16384;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &pool) != VK_SUCCESS) {
        LOG("[vk] CreateDescriptorPool failed");
        return false;
    }
    descPools.push_back(pool);
    currentDescPool = pool;
    return true;
}

bool VkBackend::CreateSwapchainObjects() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);

    extent.width = reqWidth ? reqWidth : caps.currentExtent.width;
    extent.height = reqHeight ? reqHeight : caps.currentExtent.height;
    if (extent.width < caps.minImageExtent.width) extent.width = caps.minImageExtent.width;
    if (extent.width > caps.maxImageExtent.width) extent.width = caps.maxImageExtent.width;
    if (extent.height < caps.minImageExtent.height) extent.height = caps.minImageExtent.height;
    if (extent.height > caps.maxImageExtent.height) extent.height = caps.maxImageExtent.height;
    if (extent.width == 0) extent.width = 1;
    if (extent.height == 0) extent.height = 1;

    uint32_t fcount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fcount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fcount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &fcount, formats.data());
    colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
            colorFormat = f.format;
            colorSpace = f.colorSpace;
            break;
        }
    }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount)
        imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = imgCount;
    sci.imageFormat = colorFormat;
    sci.imageColorSpace = colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    uint32_t families[2] = {gfxFamily, presentFamily};
    if (gfxFamily != presentFamily) {
        sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices = families;
    } else {
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &pmCount, presentModes.data());
    VkPresentModeKHR chosenPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsyncEnabled) {
        for (auto pm : presentModes) {
            if (pm == VK_PRESENT_MODE_MAILBOX_KHR) {
                chosenPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }
        if (chosenPresentMode == VK_PRESENT_MODE_FIFO_KHR) {
            for (auto pm : presentModes) {
                if (pm == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    chosenPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    break;
                }
            }
        }
    }
    sci.presentMode = chosenPresentMode;
    sci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain) != VK_SUCCESS) {
        LOG("[vk] swapchain creation failed");
        return false;
    }

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
    images.resize(n);
    views.resize(n);
    vkGetSwapchainImagesKHR(device, swapchain, &n, images.data());
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = colorFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &vci, nullptr, &views[i]);
    }

    for (auto s : renderFinishedSemaphores) {
        if (s) vkDestroySemaphore(device, s, nullptr);
    }
    renderFinishedSemaphores.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device, &semci, nullptr, &renderFinishedSemaphores[i]);
    }

    VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dci.imageType = VK_IMAGE_TYPE_2D;
    dci.format = depthFormat;
    dci.extent = {extent.width, extent.height, 1};
    dci.mipLevels = 1;
    dci.arrayLayers = 1;
    dci.samples = VK_SAMPLE_COUNT_1_BIT;
    dci.tiling = VK_IMAGE_TILING_OPTIMAL;
    dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    vmaCreateImage(allocator, &dci, &aci, &depthImage, &depthAlloc, nullptr);
    VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dvci.image = depthImage;
    dvci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    dvci.format = depthFormat;
    dvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                             0, 1, 0, 1};
    vkCreateImageView(device, &dvci, nullptr, &depthView);
    return true;
}

void VkBackend::DestroySwapchainObjects() {
    if (depthView) vkDestroyImageView(device, depthView, nullptr);
    if (depthImage) vmaDestroyImage(allocator, depthImage, depthAlloc);
    depthView = VK_NULL_HANDLE;
    depthImage = VK_NULL_HANDLE;
    depthAlloc = VK_NULL_HANDLE;
    for (auto s : renderFinishedSemaphores) {
        if (s) vkDestroySemaphore(device, s, nullptr);
    }
    renderFinishedSemaphores.clear();
    for (auto v : views)
        if (v) vkDestroyImageView(device, v, nullptr);
    views.clear();
    images.clear();
    layouts_.clear();  // image handles die with the swapchain
    if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
}

bool VkBackend::RecreateSwapchain(uint32_t width, uint32_t height) {
    if (!device) return false;
    reqWidth = width ? width : reqWidth;
    reqHeight = height ? height : reqHeight;
    vkDeviceWaitIdle(device);
    renderingActive = false;
    DestroySwapchainObjects();
    return CreateSwapchainObjects();
}

bool VkBackend::BeginFrame() {
    VkFrame& f = frames[frameIndex];
    vkWaitForFences(device, 1, &f.inFlight, VK_TRUE, UINT64_MAX);

    currentDescSet = VK_NULL_HANDLE;
    texturesDirty = true;

    VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                         f.imageAvailable, VK_NULL_HANDLE,
                                         &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain(extent.width, extent.height);
        return false;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) return false;

    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    VkViewport vp{0.0f, (float)extent.height, (float)extent.width,
                  -(float)extent.height, 0.0f, 1.0f};
    vkCmdSetViewport(f.cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, extent};
    vkCmdSetScissor(f.cmd, 0, 1, &sc);
    pendingClear = ClearRequest{};
    renderingActive = false;
    recording = true;
    BindSwapchainTargets();
    return true;
}

void VkBackend::BeginRendering(const ClearRequest* clear) {
    if (renderingActive) return;
    if (targetCount == 0) BindSwapchainTargets();

    // A rendering scope needs a real attachment view. A null one makes the
    // fragment shader write to address 0 and the driver loses the device
    // (VK_EXT_device_fault: WRITE_INVALID at 0x0), so fall back to the
    // swapchain when the engine binds a target whose view was not created.
    for (uint32_t i = 0; i < targetCount; ++i) {
        if (ResolveView(targetColor[i]) != VK_NULL_HANDLE) continue;
        LOG("[vk] color target %u has no view - using swapchain", i);
        BindSwapchainTargets();
        break;
    }

    VkRenderingAttachmentInfo colors[4]{};
    uint32_t count = targetCount;
    for (uint32_t i = 0; i < count; ++i) {
        colors[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colors[i].imageView = ResolveView(targetColor[i]);
        colors[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colors[i].loadOp = (clear && clear->clearColor)
                               ? VK_ATTACHMENT_LOAD_OP_CLEAR
                               : VK_ATTACHMENT_LOAD_OP_LOAD;
        colors[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (clear) colors[i].clearValue.color = clear->color;
    }

    VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    bool hasDepth = ResolveView(targetDepth) != VK_NULL_HANDLE;
    if (hasDepth) {
        depth.imageView = ResolveView(targetDepth);
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.loadOp = (clear && clear->clearDepth)
                           ? VK_ATTACHMENT_LOAD_OP_CLEAR
                           : VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (clear) depth.clearValue.depthStencil = {clear->depth, clear->stencil};
    }

    VkRenderingAttachmentInfo stencil{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    bool hasStencil = hasDepth && (targetDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
                                   targetDepthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT);
    if (hasStencil) {
        stencil.imageView = targetDepth.stencilView ? targetDepth.stencilView : ResolveView(targetDepth);
        stencil.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        stencil.loadOp = (clear && clear->clearStencil)
                             ? VK_ATTACHMENT_LOAD_OP_CLEAR
                             : VK_ATTACHMENT_LOAD_OP_LOAD;
        stencil.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (clear) stencil.clearValue.depthStencil = {clear->depth, clear->stencil};
    }

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {targetW, targetH}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = count;
    ri.pColorAttachments = colors;
    ri.pDepthAttachment = hasDepth ? &depth : nullptr;
    ri.pStencilAttachment = hasStencil ? &stencil : nullptr;

    for (uint32_t i = 0; i < count; ++i)
        TransitionImage(ResolveImage(targetColor[i]), VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (hasDepth) {
        VkImageAspectFlags a = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (targetDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
            targetDepthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT)
            a |= VK_IMAGE_ASPECT_STENCIL_BIT;
        TransitionImage(ResolveImage(targetDepth), a,
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    static int renderLogs = 0;
    if (renderLogs++ < 4)
        LOG("[render] begin %ux%u color=%u depth=%d colorView=%p depthView=%p",
            ri.renderArea.extent.width, ri.renderArea.extent.height, count,
            (int)hasDepth, (void*)(uintptr_t)colors[0].imageView,
            (void*)(uintptr_t)depth.imageView);
    vkCmdBeginRendering(Cmd(), &ri);
    renderingActive = true;
}

void VkBackend::EndRendering() {
    if (!renderingActive) return;
    vkCmdEndRendering(Cmd());
    renderingActive = false;
}

void VkBackend::EnsureRendering() {
    if (renderingActive) return;
    BeginRendering(pendingClear.valid ? &pendingClear : nullptr);
    pendingClear = ClearRequest{};
}

void VkBackend::SetPendingClear(uint32_t flags, uint32_t color, float depth,
                                uint32_t stencil) {
    if (flags & 1) {
        pendingClear.clearColor = true;
        pendingClear.color.float32[0] = ((color >> 16) & 0xFF) / 255.0f;
        pendingClear.color.float32[1] = ((color >> 8) & 0xFF) / 255.0f;
        pendingClear.color.float32[2] = (color & 0xFF) / 255.0f;
        pendingClear.color.float32[3] = ((color >> 24) & 0xFF) / 255.0f;
    }
    if (flags & 2) {
        pendingClear.clearDepth = true;
        pendingClear.depth = depth;
    }
    if (flags & 4) {
        pendingClear.clearStencil = true;
        pendingClear.stencil = stencil;
    }
    if (pendingClear.clearColor || pendingClear.clearDepth || pendingClear.clearStencil) {
        pendingClear.valid = true;
    }
}

void VkBackend::Clear(uint32_t flags, uint32_t color, float depth,
                      uint32_t stencil, const RECT* rects, uint32_t rectCount) {
    if (!recording) return;
    if (!renderingActive && (!rects || !rectCount)) {
        ClearRequest req{};
        if (flags & 1) {
            req.clearColor = true;
            req.color.float32[0] = ((color >> 16) & 0xFF) / 255.0f;
            req.color.float32[1] = ((color >> 8) & 0xFF) / 255.0f;
            req.color.float32[2] = (color & 0xFF) / 255.0f;
            req.color.float32[3] = ((color >> 24) & 0xFF) / 255.0f;
        }
        if (flags & 2) {
            req.clearDepth = true;
            req.depth = depth;
        }
        if (flags & 4) {
            req.clearStencil = true;
            req.stencil = stencil;
        }
        req.valid = req.clearColor || req.clearDepth || req.clearStencil;
        BeginRendering(req.valid ? &req : nullptr);
        return;
    }
    if (!renderingActive) {
        BeginRendering(nullptr);
    }
    VkClearAttachment atts[6]{};
    uint32_t n = 0;
    if (flags & 1) {
        uint32_t cCount = targetCount ? targetCount : 1;
        if (cCount > 4) cCount = 4;
        for (uint32_t i = 0; i < cCount; ++i) {
            atts[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            atts[n].colorAttachment = i;
            atts[n].clearValue.color.float32[0] = ((color >> 16) & 0xFF) / 255.0f;
            atts[n].clearValue.color.float32[1] = ((color >> 8) & 0xFF) / 255.0f;
            atts[n].clearValue.color.float32[2] = (color & 0xFF) / 255.0f;
            atts[n].clearValue.color.float32[3] = ((color >> 24) & 0xFF) / 255.0f;
            ++n;
        }
    }
    VkImageAspectFlags dsAspect = 0;
    const bool hasDepth = ResolveView(targetDepth) != VK_NULL_HANDLE;
    const bool hasStencil =
        hasDepth && (targetDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
                     targetDepthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT);
    if ((flags & 2) && hasDepth) dsAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if ((flags & 4) && hasStencil) dsAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (dsAspect) {
        atts[n].aspectMask = dsAspect;
        atts[n].clearValue.depthStencil = {depth, stencil};
        ++n;
    }
    if (!n) return;
    if (rects && rectCount) {
        if (rectCount <= 16) {
            VkClearRect rs[16];
            uint32_t valid = 0;
            for (uint32_t i = 0; i < rectCount; ++i) {
                int32_t l = rects[i].left > 0 ? rects[i].left : 0;
                int32_t t = rects[i].top > 0 ? rects[i].top : 0;
                int32_t r = rects[i].right < (int32_t)targetW ? rects[i].right : (int32_t)targetW;
                int32_t b = rects[i].bottom < (int32_t)targetH ? rects[i].bottom : (int32_t)targetH;
                if (l < r && t < b) {
                    rs[valid].rect = {{l, t}, {(uint32_t)(r - l), (uint32_t)(b - t)}};
                    rs[valid].baseArrayLayer = 0;
                    rs[valid].layerCount = 1;
                    ++valid;
                }
            }
            if (valid) vkCmdClearAttachments(Cmd(), n, atts, valid, rs);
        } else {
            std::vector<VkClearRect> rs;
            rs.reserve(rectCount);
            for (uint32_t i = 0; i < rectCount; ++i) {
                int32_t l = rects[i].left > 0 ? rects[i].left : 0;
                int32_t t = rects[i].top > 0 ? rects[i].top : 0;
                int32_t r = rects[i].right < (int32_t)targetW ? rects[i].right : (int32_t)targetW;
                int32_t b = rects[i].bottom < (int32_t)targetH ? rects[i].bottom : (int32_t)targetH;
                if (l < r && t < b) {
                    VkClearRect cr{};
                    cr.rect = {{l, t}, {(uint32_t)(r - l), (uint32_t)(b - t)}};
                    cr.baseArrayLayer = 0;
                    cr.layerCount = 1;
                    rs.push_back(cr);
                }
            }
            if (!rs.empty()) {
                vkCmdClearAttachments(Cmd(), n, atts, (uint32_t)rs.size(), rs.data());
            }
        }
    } else {
        VkClearRect r{};
        r.rect = {{0, 0}, {targetW, targetH}};
        r.layerCount = 1;
        vkCmdClearAttachments(Cmd(), n, atts, 1, &r);
    }
}

void VkBackend::SetTextureSlot(uint32_t slot, VkImageView view,
                               VkSampler sampler, bool cube, bool volume) {
    if (slot >= 16) return;
    if (boundViews[slot] == view && boundSamplers[slot] == sampler &&
        boundCube[slot] == cube && boundVolume[slot] == volume)
        return;
    boundCube[slot] = cube;
    boundVolume[slot] = volume;
    boundViews[slot] = view;
    boundSamplers[slot] = sampler ? sampler : defaultSampler;
    texturesDirty = true;
}

size_t VkBackend::TextureStateHash::operator()(const TextureState& s) const {
    return (size_t)XXH64(&s, sizeof(s), 0x54455853ull);
}

void VkBackend::FlushTextures() {
    if (!texturesDirty && currentDescSet != VK_NULL_HANDLE) return;
    if (!ringBuffer_) return;

    TextureState state{};
    for (uint32_t i = 0; i < 16; ++i) {
        state.views[i] =
            boundViews[i] ? boundViews[i]
                          : (boundVolume[i] ? defaultVolumeView
                                            : (boundCube[i] ? defaultCubeView : defaultView));
        state.samplers[i] = boundSamplers[i] ? boundSamplers[i] : defaultSampler;
    }

    const uint64_t h = XXH64(&state, sizeof(state), 0x54455853ull);
    for (uint32_t i = 0; i < 8; ++i) {
        if (texMru_[i].set != VK_NULL_HANDLE && texMru_[i].hash == h &&
            memcmp(&texMru_[i].state, &state, sizeof(state)) == 0) {
            currentDescSet = texMru_[i].set;
            texturesDirty = false;
            return;
        }
    }

    auto it = descCache.find(state);
    if (it != descCache.end()) {
        currentDescSet = it->second;
        const uint32_t slot = texMruNext_ & 7u;
        texMruNext_ = slot + 1u;
        texMru_[slot].hash = h;
        texMru_[slot].state = state;
        texMru_[slot].set = currentDescSet;
        texturesDirty = false;
        return;
    }

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = currentDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &setLayout;
    VkDescriptorSet newSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &dsai, &newSet) != VK_SUCCESS) {
        if (!CreateDescriptorPool()) {
            texturesDirty = false;
            return;
        }
        dsai.descriptorPool = currentDescPool;
        if (vkAllocateDescriptorSets(device, &dsai, &newSet) != VK_SUCCESS) {
            texturesDirty = false;
            return;
        }
    }

    VkDescriptorBufferInfo bufInfo{ringBuffer_, 0, ringRange_};
    VkDescriptorImageInfo imgInfos[16]{};
    VkWriteDescriptorSet writes[18]{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = newSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &bufInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = newSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[1].pBufferInfo = &bufInfo;

    for (uint32_t i = 0; i < 16; ++i) {
        imgInfos[i].imageView = state.views[i];
        imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imgInfos[i].sampler = state.samplers[i];

        writes[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2 + i].dstSet = newSet;
        writes[2 + i].dstBinding = 2 + i;
        writes[2 + i].descriptorCount = 1;
        writes[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2 + i].pImageInfo = &imgInfos[i];
    }

    vkUpdateDescriptorSets(device, 18, writes, 0, nullptr);
    descCache.emplace(state, newSet);
    const uint32_t slot = texMruNext_ & 7u;
    texMruNext_ = slot + 1u;
    texMru_[slot].hash = h;
    texMru_[slot].state = state;
    texMru_[slot].set = newSet;
    currentDescSet = newSet;
    texturesDirty = false;
}

void VkBackend::BindUniformRing(VkBuffer buffer, uint32_t range) {
    if (!buffer) return;
    ringBuffer_ = buffer;
    ringRange_ = range;
    texturesDirty = true;
    FlushTextures();
    LOG("[vk] uniform ring bound range=%u", range);
}

void VkBackend::EndFrame(bool present) {
    EndRendering();
    VkCommandBuffer cmd = frames[frameIndex].cmd;

    ++frameCounter;

    if (present && imageIndex < images.size() && imageIndex < views.size()) {
        g_fpsOverlay.TickAndDraw(cmd, images[imageIndex], views[imageIndex],
                                 colorFormat, extent, frameIndex);
    }

    TransitionImage(images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    recording = false;

    vkEndCommandBuffer(cmd);
    if (!present) {
        frameIndex = (frameIndex + 1) % 2;
        return;
    }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkSemaphore waitSem = frames[frameIndex].imageAvailable;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &waitSem;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkSemaphore renderSem = (imageIndex < renderFinishedSemaphores.size())
                                ? renderFinishedSemaphores[imageIndex]
                                : frames[frameIndex].renderFinished;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderSem;

    VkLockGuard lock(&queueLock);
    vkResetFences(device, 1, &frames[frameIndex].inFlight);
    VkResult sub = vkQueueSubmit(gfxQueue, 1, &si, frames[frameIndex].inFlight);
    if (sub != VK_SUCCESS) LogDeviceFault(device, "queue submit", sub);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &imageIndex;
    VkResult r = vkQueuePresentKHR(presentQueue, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain(extent.width, extent.height);
    }
    frameIndex = (frameIndex + 1) % 2;
}

VkResult VkBackend::ImmediateSubmit(void (*fn)(VkCommandBuffer, void*),
                                    void* user) {
    if (!device) return VK_ERROR_INITIALIZATION_FAILED;
    VkLockGuard lock(&queueLock);
    vkResetFences(device, 1, &immediateFence);
    vkResetCommandBuffer(immediateCmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(immediateCmd, &bi);
    fn(immediateCmd, user);
    vkEndCommandBuffer(immediateCmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &immediateCmd;
    VkResult r = vkQueueSubmit(gfxQueue, 1, &si, immediateFence);
    if (r == VK_SUCCESS)
        vkWaitForFences(device, 1, &immediateFence, VK_TRUE, UINT64_MAX);
    else
        LogDeviceFault(device, "immediate submit", r);
    return r;
}

void VkBackend::WaitIdle() {
    if (device) vkDeviceWaitIdle(device);
}

void VkBackend::Shutdown() {
    if (!device) return;
    vkDeviceWaitIdle(device);
    g_studShadows.Shutdown();
    g_fpsOverlay.Shutdown();
    memset(texMru_, 0, sizeof(texMru_));
    texMruNext_ = 0;
    DestroySwapchainObjects();
    for (int i = 0; i < 2; ++i) {
        if (frames[i].inFlight) vkDestroyFence(device, frames[i].inFlight, nullptr);
        if (frames[i].renderFinished)
            vkDestroySemaphore(device, frames[i].renderFinished, nullptr);
        if (frames[i].imageAvailable)
            vkDestroySemaphore(device, frames[i].imageAvailable, nullptr);
    }
    if (immediateFence) vkDestroyFence(device, immediateFence, nullptr);
    if (defaultView) vkDestroyImageView(device, defaultView, nullptr);
    if (defaultSampler) vkDestroySampler(device, defaultSampler, nullptr);
    if (defaultImage) vmaDestroyImage(allocator, defaultImage, defaultAlloc);
    if (dummyVb) vmaDestroyBuffer(allocator, dummyVb, dummyVbAlloc);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    for (auto p : descPools) {
        if (p) vkDestroyDescriptorPool(device, p, nullptr);
    }
    descPools.clear();
    descCache.clear();
    if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
    if (allocator) vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

VmaAllocationCreateInfo VkBackend::HostAlloc() {
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return aci;
}

// ---------------------------------------------------------------------------
// Attachments
// ---------------------------------------------------------------------------
VkImage VkBackend::ResolveImage(const ImageRef& r) const {
    if (r.swapchain)
        return r.depth ? depthImage
                       : (imageIndex < images.size() ? images[imageIndex]
                                                     : VK_NULL_HANDLE);
    return r.image;
}

VkImageView VkBackend::ResolveView(const ImageRef& r) const {
    if (r.swapchain)
        return r.depth ? depthView
                       : (imageIndex < views.size() ? views[imageIndex]
                                                    : VK_NULL_HANDLE);
    return r.view;
}

void VkBackend::TransitionImage(VkImage image, VkImageAspectFlags aspect,
                                VkImageLayout newLayout, uint32_t mipLevels) {
    if (!image || !recording) return;
    auto it = layouts_.find(image);
    VkImageLayout old =
        (it == layouts_.end()) ? VK_IMAGE_LAYOUT_UNDEFINED : it->second;
    if (old == newLayout) return;
    if (renderingActive) EndRendering();

    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = old;
    b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {aspect, 0, mipLevels, 0, 1};

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    switch (old) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            b.srcAccessMask = 0;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            b.srcAccessMask = 0;
            srcStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            break;
        default:
            b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            break;
    }

    switch (newLayout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                              VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            b.dstAccessMask = 0;
            dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            break;
        default:
            b.dstAccessMask = 0;
            dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            break;
    }

    vkCmdPipelineBarrier(Cmd(), srcStage, dstStage, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    layouts_[image] = newLayout;
}

bool VkBackend::SetRenderTargets(const ImageRef* colors, uint32_t count,
                                 const ImageRef* depth) {
    uint32_t cCount = count > 4 ? 4 : count;
    bool same = (cCount == targetCount);
    if (same) {
        for (uint32_t i = 0; i < cCount; ++i) {
            if (targetColor[i].image != colors[i].image ||
                targetColor[i].view != colors[i].view ||
                targetColor[i].swapchain != colors[i].swapchain) {
                same = false;
                break;
            }
        }
        if (same) {
            VkImage curD = targetDepth.swapchain ? VK_NULL_HANDLE : targetDepth.image;
            VkImage newD = depth ? (depth->swapchain ? VK_NULL_HANDLE : depth->image) : VK_NULL_HANDLE;
            if (curD != newD) same = false;
        }
    }
    if (same) return false;

    EndRendering();

    if (recording) {
        for (uint32_t i = 0; i < targetCount; ++i) {
            VkImage img = ResolveImage(targetColor[i]);
            if (!img || targetColor[i].swapchain) continue;
            VkImage newImg = depth ? ResolveImage(*depth) : VK_NULL_HANDLE;
            if (img == newImg) continue;
            TransitionImage(img, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        if (targetDepth.image && !targetDepth.swapchain && targetDepth.depth) {
            VkImageAspectFlags a = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (targetDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
                targetDepthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT)
                a |= VK_IMAGE_ASPECT_STENCIL_BIT;
            TransitionImage(targetDepth.image, a,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    targetCount = count > 4 ? 4 : count;
    for (uint32_t i = 0; i < targetCount; ++i) {
        targetColor[i] = colors[i];
        targetFormat[i] = colors[i].format;
    }
    for (uint32_t i = targetCount; i < 4; ++i) {
        targetColor[i] = ImageRef{};
        targetFormat[i] = VK_FORMAT_UNDEFINED;
    }
    targetDepth = depth ? *depth : ImageRef{};
    targetDepthFormat = targetDepth.format;

    if (targetCount) {
        targetW = colors[0].width;
        targetH = colors[0].height;
    } else {
        targetW = extent.width;
        targetH = extent.height;
    }
    if (!targetW) targetW = extent.width;
    if (!targetH) targetH = extent.height;

    if (recording) {
        VkViewport vp{0.0f, (float)targetH, (float)targetW, -(float)targetH,
                      0.0f, 1.0f};
        vkCmdSetViewport(Cmd(), 0, 1, &vp);
        VkRect2D sc{{0, 0}, {targetW, targetH}};
        vkCmdSetScissor(Cmd(), 0, 1, &sc);
    }
    pendingClear = ClearRequest{};
    return true;
}

bool VkBackend::BindSwapchainTargets() {
    ImageRef c;
    c.swapchain = true;
    c.format = colorFormat;
    c.width = extent.width;
    c.height = extent.height;
    ImageRef d;
    d.swapchain = true;
    d.depth = true;
    d.format = depthFormat;
    d.width = extent.width;
    d.height = extent.height;
    return SetRenderTargets(&c, 1, &d);
}

ImageRef VkBackend::DepthFor(uint32_t w, uint32_t h, VkFormat format) {
    if (!w) w = extent.width;
    if (!h) h = extent.height;
    VkFormat f = format != VK_FORMAT_UNDEFINED ? format : depthFormat;
    for (const DepthTarget& d : depthTargets) {
        if (d.width == w && d.height == h && d.format == f) {
            ImageRef r;
            r.image = d.image;
            r.view = d.view;
            r.format = f;
            r.width = w;
            r.height = h;
            r.depth = true;
            return r;
        }
    }

    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT)
        aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

    DepthTarget d;
    d.width = w;
    d.height = h;
    d.format = f;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = f;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(allocator, &ici, &aci, &d.image, &d.alloc, nullptr) !=
        VK_SUCCESS) {
        LOG("[rt] cannot create %ux%u depth target", w, h);
        return ImageRef{};
    }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = d.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = f;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vci, nullptr, &d.view) != VK_SUCCESS) {
        vmaDestroyImage(allocator, d.image, d.alloc);
        LOG("[rt] cannot create %ux%u depth view", w, h);
        return ImageRef{};
    }
    TransitionImage(d.image, aspect, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    depthTargets.push_back(d);
    LOG("[rt] created %ux%u depth target fmt=%d", w, h, (int)f);

    ImageRef r;
    r.image = d.image;
    r.view = d.view;
    r.format = f;
    r.width = w;
    r.height = h;
    r.depth = true;
    return r;
}

void VkBackend::BlitImage(const ImageRef& src, const RECT* srcRect,
                          const ImageRef& dst, const RECT* dstRect) {
    if (!recording) return;
    EndRendering();
    VkImage si = ResolveImage(src);
    VkImage di = ResolveImage(dst);
    if (!si || !di) return;

    VkImageAspectFlags sa =
        src.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageAspectFlags da =
        dst.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    TransitionImage(si, sa, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    TransitionImage(di, da, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageBlit blit{};
    blit.srcSubresource = {sa, 0, 0, 1};
    blit.dstSubresource = {da, 0, 0, 1};
    if (srcRect) {
        blit.srcOffsets[0] = {srcRect->left, srcRect->top, 0};
        blit.srcOffsets[1] = {srcRect->right, srcRect->bottom, 1};
    } else {
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {(int32_t)src.width, (int32_t)src.height, 1};
    }
    if (dstRect) {
        blit.dstOffsets[0] = {dstRect->left, dstRect->top, 0};
        blit.dstOffsets[1] = {dstRect->right, dstRect->bottom, 1};
    } else {
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {(int32_t)dst.width, (int32_t)dst.height, 1};
    }

    // Two colour images take the blit path; depth images cannot be blitted, so
    // they always copy. The old condition was true for every colour pair via
    // "src.depth == dst.depth", and wrong for depth pairs.
    if (!src.depth && !dst.depth) {
        vkCmdBlitImage(Cmd(), si, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, di,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                       VK_FILTER_LINEAR);
    } else {
        // Format-incompatible blit: copy only when the extents agree.
        VkImageCopy c{};
        c.srcSubresource = blit.srcSubresource;
        c.dstSubresource = blit.dstSubresource;
        c.extent = {(uint32_t)(blit.srcOffsets[1].x - blit.srcOffsets[0].x),
                    (uint32_t)(blit.srcOffsets[1].y - blit.srcOffsets[0].y),
                    1u};
        vkCmdCopyImage(Cmd(), si, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, di,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    }

    TransitionImage(si, sa, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionImage(di, da, dst.depth
                               ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                               : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

bool VkBackend::CreateDefaultCube() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 6;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(allocator, &ici, &aci, &defaultCubeImage,
                       &defaultCubeAlloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = defaultCubeImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    vci.format = ici.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
    vkCreateImageView(device, &vci, nullptr, &defaultCubeView);

    ImmediateSubmit(
        [](VkCommandBuffer cmd, void* user) {
            VkImage img = (VkImage)user;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = img;
            b.subresourceRange = range;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b);
            VkClearColorValue white{};
            for (int i = 0; i < 4; ++i) white.float32[i] = 1.0f;
            vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &white, 1, &range);
            VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.image = img;
            b2.subresourceRange = range;
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b2);
        },
        (void*)defaultCubeImage);
    return true;
}

bool VkBackend::CreateDefaultVolume() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(allocator, &ici, &aci, &defaultVolumeImage,
                       &defaultVolumeAlloc, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = defaultVolumeImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vci.format = ici.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &vci, nullptr, &defaultVolumeView);

    VkSamplerCreateInfo vsci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    vsci.magFilter = VK_FILTER_LINEAR;
    vsci.minFilter = VK_FILTER_LINEAR;
    vsci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vsci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vsci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vsci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &vsci, nullptr, &defaultVolumeSampler);

    ImmediateSubmit(
        [](VkCommandBuffer cmd, void* user) {
            VkImage img = (VkImage)user;
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = img;
            b.subresourceRange = range;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b);
            VkClearColorValue white{};
            for (int i = 0; i < 4; ++i) white.float32[i] = 1.0f;
            vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &white, 1, &range);
            VkImageMemoryBarrier b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.image = img;
            b2.subresourceRange = range;
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b2);
        },
        (void*)defaultVolumeImage);
    return true;
}
