// Numeric harness for the SM3 -> SPIR-V code generator.
//
// The structural selftest only proves the driver accepts a module. This tool
// answers the question that actually matters for rendering: does the generated
// SPIR-V compute the same values D3D9 specifies, lane for lane?
//
// Each case is a hand-encoded vs_3_0 stream. The shader writes its position
// straight from a constant (so the primitive is always inside the clip volume)
// and copies the value under test into a varying. A matching ps_3_0 shader
// forwards that varying to the colour output. The case is drawn as a single
// point into a 1x1 RGBA32F target, so the pixel that comes back is exactly the
// vertex's varying with no interpolation, and it is compared against a CPU
// reference implementing the D3D9 lane semantics.
//
// Exit code 0 means every case matched; 1 means a mismatch or a setup failure.

#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sm3_to_spirv.h"

// The compiler reports internal diagnostics through Logf; the harness only
// cares about pass/fail, so they go to stderr.
void Logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

// ---------------------------------------------------------------------------
// SM3 token encoding
// ---------------------------------------------------------------------------
enum {
    OPC_MOV = 1,
    OPC_ADD = 2,
    OPC_MAD = 4,
    OPC_MUL = 5,
    OPC_DP3 = 8,
    OPC_DCL = 31,
    OPC_END = 0xFFFF
};

enum {
    RG_TEMP = 0,
    RG_INPUT = 1,
    RG_CONST = 2,
    RG_RASTOUT = 4,
    RG_OUTPUT = 6,
    RG_COLOROUT = 8
};

enum { USAGE_POSITION = 0, USAGE_TEXCOORD = 5 };

static uint32_t Inst(uint32_t opcode, uint32_t params) {
    return opcode | (params << 24);
}

// Source operand. swz packs four 2-bit lanes, 0xE4 being the identity .xyzw.
static uint32_t Src(uint32_t type, uint32_t num, uint32_t swz = 0xE4,
                    uint32_t mod = 0) {
    return (1u << 31) | ((type & 0x7u) << 28) | ((mod & 0xFu) << 24) |
           ((swz & 0xFFu) << 16) | (((type >> 3) & 0x3u) << 11) | (num & 0x7FFu);
}

static uint32_t Dst(uint32_t type, uint32_t num, uint32_t mask = 0xF,
                    uint32_t shift = 0) {
    return ((type & 0x7u) << 28) | ((shift & 0xFu) << 24) | ((mask & 0xFu) << 16) |
           (((type >> 3) & 0x3u) << 11) | (num & 0x7FFu);
}

static uint32_t Decl(uint32_t usage, uint32_t index) {
    return 0x80000000u | ((index & 0xFu) << 16) | (usage & 0x1Fu);
}

// Swizzle pack for a repeated single lane, e.g. .yyyy == 0x55.
static uint32_t Swz(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    return (x & 3u) | ((y & 3u) << 2) | ((z & 3u) << 4) | ((w & 3u) << 6);
}

// Prints the interface a module declares: entry-point interface ids, their
// Location or BuiltIn decoration, and the type width. A vertex/fragment pair
// only links when every fragment input location has a vertex output.
static void DumpInterface(const std::vector<uint32_t>& spv, const char* label) {
    if (spv.size() < 5) return;
    std::vector<uint32_t> iface;
    std::vector<std::pair<uint32_t, uint32_t>> vars;  // id -> storage class
    std::vector<std::pair<uint32_t, uint32_t>> decs;  // id, decoration, literal
    std::vector<std::pair<uint32_t, uint32_t>> decLit;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t wc = spv[i] >> 16;
        const uint32_t op = spv[i] & 0xFFFFu;
        if (!wc || i + wc > spv.size()) break;
        if (op == 15 /*OpEntryPoint*/) {
            printf("    %s model=%u words=%zu\n", label, spv[i + 1], spv.size());
            size_t k = i + 3;
            while (k < i + wc) {
                const uint32_t w = spv[k++];
                if (w & 0xFF000000u) break;
                if (w & 0x00FF0000u) break;
                if (w & 0x0000FF00u) break;
                if (w & 0x000000FFu) break;
            }
            while (k < i + wc) iface.push_back(spv[k++]);
        }
        if (op == 59 /*OpVariable*/ && wc >= 4) vars.push_back({spv[i + 2], spv[i + 3]});
        if (op == 71 /*OpDecorate*/ && wc >= 4) {
            decs.push_back({spv[i + 1], spv[i + 2]});
            decLit.push_back({spv[i + 1], spv[i + 3]});
        }
        i += wc;
    }
    if (iface.empty()) {
        printf("    %s interface empty\n", label);
        return;
    }
    for (size_t d = 0; d < decs.size(); ++d) {
        const uint32_t target = decs[d].first, dec = decs[d].second;
        bool listed = false;
        for (uint32_t id : iface) listed |= (id == target);
        if (!listed) continue;
        if (dec != 11 && dec != 30) continue;
        uint32_t sc = 99;
        for (auto& v : vars)
            if (v.first == target) sc = v.second;
        printf("    %s %-4s %s %u\n", label,
               sc == 1 ? "in" : sc == 3 ? "out" : "var",
               dec == 30 ? "loc" : "builtin", decLit[d].second);
    }
}

static std::vector<uint32_t> MakeVs(const std::vector<uint32_t>& body) {
    std::vector<uint32_t> t;
    t.push_back(0xFFFE0300u);  // vs_3_0
    t.push_back(Inst(OPC_DCL, 2));
    t.push_back(Decl(USAGE_POSITION, 0));
    t.push_back(Dst(RG_OUTPUT, 0));
    t.push_back(Inst(OPC_DCL, 2));
    t.push_back(Decl(USAGE_TEXCOORD, 0));
    t.push_back(Dst(RG_OUTPUT, 1));
    // o0 = c0, a constant chosen to land the point on the pixel centre.
    t.push_back(Inst(OPC_MOV, 2));
    t.push_back(Dst(RG_OUTPUT, 0));
    t.push_back(Src(RG_CONST, 0));
    t.insert(t.end(), body.begin(), body.end());
    // o1 = r0, the value under test.
    t.push_back(Inst(OPC_MOV, 2));
    t.push_back(Dst(RG_OUTPUT, 1));
    t.push_back(Src(RG_TEMP, 0));
    t.push_back(OPC_END);
    return t;
}

static std::vector<uint32_t> MakePs(uint32_t texIndex = 0) {
    std::vector<uint32_t> t;
    t.push_back(0xFFFF0300u);  // ps_3_0: 0xFFFF marks the pixel stage, 0xFFFE vertex
    t.push_back(Inst(OPC_DCL, 2));
    t.push_back(Decl(USAGE_TEXCOORD, texIndex));
    t.push_back(Dst(RG_INPUT, 0));
    t.push_back(Inst(OPC_MOV, 2));
    t.push_back(Dst(RG_COLOROUT, 0));
    t.push_back(Src(RG_INPUT, 0));
    t.push_back(OPC_END);
    return t;
}

// ---------------------------------------------------------------------------
// Constant file used by every case. c0 places the point; c1..c4 are operands.
// ---------------------------------------------------------------------------
static const float kConsts[8][4] = {
    {0.0f, 0.0f, 0.5f, 1.0f},      // c0 position
    {1.5f, -2.25f, 3.75f, -4.5f},  // c1
    {2.0f, -0.5f, 4.0f, 0.25f},    // c2
    {0.75f, 1.25f, -1.5f, 2.5f},   // c3
    {0.1f, 0.2f, 0.3f, 0.4f},      // c4
    {0, 0, 0, 0},
    {0, 0, 0, 0},
    {0, 0, 0, 0},
};

// ---------------------------------------------------------------------------
// CPU reference: D3D9 evaluates each instruction per destination lane, reading
// the source lane named by that lane's swizzle component. Lanes outside the
// write mask keep whatever the register already held (zero here).
// ---------------------------------------------------------------------------
struct Ref {
    float r[4] = {0, 0, 0, 0};

    static float Lane(const float* v, uint32_t swz, uint32_t lane) {
        return v[(swz >> (lane * 2)) & 3u];
    }

    void Apply(uint32_t mask, const float* a, uint32_t aswz, const float* b,
               uint32_t bswz, int op, float scale = 1.0f) {
        for (uint32_t i = 0; i < 4; ++i) {
            if (!(mask & (1u << i))) continue;
            const float x = Lane(a, aswz, i);
            const float y = b ? Lane(b, bswz, i) : 0.0f;
            switch (op) {
                case OPC_MOV: r[i] = x; break;
                case OPC_MUL: r[i] = x * y * scale; break;
                case OPC_MAD: r[i] = x * y * scale + Lane(b, bswz, i); break;
                case OPC_ADD: r[i] = x + y * scale; break;
                case OPC_DP3: break;
                default: break;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Minimal headless Vulkan device: one 1x1 float target, one point per case.
// ---------------------------------------------------------------------------
struct Gpu {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkPhysicalDeviceMemoryProperties mem{};
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer vsUbo = VK_NULL_HANDLE, psUbo = VK_NULL_HANDLE;
    VkDeviceMemory vsMem = VK_NULL_HANDLE, psMem = VK_NULL_HANDLE;
    void* vsMap = nullptr;
    void* psMap = nullptr;
    VkBuffer readBuf = VK_NULL_HANDLE;
    VkDeviceMemory readMem = VK_NULL_HANDLE;
    void* readMap = nullptr;

    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
    PFN_vkCreateImage CreateImage = nullptr;
    PFN_vkDestroyImage DestroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements = nullptr;
    PFN_vkBindImageMemory BindImageMemory = nullptr;
    PFN_vkCreateImageView CreateImageView = nullptr;
    PFN_vkDestroyImageView DestroyImageView = nullptr;
    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers = nullptr;
    PFN_vkCmdBeginRendering CmdBeginRendering = nullptr;
    PFN_vkCmdEndRendering CmdEndRendering = nullptr;
    PFN_vkCmdDraw CmdDraw = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
    PFN_vkResetFences ResetFences = nullptr;
};

template <typename T>
static bool Load(VkDevice dev, PFN_vkGetDeviceProcAddr gdpa, const char* name, T& out) {
    out = (T)gdpa(dev, name);
    return out != nullptr;
}

static uint32_t PickMemory(Gpu& g, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

static bool GpuInit(Gpu& g, std::string& err) {
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) {
        err = "vulkan-1.dll not found";
        return false;
    }
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    if (!gipa) {
        err = "vkGetInstanceProcAddr missing";
        return false;
    }
    auto CreateInstance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    if (!CreateInstance) {
        err = "vkCreateInstance missing";
        return false;
    }

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    ai.pApplicationName = "NuVulkan numeric VS harness";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    if (CreateInstance(&ici, nullptr, &g.inst) != VK_SUCCESS) {
        err = "vkCreateInstance failed";
        return false;
    }

    auto EnumPhys = (PFN_vkEnumeratePhysicalDevices)gipa(g.inst, "vkEnumeratePhysicalDevices");
    auto GetProps = (PFN_vkGetPhysicalDeviceProperties)gipa(g.inst, "vkGetPhysicalDeviceProperties");
    auto GetQ = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(
        g.inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    auto GetMem = (PFN_vkGetPhysicalDeviceMemoryProperties)gipa(
        g.inst, "vkGetPhysicalDeviceMemoryProperties");
    if (!EnumPhys || !GetProps || !GetQ || !GetMem) {
        err = "instance entry points missing";
        return false;
    }

    uint32_t n = 0;
    EnumPhys(g.inst, &n, nullptr);
    if (!n) {
        err = "no physical devices";
        return false;
    }
    std::vector<VkPhysicalDevice> devs(n);
    EnumPhys(g.inst, &n, devs.data());

    VkPhysicalDeviceProperties props{};
    for (VkPhysicalDevice d : devs) {
        GetProps(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g.phys = d;
            break;
        }
        if (!g.phys) g.phys = d;
    }
    GetProps(g.phys, &props);
    printf("device: %s\n", props.deviceName);
    GetMem(g.phys, &g.mem);

    uint32_t qn = 0;
    GetQ(g.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    GetQ(g.phys, &qn, qs.data());
    g.qf = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i)
        if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qs[i].queueCount) {
            g.qf = i;
            break;
        }
    if (g.qf == UINT32_MAX) {
        err = "no graphics queue family";
        return false;
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    auto CreateDevice = (PFN_vkCreateDevice)gipa(g.inst, "vkCreateDevice");
    if (!CreateDevice || CreateDevice(g.phys, &dci, nullptr, &g.dev) != VK_SUCCESS) {
        err = "vkCreateDevice failed";
        return false;
    }
    auto GetQueue = (PFN_vkGetDeviceQueue)gipa(g.inst, "vkGetDeviceQueue");
    auto gdpa = (PFN_vkGetDeviceProcAddr)gipa(g.inst, "vkGetDeviceProcAddr");
    if (!GetQueue || !gdpa) {
        err = "device lookup missing";
        return false;
    }
    GetQueue(g.dev, g.qf, 0, &g.queue);

    struct Entry {
        const char* name;
        bool ok;
    };
    bool ok = true;
#pragma warning(push)
#pragma warning(disable : 4127)
#define LOAD(field) ok &= Load(g.dev, gdpa, "vk" #field, g.field)
    LOAD(CreateShaderModule);
    LOAD(DestroyShaderModule);
    LOAD(CreateGraphicsPipelines);
    LOAD(DestroyPipeline);
    LOAD(CreateImage);
    LOAD(DestroyImage);
    LOAD(GetImageMemoryRequirements);
    LOAD(BindImageMemory);
    LOAD(CreateImageView);
    LOAD(DestroyImageView);
    LOAD(CreateBuffer);
    LOAD(DestroyBuffer);
    LOAD(GetBufferMemoryRequirements);
    LOAD(BindBufferMemory);
    LOAD(AllocateMemory);
    LOAD(FreeMemory);
    LOAD(MapMemory);
    LOAD(UnmapMemory);
    LOAD(CreateDescriptorPool);
    LOAD(AllocateDescriptorSets);
    LOAD(UpdateDescriptorSets);
    LOAD(CreateCommandPool);
    LOAD(AllocateCommandBuffers);
    LOAD(BeginCommandBuffer);
    LOAD(EndCommandBuffer);
    LOAD(CmdBindPipeline);
    LOAD(CmdBindDescriptorSets);
    LOAD(CmdBindVertexBuffers);
    LOAD(CmdBeginRendering);
    LOAD(CmdEndRendering);
    LOAD(CmdDraw);
    LOAD(CmdPipelineBarrier);
    LOAD(CmdCopyImageToBuffer);
    LOAD(QueueSubmit);
    LOAD(CreateFence);
    LOAD(WaitForFences);
    LOAD(ResetFences);
#undef LOAD
#pragma warning(pop)
    if (!ok) {
        err = "device entry points missing";
        return false;
    }

    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 2;
    slci.pBindings = binds;
    // Layout entry points are fetched inline; the harness needs each only once.
    auto CreateSetLayout =
        (PFN_vkCreateDescriptorSetLayout)gdpa(g.dev, "vkCreateDescriptorSetLayout");
    auto CreatePipeLayout = (PFN_vkCreatePipelineLayout)gdpa(g.dev, "vkCreatePipelineLayout");
    auto DestroySetLayout =
        (PFN_vkDestroyDescriptorSetLayout)gdpa(g.dev, "vkDestroyDescriptorSetLayout");
    auto DestroyPipeLayout =
        (PFN_vkDestroyPipelineLayout)gdpa(g.dev, "vkDestroyPipelineLayout");
    (void)DestroySetLayout;
    (void)DestroyPipeLayout;
    if (!CreateSetLayout || !CreatePipeLayout) {
        err = "layout entry points missing";
        return false;
    }
    if (CreateSetLayout(g.dev, &slci, nullptr, &g.setLayout) != VK_SUCCESS) {
        err = "descriptor set layout failed";
        return false;
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.setLayout;
    if (CreatePipeLayout(g.dev, &plci, nullptr, &g.pipeLayout) != VK_SUCCESS) {
        err = "pipeline layout failed";
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = 4112;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf = VK_NULL_HANDLE;
        if (g.CreateBuffer(g.dev, &bci, nullptr, &buf) != VK_SUCCESS) {
            err = "uniform buffer failed";
            return false;
        }
        VkMemoryRequirements req{};
        g.GetBufferMemoryRequirements(g.dev, buf, &req);
        const uint32_t mt = PickMemory(g, req.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mt;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (mt == UINT32_MAX || g.AllocateMemory(g.dev, &mai, nullptr, &mem) != VK_SUCCESS) {
            err = "uniform memory failed";
            return false;
        }
        g.BindBufferMemory(g.dev, buf, mem, 0);
        void* map = nullptr;
        g.MapMemory(g.dev, mem, 0, 4112, 0, &map);
        memset(map, 0, 4112);
        memcpy(map, kConsts, sizeof(kConsts));
        if (i == 0) {
            g.vsUbo = buf;
            g.vsMem = mem;
            g.vsMap = map;
        } else {
            g.psUbo = buf;
            g.psMem = mem;
            g.psMap = map;
        }
    }

    VkBufferCreateInfo rb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    rb.size = 16;
    rb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (g.CreateBuffer(g.dev, &rb, nullptr, &g.readBuf) != VK_SUCCESS) {
        err = "readback buffer failed";
        return false;
    }
    VkMemoryRequirements rr{};
    g.GetBufferMemoryRequirements(g.dev, g.readBuf, &rr);
    const uint32_t rmt = PickMemory(g, rr.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo rma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    rma.allocationSize = rr.size;
    rma.memoryTypeIndex = rmt;
    if (rmt == UINT32_MAX || g.AllocateMemory(g.dev, &rma, nullptr, &g.readMem) != VK_SUCCESS) {
        err = "readback memory failed";
        return false;
    }
    g.BindBufferMemory(g.dev, g.readBuf, g.readMem, 0);
    g.MapMemory(g.dev, g.readMem, 0, 16, 0, &g.readMap);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (g.CreateDescriptorPool(g.dev, &dpci, nullptr, &g.descPool) != VK_SUCCESS) {
        err = "descriptor pool failed";
        return false;
    }
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = g.descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g.setLayout;
    if (g.AllocateDescriptorSets(g.dev, &dsai, &g.set) != VK_SUCCESS) {
        err = "descriptor set failed";
        return false;
    }
    VkDescriptorBufferInfo vb{g.vsUbo, 0, 4112};
    VkDescriptorBufferInfo pb{g.psUbo, 0, 4112};
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = g.set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[0].pBufferInfo = &vb;
    w[1] = w[0];
    w[1].dstBinding = 1;
    w[1].pBufferInfo = &pb;
    g.UpdateDescriptorSets(g.dev, 2, w, 0, nullptr);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = g.qf;
    if (g.CreateCommandPool(g.dev, &cpci, nullptr, &g.pool) != VK_SUCCESS) {
        err = "command pool failed";
        return false;
    }
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (g.AllocateCommandBuffers(g.dev, &cbai, &g.cmd) != VK_SUCCESS) {
        err = "command buffer failed";
        return false;
    }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (g.CreateFence(g.dev, &fci, nullptr, &g.fence) != VK_SUCCESS) {
        err = "fence failed";
        return false;
    }
    return true;
}

// Everything a single draw needs: constant file, optional vertex stream.
struct DrawCfg {
    std::vector<float> consts;  // 256 vec4
    VkBuffer vbuf = VK_NULL_HANDLE;
    uint32_t vstride = 0;
    std::vector<VkVertexInputAttributeDescription> attrs;
};

static DrawCfg DefaultCfg() {
    DrawCfg c;
    c.consts.assign(1024, 0.0f);
    memcpy(c.consts.data(), kConsts, sizeof(kConsts));
    return c;
}

// Runs one vertex shader over a single point and returns the pixel it wrote.
static bool RunCase(Gpu& g, const std::vector<uint32_t>& vs, const std::vector<uint32_t>& ps,
                    const DrawCfg& cfg, float out[4], std::string& err) {
    memcpy(g.vsMap, cfg.consts.data(), 4096);
    memset(g.readMap, 0, 16);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = vs.size() * 4;
    smci.pCode = vs.data();
    VkShaderModule vm = VK_NULL_HANDLE;
    if (g.CreateShaderModule(g.dev, &smci, nullptr, &vm) != VK_SUCCESS) {
        err = "VS module rejected";
        return false;
    }
    smci.pCode = ps.data();
    smci.codeSize = ps.size() * 4;
    VkShaderModule pm = VK_NULL_HANDLE;
    if (g.CreateShaderModule(g.dev, &smci, nullptr, &pm) != VK_SUCCESS) {
        g.DestroyShaderModule(g.dev, vm, nullptr);
        err = "PS module rejected";
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vm;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = pm;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkVertexInputBindingDescription vbind{0, cfg.vstride, VK_VERTEX_INPUT_RATE_VERTEX};
    if (cfg.vbuf) {
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vbind;
        vi.vertexAttributeDescriptionCount = (uint32_t)cfg.attrs.size();
        vi.pVertexAttributeDescriptions = cfg.attrs.data();
    }
    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {1, 1}};
    VkPipelineViewportStateCreateInfo vs2{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs2.viewportCount = 1;
    vs2.pViewports = &vp;
    vs2.scissorCount = 1;
    vs2.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // With dynamic rendering there is no render pass to supply defaults, and
    // the driver faults while compiling a pipeline whose depth state is fully
    // disabled but whose rendering info names a depth format.
    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkPipelineRenderingCreateInfo prci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prci.colorAttachmentCount = 1;
    prci.pColorAttachmentFormats = &fmt;
    prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &prci;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vs2;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.layout = g.pipeLayout;

    VkPipeline pipe = VK_NULL_HANDLE;
    const VkResult pr =
        g.CreateGraphicsPipelines(g.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe);
    if (pr != VK_SUCCESS) {
        // Localise the failure: a vertex-only pipeline with rasterisation
        // discarded is legal, so if that builds the vertex module is fine and
        // the problem is the vertex/fragment interface.
        VkGraphicsPipelineCreateInfo only = gp;
        only.stageCount = 1;
        VkPipelineRasterizationStateCreateInfo discard = rs;
        discard.rasterizerDiscardEnable = VK_TRUE;
        only.pRasterizationState = &discard;
        VkPipeline probe = VK_NULL_HANDLE;
        const VkResult pr2 =
            g.CreateGraphicsPipelines(g.dev, VK_NULL_HANDLE, 1, &only, nullptr, &probe);
        if (pr2 == VK_SUCCESS) g.DestroyPipeline(g.dev, probe, nullptr);
        g.DestroyShaderModule(g.dev, vm, nullptr);
        g.DestroyShaderModule(g.dev, pm, nullptr);
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "pipeline failed (%d); vertex-only pipeline %s", (int)pr,
                 pr2 == VK_SUCCESS ? "succeeded -> interface problem"
                                   : "also failed -> vertex module problem");
        err = buf;
        return false;
    }

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {1, 1, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage img = VK_NULL_HANDLE;
    if (g.CreateImage(g.dev, &ici, nullptr, &img) != VK_SUCCESS) {
        g.DestroyPipeline(g.dev, pipe, nullptr);
        g.DestroyShaderModule(g.dev, vm, nullptr);
        g.DestroyShaderModule(g.dev, pm, nullptr);
        err = "target image failed";
        return false;
    }
    VkMemoryRequirements ir{};
    g.GetImageMemoryRequirements(g.dev, img, &ir);
    VkMemoryAllocateInfo ima{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ima.allocationSize = ir.size;
    ima.memoryTypeIndex = PickMemory(g, ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory imem = VK_NULL_HANDLE;
    g.AllocateMemory(g.dev, &ima, nullptr, &imem);
    g.BindImageMemory(g.dev, img, imem, 0);

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = img;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = fmt;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    g.CreateImageView(g.dev, &ivci, nullptr, &view);

    // A depth attachment must be present because the pipeline declares a depth
    // format; the test writes no depth, so it only has to be cleared.
    VkImageCreateInfo dci = ici;
    dci.format = VK_FORMAT_D32_SFLOAT;
    dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    VkImage dimg = VK_NULL_HANDLE;
    g.CreateImage(g.dev, &dci, nullptr, &dimg);
    VkMemoryRequirements dr{};
    g.GetImageMemoryRequirements(g.dev, dimg, &dr);
    VkMemoryAllocateInfo dma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    dma.allocationSize = dr.size;
    dma.memoryTypeIndex = PickMemory(g, dr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory dmem = VK_NULL_HANDLE;
    g.AllocateMemory(g.dev, &dma, nullptr, &dmem);
    g.BindImageMemory(g.dev, dimg, dmem, 0);
    VkImageViewCreateInfo dvci = ivci;
    dvci.image = dimg;
    dvci.format = VK_FORMAT_D32_SFLOAT;
    dvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VkImageView dview = VK_NULL_HANDLE;
    g.CreateImageView(g.dev, &dvci, nullptr, &dview);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    g.BeginCommandBuffer(g.cmd, &bi);

    VkImageMemoryBarrier toColor{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toColor.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColor.image = img;
    toColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    g.CmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toColor);

    VkImageMemoryBarrier toDepth = toColor;
    toDepth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDepth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    toDepth.image = dimg;
    toDepth.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    toDepth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    g.CmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &toDepth);

    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    // A sentinel rather than 0: it distinguishes "the point never covered the
    // pixel" from "the shader wrote zeros".
    for (int k = 0; k < 4; ++k) att.clearValue.color.float32[k] = 1234.0f;
    VkRenderingAttachmentInfo datt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    datt.imageView = dview;
    datt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    datt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    datt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    datt.clearValue.depthStencil.depth = 1.0f;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = sc;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    ri.pDepthAttachment = &datt;
    g.CmdBeginRendering(g.cmd, &ri);
    g.CmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    g.CmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeLayout, 0, 1,
                            &g.set, 0, nullptr);
    if (cfg.vbuf) {
        VkDeviceSize offset = 0;
        g.CmdBindVertexBuffers(g.cmd, 0, 1, &cfg.vbuf, &offset);
    }
    g.CmdDraw(g.cmd, 1, 1, 0, 0);
    g.CmdEndRendering(g.cmd);

    VkImageMemoryBarrier toCopy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toCopy.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopy.image = img;
    toCopy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toCopy.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toCopy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    g.CmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &toCopy);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {1, 1, 1};
    g.CmdCopyImageToBuffer(g.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g.readBuf, 1,
                           &region);
    g.EndCommandBuffer(g.cmd);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd;
    g.ResetFences(g.dev, 1, &g.fence);
    g.QueueSubmit(g.queue, 1, &si, g.fence);
    g.WaitForFences(g.dev, 1, &g.fence, VK_TRUE, 1000000000ull);

    memcpy(out, g.readMap, sizeof(float) * 4);

    g.DestroyImageView(g.dev, dview, nullptr);
    g.DestroyImage(g.dev, dimg, nullptr);
    g.FreeMemory(g.dev, dmem, nullptr);
    g.DestroyImageView(g.dev, view, nullptr);
    g.DestroyImage(g.dev, img, nullptr);
    g.FreeMemory(g.dev, imem, nullptr);
    g.DestroyPipeline(g.dev, pipe, nullptr);
    g.DestroyShaderModule(g.dev, vm, nullptr);
    g.DestroyShaderModule(g.dev, pm, nullptr);
    return true;
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------
struct Case {
    const char* name;
    std::vector<uint32_t> body;
    uint32_t mask;   // lanes the body writes
    int op;          // semantic used for the reference
    const float* a;  // source 1
    uint32_t aswz;
    const float* b;  // source 2 (MAD: addend)
    uint32_t bswz;
};

static bool Same(const float* a, const float* b) {
    for (int i = 0; i < 4; ++i) {
        const float d = fabsf(a[i] - b[i]);
        const float s = fmaxf(1.0f, fmaxf(fabsf(a[i]), fabsf(b[i])));
        if (!(d <= 1e-5f * s)) return false;
    }
    return true;
}

#include "vs_numeric_real.inc"

int main(int argc, char** argv) {
    Gpu g;
    std::string err;
    if (!GpuInit(g, err)) {
        printf("SETUP FAILED: %s\n", err.c_str());
        return 1;
    }

    std::vector<uint32_t> psTokens = MakePs(0);
    std::vector<uint32_t> psSpv;
    std::string cerr;
    if (!Sm3CompileToSpirv(true, psTokens.data(), (uint32_t)psTokens.size(), psSpv, cerr)) {
        printf("PS compile failed: %s\n", cerr.c_str());
        return 1;
    }

    const float* c1 = kConsts[1];
    const float* c2 = kConsts[2];
    const float* c3 = kConsts[3];
    const float* c4 = kConsts[4];

    std::vector<Case> cases;

    // mul r0.yzw, c1.xxyz, c2.y  -- replicated swizzle on operator A, scalar
    // replicate on operator B, partial write mask.
    {
        Case k{};
        k.name = "mul r0.yzw, c1.xxyz, c2.y";
        k.body = {Inst(OPC_MUL, 3), Dst(RG_TEMP, 0, 0xE), Src(RG_CONST, 1, Swz(0, 0, 1, 2)),
                  Src(RG_CONST, 2, Swz(1, 1, 1, 1))};
        k.mask = 0xE;
        k.op = OPC_MUL;
        k.a = c1;
        k.aswz = Swz(0, 0, 1, 2);
        k.b = c2;
        k.bswz = Swz(1, 1, 1, 1);
        cases.push_back(k);
    }
    // mad r0.xyw, c1.x, c3.xyzz, c4  -- three-lane mask, replicated A, swizzled
    // addend whose swizzle must be applied per destination lane.
    {
        Case k{};
        k.name = "mad r0.xyw, c1.x, c3.xyzz, c4";
        k.body = {Inst(OPC_MAD, 4), Dst(RG_TEMP, 0, 0xB), Src(RG_CONST, 1, Swz(0, 0, 0, 0)),
                  Src(RG_CONST, 3, Swz(0, 1, 2, 2)), Src(RG_CONST, 4, Swz(0, 1, 2, 3))};
        k.mask = 0xB;
        k.op = OPC_MAD;
        k.a = c1;
        k.aswz = Swz(0, 0, 0, 0);
        k.b = c3;
        k.bswz = Swz(0, 1, 2, 2);
        cases.push_back(k);
    }
    // mov r0.xz, c1.yxwz  -- reversed swizzle under a holey mask.
    {
        Case k{};
        k.name = "mov r0.xz, c1.yxwz";
        k.body = {Inst(OPC_MOV, 2), Dst(RG_TEMP, 0, 0x5), Src(RG_CONST, 1, Swz(1, 0, 3, 2))};
        k.mask = 0x5;
        k.op = OPC_MOV;
        k.a = c1;
        k.aswz = Swz(1, 0, 3, 2);
        k.b = nullptr;
        k.bswz = 0;
        cases.push_back(k);
    }
    // add r0.zw, c1, -c2  -- negate modifier with a two-lane mask.
    {
        Case k{};
        k.name = "add r0.zw, c1, -c2";
        // D3DSPSM_NEG is 1; 2 is D3DSPSM_BIAS, which subtracts 0.5 instead.
        k.body = {Inst(OPC_ADD, 3), Dst(RG_TEMP, 0, 0xC), Src(RG_CONST, 1),
                  Src(RG_CONST, 2, 0xE4, 1 /* negate */)};
        k.mask = 0xC;
        k.op = OPC_ADD;
        k.a = c1;
        k.aswz = 0xE4;
        k.b = c2;
        k.bswz = 0xE4;
        cases.push_back(k);
    }
    // dp3 r0.x, c1, c2  -- reduction into a single lane.
    {
        Case k{};
        k.name = "dp3 r0.x, c1, c2";
        k.body = {Inst(OPC_DP3, 3), Dst(RG_TEMP, 0, 0x1), Src(RG_CONST, 1),
                  Src(RG_CONST, 2)};
        k.mask = 0x1;
        k.op = OPC_DP3;
        k.a = c1;
        k.aswz = 0xE4;
        k.b = c2;
        k.bswz = 0xE4;
        cases.push_back(k);
    }

    int failed = 0;
    for (const Case& k : cases) {
        std::vector<uint32_t> vsTokens = MakeVs(k.body);
        std::vector<uint32_t> vsSpv;
        if (!Sm3CompileToSpirv(false, vsTokens.data(), (uint32_t)vsTokens.size(), vsSpv,
                               cerr)) {
            printf("FAIL %-38s compile: %s\n", k.name, cerr.c_str());
            ++failed;
            continue;
        }

        float gpu[4] = {NAN, NAN, NAN, NAN};
        if (!RunCase(g, vsSpv, psSpv, DefaultCfg(), gpu, err)) {
            printf("FAIL %-38s run: %s\n", k.name, err.c_str());
            if (failed == 0) {
                DumpInterface(vsSpv, "vs");
                DumpInterface(psSpv, "ps");
            }
            ++failed;
            continue;
        }

        Ref ref;
        if (k.op == OPC_DP3) {
            ref.r[0] = k.a[0] * k.b[0] + k.a[1] * k.b[1] + k.a[2] * k.b[2];
        } else if (k.op == OPC_MAD) {
            // MAD: per-lane a*b + addend. The third operand is the addend, so
            // evaluate it directly rather than through Apply's two-operand path.
            const float addend[4] = {c4[0], c4[1], c4[2], c4[3]};
            for (uint32_t i = 0; i < 4; ++i) {
                if (!(k.mask & (1u << i))) continue;
                const float x = Ref::Lane(k.a, k.aswz, i);
                const float y = Ref::Lane(k.b, k.bswz, i);
                ref.r[i] = x * y + Ref::Lane(addend, 0xE4, i);
            }
        } else if (k.op == OPC_ADD) {
            for (uint32_t i = 0; i < 4; ++i) {
                if (!(k.mask & (1u << i))) continue;
                ref.r[i] = Ref::Lane(k.a, k.aswz, i) - Ref::Lane(k.b, k.bswz, i);
            }
        } else {
            ref.Apply(k.mask, k.a, k.aswz, k.b, k.bswz, k.op);
        }

        if (Same(gpu, ref.r)) {
            printf("PASS %-38s gpu=(% .4f % .4f % .4f % .4f)\n", k.name, gpu[0], gpu[1],
                   gpu[2], gpu[3]);
        } else {
            printf("FAIL %-38s gpu=(% .4f % .4f % .4f % .4f) ref=(% .4f % .4f % .4f % .4f)\n",
                   k.name, gpu[0], gpu[1], gpu[2], gpu[3], ref.r[0], ref.r[1], ref.r[2],
                   ref.r[3]);
            ++failed;
        }
    }

    printf("\n%d/%zu pattern cases matched\n", (int)cases.size() - failed, cases.size());

    const char* shaderPath = (argc > 1 && argv[1][0] != '-') ? argv[1] : kDefaultShader;
    std::vector<float> capBlock;
    uint64_t capHash = 0;
    bool haveCap = false;
    if (argc > 2 && strcmp(argv[1], "--capture") == 0) {
        haveCap = LoadCapture(argv[2], capBlock, capHash);
        if (haveCap) {
            static char capPath[512];
            snprintf(capPath, sizeof(capPath),
                     "../The LEGO Movie - Videogame/shaders/v32_%016llx_vs.sm3",
                     (unsigned long long)capHash);
            shaderPath = capPath;
            printf("  capture %s -> vs %016llx", argv[2], (unsigned long long)capHash);
            puts("");
        } else {
            printf("  capture %s could not be read", argv[2]);
            puts("");
        }
    }
    int real = 0;
    if (haveCap) {
        real = RunRealCase(g, shaderPath, &capBlock);
    } else {
        puts("");
        puts("  real world shader: skipped (pass --capture <file.bin>)");
    }
    printf("\n%s\n", real ? "FAILURES PRESENT" : "all cases matched");
    return (failed || real) ? 1 : 0;
}
