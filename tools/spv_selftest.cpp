// End-to-end test for the SM3 -> SPIR-V emitter.
//
// Three tiers:
//   1. hand-written token streams (structural, no external tools)
//   2. real HLSL compiled with d3dcompiler_47.dll (the same compiler the engine
//      calls in sub_706080), fed through the JIT exactly as at runtime
//   3. the emitted modules handed to the actual Vulkan driver, which performs
//      full semantic SPIR-V validation before building SASS
//
// Tier 3 is the real test: vkCreateShaderModule/vkCreateGraphicsPipelines reject
// operand type and dimensionality errors that no structural walk can see.
#define _CRT_SECURE_NO_WARNINGS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3dcompiler.h>

#define VK_USE_PLATFORM_WIN32_KHR 1
#define VK_NO_PROTOTYPES 1
#include <vulkan/vulkan.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <process.h>
#include <string>
#include <vector>

#include "sm3_to_spirv.h"

void Logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

// ---------------------------------------------------------------------------
// Tier 1: hand-encoded token streams
// ---------------------------------------------------------------------------
// Register types use the real split encoding: bits 28..30 give 0..7 and bit 11
// extends them to 8..15 (D3DSP_REGTYPE_MASK/0x70000000 | MASK2/0x00001800).
//   TEMP 0x80000000+n   INPUT 0x90000000+n   CONST 0xA0000000+n
//   ADDR 0xB0000000+n   OUTPUT 0xE0000000+n  CONSTINT 0x70000000+n
//   COLOROUT 0x00000800+n (type 8)          SAMPLER 0x20000800+n (type 10)
static uint32_t kVs[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x02000001u, 0xE00F0000u, 0x90E40000u,  // mov o0, v0
    0x0000FFFFu,
};

// Minimal constant-buffer read: mov o0, c0. Isolates the UBO path.
static uint32_t kUbo[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x02000001u, 0xE00F0000u, 0xA0E40000u,  // mov o0, c0
    0x0000FFFFu,
};

static uint32_t kPsi[] = {
    0xFFFF0300u,
    0x0200001Fu, 0x00000005u, 0x90030000u,  // dcl_texcoord0 v0
    0x0200001Fu, 0x10000000u, 0x20000800u,  // dcl_2d s0
    0x03000042u, 0x800F0000u, 0x90E40000u, 0x20000800u,  // texld r0, v0, s0
    0x02000001u, 0x000F0800u, 0x80040000u,  // mov oC0, r0
    0x0000FFFFu,
};

// vs_3_0 relative constant indexing, the skinning pattern:
//   mova a0.x, v0.x / dp4 r0, v0, c[a0.x + 0] / mov o0, r0
static uint32_t kRel[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x0200002Eu, 0xB0030000u, 0x90000000u,  // mova a0.x, v0.x
    0x05000009u, 0x800F0000u, 0x90E40000u, 0xA0E42000u, 0xB0000000u,  // dp4
    0x02000001u, 0xE00F0000u, 0x80E40000u,  // mov o0, r0
    0x0000FFFFu,
};

// mov r0, v0 / mov o0.xyz, r0 — exercises the partial-write path.
static uint32_t kPartial[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,
    0x0200001Fu, 0x00000000u, 0xE00F0000u,
    0x02000001u, 0x800F0000u, 0x90E40000u,  // mov r0, v0
    0x02000001u, 0xE0070000u, 0x80E40000u,  // mov o0.xyz, r0
    0x0000FFFFu,
};

// mov r0, c0 / mov o0, r0 — UBO read via a temp.
static uint32_t kUbo2[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,
    0x0200001Fu, 0x00000000u, 0xE00F0000u,
    0x02000001u, 0x800F0000u, 0xA0E40000u,  // mov r0, c0
    0x02000001u, 0xE00F0000u, 0x80E40000u,  // mov o0, r0
    0x0000FFFFu,
};

// dp4 r0, v0, c0 — isolates OpDot.
static uint32_t kDp4[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,
    0x0200001Fu, 0x00000000u, 0xE00F0000u,
    0x03000009u, 0x800F0000u, 0x90E40000u, 0xA0E40000u,  // dp4 r0, v0, c0
    0x02000001u, 0xE00F0000u, 0x80E40000u,
    0x0000FFFFu,
};

// m4x3 with a full write mask, so WriteDst takes the whole-vector store path.
static uint32_t kM4x3Full[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,
    0x0200001Fu, 0x00000000u, 0xE00F0000u,
    0x03000015u, 0x800F0000u, 0x90E40000u, 0xA0E40000u,  // m4x3 r0, v0, c0
    0x02000001u, 0xE00F0000u, 0x80E40000u,
    0x0000FFFFu,
};

static uint32_t kCf[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x0200001Fu, 0x00000005u, 0xE00F0001u,  // dcl_texcoord0 o1
    0x05000051u, 0xA00F0000u, 0x3F800000u, 0x3F800000u, 0x3F800000u,
    0x3F800000u,                            // def c0, 1,1,1,1
    0x02000001u, 0x800F0000u, 0x90E40000u,  // mov r0, v0
    0x01000026u, 0x70000000u,               // rep i0
    0x03000002u, 0x800F0000u, 0x80040000u, 0x90E40000u,  // add r0, r0, v0
    0x00000027u,                            // endrep
    0x02000001u, 0xE00F0000u, 0x80040000u,  // mov o0, r0
    0x0000FFFFu,
};

static uint32_t kCube[] = {
    0xFFFF0300u,
    0x0200001Fu, 0x00000005u, 0x90030000u,  // dcl_texcoord0 v0
    0x0200001Fu, 0x18000000u, 0x20000800u,  // dcl_cube s0
    0x03010042u, 0x800F0000u, 0x90E40000u, 0x20000800u,  // texldp
    0x03000002u, 0x800F0001u, 0x80040000u, 0xA4E40000u,  // add r1, r0, c0_bx2
    0x02000001u, 0x000F0800u, 0x80040001u,  // mov oC0, r1
    0x0000FFFFu,
};

static uint32_t kM4x3[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x0200001Fu, 0x00000005u, 0xE00F0001u,  // dcl_texcoord0 o1
    0x02000001u, 0x800F0000u, 0x90E40000u,  // mov r0, v0
    0x03000015u, 0x80070000u, 0x90E40000u, 0xA0E40000u,  // m4x3 r0.xyz, v0, c0
    0x02000001u, 0xE00F0000u, 0x80040000u,  // mov o0, r0
    0x0000FFFFu,
};

// slt r0, v0, c0 / mov o0, r0 — isolates the OpFOrdLess + OpSelect pair that
// OP_SLT expands to.
static uint32_t kSlt[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x0300000Cu, 0x800F0000u, 0x90E40000u, 0xA0E40000u,  // slt r0, v0, c0
    0x02000001u, 0xE00F0000u, 0x80040000u,  // mov o0, r0
    0x0000FFFFu,
};

// lit r0, v0 / mov o0, r0 — lit is the one opcode that expands to a
// scalar-condition OpSelect, so it tells vector and scalar selects apart.
static uint32_t kLit[] = {
    0xFFFE0300u,
    0x0200001Fu, 0x00000000u, 0x900F0000u,  // dcl_position v0
    0x0200001Fu, 0x00000000u, 0xE00F0000u,  // dcl_position o0
    0x02000010u, 0x800F0000u, 0x90E40000u,  // lit r0, v0
    0x02000001u, 0xE00F0000u, 0x80040000u,  // mov o0, r0
    0x0000FFFFu,
};

// mov oDepth, c0 — depth-only pixel shader. The emitter drops the oDepth write
// (see WriteDst): every gl_FragDepth store faulted nvgpucomp, so the module must
// come out as a valid shader with no fragment outputs at all.
static uint32_t kDepth[] = {
    0xFFFF0300u,
    0x02000001u, 0x900F0800u, 0xA0000000u,  // mov oDepth, c0
    0x0000FFFFu,
};

// ---------------------------------------------------------------------------
// Tier 2: real HLSL, compiled by the system fxc at runtime
// ---------------------------------------------------------------------------
static const char* kHlslSkin = R"(
float4x3 bones[50] : register(c0);
float4x4 viewProj  : register(c150);

struct VSIn {
    float4 pos     : POSITION;
    float3 normal  : NORMAL;
    float2 uv      : TEXCOORD0;
    float4 weights : BLENDWEIGHT;
    float4 indices : BLENDINDICES;
};

struct VSOut {
    float4 pos   : POSITION;
    float2 uv    : TEXCOORD0;
    float4 color : COLOR0;
};

VSOut main(VSIn i)
{
    VSOut o;
    int i0 = (int)i.indices.x;
    int i1 = (int)i.indices.y;
    float3 p = mul(i.pos, bones[i0]) * i.weights.x
             + mul(i.pos, bones[i1]) * i.weights.y;
    o.pos = mul(float4(p, 1.0), viewProj);
    o.uv = i.uv;
    o.color = float4(i.normal * 0.5 + 0.5, 1.0);
    return o;
}
)";

static const char* kHlslSamplers = R"(
sampler2D   s0 : register(s0);
samplerCUBE s1 : register(s1);

float4 main(float2 uv : TEXCOORD0, float4 proj : TEXCOORD1,
            float3 dir : TEXCOORD2) : COLOR
{
    float4 a = tex2D(s0, uv);
    float4 b = texCUBE(s1, dir);
    float4 c = tex2Dproj(s0, proj);
    float3 n = normalize(dir);
    float  d = saturate(dot(n, a.rgb));
    return lerp(a, b, d) + c * float4(0.25, 0.5, 0.75, 1.0);
}
)";

static const char* kHlslLoop = R"(
float4 main(float2 uv : TEXCOORD0) : COLOR
{
    float4 acc = 0;
    [loop]
    for (int i = 0; i < 4; ++i)
        acc += pow(uv.xyyy, (float)(i + 1));

    if (acc.r > 0.5)
        acc.b = rsqrt(abs(acc.g) + 0.0001);

    clip(acc.a - 0.01);
    return acc;
}
)";

// sincos is a vertex-shader-only SM3 opcode, so it lives in its own vs_3_0 case.
static const char* kHlslSincos = R"(
float4 main(float4 pos : POSITION, float2 uv : TEXCOORD0) : POSITION
{
    float s, c;
    sincos(uv.x, s, c);
    float4 o = pos;
    o.xy += float2(s, c) * 0.001;
    return o;
}
)";

// ---------------------------------------------------------------------------
// Token dump: decodes the stream with the shared length helper so a bad
// encoding shows up as a desynchronised walk rather than a silent misparse.
// ---------------------------------------------------------------------------
static void DumpTokens(const char* label, const std::vector<uint32_t>& t) {
    printf("---- %s: %zu DWORDs ----\n", label, t.size());
    size_t i = 1;
    while (i < t.size()) {
        uint32_t head = t[i];
        uint32_t op = head & 0xFFFFu;
        if (op == 0xFFFFu) {
            printf("  [%04zu] END\n", i);
            break;
        }
        uint32_t n = Sm3InstructionDwords(head);
        printf("  [%04zu] op=0x%02X n=%u", i, op, n);
        for (uint32_t k = 1; k < n && i + k < t.size(); ++k)
            printf(" %08X", t[i + k]);
        printf("\n");
        i += n;
    }
}

// ---------------------------------------------------------------------------
// Vulkan device used purely as a validator: the driver compiles every module
// and reports typing/dimensionality problems that a structural check cannot.
// ---------------------------------------------------------------------------
struct VkCtx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
};

static bool VkInit(VkCtx& c, std::string& diag) {
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) {
        diag = "vulkan-1.dll not found";
        return false;
    }
    auto gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
    if (!gipa) {
        diag = "vkGetInstanceProcAddr missing";
        return false;
    }
    // vkGetInstanceProcAddr(NULL, ...) resolves only global entry points;
    // vkEnumeratePhysicalDevices and the query functions are instance-level
    // and must be fetched from the created instance.
    auto CreateInstance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
    if (!CreateInstance) {
        diag = "vkCreateInstance missing";
        return false;
    }

    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    ai.pApplicationName = "NuVulkan SPIR-V validator";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    if (CreateInstance(&ici, nullptr, &c.inst) != VK_SUCCESS) {
        diag = "vkCreateInstance failed";
        return false;
    }

    auto EnumPhys =
        (PFN_vkEnumeratePhysicalDevices)gipa(c.inst, "vkEnumeratePhysicalDevices");
    auto GetProps =
        (PFN_vkGetPhysicalDeviceProperties)gipa(c.inst, "vkGetPhysicalDeviceProperties");
    auto GetQ = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(
        c.inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    if (!EnumPhys || !GetProps || !GetQ) {
        diag = "instance entry points missing";
        return false;
    }

    uint32_t n = 0;
    EnumPhys(c.inst, &n, nullptr);
    if (!n) {
        diag = "no physical devices";
        return false;
    }
    std::vector<VkPhysicalDevice> devs(n);
    EnumPhys(c.inst, &n, devs.data());

    VkPhysicalDeviceProperties props{};
    for (VkPhysicalDevice d : devs) {
        GetProps(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            c.phys = d;
            break;
        }
        if (!c.phys) c.phys = d;
    }
    GetProps(c.phys, &props);

    printf("Vulkan device: %s (API %u.%u.%u)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));
    printf("  limits: vertexInputAttrs=%u vertexOutputComps=%u "
           "fragmentInputComps=%u fragmentOutputAttachments=%u\n",
           props.limits.maxVertexInputAttributes,
           props.limits.maxVertexOutputComponents,
           props.limits.maxFragmentInputComponents,
           props.limits.maxFragmentOutputAttachments);

    uint32_t qn = 0;
    GetQ(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    GetQ(c.phys, &qn, qs.data());
    uint32_t qf = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i)
        if ((qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qs[i].queueCount) {
            qf = i;
            break;
        }
    if (qf == UINT32_MAX) {
        diag = "no graphics queue family";
        return false;
    }

    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (gipa && c.inst) {
        auto CreateDevice = (PFN_vkCreateDevice)gipa(c.inst, "vkCreateDevice");
        if (!CreateDevice || CreateDevice(c.phys, &dci, nullptr, &c.dev) != VK_SUCCESS) {
            diag = "vkCreateDevice failed";
            return false;
        }
    }

    auto gdpa = (PFN_vkGetDeviceProcAddr)gipa(c.inst, "vkGetDeviceProcAddr");
    if (!gdpa) {
        diag = "vkGetDeviceProcAddr missing";
        return false;
    }
    c.CreateShaderModule = (PFN_vkCreateShaderModule)gdpa(c.dev, "vkCreateShaderModule");
    c.DestroyShaderModule = (PFN_vkDestroyShaderModule)gdpa(c.dev, "vkDestroyShaderModule");
    c.CreateDescriptorSetLayout =
        (PFN_vkCreateDescriptorSetLayout)gdpa(c.dev, "vkCreateDescriptorSetLayout");
    c.DestroyDescriptorSetLayout =
        (PFN_vkDestroyDescriptorSetLayout)gdpa(c.dev, "vkDestroyDescriptorSetLayout");
    c.CreatePipelineLayout = (PFN_vkCreatePipelineLayout)gdpa(c.dev, "vkCreatePipelineLayout");
    c.DestroyPipelineLayout =
        (PFN_vkDestroyPipelineLayout)gdpa(c.dev, "vkDestroyPipelineLayout");
    c.CreateGraphicsPipelines =
        (PFN_vkCreateGraphicsPipelines)gdpa(c.dev, "vkCreateGraphicsPipelines");
    c.DestroyPipeline = (PFN_vkDestroyPipeline)gdpa(c.dev, "vkDestroyPipeline");
    if (!c.CreateShaderModule || !c.CreateGraphicsPipelines || !c.CreatePipelineLayout) {
        diag = "device entry points missing";
        return false;
    }
    return true;
}

static bool LoadSpv(const char* path, std::vector<uint32_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.assign((size_t)bytes / 4, 0);
    size_t got = out.empty() ? 0 : fread(out.data(), 4, out.size(), f);
    fclose(f);
    return got == out.size();
}

static bool LoadSm3(const char* path, std::vector<uint32_t>& out) {
    return LoadSpv(path, out) && Sm3IsSupported(out.data(), (uint32_t)out.size());
}

// A known-good vertex module: a pixel-only case still needs a real module in
// the vertex stage, because a NULL module in a VkPipelineShaderStageCreateInfo
// is a driver fault rather than a validation error.
static VkShaderModule MakeVsFallback(VkCtx& c) {
    std::vector<uint32_t> spv;
    std::string err;
    Sm3CompileToSpirv(false, kVs, sizeof(kVs) / 4, spv, err);
    if (spv.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size() * 4;
    ci.pCode = spv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    if (c.CreateShaderModule(c.dev, &ci, nullptr, &m) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return m;
}

// The descriptor ABI the compiler emits: bindings 0/1 constant files, 2..17 one
// combined image sampler each.
static VkDescriptorSetLayout MakeSetLayout(VkCtx& c) {
    // Mirrors VkBackend's layout exactly: dynamic UBOs with one stage each, so
    // the shim's interface is what the driver sees here.
    VkDescriptorSetLayoutBinding b[18]{};
    for (uint32_t i = 0; i < 18; ++i) {
        b[i].binding = i;
        b[i].descriptorType =
            i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                  : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = i == 0   ? VK_SHADER_STAGE_VERTEX_BIT
                          : i == 1 ? VK_SHADER_STAGE_FRAGMENT_BIT
                                   : VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 18;
    ci.pBindings = b;
    VkDescriptorSetLayout out = VK_NULL_HANDLE;
    if (c.CreateDescriptorSetLayout(c.dev, &ci, nullptr, &out) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return out;
}

// The in-game state of the first draw: a real vertex declaration, alpha
// blending, depth test against a D24S8 target and a triangle strip. Each piece
// can be switched off to bisect a driver fault down to one field.
struct GamePipelineState {
    bool on = false;
    bool vi = true;
    bool blend = true;
    bool depth = true;
};
static GamePipelineState gGame;

static bool TestModule(VkCtx& c, const std::vector<uint32_t>& spv, bool pixel,
                       VkShaderModule vsFallback, const char* name) {
    if (spv.empty() || spv[0] != 0x07230203u) {
        printf("  %-10s NO MODULE\n", name);
        return false;
    }
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4;
    smci.pCode = spv.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult r = c.CreateShaderModule(c.dev, &smci, nullptr, &mod);
    printf("  %-10s vkCreateShaderModule=%d", name, (int)r);
    if (r != VK_SUCCESS) {
        printf("  <-- DRIVER REJECTED SPIR-V\n");
        return false;
    }

    fprintf(stderr, "  [module ok]\n");
    VkShaderModule vs = pixel ? vsFallback : mod;
    VkShaderModule fs = pixel ? mod : VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    uint32_t stageCount = 1;
    if (fs) {
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";
        stageCount = 2;
    }

    fprintf(stderr, "  [setlayout]\n");
    VkDescriptorSetLayout set = MakeSetLayout(c);
    if (!set) {
        printf("  set layout failed\n");
        return false;
    }
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set;
    fprintf(stderr, "  [pipelayout]\n");
    VkPipelineLayout layout = VK_NULL_HANDLE;
    r = c.CreatePipelineLayout(c.dev, &plci, nullptr, &layout);
    if (r != VK_SUCCESS) {
        printf("  pipeline layout failed (%d)\n", (int)r);
        c.DestroyDescriptorSetLayout(c.dev, set, nullptr);
        return false;
    }

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkVertexInputBindingDescription gBinding{};
    VkVertexInputAttributeDescription gAttrs[4]{};
    if (gGame.on && gGame.vi) {
        gBinding = {0, 36, VK_VERTEX_INPUT_RATE_VERTEX};
        // Locations follow Sm3LocationFor: POSITION0 0, NORMAL0 1, TEXCOORD0 2,
        // COLOR0 3. usage * 4 + index would put COLOR0 at 40, past the 32
        // attribute limit, and the draw faults the driver.
        gAttrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
        gAttrs[1] = {1, 0, VK_FORMAT_R8G8B8A8_UNORM, 12};
        gAttrs[2] = {3, 0, VK_FORMAT_B8G8R8A8_UNORM, 16};
        gAttrs[3] = {2, 0, VK_FORMAT_R32_SFLOAT, 20};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &gBinding;
        vi.vertexAttributeDescriptionCount = 4;
        vi.pVertexAttributeDescriptions = gAttrs;
    }
    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = gGame.on ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
                           : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    if (gGame.on && gGame.blend) {
        ba.blendEnable = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.colorBlendOp = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;

    // With dynamic rendering there is no render pass to supply defaults, so
    // pDepthStencilState must be valid as soon as the pipeline declares a depth
    // attachment; leaving it NULL faults the driver on any depth-writing shader.
    VkPipelineDepthStencilStateCreateInfo dss{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dss.depthTestEnable = VK_TRUE;
    dss.depthWriteEnable = VK_TRUE;
    dss.depthCompareOp = (gGame.on && gGame.depth) ? VK_COMPARE_OP_LESS_OR_EQUAL
                                                  : VK_COMPARE_OP_ALWAYS;
    if (gGame.on) dss.depthWriteEnable = VK_FALSE;

    VkPipelineRenderingCreateInfo prc{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prc.colorAttachmentCount = 1;
    VkFormat colorFmt = VK_FORMAT_B8G8R8A8_UNORM;
    prc.pColorAttachmentFormats = &colorFmt;
    // A fragment shader that writes gl_FragDepth (D3D9 oDepth) needs a depth
    // attachment on the pipeline; without one the driver faults while
    // compiling instead of failing validation.
    prc.depthAttachmentFormat = (gGame.on && gGame.depth)
                                   ? VK_FORMAT_D24_UNORM_S8_UINT
                                   : VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &prc;
    gp.stageCount = stageCount;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &dss;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    fprintf(stderr, "  [create pipeline]\n");
    r = c.CreateGraphicsPipelines(c.dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe);
    printf(" pipeline=%d%s\n", (int)r,
           r == VK_SUCCESS ? " (driver accepted)" : " <-- DRIVER REJECTED PIPELINE");

    if (pipe && c.DestroyPipeline) c.DestroyPipeline(c.dev, pipe, nullptr);
    if (c.DestroyPipelineLayout) c.DestroyPipelineLayout(c.dev, layout, nullptr);
    if (c.DestroyDescriptorSetLayout)
        c.DestroyDescriptorSetLayout(c.dev, set, nullptr);
    // The fallback module belongs to the caller and outlives this call.
    if (pixel && c.DestroyShaderModule) c.DestroyShaderModule(c.dev, mod, nullptr);
    if (!pixel && c.DestroyShaderModule) c.DestroyShaderModule(c.dev, mod, nullptr);
    return r == VK_SUCCESS;
}

// ---------------------------------------------------------------------------
static void DumpSpv(const char* name, const std::vector<uint32_t>& spv) {
    char path[256];
    snprintf(path, sizeof(path), "build/dump_%s.txt", name);
    FILE* f = fopen(path, "w");
    if (!f) return;
    size_t i = 5;
    while (i < spv.size()) {
        uint32_t wc = spv[i] >> 16, op = spv[i] & 0xFFFF;
        if (!wc || i + wc > spv.size()) {
            fprintf(f, "[%zu] BAD wc=%u op=%u", i, wc, op);
            fputc(10, f);
            break;
        }
        bool zero = false;
        for (uint32_t k = 1; k < wc; ++k)
            if (spv[i + k] == 0) zero = true;
        if (zero) {
            fprintf(f, "[%zu] op=%u wc=%u:", i, op, wc);
            for (uint32_t k = 0; k < wc; ++k) fprintf(f, " %u", spv[i + k]);
            fputc(10, f);
        }
        i += wc;
    }
    fclose(f);
}

int main(int argc, char** argv) {
    bool dump = false;
    const char* only = nullptr;
    const char* spvPath = nullptr;
    bool spvPixel = false;
    const char* bisectName = nullptr;
    const char* bisectFile = nullptr;
    const char* compileIn = nullptr;
    const char* compileOut = nullptr;
    const char* pairVs = nullptr;
    const char* pairPs = nullptr;
    int repeat = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump")) dump = true;
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--spv") && i + 1 < argc) spvPath = argv[++i];
        else if (!strcmp(argv[i], "--bisect") && i + 1 < argc) bisectName = argv[++i];
        else if (!strcmp(argv[i], "--bisectfile") && i + 1 < argc) bisectFile = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compile") && i + 2 < argc) {
            compileIn = argv[++i];
            compileOut = argv[++i];
        } else if (!strcmp(argv[i], "--pair") && i + 2 < argc) {
            pairVs = argv[++i];
            pairPs = argv[++i];
        } else if (!strcmp(argv[i], "--pixel")) spvPixel = true;
        else if (!strcmp(argv[i], "--game")) gGame.on = true;
        else if (!strcmp(argv[i], "--xvi")) gGame.vi = false;
        else if (!strcmp(argv[i], "--xblend")) gGame.blend = false;
        else if (!strcmp(argv[i], "--xdepth")) gGame.depth = false;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Compile one blob and write the module: --compile in.sm3 out.spv
    if (compileIn && compileOut) {
        std::vector<uint32_t> tok;
        if (!LoadSm3(compileIn, tok)) {
            printf("cannot read SM3: %s\n", compileIn);
            return 2;
        }
        bool px = (tok[0] & 0xFFFF0000u) == 0xFFFF0000u;
        std::vector<uint32_t> spv;
        std::string err;
        if (!Sm3CompileToSpirv(px, tok.data(), (uint32_t)tok.size(), spv, err)) {
            printf("compile failed: %s\n", err.c_str());
            return 1;
        }
        FILE* f = fopen(compileOut, "wb");
        if (!f) {
            printf("cannot write %s\n", compileOut);
            return 2;
        }
        fwrite(spv.data(), 4, spv.size(), f);
        fclose(f);
        printf("wrote %s (%zu words, %s)\n", compileOut, spv.size(),
               px ? "ps" : "vs");
        return 0;
    }

    // Drive a specific vertex/pixel module pair: --pair vs.spv ps.spv
    if (pairVs && pairPs) {
        std::vector<uint32_t> vs, ps;
        if (!LoadSpv(pairVs, vs) || !LoadSpv(pairPs, ps)) {
            printf("cannot read pair\n");
            return 2;
        }
        VkCtx ctx;
        std::string diag;
        if (!VkInit(ctx, diag)) {
            printf("SKIP: %s\n", diag.c_str());
            return 2;
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = vs.size() * 4;
        ci.pCode = vs.data();
        VkShaderModule vsm = VK_NULL_HANDLE;
        if (ctx.CreateShaderModule(ctx.dev, &ci, nullptr, &vsm) != VK_SUCCESS) {
            printf("FAIL: VS module rejected\n");
            return 1;
        }
        bool ok = TestModule(ctx, ps, true, vsm, "pair");
        printf("%s: pair\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    // Repeat one known-good module N times in a single device: tells a genuine
    // module fault apart from state that accumulates across pipelines.
    if (repeat) {
        VkCtx rctx;
        std::string rdiag;
        if (!VkInit(rctx, rdiag)) {
            printf("SKIP: %s\n", rdiag.c_str());
            return 2;
        }
        std::vector<uint32_t> rspv;
        std::string rerr;
        Sm3CompileToSpirv(false, kVs, sizeof(kVs) / 4, rspv, rerr);
        int failed = 0;
        for (int k = 0; k < repeat; ++k) {
            printf("repeat %d\n", k);
            if (!TestModule(rctx, rspv, false, VK_NULL_HANDLE, "repeat")) ++failed;
        }
        printf("REPEAT done: %d pipeline(s), %d failed\n", repeat, failed);
        return failed ? 1 : 0;
    }

    // Prefix bisect for a shader blob on disk: prints each prefix before
    // handing it to the driver, so the last line before a crash names the SM3
    // instruction that triggers it.
    if (bisectFile) {
        FILE* f = fopen(bisectFile, "rb");
        if (!f) {
            printf("cannot open %s\n", bisectFile);
            return 2;
        }
        fseek(f, 0, SEEK_END);
        long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> tok((size_t)bytes / 4);
        size_t got = tok.empty() ? 0 : fread(tok.data(), 4, tok.size(), f);
        fclose(f);
        if (got != tok.size() || !Sm3IsSupported(tok.data(), (uint32_t)tok.size())) {
            printf("not an SM3 blob: %s\n", bisectFile);
            return 2;
        }
        bool px = (tok[0] & 0xFFFF0000u) == 0xFFFF0000u;
        VkCtx bctx;
        std::string bdiag;
        if (!VkInit(bctx, bdiag)) {
            printf("SKIP: %s\n", bdiag.c_str());
            return 2;
        }
        std::vector<uint32_t> vsSpv;
        std::string verr;
        Sm3CompileToSpirv(false, kVs, sizeof(kVs) / 4, vsSpv, verr);
        VkShaderModuleCreateInfo vci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vci.codeSize = vsSpv.size() * 4;
        vci.pCode = vsSpv.data();
        VkShaderModule fallback = VK_NULL_HANDLE;
        bctx.CreateShaderModule(bctx.dev, &vci, nullptr, &fallback);
        std::vector<size_t> bounds;
        for (size_t i = 1; i < tok.size();) {
            uint32_t head = tok[i];
            if ((head & 0xFFFFu) == 0xFFFFu) break;
            bounds.push_back(i);
            i += Sm3InstructionDwords(head);
        }
        // One extra pass with the whole stream, so the final instruction is
        // covered as well as every prefix before it.
        bounds.push_back(tok.size());
        for (size_t b : bounds) {
            std::vector<uint32_t> cut(tok.begin(), tok.begin() + (long)b);
            if (b < tok.size()) {
                cut.push_back(0x0000FFFFu);
                printf("BISECT prefix=%zu/%zu next_op=0x%04X\n", b, tok.size(),
                       tok[b] & 0xFFFFu);
            } else {
                printf("BISECT full=%zu/%zu\n", b, tok.size());
            }
            std::vector<uint32_t> bspv;
            std::string berr;
            if (!Sm3CompileToSpirv(px, cut.data(), (uint32_t)cut.size(), bspv, berr)) {
                printf("  compile failed: %s\n", berr.c_str());
                continue;
            }
            FILE* pf = fopen("build/prefix.spv", "wb");
            if (pf) {
                fwrite(bspv.data(), 4, bspv.size(), pf);
                fclose(pf);
            }
            TestModule(bctx, bspv, px, fallback, "prefix");
        }
        printf("BISECT done: no fault\n");
        return 0;
    }

    // Standalone check for one prebuilt module: spv_selftest --spv file.spv
    // [--pixel]. Lets a suspect module be bisected by hand-editing its words.
    if (spvPath) {
        FILE* f = fopen(spvPath, "rb");
        if (!f) {
            printf("cannot open %s\n", spvPath);
            return 2;
        }
        fseek(f, 0, SEEK_END);
        long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> spv((size_t)bytes / 4);
        size_t got = spv.empty() ? 0 : fread(spv.data(), 4, spv.size(), f);
        fclose(f);
        if (got != spv.size()) {
            printf("short read: %s\n", spvPath);
            return 2;
        }
        VkCtx ctx;
        std::string diag;
        if (!VkInit(ctx, diag)) {
            printf("SKIP: %s\n", diag.c_str());
            return 2;
        }
        VkShaderModule fallback = spvPixel ? MakeVsFallback(ctx) : VK_NULL_HANDLE;
        bool ok = TestModule(ctx, spv, spvPixel, fallback, spvPath);
        if (fallback && ctx.DestroyShaderModule)
            ctx.DestroyShaderModule(ctx.dev, fallback, nullptr);
        printf("%s: %s\n", ok ? "PASS" : "FAIL", spvPath);
        return ok ? 0 : 1;
    }

    struct Out {
        const char* name;
        bool pixel;
        std::vector<uint32_t> spv;
    };
    std::vector<Out> mods;
    int bad = 0;

    // ---- tier 1 -----------------------------------------------------------
    struct Hand {
        const char* name;
        const char* path;
        bool pixel;
        uint32_t* code;
        uint32_t count;
    } hand[] = {
        {"hand_vs", "build/selftest_vs.spv", false, kVs, sizeof(kVs) / 4},
        {"hand_ubo", "build/selftest_ubo.spv", false, kUbo, sizeof(kUbo) / 4},
        {"hand_partial", "build/selftest_partial.spv", false, kPartial,
         sizeof(kPartial) / 4},
        {"hand_rel", "build/selftest_rel.spv", false, kRel, sizeof(kRel) / 4},
        {"hand_ubo2", "build/selftest_ubo2.spv", false, kUbo2, sizeof(kUbo2) / 4},
        {"hand_dp4", "build/selftest_dp4.spv", false, kDp4, sizeof(kDp4) / 4},
        {"hand_m4full", "build/selftest_m4full.spv", false, kM4x3Full,
         sizeof(kM4x3Full) / 4},
        {"hand_ps", "build/selftest_ps.spv", true, kPsi, sizeof(kPsi) / 4},
        {"hand_cf", "build/selftest_cf.spv", false, kCf, sizeof(kCf) / 4},
        {"hand_cube", "build/selftest_cube.spv", true, kCube, sizeof(kCube) / 4},
        {"hand_m4x3", "build/selftest_m4x3.spv", false, kM4x3, sizeof(kM4x3) / 4},
        {"hand_slt", "build/selftest_slt.spv", false, kSlt, sizeof(kSlt) / 4},
        {"hand_lit", "build/selftest_lit.spv", false, kLit, sizeof(kLit) / 4},
        {"hand_depth", "build/selftest_depth.spv", true, kDepth, sizeof(kDepth) / 4},
    };
    printf("== tier 1: hand-encoded token streams ==\n");
    for (auto& h : hand) {
        if (only && strcmp(only, h.name)) continue;
        fprintf(stderr, "== compiling %s ==\n", h.name);
        std::vector<uint32_t> spv;
        std::string err;
        bool ok = Sm3CompileToSpirv(h.pixel, h.code, h.count, spv, err);
        printf("%-12s ok=%d words=%zu %s\n", h.name, (int)ok, spv.size(), err.c_str());
        if (ok) DumpSpv(h.name, spv);
        if (!ok) {
            ++bad;
            continue;
        }
        FILE* f = fopen(h.path, "wb");
        if (!f) {
            printf("  cannot write %s\n", h.path);
            ++bad;
            continue;
        }
        fwrite(spv.data(), 4, spv.size(), f);
        fclose(f);
        mods.push_back({h.name, h.pixel, spv});
    }

    // ---- tier 2 -----------------------------------------------------------
    printf("== tier 2: real fxc (d3dcompiler_47.dll) ==\n");
    HMODULE fxcLib = LoadLibraryA("d3dcompiler_47.dll");
    pD3DCompile fxc = fxcLib ? (pD3DCompile)GetProcAddress(fxcLib, "D3DCompile") : nullptr;
    if (!fxc) {
        printf("SKIP: d3dcompiler_47.dll / D3DCompile unavailable\n");
        ++bad;
    } else {
        struct Hlsl {
            const char* name;
            const char* path;
            const char* src;
            const char* target;
            bool pixel;
        } hlsl[] = {
            {"skin", "build/selftest_skin.spv", kHlslSkin, "vs_3_0", false},
            {"samplers", "build/selftest_samplers.spv", kHlslSamplers, "ps_3_0", true},
            {"loop", "build/selftest_loop.spv", kHlslLoop, "ps_3_0", true},
            {"sincos", "build/selftest_sincos.spv", kHlslSincos, "vs_3_0", false},
        };
        for (auto& h : hlsl) {
            if (only && strcmp(only, h.name)) continue;
            fprintf(stderr, "== compiling %s ==\n", h.name);
            ID3DBlob* code = nullptr;
            ID3DBlob* msg = nullptr;
            HRESULT hr = fxc(h.src, strlen(h.src), "selftest.hlsl", nullptr, nullptr,
                             "main", h.target,
                             0x200 /* D3DCOMPILE_AVOID_FLOW_CONTROL */, 0, &code, &msg);
            if (FAILED(hr)) {
                printf("%-12s D3DCompile FAILED hr=0x%08lX\n", h.name,
                       (unsigned long)hr);
                if (msg) {
                    printf("  %.*s\n", (int)msg->GetBufferSize(),
                           (const char*)msg->GetBufferPointer());
                    msg->Release();
                }
                ++bad;
                continue;
            }
            std::vector<uint32_t> tok(
                (const uint32_t*)code->GetBufferPointer(),
                (const uint32_t*)code->GetBufferPointer() + code->GetBufferSize() / 4);
            printf("%-12s fxc ok, %zu DWORDs (version 0x%08X)\n", h.name, tok.size(),
                   tok.empty() ? 0u : tok[0]);
            if (dump) DumpTokens(h.name, tok);

            // Prefix bisect: compile and drive every instruction prefix, so a
            // driver fault can be attributed to one SM3 instruction. The last
            // line printed before a crash names the instruction that triggers
            // it (stdout and stderr are unbuffered).
            if (bisectName && !strcmp(bisectName, h.name)) {
                VkCtx bctx;
                std::string bdiag;
                if (!VkInit(bctx, bdiag)) {
                    printf("SKIP bisect: %s\n", bdiag.c_str());
                    code->Release();
                    if (msg) msg->Release();
                    return 2;
                }
                std::vector<size_t> bounds;
                for (size_t i = 1; i < tok.size();) {
                    uint32_t head = tok[i];
                    if ((head & 0xFFFFu) == 0xFFFFu) break;
                    bounds.push_back(i);
                    i += Sm3InstructionDwords(head);
                }
                for (size_t b : bounds) {
                    std::vector<uint32_t> cut(tok.begin(), tok.begin() + (long)b);
                    cut.push_back(0x0000FFFFu);
                    printf("BISECT prefix=%zu/%zu next_op=0x%04X\n", b, tok.size(),
                           tok[b] & 0xFFFFu);
                    std::vector<uint32_t> bspv;
                    std::string berr;
                    if (!Sm3CompileToSpirv(h.pixel, cut.data(), (uint32_t)cut.size(),
                                           bspv, berr)) {
                        printf("  compile failed: %s\n", berr.c_str());
                        continue;
                    }
                    FILE* pf = fopen("build/prefix.spv", "wb");
                    if (pf) {
                        fwrite(bspv.data(), 4, bspv.size(), pf);
                        fclose(pf);
                    }
                    TestModule(bctx, bspv, h.pixel, VK_NULL_HANDLE, "prefix");
                }
                printf("BISECT done: no fault\n");
                code->Release();
                if (msg) msg->Release();
                return 0;
            }

            std::vector<uint32_t> spv;
            std::string err;
            bool ok = Sm3CompileToSpirv(h.pixel, tok.data(), (uint32_t)tok.size(), spv, err);
            printf("%-12s jit ok=%d words=%zu %s\n", h.name, (int)ok, spv.size(),
                   err.c_str());
            code->Release();
            if (msg) msg->Release();
            if (!ok) {
                ++bad;
                continue;
            }
            FILE* f = fopen(h.path, "wb");
            if (f) {
                fwrite(spv.data(), 4, spv.size(), f);
                fclose(f);
            }
            mods.push_back({h.name, h.pixel, spv});
        }
    }

    // ---- tier 4 -----------------------------------------------------------
    // Every shader blob the game ships, compiled and handed to the driver. The
    // two the engine reaches first are named explicitly so a regression there
    // shows up in the log even when the directory walk is unavailable.
    printf("== tier 4: game shader blobs ==\n");
    {
        int compiled = 0, rejected = 0, broken = 0;
        const char* dir = "../The LEGO Movie - Videogame/shaders";
        const char* must[] = {"b40b4e49c6af0e9d_vs.sm3",
                              "ae3d5d0f1c4741a0_ps.sm3"};
        auto addBlob = [&](const char* file) {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", dir, file);
            FILE* f = fopen(path, "rb");
            if (!f) {
                printf("%-30s missing %s\n", file, path);
                ++rejected;
                return;
            }
            fseek(f, 0, SEEK_END);
            long bytes = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::vector<uint32_t> tok((size_t)bytes / 4);
            size_t got = tok.empty() ? 0 : fread(tok.data(), 4, tok.size(), f);
            fclose(f);
            if (got != tok.size() ||
                !Sm3IsSupported(tok.data(), (uint32_t)tok.size())) {
                printf("%-30s not an SM3 blob\n", file);
                ++rejected;
                return;
            }
            size_t n = strlen(file);
            bool pixel = n > 7 && !strcmp(file + n - 7, "_ps.sm3");
            std::vector<uint32_t> spv;
            std::string err;
            if (!Sm3CompileToSpirv(pixel, tok.data(), (uint32_t)tok.size(), spv,
                                   err)) {
                printf("%-30s compile failed: %s\n", file, err.c_str());
                ++rejected;
                return;
            }
            // One child process per blob: nvgpucomp32 faults after a few
            // hundred distinct compiles inside a single process (the same
            // module repeated 400 times is fine), and the game never compiles
            // that many at once. A child keeps every verdict independent and
            // attributes a fault to the shader that caused it.
            FILE* out = fopen("build/game_check.spv", "wb");
            if (!out) {
                printf("%-30s cannot write build/game_check.spv\n", file);
                ++rejected;
                return;
            }
            fwrite(spv.data(), 4, spv.size(), out);
            fclose(out);
            intptr_t rc = _spawnl(_P_WAIT, argv[0], argv[0], "--spv",
                                  "build/game_check.spv",
                                  pixel ? "--pixel" : (const char*)nullptr,
                                  (const char*)nullptr);
            if (rc != 0) {
                printf("%-30s DRIVER FAULT (child=%ld)\n", file, (long)rc);
                ++broken;
            }
            ++compiled;
        };
        for (const char* m : must) addBlob(m);
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA("../The LEGO Movie - Videogame/shaders/*.sm3", &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
                addBlob(fd.cFileName);
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        } else {
            printf("SKIP: no shader directory\n");
        }
        printf("game shaders: %d compiled, %d rejected, %d driver fault(s)\n",
               compiled, rejected, broken);
        if (rejected || broken) ++bad;
    }

    // ---- tier 3 -----------------------------------------------------------
    printf("== tier 3: driver validation (vkCreateShaderModule + pipeline) ==\n");
    VkCtx ctx;
    std::string diag;
    if (!VkInit(ctx, diag)) {
        printf("SKIP: %s\n", diag.c_str());
        ++bad;
    } else {
        // A known-good vertex module stands in for pixel-only test cases.
        std::vector<uint32_t> vsSpv;
        std::string verr;
        Sm3CompileToSpirv(false, kVs, sizeof(kVs) / 4, vsSpv, verr);
        VkShaderModuleCreateInfo vci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        vci.codeSize = vsSpv.size() * 4;
        vci.pCode = vsSpv.data();
        VkShaderModule fallback = VK_NULL_HANDLE;
        if (ctx.CreateShaderModule(ctx.dev, &vci, nullptr, &fallback) != VK_SUCCESS)
            fallback = VK_NULL_HANDLE;

        for (auto& m : mods)
            if (!TestModule(ctx, m.spv, m.pixel, fallback, m.name)) ++bad;

        if (fallback && ctx.DestroyShaderModule)
            ctx.DestroyShaderModule(ctx.dev, fallback, nullptr);
    }

    printf("\n%s: %d module(s), %d failure(s)\n", bad ? "FAIL" : "PASS",
           (int)mods.size(), bad);
    return bad ? 1 : 0;
}
