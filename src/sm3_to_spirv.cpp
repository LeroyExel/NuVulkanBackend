// In-process D3D9 SM3 token-stream -> SPIR-V 1.3 compiler.
//
// Input is exactly what the engine hands to CreateVertexShader/CreatePixelShader:
// a DWORD array starting with 0xFFFE0300 (vs_3_0) or 0xFFFF0300 (ps_3_0) and
// ending with the 0x0000FFFF END token. This is the pre-DX10 shader format, not
// a DXBC container.
//
// Verified from the binary: sub_706080 passes HLSL *text* to D3DPreprocess, then
// D3DCompile(text, size, NULL, defines, NULL, "vertexProgram", "vs_3_0"|"ps_3_0",
// 0x200, 0, ...). D3DCompile with an SM3 target emits this token stream, so the
// blob the engine later gives us is always SM3 tokens.
//
// Register model: registers are Function-storage vec4 variables; every
// instruction loads, computes and stores. Partial write masks, per-lane
// swizzles and structured control flow all become trivial, and the driver's SSA
// construction removes the redundant loads and stores.
//
// Clip space: the backend uses a negative-height viewport, which already maps
// D3D9's y-down NDC onto Vulkan. Negating y here as well would flip twice, and
// D3D9 z is [0,w] -> [0,1] after the divide, which is Vulkan's range already, so
// no z remap is emitted either. This is what DXVK does for D3D9.

#include "sm3_to_spirv.h"

#include "vk_common.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_map>

// Interface locations must stay below the device's attribute limit (32 here),
// so a D3D9 (usage, usageIndex) pair cannot use usage * 4 + index: COLOR0 alone
// would be location 40 and a draw with it faults the driver, which creates the
// pipeline without validating the input interface first.
// Defined outside the anonymous namespace so resources.cpp shares the mapping.
uint32_t Sm3LocationFor(uint32_t usage, uint32_t usageIndex) {
    switch (usage) {
        case 0: return 0 + (usageIndex > 7 ? 7 : usageIndex);
        case 1: return 8 + (usageIndex > 1 ? 1 : usageIndex);
        case 2: return 10 + (usageIndex > 1 ? 1 : usageIndex);
        case 3: return 12 + (usageIndex > 1 ? 1 : usageIndex);
        case 4: return 14;
        case 6: return 15 + (usageIndex > 1 ? 1 : usageIndex);
        case 7: return 17 + (usageIndex > 1 ? 1 : usageIndex);
        case 10: return 19 + (usageIndex > 1 ? 1 : usageIndex);
        case 11: return 21;
        case 5: return 22 + (usageIndex > 7 ? 7 : usageIndex);
        case 9: return 30;
        default: return 31;
    }
}

namespace {

// ---------------------------------------------------------------------------
// D3D9 SM3 opcodes
// ---------------------------------------------------------------------------
enum : uint32_t {
    OP_NOP = 0, OP_MOV = 1, OP_ADD = 2, OP_SUB = 3, OP_MAD = 4, OP_MUL = 5,
    OP_RCP = 6, OP_RSQ = 7, OP_DP3 = 8, OP_DP4 = 9, OP_MIN = 10, OP_MAX = 11,
    OP_SLT = 12, OP_SGE = 13, OP_EXP = 14, OP_LOG = 15, OP_LIT = 16,
    OP_DST = 17, OP_LRP = 18, OP_FRC = 19, OP_M4x4 = 20, OP_M4x3 = 21,
    OP_M3x4 = 22, OP_M3x3 = 23, OP_M3x2 = 24, OP_CALL = 25, OP_CALLNZ = 26,
    OP_LOOP = 27, OP_RET = 28, OP_ENDLOOP = 29, OP_LABEL = 30, OP_DCL = 31,
    OP_POW = 32,
    OP_CRS = 33, OP_SGN = 34, OP_ABS = 35, OP_NRM = 36, OP_SINCOS = 37,
    OP_REP = 38, OP_ENDREP = 39, OP_IF = 40, OP_IFC = 41, OP_ELSE = 42,
    OP_ENDIF = 43, OP_BREAK = 44, OP_BREAKC = 45, OP_MOVA = 46, OP_DEFB = 47,
    OP_DEFI = 48, OP_TEXCOORD = 64, OP_TEXKILL = 65, OP_TEX = 66,
    OP_CND = 79, OP_DEF = 81, OP_CMP = 88, OP_DP2ADD = 90,
    OP_DSX = 91, OP_DSY = 92,
    OP_TEXLDL = 95, OP_COMMENT = 0xFFFE, OP_END = 0xFFFF
};

// _D3DSHADER_PARAM_REGISTER_TYPE, verbatim. Note OUTPUT and TEXCRDOUT are the
// same value (6): vs_3_0 reuses the slot, so switches must not list both.
enum : uint32_t {
    RT_TEMP = 0, RT_INPUT = 1, RT_CONST = 2, RT_ADDR = 3, RT_TEXTURE = 3,
    RT_RASTOUT = 4, RT_ATTROUT = 5, RT_OUTPUT = 6, RT_TEXCRDOUT = 6,
    RT_CONSTINT = 7, RT_COLOROUT = 8, RT_DEPTHOUT = 9, RT_SAMPLER = 10,
    RT_CONST2 = 11, RT_CONST3 = 12, RT_CONST4 = 13, RT_CONSTBOOL = 14,
    RT_LOOP = 15
};

enum : uint32_t {
    SMOD_NONE = 0, SMOD_NEG = 1, SMOD_BIAS = 2, SMOD_BIASNEG = 3,
    SMOD_SIGN = 4, SMOD_SIGNNEG = 5, SMOD_COMP = 6,
    SMOD_X2 = 7, SMOD_X2NEG = 8, SMOD_DZ = 9, SMOD_DW = 10,
    SMOD_ABS = 11, SMOD_ABSNEG = 12, SMOD_NOT = 13
};

// D3DSPC comparison functions (instruction token bits 16..23)
// D3DSPC_* from d3d9types.h: GT=1, EQ=2, GE=3, LT=4, NE=5, LE=6. This is not a
// 0-based sequence, and shifting it silently inverts every conditional.
enum : uint32_t { CMP_GT = 1, CMP_EQ = 2, CMP_GE = 3, CMP_LT = 4, CMP_NE = 5,
                  CMP_LE = 6 };

// ---------------------------------------------------------------------------
// SPIR-V opcodes
// ---------------------------------------------------------------------------
enum : uint32_t {
    OPC_CAPABILITY = 17, OPC_EXT_INST_IMPORT = 11, OPC_EXT_INST = 12,
    OPC_MEMORY_MODEL = 14, OPC_ENTRY_POINT = 15, OPC_EXECUTION_MODE = 16,
    OPC_TYPE_VOID = 19, OPC_TYPE_BOOL = 20, OPC_TYPE_INT = 21,
    OPC_TYPE_FLOAT = 22, OPC_TYPE_VECTOR = 23, OPC_TYPE_ARRAY = 28,
    OPC_TYPE_STRUCT = 30, OPC_TYPE_POINTER = 32, OPC_TYPE_FUNCTION = 33,
    OPC_CONSTANT = 43, OPC_CONSTANT_COMPOSITE = 44, OPC_SPEC_CONSTANT = 50, OPC_FUNCTION = 54,
    OPC_FUNCTION_END = 56, OPC_VARIABLE = 59, OPC_LOAD = 61, OPC_STORE = 62,
    OPC_ACCESS_CHAIN = 65, OPC_CONVERT_F_TO_U = 109, OPC_CONVERT_F_TO_S = 110,
    OPC_CONVERT_S_TO_U = 113, OPC_BITCAST = 124, OPC_S_NEGATE = 126,
    OPC_IADD = 128, OPC_I_SUB = 130, OPC_DECORATE = 71, OPC_MEMBER_DECORATE = 72,
    OPC_COMPOSITE_CONSTRUCT = 80, OPC_COMPOSITE_EXTRACT = 81,
    OPC_COMPOSITE_INSERT = 82, OPC_IMAGE_SAMPLE_IMPLICIT = 87,
    OPC_IMAGE_SAMPLE_EXPLICIT = 88, OPC_IMAGE_SAMPLE_DREF_IMPLICIT = 89,
    OPC_IMAGE_SAMPLE_DREF_EXPLICIT = 90, OPC_FNEGATE = 127, OPC_FADD = 129,
    OPC_FSUB = 131, OPC_FMUL = 133, OPC_FDIV = 136, OPC_FDOT = 148,
    OPC_ANY = 154, OPC_LOGICAL_OR = 166, OPC_LOGICAL_AND = 167, OPC_LOGICAL_NOT = 168, OPC_SELECT = 169,
    OPC_I_EQUAL = 170, OPC_I_NOT_EQUAL = 171,
    OPC_FORD_EQUAL = 180, OPC_FORD_NOT_EQUAL = 182, OPC_FORD_LESS = 184,
    OPC_FORD_GREATER = 186, OPC_FORD_LESS_EQUAL = 188,
    OPC_FORD_GREATER_EQUAL = 190,
    OPC_SHIFT_RIGHT_LOGICAL = 194, OPC_SHIFT_RIGHT_ARITHMETIC = 195,
    OPC_SHIFT_LEFT_LOGICAL = 196, OPC_BITWISE_OR = 197, OPC_BITWISE_XOR = 198,
    OPC_BITWISE_AND = 199, OPC_NOT = 200,
    OPC_LOOP_MERGE = 246,
    OPC_SELECTION_MERGE = 247, OPC_LABEL = 248, OPC_BRANCH = 249,
    OPC_BRANCH_CONDITIONAL = 250, OPC_KILL = 252, OPC_RETURN = 253,
    OPC_TYPE_SAMPLED_IMAGE = 27, OPC_TYPE_IMAGE = 25
};

enum : uint32_t { SC_UNIFORM_CONSTANT = 0, SC_INPUT = 1, SC_UNIFORM = 2,
                  SC_OUTPUT = 3, SC_FUNCTION = 7 };

enum : uint32_t { DEC_SPEC_ID = 1, DEC_BLOCK = 2, DEC_ARRAY_STRIDE = 6, DEC_BUILTIN = 11,
                  DEC_LOCATION = 30, DEC_BINDING = 33, DEC_DESCRIPTOR_SET = 34,
                  DEC_OFFSET = 35 };

enum : uint32_t { BI_POSITION = 0, BI_FRAG_DEPTH = 7 };

// GLSL.std.450 numbers (these are NOT sequential by name, easy to get wrong)
enum : uint32_t {
    GLSL_Round = 1, GLSL_FAbs = 4, GLSL_Sign = 6, GLSL_Fract = 10, GLSL_Sin = 13,
    GLSL_Cos = 14,
    GLSL_Pow = 26, GLSL_Exp2 = 29, GLSL_Log2 = 30, GLSL_InverseSqrt = 32,
    GLSL_FMin = 37, GLSL_FMax = 40, GLSL_FClamp = 43, GLSL_Cross = 68,
    GLSL_Normalize = 69
};

// MSVC does not implement C compound literals, so the identity swizzle lives
// here rather than being written inline at each call site.
const uint32_t kIdentitySwz[4] = {0, 1, 2, 3};

constexpr uint32_t kConstRegs = 256;
constexpr uint32_t kUboRegs = 257;
constexpr uint32_t kSamplerSlots = 16;
constexpr uint32_t kTempCount = 32;
constexpr uint32_t kCfScalars = 4;
constexpr uint32_t kShadowVecs = 8;

// ---------------------------------------------------------------------------
// A SPIR-V section: just a word list with helpers.
// ---------------------------------------------------------------------------
struct Sect {
    std::vector<uint32_t> code;
    void op(uint32_t opcode, std::initializer_list<uint32_t> args) {
        code.push_back(((uint32_t)(args.size() + 1) << 16) | opcode);
        for (uint32_t a : args) code.push_back(a);
    }
    void opv(uint32_t opcode, const std::vector<uint32_t>& args) {
        code.push_back(((uint32_t)(args.size() + 1) << 16) | opcode);
        for (uint32_t a : args) code.push_back(a);
    }
};

struct SrcP {
    uint32_t regType = 0;
    uint32_t regNum = 0;
    uint32_t swz[4] = {0, 1, 2, 3};
    uint32_t mod = 0;
    // Present when the token carried D3DSHADER_ADDRMODE_RELATIVE; the address
    // register token follows it in the stream and is NOT an operand.
    bool rel = false;
    uint32_t relRegType = 0, relRegNum = 0, relSwz = 0;
};

struct DstP {
    uint32_t regType = 0;
    uint32_t regNum = 0;
    uint32_t mask = 0xF;
    uint32_t dm = 0;
    bool rel = false;
};

// The register type is split across the token:
//   D3DSP_REGTYPE_MASK 0x70000000 >> 28  |  D3DSP_REGTYPE_MASK2 0x00001800 >> 8
// so every type >= 8 (COLOROUT, SAMPLER, CONST2..4, CONSTBOOL, LOOP) sets bit 11.
// Reading bits 13:11 alone silently turns oC0 into INPUT and s0 into TEXCRDOUT.
uint32_t RegTypeOf(uint32_t t) {
    return ((t & 0x70000000u) >> 28) | ((t & 0x00001800u) >> 8);
}

SrcP ParseSrc(uint32_t t, uint32_t relToken = 0) {
    SrcP s;
    s.regNum = t & 0x7FF;
    s.regType = RegTypeOf(t);
    for (int i = 0; i < 4; ++i) s.swz[i] = (t >> (16 + i * 2)) & 3;
    s.mod = (t >> 24) & 0xF;
    if (t & 0x2000u) {  // D3DSHADER_ADDRMODE_RELATIVE
        s.rel = true;
        s.relRegType = RegTypeOf(relToken);
        s.relRegNum = relToken & 0x7FF;
        s.relSwz = (relToken >> 16) & 3;
    }
    return s;
}

DstP ParseDst(uint32_t t, uint32_t relToken = 0) {
    DstP d;
    d.regNum = t & 0x7FF;
    d.regType = RegTypeOf(t);
    d.mask = (t >> 16) & 0xF;
    d.dm = (t >> 20) & 0xF;
    d.rel = (t & 0x2000u) != 0;
    (void)relToken;
    return d;
}

// ---------------------------------------------------------------------------
struct Ctx {
    uint32_t nextId = 1;
    uint32_t id() { return nextId++; }

    Sect pre, ep, ann, glob, fn;
    bool pixel = false;
    bool failed = false;
    uint32_t drefMask = 0;
    std::string* err = nullptr;

    uint32_t tVoid = 0, tFloat = 0, tUint = 0, tSint = 0, tBool = 0, tVec2 = 0, tVec3 = 0,
             tVec4 = 0, tBoolVec4 = 0, tSampled = 0;
    uint32_t              cZero = 0, cOne = 0, cTwo = 0, cUInt0 = 0, cUInt1 = 0, cUInt2 = 0,
             cUInt3 = 0, cUInt4 = 0, cUInt5 = 0, cUInt6 = 0, cUInt7 = 0, cUInt8 = 0,
             cUInt256 = 0, cUInt257 = 0, cUInt16 = 0, v4Zero = 0, v4One = 0,
             v4PosZero = 0,
             cUInt7FFFFFFF = 0, cUInt23 = 0, cUInt31 = 0, cSint0 = 0, cSint254 = 0;
    uint32_t specAlphaTest = 0, specAlphaFunc = 0, specAlphaRef = 0;
    uint32_t glsl = 0;
    uint32_t ptrFv4 = 0, ptrFfloat = 0, ptrInV4 = 0, ptrOutV4 = 0,
             ptrUv4 = 0, ptrSampled = 0;
    uint32_t uboVar = 0;
    uint32_t addrVar = 0, loopVar = 0;  // SM3 a0 and aL
    std::unordered_map<uint32_t, uint32_t> defInts, defBools;

    std::vector<uint32_t> temps;
    std::vector<uint32_t> cfScalars;
    std::vector<uint32_t> shadows;
    uint32_t shadowNext = 0, cfNext = 0;

    std::unordered_map<uint32_t, uint32_t> inVars, realInVars, outVars, realOutVars;

    // Two outputs declared with the same (usage, usageIndex) used to receive
    // the same Location (20/20), an invalid interface that faulted the
    // driver compiler at NULL+8.
    // Vulkan vertex input locations and vertex output locations are separate
    // namespaces. They must not share one "taken" set: a vertex declaration that
    // has TEXCOORD0 at location 2 would push the matching *output* varying to 3,
    // so the pixel shader kept reading a location the vertex stage never wrote
    // and every texture sample used a constant, garbage coordinate.
    std::vector<uint32_t> inLocUsed;
    std::vector<uint32_t> outLocUsed;
    std::unordered_map<uint32_t, uint32_t> samplerTypes;
    std::unordered_map<uint32_t, uint32_t> samplerVar;
    std::unordered_map<uint32_t, uint32_t> samplerPtr;
    std::unordered_map<uint32_t, uint32_t> samplerSampled;
    std::map<uint32_t, uint32_t> defConsts;
    std::vector<uint32_t> interfaces;
    uint32_t posVar = 0;
    uint32_t realPosVar = 0;
    uint32_t mainId = 0;

    struct Cf {
        uint32_t merge = 0, elseLabel = 0, contLabel = 0, header = 0;
        bool hasElse = false, isLoop = false;
        uint32_t counterVar = 0, countValue = 0;
        bool usesLoopReg = false;  // vs_3_0 `loop aL, i0`
    };
    std::vector<Cf> cf;
    uint32_t unsupported = 0;
    bool depthWarned = false;
    std::vector<uint32_t> colorUsed;

    void fail(const std::string& why) {
        if (!failed && err) *err = why;
        failed = true;
    }

    // Lazily materialise vec2/vec3: returning a zero type id puts literal 0 in
    // an <id> operand slot, and NVIDIA's parser then reads [NULL+8]. ConstF and
    // ConstU already append to the module section during function emission, so
    // declaring types here is equally safe.
    uint32_t TypeVec(uint32_t n) {
        if (n == 2) {
            if (!tVec2) {
                tVec2 = id();
                glob.op(OPC_TYPE_VECTOR, {tVec2, tFloat, 2});
            }
            return tVec2;
        }
        if (n == 3) {
            if (!tVec3) {
                tVec3 = id();
                glob.op(OPC_TYPE_VECTOR, {tVec3, tFloat, 3});
            }
            return tVec3;
        }
        return tVec4;
    }

    uint32_t PtrF(uint32_t ty) {
        uint32_t p = id();
        glob.op(OPC_TYPE_POINTER, {p, SC_FUNCTION, ty});
        return p;
    }

    uint32_t ConstF(float f) {
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t c = id();
        glob.op(OPC_CONSTANT, {tFloat, c, bits});
        return c;
    }

    uint32_t ConstU(uint32_t v) {
        uint32_t c = id();
        glob.op(OPC_CONSTANT, {tUint, c, v});
        return c;
    }

    uint32_t ConstS(int32_t v) {
        uint32_t c = id();
        glob.op(OPC_CONSTANT, {tSint, c, (uint32_t)v});
        return c;
    }

    uint32_t Splat(uint32_t scalar, uint32_t width) {
        std::vector<uint32_t> a = {TypeVec(width), id()};
        for (uint32_t i = 0; i < width; ++i) a.push_back(scalar);
        uint32_t r = a[1];
        fn.opv(OPC_COMPOSITE_CONSTRUCT, a);
        return r;
    }

    // ---- module scaffolding ------------------------------------------------
    void BuildPreamble() {
        pre.op(OPC_CAPABILITY, {1});  // Shader
        glsl = id();
        {
            const char* s = "GLSL.std.450";
            size_t n = strlen(s);
            uint32_t words = (uint32_t)(n / 4) + 1;
            pre.code.push_back(((2 + words) << 16) | OPC_EXT_INST_IMPORT);
            pre.code.push_back(glsl);
            for (size_t i = 0; i <= n; i += 4) {
                uint32_t w = 0;
                for (size_t k = 0; k < 4 && i + k <= n; ++k)
                    w |= (uint32_t)(uint8_t)s[i + k] << (8 * k);
                pre.code.push_back(w);
            }
        }
        pre.op(OPC_MEMORY_MODEL, {0, 1});  // Logical GLSL450
    }

    void BuildTypes() {
        tVoid = id();
        glob.op(OPC_TYPE_VOID, {tVoid});
        tFloat = id();
        glob.op(OPC_TYPE_FLOAT, {tFloat, 32});
        tUint = id();
        glob.op(OPC_TYPE_INT, {tUint, 32, 0});
        tSint = id();
        glob.op(OPC_TYPE_INT, {tSint, 32, 1});
        tBool = id();
        glob.op(OPC_TYPE_BOOL, {tBool});
        tVec2 = id();
        glob.op(OPC_TYPE_VECTOR, {tVec2, tFloat, 2});
        tVec3 = id();
        glob.op(OPC_TYPE_VECTOR, {tVec3, tFloat, 3});
        tVec4 = id();
        glob.op(OPC_TYPE_VECTOR, {tVec4, tFloat, 4});
        tBoolVec4 = id();
        glob.op(OPC_TYPE_VECTOR, {tBoolVec4, tBool, 4});

        cZero = ConstF(0.0f);
        cOne = ConstF(1.0f);
        cTwo = ConstF(2.0f);
        cUInt0 = ConstU(0);
        cUInt1 = ConstU(1);
        cUInt2 = ConstU(2);
        cUInt3 = ConstU(3);
        cUInt4 = ConstU(4);
        cUInt5 = ConstU(5);
        cUInt6 = ConstU(6);
        cUInt7 = ConstU(7);
        cUInt8 = ConstU(8);
        cUInt256 = ConstU(kConstRegs);
        cUInt257 = ConstU(kUboRegs);
        cUInt16 = ConstU(kSamplerSlots);
        cUInt7FFFFFFF = ConstU(0x7FFFFFFFu);
        cUInt23 = ConstU(23);
        cUInt31 = ConstU(31);
        cSint0 = ConstS(0);
        cSint254 = ConstS(254);
        v4Zero = id();
        glob.op(OPC_CONSTANT_COMPOSITE, {tVec4, v4Zero, cZero, cZero, cZero, cZero});
        v4One = id();
        glob.op(OPC_CONSTANT_COMPOSITE, {tVec4, v4One, cOne, cOne, cOne, cOne});
        // Identity clip position: an unwritten gl_Position must not divide by a
        // zero w during rasterisation.
        v4PosZero = id();
        glob.op(OPC_CONSTANT_COMPOSITE,
                {tVec4, v4PosZero, cZero, cZero, cZero, cOne});

        if (pixel) {
            specAlphaTest = id();
            glob.op(OPC_SPEC_CONSTANT, {tUint, specAlphaTest, 0u});
            ann.op(OPC_DECORATE, {specAlphaTest, DEC_SPEC_ID, 0u});

            specAlphaFunc = id();
            glob.op(OPC_SPEC_CONSTANT, {tUint, specAlphaFunc, 8u});
            ann.op(OPC_DECORATE, {specAlphaFunc, DEC_SPEC_ID, 1u});

            specAlphaRef = id();
            glob.op(OPC_SPEC_CONSTANT, {tFloat, specAlphaRef, 0u});
            ann.op(OPC_DECORATE, {specAlphaRef, DEC_SPEC_ID, 2u});
        }

        ptrFv4 = PtrF(tVec4);
        ptrFfloat = PtrF(tFloat);
        ptrInV4 = id();
        glob.op(OPC_TYPE_POINTER, {ptrInV4, SC_INPUT, tVec4});
        ptrOutV4 = id();
        glob.op(OPC_TYPE_POINTER, {ptrOutV4, SC_OUTPUT, tVec4});

        // set 0 binding 0/1 : std430 vec4 c[256] (4096 bytes exactly)
        uint32_t arr = id();
        glob.op(OPC_TYPE_ARRAY, {arr, tVec4, cUInt257});
        ann.op(OPC_DECORATE, {arr, DEC_ARRAY_STRIDE, 16});
        uint32_t st = id();
        glob.op(OPC_TYPE_STRUCT, {st, arr});
        ann.op(OPC_DECORATE, {st, DEC_BLOCK});
        ann.op(OPC_MEMBER_DECORATE, {st, 0, DEC_OFFSET, 0});
        uint32_t ptrUbo = id();
        glob.op(OPC_TYPE_POINTER, {ptrUbo, SC_UNIFORM, st});
        ptrUv4 = id();
        glob.op(OPC_TYPE_POINTER, {ptrUv4, SC_UNIFORM, tVec4});
        uboVar = id();
        glob.op(OPC_VARIABLE, {ptrUbo, uboVar, SC_UNIFORM});
        ann.op(OPC_DECORATE, {uboVar, DEC_DESCRIPTOR_SET, 0});
        ann.op(OPC_DECORATE, {uboVar, DEC_BINDING, pixel ? 1u : 0u});

        // Samplers are individual variables at bindings 2..17, created when the
        // shader declares them: a descriptor array is homogeneous and could not
        // mix sampler2D with samplerCube.

        if (!pixel) {
            realPosVar = id();
            glob.op(OPC_VARIABLE, {ptrOutV4, realPosVar, SC_OUTPUT});
            ann.op(OPC_DECORATE, {realPosVar, DEC_BUILTIN, BI_POSITION});
            if (realPosVar) interfaces.push_back(realPosVar);
        }
    }

    // Pixel outputs are declared only once the instruction scan knows which
    // ones the shader writes. Declaring all four colour outputs plus
    // gl_FragDepth up front gave modules an interface wider than the pipeline's
    // attachment list, which nvgpucomp walks off the end of and faults on.
    // gl_FragDepth itself must be a float, not the vec4 the D3D9 oDepth
    // register is: a vec4 here leaves the module invalid.
    void DeclarePixelOutputs() {
        std::sort(colorUsed.begin(), colorUsed.end());
        colorUsed.erase(std::unique(colorUsed.begin(), colorUsed.end()),
                        colorUsed.end());
        for (uint32_t reg : colorUsed) {
            uint32_t v = id();
            glob.op(OPC_VARIABLE, {ptrOutV4, v, SC_OUTPUT});
            ann.op(OPC_DECORATE, {v, DEC_LOCATION, reg});
            realOutVars[0x100 + reg] = v;
            interfaces.push_back(v);
        }
    }

    // Function-storage variables must live in the first block of the function.
    void DeclareLocals() {
        temps.resize(kTempCount);
        for (uint32_t i = 0; i < kTempCount; ++i) {
            uint32_t v = id();
            fn.op(OPC_VARIABLE, {ptrFv4, v, SC_FUNCTION, v4Zero});
            temps[i] = v;
        }
        cfScalars.resize(kCfScalars);
        for (uint32_t i = 0; i < kCfScalars; ++i) {
            uint32_t v = id();
            fn.op(OPC_VARIABLE, {ptrFfloat, v, SC_FUNCTION, cZero});
            cfScalars[i] = v;
        }
        shadows.resize(kShadowVecs);
        for (uint32_t i = 0; i < kShadowVecs; ++i) {
            uint32_t v = id();
            fn.op(OPC_VARIABLE, {ptrFv4, v, SC_FUNCTION, v4Zero});
            shadows[i] = v;
        }
        // Address register file. Kept as floats because MOVA rounds but does
        // not truncate; the conversion to an integer index happens at the use
        // site, where a clamp into the 256-entry constant file is also applied.
        addrVar = id();
        fn.op(OPC_VARIABLE, {ptrFv4, addrVar, SC_FUNCTION, v4Zero});
        loopVar = id();
        fn.op(OPC_VARIABLE, {ptrFv4, loopVar, SC_FUNCTION, v4Zero});

        for (auto& kv : realInVars) {
            uint32_t v = id();
            fn.op(OPC_VARIABLE, {ptrFv4, v, SC_FUNCTION, v4Zero});
            inVars[kv.first] = v;
        }

        if (pixel) {
            for (uint32_t reg : colorUsed) {
                uint32_t v = id();
                fn.op(OPC_VARIABLE, {ptrFv4, v, SC_FUNCTION, v4Zero});
                outVars[0x100 + reg] = v;
            }
        } else {
            posVar = id();
            fn.op(OPC_VARIABLE, {ptrFv4, posVar, SC_FUNCTION, v4PosZero});
            for (auto& kv : realOutVars) {
                if (kv.second == realPosVar) {
                    outVars[kv.first] = posVar;
                } else {
                    uint32_t v = id();
                    fn.op(OPC_VARIABLE, {ptrFv4, v, SC_FUNCTION, v4Zero});
                    outVars[kv.first] = v;
                }
            }
        }

        for (auto& kv : realInVars) {
            uint32_t globalVar = kv.second;
            uint32_t localVar = inVars[kv.first];
            uint32_t rawVal = LoadVar(globalVar, tVec4);
            std::vector<uint32_t> sanLanes = {tVec4, id()};
            for (uint32_t i = 0; i < 4; ++i) {
                uint32_t laneVal = id();
                fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, laneVal, rawVal, i});
                uint32_t bVal = id();
                fn.op(OPC_BITCAST, {tUint, bVal, laneVal});
                uint32_t bExp = id();
                fn.op(OPC_SHIFT_RIGHT_LOGICAL, {tUint, bExp, bVal, ConstU(23)});
                uint32_t bExp8 = id();
                fn.op(OPC_BITWISE_AND, {tUint, bExp8, bExp, ConstU(0xFF)});
                uint32_t bExp8S = id();
                fn.op(OPC_BITCAST, {tSint, bExp8S, bExp8});
                uint32_t diff = id();
                fn.op(OPC_I_SUB, {tSint, diff, ConstS(254), bExp8S});
                uint32_t badFloat = id();
                fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, badFloat, diff, ConstU(31)});
                uint32_t badFloatU = id();
                fn.op(OPC_BITCAST, {tUint, badFloatU, badFloat});
                uint32_t notBadFloatU = id();
                fn.op(OPC_NOT, {tUint, notBadFloatU, badFloatU});
                uint32_t keptBits = id();
                fn.op(OPC_BITWISE_AND, {tUint, keptBits, bVal, notBadFloatU});
                uint32_t cleanF = id();
                fn.op(OPC_BITCAST, {tFloat, cleanF, keptBits});
                sanLanes.push_back(cleanF);
            }
            uint32_t sanVec = sanLanes[1];
            fn.opv(OPC_COMPOSITE_CONSTRUCT, sanLanes);
            fn.op(OPC_STORE, {localVar, sanVec});
        }
    }

    // ---- register access ---------------------------------------------------
    uint32_t LoadVar(uint32_t var, uint32_t ty) {
        uint32_t l = id();
        fn.op(OPC_LOAD, {ty, l, var});
        return l;
    }

    uint32_t ConstReg(uint32_t n) {
        auto it = defConsts.find(n);
        if (it != defConsts.end()) return it->second;
        if (n >= kConstRegs) return v4Zero;
        uint32_t k = ConstU(n);
        uint32_t p = id();
        fn.op(OPC_ACCESS_CHAIN, {ptrUv4, p, uboVar, cUInt0, k});
        return LoadVar(p, tVec4);
    }

    uint32_t Shadow() {
        uint32_t v = shadows[shadowNext % kShadowVecs];
        ++shadowNext;
        return v;
    }

    uint32_t VarForWrite(const DstP& d) {
        switch (d.regType) {
            case RT_TEMP:
                return d.regNum < kTempCount ? temps[d.regNum] : Shadow();
            case RT_ADDR:
                return addrVar;
            case RT_LOOP:
                return loopVar;
            case RT_OUTPUT:
            case RT_ATTROUT: {
                auto it = outVars.find(d.regNum);
                return it != outVars.end() ? it->second : Shadow();
            }
            case RT_RASTOUT: {
                // oFog is written without a declaration -- the register itself
                // carries the semantic -- so its varying has to be created on
                // the first write. Without this the fog factor was stored into
                // a shadow variable and the pixel stage read an unwritten input
                // as 0, fogging every lit surface to the (black) fog colour.
                auto it = outVars.find(d.regNum);
                if (it != outVars.end()) return it->second;
                if (d.regNum != 1) return Shadow();  // oPts has no consumer
                uint32_t v = id();
                glob.op(OPC_VARIABLE, {ptrOutV4, v, SC_OUTPUT});
                // Same location the pixel stage derives for usage 9 (FOG).
                uint32_t fogLoc = Sm3LocationFor(11u, 0u);
                ann.op(OPC_DECORATE, {v, DEC_LOCATION, fogLoc});
                LOG("[sm3] vs oFog output created at loc %u", fogLoc);
                outVars[d.regNum] = v;
                interfaces.push_back(v);
                return v;
            }
            case RT_COLOROUT: {
                auto it = outVars.find(0x100 + d.regNum);
                return it != outVars.end() ? it->second : Shadow();
            }
            case RT_DEPTHOUT: {
                auto it = outVars.find(0x200);
                return it != outVars.end() ? it->second : Shadow();
            }
            default:
                return Shadow();
        }
    }

    // c[a0.x + N]: dynamic constant indexing for vertex skinning. The address
    // register is float, so round at MOVA, clamp here, then convert to an
    // unsigned index and add the base register.
    uint32_t ConstRegDynamic(uint32_t base, const SrcP& rel) {
        uint32_t var = (rel.relRegType == RT_LOOP) ? loopVar : addrVar;
        if (!var) var = addrVar;
        if (!var) return ConstReg(base);
        uint32_t aVec = LoadVar(var, tVec4);
        uint32_t lane = id();
        fn.op(OPC_COMPOSITE_EXTRACT,
              {tFloat, lane, aVec, (uint32_t)(rel.relSwz & 3u)});
        uint32_t sum = id();
        fn.op(OPC_FADD, {tFloat, sum, lane, ConstF((float)base)});

        uint32_t sumBits = id();
        fn.op(OPC_BITCAST, {tUint, sumBits, sum});
        uint32_t sumExp = id();
        fn.op(OPC_SHIFT_RIGHT_LOGICAL, {tUint, sumExp, sumBits, ConstU(23)});
        uint32_t sumExp8 = id();
        fn.op(OPC_BITWISE_AND, {tUint, sumExp8, sumExp, ConstU(0xFF)});
        uint32_t sumExp8S = id();
        fn.op(OPC_BITCAST, {tSint, sumExp8S, sumExp8});
        uint32_t diff = id();
        fn.op(OPC_I_SUB, {tSint, diff, ConstS(254), sumExp8S});
        uint32_t badFloat = id();
        fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, badFloat, diff, ConstU(31)});
        uint32_t badFloatU = id();
        fn.op(OPC_BITCAST, {tUint, badFloatU, badFloat});
        uint32_t notBadFloatU = id();
        fn.op(OPC_NOT, {tUint, notBadFloatU, badFloatU});
        uint32_t keptBits = id();
        fn.op(OPC_BITWISE_AND, {tUint, keptBits, sumBits, notBadFloatU});
        uint32_t replBits = id();
        fn.op(OPC_BITWISE_AND, {tUint, replBits, ConstU(0x43800000u), badFloatU});
        uint32_t sanBits = id();
        fn.op(OPC_BITWISE_OR, {tUint, sanBits, keptBits, replBits});
        uint32_t sanSum = id();
        fn.op(OPC_BITCAST, {tFloat, sanSum, sanBits});

        uint32_t rnd = id();
        fn.op(OPC_EXT_INST, {tFloat, rnd, glsl, GLSL_Round, sanSum});
        uint32_t cl = id();
        fn.op(OPC_EXT_INST,
              {tFloat, cl, glsl, GLSL_FClamp, rnd, ConstF(-1.0f), ConstF(256.0f)});
        uint32_t idxS = id();
        fn.op(OPC_CONVERT_F_TO_S, {tSint, idxS, cl});
        uint32_t negMask = id();
        fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, negMask, idxS, ConstU(31)});
        uint32_t diff255 = id();
        fn.op(OPC_I_SUB, {tSint, diff255, ConstS(255), idxS});
        uint32_t overMask = id();
        fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, overMask, diff255, ConstU(31)});
        uint32_t oobS = id();
        fn.op(OPC_BITWISE_OR, {tSint, oobS, negMask, overMask});
        uint32_t oobU = id();
        fn.op(OPC_BITCAST, {tUint, oobU, oobS});
        uint32_t inBoundsU = id();
        fn.op(OPC_NOT, {tUint, inBoundsU, oobU});
        uint32_t idxU = id();
        fn.op(OPC_BITCAST, {tUint, idxU, idxS});
        uint32_t keptIdx = id();
        fn.op(OPC_BITWISE_AND, {tUint, keptIdx, idxU, inBoundsU});
        uint32_t oobIdx = id();
        fn.op(OPC_BITWISE_AND, {tUint, oobIdx, cUInt256, oobU});
        uint32_t finalIdx = id();
        fn.op(OPC_BITWISE_OR, {tUint, finalIdx, keptIdx, oobIdx});

        uint32_t p = id();
        fn.op(OPC_ACCESS_CHAIN, {ptrUv4, p, uboVar, cUInt0, finalIdx});
        uint32_t rawVal = LoadVar(p, tVec4);
        std::vector<uint32_t> sanLanes = {tVec4, id()};
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t laneVal = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, laneVal, rawVal, i});
            uint32_t bVal = id();
            fn.op(OPC_BITCAST, {tUint, bVal, laneVal});
            uint32_t bExp = id();
            fn.op(OPC_SHIFT_RIGHT_LOGICAL, {tUint, bExp, bVal, cUInt23});
            uint32_t bExp8 = id();
            fn.op(OPC_BITWISE_AND, {tUint, bExp8, bExp, ConstU(0xFF)});
            uint32_t bExp8S = id();
            fn.op(OPC_BITCAST, {tSint, bExp8S, bExp8});
            uint32_t diff = id();
            fn.op(OPC_I_SUB, {tSint, diff, cSint254, bExp8S});
            uint32_t badFloat = id();
            fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, badFloat, diff, cUInt31});
            uint32_t badFloatU = id();
            fn.op(OPC_BITCAST, {tUint, badFloatU, badFloat});
            uint32_t notBadFloatU = id();
            fn.op(OPC_NOT, {tUint, notBadFloatU, badFloatU});
            uint32_t keptBits = id();
            fn.op(OPC_BITWISE_AND, {tUint, keptBits, bVal, notBadFloatU});
            uint32_t cleanF = id();
            fn.op(OPC_BITCAST, {tFloat, cleanF, keptBits});
            sanLanes.push_back(cleanF);
        }
        uint32_t sanVec = sanLanes[1];
        fn.opv(OPC_COMPOSITE_CONSTRUCT, sanLanes);
        return sanVec;
    }

    uint32_t ReadRaw(const SrcP& s) {
        switch (s.regType) {
            case RT_TEMP:
                return s.regNum < kTempCount ? LoadVar(temps[s.regNum], tVec4)
                                             : v4Zero;
            case RT_INPUT: {
                auto it = inVars.find(s.regNum);
                return it != inVars.end() ? LoadVar(it->second, tVec4) : v4Zero;
            }
            case RT_OUTPUT:
            case RT_RASTOUT:
            case RT_ATTROUT: {
                auto it = outVars.find(s.regNum);
                return it != outVars.end() ? LoadVar(it->second, tVec4) : v4Zero;
            }
            case RT_COLOROUT: {
                auto it = outVars.find(0x100 + s.regNum);
                return it != outVars.end() ? LoadVar(it->second, tVec4) : v4Zero;
            }
            case RT_ADDR:
                return LoadVar(addrVar, tVec4);
            case RT_LOOP:
                return LoadVar(loopVar, tVec4);
            case RT_CONST:
                return s.rel ? ConstRegDynamic(s.regNum, s) : ConstReg(s.regNum);
            case RT_CONST2: case RT_CONST3: case RT_CONST4:
                // c2048.. live above the 256-entry file this ABI exposes; NuEngine
                // never writes that high, so an index into c[] is the safe read.
                return ConstReg(s.regNum);
            case RT_CONSTINT: {
                auto it = defInts.find(s.regNum);
                return it != defInts.end() ? it->second : v4Zero;
            }
            case RT_CONSTBOOL: {
                auto it = defBools.find(s.regNum);
                return it != defBools.end() ? it->second : v4Zero;
            }
            default:
                return v4Zero;
        }
    }

    // Extract n lanes with the given swizzle into a vecN.
    uint32_t ToVecN(uint32_t v4, const uint32_t swz[4], uint32_t n) {
        std::vector<uint32_t> a = {TypeVec(n), id()};
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t e = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, e, v4, swz[i]});
            a.push_back(e);
        }
        uint32_t r = a[1];
        fn.opv(OPC_COMPOSITE_CONSTRUCT, a);
        return r;
    }

    uint32_t ApplyMod(uint32_t v, uint32_t ty, uint32_t width, uint32_t mod) {
        switch (mod) {
            case SMOD_NONE:
                return v;
            case SMOD_NEG: {
                uint32_t r = id();
                fn.op(OPC_FNEGATE, {ty, r, v});
                return r;
            }
            case SMOD_ABS: {
                uint32_t r = id();
                fn.op(OPC_EXT_INST, {ty, r, glsl, GLSL_FAbs, v});
                return r;
            }
            case SMOD_ABSNEG: {
                uint32_t r = id();
                fn.op(OPC_EXT_INST, {ty, r, glsl, GLSL_FAbs, v});
                uint32_t n = id();
                fn.op(OPC_FNEGATE, {ty, n, r});
                return n;
            }
            case SMOD_X2: case SMOD_X2NEG: {
                uint32_t two = Splat(cTwo, width);
                uint32_t r = id();
                fn.op(OPC_FMUL, {ty, r, v, two});
                if (mod == SMOD_X2NEG) {
                    uint32_t n = id();
                    fn.op(OPC_FNEGATE, {ty, n, r});
                    return n;
                }
                return r;
            }
            case SMOD_BIAS: case SMOD_BIASNEG: {
                uint32_t r = id();
                fn.op(OPC_FSUB, {ty, r, v, Splat(ConstF(0.5f), width)});
                if (mod == SMOD_BIASNEG) {
                    uint32_t n = id();
                    fn.op(OPC_FNEGATE, {ty, n, r});
                    return n;
                }
                return r;
            }
            case SMOD_SIGN: case SMOD_SIGNNEG: {
                uint32_t r = id();
                fn.op(OPC_FSUB, {ty, r, v, Splat(ConstF(0.5f), width)});
                uint32_t r2 = id();
                fn.op(OPC_FMUL, {ty, r2, r, Splat(cTwo, width)});
                if (mod == SMOD_SIGNNEG) {
                    uint32_t n = id();
                    fn.op(OPC_FNEGATE, {ty, n, r2});
                    return n;
                }
                return r2;
            }
            case SMOD_COMP: {
                uint32_t r = id();
                fn.op(OPC_FSUB, {ty, r, Splat(cOne, width), v});
                return r;
            }
            default:
                // DZ/DW are projections, applied at the sample site where the
                // coordinate width is known; NOT has no float meaning.
                return v;
        }
    }

    uint32_t ReadSrc(const SrcP& s) {
        uint32_t raw = ReadRaw(s);
        uint32_t sw = ToVecN(raw, s.swz, 4);
        return ApplyMod(sw, tVec4, 4, s.mod);
    }

    void WriteDst(const DstP& d, uint32_t val) {
        if (d.dm & 1) {  // _sat
            uint32_t r = id();
            fn.op(OPC_EXT_INST, {tVec4, r, glsl, GLSL_FClamp, val, v4Zero, v4One});
            val = r;
        }
        if (d.regType == RT_DEPTHOUT) {
            if (!depthWarned) {
                depthWarned = true;
                LOG("[sm3] oDepth write ignored: driver faults on gl_FragDepth");
            }
            return;
        }
        uint32_t var = VarForWrite(d);
        uint32_t mask = d.mask & 0xF;
        if (!mask) return;
        uint32_t toStore = val;
        if (mask != 0xF) {
            uint32_t cur = LoadVar(var, tVec4);
            std::vector<uint32_t> lanes = {tVec4, id()};
            for (uint32_t i = 0; i < 4; ++i) {
                uint32_t lane = id();
                fn.op(OPC_COMPOSITE_EXTRACT,
                      {tFloat, lane, (mask & (1u << i)) ? val : cur, i});
                lanes.push_back(lane);
            }
            toStore = lanes[1];
            fn.opv(OPC_COMPOSITE_CONSTRUCT, lanes);
        }
        fn.op(OPC_STORE, {var, toStore});
    }

    uint32_t Bin(uint32_t opc, uint32_t a, uint32_t b) {
        uint32_t r = id();
        fn.op(opc, {tVec4, r, a, b});
        return r;
    }

    uint32_t Ext1(uint32_t ext, uint32_t a) {
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, ext, a});
        return r;
    }

    uint32_t Ext2(uint32_t ext, uint32_t a, uint32_t b) {
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, ext, a, b});
        return r;
    }

    uint32_t Ext3(uint32_t ext, uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, ext, a, b, c});
        return r;
    }

    // nvgpucomp32 faults inside vkCreateGraphicsPipelines on any OpSelect in a
    // vertex shader, including the scalar-condition form; vkCreateShaderModule
    // accepts the module, and the same instruction is fine in a pixel shader.
    // D3D9's component-wise compares, sgn and lit are therefore expressed
    // arithmetically, which is exact for every finite input (each verification
    // of an OpSelect emission reproduced the fault, hand_lit included).
    uint32_t SignOf(uint32_t v) {
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, GLSL_Sign, v});
        return r;
    }

    // 1.0 in lanes where v > 0, 0.0 elsewhere.
    uint32_t PosMask(uint32_t v) {
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, GLSL_FMax, SignOf(v), v4Zero});
        return r;
    }

    uint32_t D3dMul(uint32_t a, uint32_t b) {
        std::vector<uint32_t> lanes = {tVec4, id()};
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t fa = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, fa, a, i});
            uint32_t fb = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, fb, b, i});
            uint32_t fprod = id();
            fn.op(OPC_FMUL, {tFloat, fprod, fa, fb});

            uint32_t ua = id();
            fn.op(OPC_BITCAST, {tUint, ua, fa});
            uint32_t ub = id();
            fn.op(OPC_BITCAST, {tUint, ub, fb});
            uint32_t uprod = id();
            fn.op(OPC_BITCAST, {tUint, uprod, fprod});

            uint32_t magA = id();
            fn.op(OPC_BITWISE_AND, {tUint, magA, ua, cUInt7FFFFFFF});
            uint32_t sa = id();
            fn.op(OPC_BITCAST, {tSint, sa, magA});
            uint32_t negA = id();
            fn.op(OPC_S_NEGATE, {tSint, negA, sa});
            uint32_t orA = id();
            fn.op(OPC_BITWISE_OR, {tSint, orA, sa, negA});
            uint32_t maskA_s = id();
            fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, maskA_s, orA, cUInt31});
            uint32_t maskA = id();
            fn.op(OPC_BITCAST, {tUint, maskA, maskA_s});

            uint32_t magB = id();
            fn.op(OPC_BITWISE_AND, {tUint, magB, ub, cUInt7FFFFFFF});
            uint32_t sb = id();
            fn.op(OPC_BITCAST, {tSint, sb, magB});
            uint32_t negB = id();
            fn.op(OPC_S_NEGATE, {tSint, negB, sb});
            uint32_t orB = id();
            fn.op(OPC_BITWISE_OR, {tSint, orB, sb, negB});
            uint32_t maskB_s = id();
            fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, maskB_s, orB, cUInt31});
            uint32_t maskB = id();
            fn.op(OPC_BITCAST, {tUint, maskB, maskB_s});

            uint32_t maskValid = id();
            fn.op(OPC_BITWISE_AND, {tUint, maskValid, maskA, maskB});

            uint32_t prodExp = id();
            fn.op(OPC_SHIFT_RIGHT_LOGICAL, {tUint, prodExp, uprod, cUInt23});
            uint32_t prodExp8 = id();
            fn.op(OPC_BITWISE_AND, {tUint, prodExp8, prodExp, ConstU(0xFF)});
            uint32_t prodExp8S = id();
            fn.op(OPC_BITCAST, {tSint, prodExp8S, prodExp8});
            uint32_t diff = id();
            fn.op(OPC_I_SUB, {tSint, diff, cSint254, prodExp8S});
            uint32_t badExp = id();
            fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, badExp, diff, cUInt31});
            uint32_t badExpU = id();
            fn.op(OPC_BITCAST, {tUint, badExpU, badExp});
            uint32_t notBadExp = id();
            fn.op(OPC_NOT, {tUint, notBadExp, badExpU});

            uint32_t finalMask = id();
            fn.op(OPC_BITWISE_AND, {tUint, finalMask, maskValid, notBadExp});

            uint32_t cleanBits = id();
            fn.op(OPC_BITWISE_AND, {tUint, cleanBits, uprod, finalMask});
            uint32_t cleanF = id();
            fn.op(OPC_BITCAST, {tFloat, cleanF, cleanBits});
            lanes.push_back(cleanF);
        }
        uint32_t res = lanes[1];
        fn.opv(OPC_COMPOSITE_CONSTRUCT, lanes);
        return res;
    }

    uint32_t DotN(const SrcP& a, const SrcP& b, uint32_t n) {
        uint32_t ty = TypeVec(n);
        uint32_t va = ApplyMod(ToVecN(ReadRaw(a), a.swz, n), ty, n, a.mod);
        uint32_t vb = ApplyMod(ToVecN(ReadRaw(b), b.swz, n), ty, n, b.mod);
        uint32_t s = id();
        fn.op(OPC_FDOT, {tFloat, s, va, vb});
        if (!pixel) {
            uint32_t uS = id();
            fn.op(OPC_BITCAST, {tUint, uS, s});
            uint32_t sExp = id();
            fn.op(OPC_SHIFT_RIGHT_LOGICAL, {tUint, sExp, uS, cUInt23});
            uint32_t sExp8 = id();
            fn.op(OPC_BITWISE_AND, {tUint, sExp8, sExp, ConstU(0xFF)});
            uint32_t sExp8S = id();
            fn.op(OPC_BITCAST, {tSint, sExp8S, sExp8});
            uint32_t diff = id();
            fn.op(OPC_I_SUB, {tSint, diff, cSint254, sExp8S});
            uint32_t badExp = id();
            fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {tSint, badExp, diff, cUInt31});
            uint32_t badExpU = id();
            fn.op(OPC_BITCAST, {tUint, badExpU, badExp});
            uint32_t notBadExp = id();
            fn.op(OPC_NOT, {tUint, notBadExp, badExpU});
            uint32_t cleanBits = id();
            fn.op(OPC_BITWISE_AND, {tUint, cleanBits, uS, notBadExp});
            uint32_t cleanS = id();
            fn.op(OPC_BITCAST, {tFloat, cleanS, cleanBits});
            s = cleanS;
        }
        return Splat(s, 4);
    }

    // Build a bool from a float comparison.
    uint32_t FCompare(uint32_t cmp, uint32_t a, uint32_t b) {
        uint32_t bo = id();
        switch (cmp) {
            default:
            case CMP_EQ: fn.op(OPC_FORD_EQUAL, {tBoolVec4, bo, a, b}); break;
            case CMP_NE: fn.op(OPC_FORD_NOT_EQUAL, {tBoolVec4, bo, a, b}); break;
            case CMP_GT: fn.op(OPC_FORD_GREATER, {tBoolVec4, bo, a, b}); break;
            case CMP_GE:
                fn.op(OPC_FORD_GREATER_EQUAL, {tBoolVec4, bo, a, b});
                break;
            case CMP_LT: fn.op(OPC_FORD_LESS, {tBoolVec4, bo, a, b}); break;
            case CMP_LE: fn.op(OPC_FORD_LESS_EQUAL, {tBoolVec4, bo, a, b}); break;
        }
        uint32_t any = id();
        fn.op(OPC_ANY, {tBool, any, bo});
        return any;
    }

    void Label(uint32_t l) { fn.op(OPC_LABEL, {l}); }

    uint32_t MakeSampler(uint32_t slot, uint32_t stype);
    void EmitDcl(const uint32_t* args, uint32_t nargs, bool declareOnly);
    void EmitTex(const uint32_t* args, uint32_t nargs, uint32_t opcode,
                 uint32_t head);
    void EmitInstr(uint32_t opcode, uint32_t head, const uint32_t* args,
                   uint32_t nargs);
};

// ---------------------------------------------------------------------------
void Ctx::EmitDcl(const uint32_t* args, uint32_t nargs, bool declareOnly) {
    if (nargs < 2) return;
    uint32_t decl = args[0], dstTok = args[1];
    // Declaration token first, destination second. If a shader disagrees, the
    // register-type field gives it away.
    // Compare with the split-field decoder, not bits 13:11: every register
    // type >= 8 (SAMPLER, CONST2..4, COLOROUT, LOOP) sets bit 11, so the raw
    // shift reports 0 for exactly the tokens that do carry a type.
    // D3DSIO_DCL is fixed-order by the D3D9 spec: args[0] is the usage/sampler
    // declaration token, args[1] is the register token. The old heuristic
    // swapped them whenever the declaration token happened to decode as a
    // register type, which produced bogus Input/Output variables at the wrong
    // register numbers and duplicate interface entries.
    (void)0;
    DstP d = ParseDst(dstTok);
    uint32_t usage = decl & 0xF;
    uint32_t usageIndex = (decl >> 16) & 0xF;
    uint32_t stype = (decl >> 27) & 0xF;

    if (d.regType == RT_SAMPLER) {
        uint32_t ty = stype ? stype : 2;
        samplerTypes[d.regNum] = ty;
        if (declareOnly) MakeSampler(d.regNum, ty);
        return;
    }
    if (!declareOnly) return;

    if (d.regType == RT_INPUT) {
        if (realInVars.find(d.regNum) != realInVars.end()) return;
        uint32_t v = id();
        glob.op(OPC_VARIABLE, {ptrInV4, v, SC_INPUT});
        // The location must be a pure function of (usage, usageIndex). Bumping it
        // when a location was already taken made the assignment depend on which
        // semantics each stage happened to declare, so a vertex and pixel shader
        // could derive different locations for the same semantic. The pixel
        // shader then read a varying the vertex stage never wrote and every
        // texture sample used a constant, garbage coordinate.
        uint32_t loc = Sm3LocationFor(usage, usageIndex);
        inLocUsed.push_back(loc);
        ann.op(OPC_DECORATE, {v, DEC_LOCATION, loc});
        realInVars[d.regNum] = v;
        if (v) interfaces.push_back(v);
        return;
    }

    if (d.regType == RT_OUTPUT || d.regType == RT_RASTOUT) {
        // Only oPos is gl_Position. Every RASTOUT register used to be aliased
        // onto it, so a vs_3_0 shader that wrote oFog (the 3D materials do, to
        // carry a per-vertex fog factor) wrote that scalar over the vertex
        // position and never produced a fog varying. The pixel shader then read
        // an unwritten input as 0, which fogs every lit surface to the fog
        // colour -- black -- while unfogged additive sprites such as the sun
        // still drew.
        uint32_t outUsage = usage;
        uint32_t outIndex = usageIndex;
        if (d.regType == RT_RASTOUT) {
            if (!pixel && d.regNum == 0) {
                realOutVars[d.regNum] = realPosVar;
                return;
            }
            outUsage = 11u;
            outIndex = 0u;
        } else if (!pixel && usage == 0) {
            realOutVars[d.regNum] = realPosVar;
            return;
        }
        uint32_t v = id();
        glob.op(OPC_VARIABLE, {ptrOutV4, v, SC_OUTPUT});
        uint32_t loc = Sm3LocationFor(outUsage, outIndex);
        outLocUsed.push_back(loc);
        ann.op(OPC_DECORATE, {v, DEC_LOCATION, loc});
        realOutVars[d.regNum] = v;
        if (v) interfaces.push_back(v);
    }
}

uint32_t Ctx::MakeSampler(uint32_t slot, uint32_t stype) {
    auto it = samplerVar.find(slot);
    if (it != samplerVar.end()) return it->second;
    // OpTypeImage Dim: 1 = 2D, 2 = 3D, 3 = Cube.
    uint32_t dim = (stype == 3) ? 3u : (stype == 4 ? 2u : 1u);
    uint32_t depth =
        (pixel && stype != 3 && stype != 4 && slot < 16 && (drefMask & (1u << slot)))
            ? 1u
            : 0u;
    uint32_t img = id();
    glob.op(OPC_TYPE_IMAGE, {img, tFloat, dim, depth, 0u, 0u, 1u, 1u});
    uint32_t samp = id();
    glob.op(OPC_TYPE_SAMPLED_IMAGE, {samp, img});
    uint32_t ptr = id();
    glob.op(OPC_TYPE_POINTER, {ptr, SC_UNIFORM_CONSTANT, samp});
    uint32_t var = id();
    glob.op(OPC_VARIABLE, {ptr, var, SC_UNIFORM_CONSTANT});
    ann.op(OPC_DECORATE, {var, DEC_DESCRIPTOR_SET, 0});
    ann.op(OPC_DECORATE, {var, DEC_BINDING, 2u + slot});
    samplerVar[slot] = var;
    samplerPtr[slot] = ptr;
    samplerSampled[slot] = samp;
    return var;
}

void Ctx::EmitTex(const uint32_t* args, uint32_t nargs, uint32_t opcode,
                  uint32_t head) {
    if (nargs < 3) return;
    DstP d = ParseDst(args[0]);
    SrcP coord = ParseSrc(args[1]);
    SrcP samp = ParseSrc(args[2]);

    uint32_t stype = 2;
    auto it = samplerTypes.find(samp.regNum);
    if (it != samplerTypes.end()) stype = it->second;
    uint32_t var = MakeSampler(samp.regNum, stype);
    uint32_t si = id();
    fn.op(OPC_LOAD, {samplerSampled[samp.regNum], si, var});

    uint32_t c4 = ReadSrc(coord);
    // texld with _dz / _dw is a projective sample.
    if (coord.mod == SMOD_DZ || coord.mod == SMOD_DW) {
        uint32_t lane = id();
        fn.op(OPC_COMPOSITE_EXTRACT,
              {tFloat, lane, c4, coord.mod == SMOD_DZ ? 2u : 3u});
        c4 = Bin(OPC_FDIV, c4, Splat(lane, 4));
    }
    if (head & 0x00010000u) {  // texldp: bit 16 of the opcode token
        uint32_t w = id();
        fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, w, c4, 3});
        c4 = Bin(OPC_FDIV, c4, Splat(w, 4));
    }

    uint32_t n = (stype == 3 || stype == 4) ? 3u : 2u;
    uint32_t cvec = ToVecN(c4, kIdentitySwz, n);

    uint32_t r = id();
    if (pixel && opcode == OP_TEX && (head & 0x00010000u) != 0 &&
        stype != 3 && stype != 4 && samp.regNum < 16 &&
        (drefMask & (1u << samp.regNum)) != 0) {
        uint32_t dref = id();
        fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, dref, c4, 2});
        uint32_t dr = id();
        fn.op(OPC_IMAGE_SAMPLE_DREF_IMPLICIT, {tFloat, dr, si, cvec, dref});
        fn.op(OPC_COMPOSITE_CONSTRUCT, {tVec4, r, dr, dr, dr, cOne});
    } else if (opcode == OP_TEXLDL) {
        uint32_t lod = id();
        fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, lod, c4, 3});
        fn.op(OPC_IMAGE_SAMPLE_EXPLICIT, {tVec4, r, si, cvec, 0x2u, lod});
    } else if (head & 0x00020000u) {  // texldb: bit 17, LOD bias
        uint32_t bias = id();
        fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, bias, c4, 3});
        fn.op(OPC_IMAGE_SAMPLE_IMPLICIT, {tVec4, r, si, cvec, 0x1u, bias});
    } else {
        fn.op(OPC_IMAGE_SAMPLE_IMPLICIT, {tVec4, r, si, cvec});
    }
    WriteDst(d, r);
}

void Ctx::EmitInstr(uint32_t opcode, uint32_t head, const uint32_t* args,
                    uint32_t nargs) {
    if (opcode == OP_NOP || opcode == OP_TEXCOORD) return;
    if (opcode == OP_DCL) {
        EmitDcl(args, nargs, false);
        return;
    }
    if (opcode == OP_TEX || opcode == OP_TEXLDL) {
        EmitTex(args, nargs, opcode, head);
        return;
    }
    if (opcode == OP_MOVA) {
        // Rounds to nearest, not toward zero, then stores into a0/aL as float.
        // The truncating conversion to an index happens at the c[] use site.
        if (nargs < 2) return;
        DstP d = ParseDst(args[0]);
        SrcP src = ParseSrc(args[1], nargs > 2 ? args[2] : 0u);
        uint32_t v = ReadSrc(src);
        uint32_t r = id();
        fn.op(OPC_EXT_INST, {tVec4, r, glsl, GLSL_Round, v});
        WriteDst(d, r);
        return;
    }
    if (opcode == OP_DEFI) {
        // defi i#, a, b, c, d
        if (nargs < 5) return;
        DstP d = ParseDst(args[0]);
        std::vector<uint32_t> a = {tVec4, id()};
        for (int i = 0; i < 4; ++i)
            a.push_back(ConstF((float)(int32_t)args[1 + i]));
        uint32_t c = a[1];
        glob.opv(OPC_CONSTANT_COMPOSITE, a);
        defInts[d.regNum] = c;
        return;
    }
    if (opcode == OP_DEFB) {
        if (nargs < 2) return;
        DstP d = ParseDst(args[0]);
        defBools[d.regNum] = Splat(ConstF(args[1] ? 1.0f : 0.0f), 4);
        return;
    }
    // ret in a main shader would make the epilogue unreachable, which SPIR-V
    // forbids; the single return at the end of the function already covers it.
    if (opcode == OP_RET || opcode == OP_LABEL) return;

    if (opcode == OP_DEF) {
        if (nargs < 5) return;
        DstP d = ParseDst(args[0]);
        if (d.regType != RT_CONST) return;
        float v[4];
        memcpy(v, args + 1, 16);
        std::vector<uint32_t> a = {tVec4, id()};
        for (int i = 0; i < 4; ++i) a.push_back(ConstF(v[i]));
        uint32_t c = a[1];
        glob.opv(OPC_CONSTANT_COMPOSITE, a);
        defConsts[d.regNum] = c;
        return;
    }

    // ---- control flow ------------------------------------------------------
    if (opcode == OP_IF || opcode == OP_IFC) {
        uint32_t cond;
        if (opcode == OP_IF) {
            uint32_t v = ReadSrc(ParseSrc(args[0]));
            uint32_t lane = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, lane, v, 0});
            cond = id();
            fn.op(OPC_FORD_NOT_EQUAL, {tBool, cond, lane, cZero});
        } else {
            uint32_t cmp = (head >> 16) & 0xFF;
            cond = FCompare(cmp, ReadSrc(ParseSrc(args[0])),
                            ReadSrc(ParseSrc(args[1])));
        }
        Cf f;
        f.merge = id();
        f.elseLabel = id();
        uint32_t thenBlock = id();
        fn.op(OPC_SELECTION_MERGE, {f.merge, 0u});
        // true -> then block. Passing merge here would execute the body only
        // when the condition is false, i.e. invert every if in the shader.
        fn.op(OPC_BRANCH_CONDITIONAL, {cond, thenBlock, f.elseLabel});
        Label(thenBlock);
        cf.push_back(f);
        return;
    }
    if (opcode == OP_ELSE) {
        if (cf.empty()) return;
        Cf& f = cf.back();
        fn.op(OPC_BRANCH, {f.merge});
        Label(f.elseLabel);
        f.hasElse = true;
        return;
    }
    if (opcode == OP_ENDIF) {
        if (cf.empty()) return;
        Cf f = cf.back();
        cf.pop_back();
        if (!f.hasElse) {
            fn.op(OPC_BRANCH, {f.merge});
            Label(f.elseLabel);
        }
        fn.op(OPC_BRANCH, {f.merge});
        Label(f.merge);
        return;
    }
    if (opcode == OP_REP || opcode == OP_LOOP) {
        Cf f;
        f.isLoop = true;
        f.merge = id();
        f.contLabel = id();
        uint32_t v = ReadSrc(ParseSrc(args[0]));
        uint32_t lane = id();
        fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, lane, v, 0});
        f.countValue = lane;
        f.counterVar = cfScalars[cfNext++ % kCfScalars];
        fn.op(OPC_STORE, {f.counterVar, cZero});
        if (opcode == OP_LOOP) {
            f.usesLoopReg = true;
            fn.op(OPC_STORE, {loopVar, v4Zero});
        }

        f.header = id();
        fn.op(OPC_BRANCH, {f.header});
        Label(f.header);
        fn.op(OPC_LOOP_MERGE, {f.merge, f.contLabel, 0u});
        uint32_t body = id();
        fn.op(OPC_BRANCH, {body});
        Label(body);
        cf.push_back(f);
        return;
    }
    if (opcode == OP_ENDREP || opcode == OP_ENDLOOP) {
        if (cf.empty() || !cf.back().isLoop) return;
        Cf f = cf.back();
        cf.pop_back();
        fn.op(OPC_BRANCH, {f.contLabel});
        Label(f.contLabel);
        uint32_t c = LoadVar(f.counterVar, tFloat);
        uint32_t nx = id();
        fn.op(OPC_FADD, {tFloat, nx, c, cOne});
        fn.op(OPC_STORE, {f.counterVar, nx});
        if (f.usesLoopReg) {
            uint32_t sp = Splat(nx, 4);
            fn.op(OPC_STORE, {loopVar, sp});
        }
        uint32_t cmp = id();
        fn.op(OPC_FORD_LESS, {tBool, cmp, nx, f.countValue});
        fn.op(OPC_BRANCH_CONDITIONAL, {cmp, f.header, f.merge});
        Label(f.merge);
        return;
    }
    if (opcode == OP_BREAK || opcode == OP_BREAKC) {
        for (size_t i = cf.size(); i-- > 0;) {
            if (!cf[i].isLoop) continue;
            uint32_t target = cf[i].merge;
            if (opcode == OP_BREAK) {
                fn.op(OPC_BRANCH, {target});
            } else {
                uint32_t cmp = (head >> 16) & 0xFF;
                uint32_t cond = FCompare(cmp, ReadSrc(ParseSrc(args[0])),
                                         ReadSrc(ParseSrc(args[1])));
                uint32_t cont = id();
                fn.op(OPC_BRANCH_CONDITIONAL, {cond, target, cont});
                Label(cont);
            }
            return;
        }
        if (opcode == OP_BREAKC) {  // keep the compare's side effects minimal
            (void)ReadSrc(ParseSrc(args[0]));
        }
        return;
    }
    if (opcode == OP_TEXKILL) {
        DstP d = ParseDst(args[0]);
        SrcP s;
        s.regType = d.regType;
        s.regNum = d.regNum;
        s.rel = d.rel;
        for (int i = 0; i < 4; ++i) s.swz[i] = i;
        s.mod = SMOD_NONE;
        uint32_t v = ReadRaw(s);
        uint8_t mask = d.mask ? (d.mask & 0xFu) : 0xFu;
        uint32_t cond = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            if (!(mask & (1u << i))) continue;
            uint32_t comp = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, comp, v, i});
            uint32_t lt = id();
            fn.op(OPC_FORD_LESS, {tBool, lt, comp, cZero});
            if (cond == 0) {
                cond = lt;
            } else {
                uint32_t nextCond = id();
                fn.op(OPC_LOGICAL_OR, {tBool, nextCond, cond, lt});
                cond = nextCond;
            }
        }
        if (cond != 0) {
            uint32_t merge = id(), killB = id();
            fn.op(OPC_SELECTION_MERGE, {merge, 0u});
            fn.op(OPC_BRANCH_CONDITIONAL, {cond, killB, merge});
            Label(killB);
            fn.op(OPC_KILL, {});
            Label(merge);
        }
        return;
    }

    // ---- arithmetic --------------------------------------------------------
    if (nargs < 1) return;
    DstP d = ParseDst(args[0], nargs > 1 ? args[1] : 0u);
    std::vector<SrcP> s;
    // A relative operand is followed by an extra address-register token that is
    // part of the instruction length but is not an operand. Failing to skip it
    // makes src1 read src0's address token.
    for (uint32_t i = d.rel ? 2u : 1u; i < nargs;) {
        SrcP one = ParseSrc(args[i], i + 1 < nargs ? args[i + 1] : 0u);
        s.push_back(one);
        i += one.rel ? 2u : 1u;
    }
    auto S = [&](uint32_t i) {
        static const SrcP kMissing{};
        return ReadSrc(i < s.size() ? s[i] : kMissing);
    };
    uint32_t r = 0;

    switch (opcode) {
        case OP_MOV: r = S(0); break;
        case OP_ADD: r = Bin(OPC_FADD, S(0), S(1)); break;
        case OP_SUB: r = Bin(OPC_FSUB, S(0), S(1)); break;
        case OP_MUL: r = pixel ? Bin(OPC_FMUL, S(0), S(1)) : D3dMul(S(0), S(1)); break;
        case OP_MAD: {
            uint32_t m = pixel ? Bin(OPC_FMUL, S(0), S(1)) : D3dMul(S(0), S(1));
            r = Bin(OPC_FADD, m, S(2));
            break;
        }
        case OP_RCP: {
            uint32_t absS = Ext1(GLSL_FAbs, S(0));
            uint32_t safeAbs = Ext2(GLSL_FMax, absS, Splat(ConstF(1.17549435e-38f), 4));
            uint32_t rcpAbs = Bin(OPC_FDIV, v4One, safeAbs);
            uint32_t sgn = SignOf(S(0));
            r = Bin(OPC_FMUL, rcpAbs, sgn);
            break;
        }
        case OP_RSQ: {
            uint32_t absS = Ext1(GLSL_FAbs, S(0));
            uint32_t eps = Splat(ConstF(1.17549435e-38f), 4);
            uint32_t safeS = Ext2(GLSL_FMax, absS, eps);
            r = Ext1(GLSL_InverseSqrt, safeS);
            break;
        }
        case OP_DP3: r = DotN(s[0], s[1], 3); break;
        case OP_DP4: r = DotN(s[0], s[1], 4); break;
        case OP_DP2ADD: {
            uint32_t dot = DotN(s[0], s[1], 2);
            uint32_t add = S(2);
            uint32_t lane = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, lane, add, 0});
            r = Bin(OPC_FADD, dot, Splat(lane, 4));
            break;
        }
        // ps_3_0 dsx / dsy: screen-space derivatives, which the surface shaders
        // use to build tangent frames for normal mapping. They were reported as
        // unsupported opcodes and dropped, leaving those shaders without the
        // perturbed normal -- every lit surface in the menu scene came out black
        // while unlit sprites (the sun flare) still drew.
        case OP_DSX: {
            uint32_t out = id();
            fn.op(207u, {tVec4, out, S(0)});  // OpDPdx
            r = out;
            break;
        }
        case OP_DSY: {
            uint32_t out = id();
            fn.op(208u, {tVec4, out, S(0)});  // OpDPdy
            r = out;
            break;
        }
        case OP_MIN: r = Ext2(GLSL_FMin, S(0), S(1)); break;
        case OP_MAX: r = Ext2(GLSL_FMax, S(0), S(1)); break;
        case OP_SLT: {
            if (pixel) {
                uint32_t cond = id();
                fn.op(OPC_FORD_LESS, {tBoolVec4, cond, S(0), S(1)});
                r = id();
                fn.op(OPC_SELECT, {tVec4, r, cond, v4One, v4Zero});
            } else {
                r = PosMask(Bin(OPC_FSUB, S(1), S(0)));
            }
            break;
        }
        case OP_SGE: {
            if (pixel) {
                uint32_t cond = id();
                fn.op(OPC_FORD_GREATER_EQUAL, {tBoolVec4, cond, S(0), S(1)});
                r = id();
                fn.op(OPC_SELECT, {tVec4, r, cond, v4One, v4Zero});
            } else {
                uint32_t lt = PosMask(Bin(OPC_FSUB, S(1), S(0)));
                r = Bin(OPC_FSUB, v4One, lt);
            }
            break;
        }
        case OP_CMP: {
            if (pixel) {
                uint32_t a = S(0), b = S(1), c = S(2);
                uint32_t cond = id();
                fn.op(OPC_FORD_GREATER_EQUAL, {tBoolVec4, cond, a, v4Zero});
                r = id();
                fn.op(OPC_SELECT, {tVec4, r, cond, b, c});
            } else {
                uint32_t a = S(0), b = S(1), c = S(2);
                uint32_t neg = id();
                fn.op(OPC_FNEGATE, {tVec4, neg, a});
                uint32_t m = Bin(OPC_FSUB, v4One, PosMask(neg));
                r = Bin(OPC_FADD, c, Bin(OPC_FMUL, Bin(OPC_FSUB, b, c), m));
            }
            break;
        }
        case OP_EXP: r = Ext1(GLSL_Exp2, S(0)); break;
        case OP_LOG: {
            uint32_t safeS = Ext2(GLSL_FMax, Ext1(GLSL_FAbs, S(0)), Splat(ConstF(1.17549435e-38f), 4));
            r = Ext1(GLSL_Log2, safeS);
            break;
        }
        case OP_POW: {
            uint32_t safeS = Ext2(GLSL_FMax, Ext1(GLSL_FAbs, S(0)), Splat(ConstF(1.17549435e-38f), 4));
            r = Ext2(GLSL_Pow, safeS, S(1));
            break;
        }
        case OP_ABS: r = Ext1(GLSL_FAbs, S(0)); break;
        case OP_FRC: r = Ext1(GLSL_Fract, S(0)); break;
        case OP_NRM: {
            uint32_t v3 = ToVecN(S(0), kIdentitySwz, 3);
            uint32_t dot = id();
            fn.op(OPC_FDOT, {tFloat, dot, v3, v3});
            uint32_t safeDot = id();
            fn.op(OPC_EXT_INST, {tFloat, safeDot, glsl, GLSL_FMax, dot, ConstF(1.17549435e-38f)});
            uint32_t invLen = id();
            fn.op(OPC_EXT_INST, {tFloat, invLen, glsl, GLSL_InverseSqrt, safeDot});
            uint32_t diff = id();
            fn.op(OPC_FSUB, {tFloat, diff, dot, ConstF(1e-6f)});
            uint32_t sgn = id();
            fn.op(OPC_EXT_INST, {tFloat, sgn, glsl, GLSL_Sign, diff});
            uint32_t valid = id();
            fn.op(OPC_EXT_INST, {tFloat, valid, glsl, GLSL_FMax, sgn, ConstF(0.0f)});
            uint32_t cleanInvLen = id();
            fn.op(OPC_FMUL, {tFloat, cleanInvLen, invLen, valid});
            uint32_t vx = id(), vy = id(), vz = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, vx, v3, 0});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, vy, v3, 1});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, vz, v3, 2});
            uint32_t nx = id(), ny = id(), nz = id();
            fn.op(OPC_FMUL, {tFloat, nx, vx, cleanInvLen});
            fn.op(OPC_FMUL, {tFloat, ny, vy, cleanInvLen});
            fn.op(OPC_FMUL, {tFloat, nz, vz, cleanInvLen});
            r = id();
            fn.op(OPC_COMPOSITE_CONSTRUCT, {tVec4, r, nx, ny, nz, cZero});
            break;
        }
        case OP_CRS: r = Ext2(GLSL_Cross, S(0), S(1)); break;
        case OP_LRP: {
            uint32_t a = S(0), b = S(1), c = S(2);
            uint32_t inv = Bin(OPC_FSUB, v4One, a);
            r = Bin(OPC_FADD, Bin(OPC_FMUL, a, b), Bin(OPC_FMUL, inv, c));
            break;
        }
        case OP_SINCOS: {
            uint32_t a = S(0);
            uint32_t cs = Ext1(GLSL_Cos, a);
            uint32_t sn = Ext1(GLSL_Sin, a);
            uint32_t cx = id(), sy = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, cx, cs, 0});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, sy, sn, 0});
            r = id();
            fn.op(OPC_COMPOSITE_CONSTRUCT, {tVec4, r, cx, sy, cZero, cZero});
            break;
        }
        case OP_M4x4: case OP_M4x3: case OP_M3x4: case OP_M3x3: case OP_M3x2: {
            uint32_t rows = (opcode == OP_M4x4 || opcode == OP_M4x3) ? 4u : 3u;
            uint32_t cols = (opcode == OP_M4x4 || opcode == OP_M3x4) ? 4u : ((opcode == OP_M3x2) ? 2u : 3u);
            SrcP mtx = s.size() > 1 ? s[1] : s[0];
            uint32_t v = S(0);
            uint32_t lanes[4] = {0, 0, 0, 0};
            for (uint32_t c = 0; c < cols; ++c) {
                SrcP row;
                row.regType = mtx.regType;
                row.regNum = mtx.regNum + c;
                row.rel = mtx.rel;
                row.relRegType = mtx.relRegType;
                row.relSwz = mtx.relSwz;
                uint32_t mv = row.rel ? ConstRegDynamic(row.regNum, row)
                                      : ReadRaw(row);
                uint32_t a = ToVecN(v, kIdentitySwz, rows);
                uint32_t b = ToVecN(mv, kIdentitySwz, rows);
                lanes[c] = id();
                fn.op(OPC_FDOT, {tFloat, lanes[c], a, b});
            }
            if (cols == 3 && d.mask == 0xF) d.mask = 0x7;
            if (cols == 2 && d.mask == 0xF) d.mask = 0x3;
            uint32_t z = id(), w = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, z, v, 2});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, w, v, 3});
            if (cols == 4) {
                r = id();
                fn.op(OPC_COMPOSITE_CONSTRUCT,
                      {tVec4, r, lanes[0], lanes[1], lanes[2], lanes[3]});
            } else if (cols == 3) {
                r = id();
                fn.op(OPC_COMPOSITE_CONSTRUCT,
                      {tVec4, r, lanes[0], lanes[1], lanes[2], w});
            } else {
                r = id();
                fn.op(OPC_COMPOSITE_CONSTRUCT,
                      {tVec4, r, lanes[0], lanes[1], z, w});
            }
            break;
        }
        case OP_LIT: {
            uint32_t a = S(0);
            uint32_t x = id(), y = id(), w = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, x, a, 0});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, y, a, 1});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, w, a, 3});
            uint32_t mx = id();
            fn.op(OPC_EXT_INST, {tFloat, mx, glsl, GLSL_FMax, x, cZero});
            uint32_t my = id();
            fn.op(OPC_EXT_INST, {tFloat, my, glsl, GLSL_FMax, y, ConstF(1.17549435e-38f)});
            uint32_t cl = id();
            fn.op(OPC_EXT_INST, {tFloat, cl, glsl, GLSL_FClamp, w,
                                 ConstF(-128.0f), ConstF(128.0f)});
            uint32_t pw = id();
            fn.op(OPC_EXT_INST, {tFloat, pw, glsl, GLSL_Pow, my, cl});
            uint32_t sgy = id();
            fn.op(OPC_EXT_INST, {tFloat, sgy, glsl, GLSL_Sign, y});
            uint32_t pmy = id();
            fn.op(OPC_EXT_INST, {tFloat, pmy, glsl, GLSL_FMax, sgy, cZero});
            uint32_t sg = id();
            fn.op(OPC_EXT_INST, {tFloat, sg, glsl, GLSL_Sign, x});
            uint32_t pm = id();
            fn.op(OPC_EXT_INST, {tFloat, pm, glsl, GLSL_FMax, sg, cZero});
            uint32_t pm2 = id();
            fn.op(OPC_FMUL, {tFloat, pm2, pm, pmy});
            uint32_t prod = id();
            fn.op(OPC_FMUL, {tFloat, prod, pw, pm2});
            uint32_t z = id();
            fn.op(OPC_EXT_INST, {tFloat, z, glsl, GLSL_FMax, prod, cZero});
            r = id();
            fn.op(OPC_COMPOSITE_CONSTRUCT, {tVec4, r, cOne, mx, z, cOne});
            break;
        }
        case OP_DST: {
            uint32_t a = S(0), bb = S(1);
            uint32_t x = id(), y = id(), z = id(), w = id();
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, x, a, 1});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, y, bb, 1});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, z, a, 2});
            fn.op(OPC_COMPOSITE_EXTRACT, {tFloat, w, bb, 3});
            uint32_t p = id();
            fn.op(OPC_FMUL, {tFloat, p, x, y});
            r = id();
            fn.op(OPC_COMPOSITE_CONSTRUCT, {tVec4, r, cOne, p, z, w});
            break;
        }
        case OP_SGN: {
            // SM3 form: sgn dst, src0, src1, src2
            if (nargs >= 4) {
                // src0 < 0 ? src1 : (src0 > 0 ? src2 : 0), as two masks.
                uint32_t c = S(0), negv = S(1), posv = S(2);
                uint32_t nc = id();
                fn.op(OPC_FNEGATE, {tVec4, nc, c});
                r = Bin(OPC_FADD, Bin(OPC_FMUL, negv, PosMask(nc)),
                        Bin(OPC_FMUL, posv, PosMask(c)));
            } else {
                // Two-source form is the plain component-wise sign.
                r = SignOf(S(0));
            }
            break;
        }
        default:
            if (++unsupported <= 6)
                LOG("[sm3] %s shader: unsupported opcode %u at r%u",
                    pixel ? "pixel" : "vertex", opcode, d.regNum);
            return;
    }
    WriteDst(d, r);
}

}  // namespace

// ---------------------------------------------------------------------------
bool Sm3IsSupported(const uint32_t* tokens, uint32_t dwordCount) {
    if (!tokens || dwordCount < 2) return false;
    return tokens[0] == 0xFFFE0300u || tokens[0] == 0xFFFF0300u;
}

bool Sm3CompileToSpirv(bool pixelHint, const uint32_t* tokens,
                       uint32_t dwordCount, std::vector<uint32_t>& out,
                       std::string& error, uint32_t drefMask) {
    if (!tokens || dwordCount < 2) {
        error = "empty token stream";
        return false;
    }
    if (!Sm3IsSupported(tokens, dwordCount)) {
        char b[96];
        snprintf(b, sizeof(b), "unsupported shader version 0x%08X", tokens[0]);
        error = b;
        return false;
    }
    Ctx c;
    c.pixel = (tokens[0] & 0xFFFF0000u) == 0xFFFF0000u;
    (void)pixelHint;  // the version token is authoritative
    c.drefMask = drefMask;
    c.err = &error;
    c.nextId = 1;

    c.BuildPreamble();
    c.BuildTypes();

    // Pass 1: declarations. In SM3 every DCL precedes the first executable
    // instruction, so all Input/Output globals exist before the function body
    // and the entry-point interface list can be built exactly.
    {
        uint32_t i = 1;
        while (i < dwordCount) {
            uint32_t head = tokens[i];
            uint32_t opcode = head & 0xFFFF;
            if (opcode == OP_END) break;
            uint32_t len = Sm3InstructionDwords(head);
            if (i + len > dwordCount) break;
            if (opcode == OP_COMMENT) {
                i += len;
                continue;
            }
            if (opcode == OP_DCL && len >= 3)
                c.EmitDcl(tokens + i + 1, len - 1, true);
            // A FragDepth store needs OpExecutionMode DepthReplacing; without it
            // nvgpucomp reads an empty execution-mode entry and faults.
            if (opcode != OP_DCL && len >= 2) {
                DstP d = ParseDst(tokens[i + 1]);
                if (d.regType == RT_COLOROUT) c.colorUsed.push_back(d.regNum);
            }
            // A sampler used without dcl_* would otherwise be created mid-body,
            // after the function started, which SPIR-V forbids.
            if ((opcode == OP_TEX || opcode == OP_TEXLDL) && len >= 4) {
                SrcP s = ParseSrc(tokens[i + 3]);
                if (!c.samplerVar.count(s.regNum)) c.MakeSampler(s.regNum, 2);
            }
            i += len;
        }
    }
    c.DeclarePixelOutputs();

    c.mainId = c.id();
    {
        Sect& e = c.ep;
        const char* nm = "main";
        size_t n = strlen(nm);
        uint32_t words = (uint32_t)(n / 4) + 1;
        // An undefined id 0 in OpEntryPoint made nvgpucomp read [NULL+8].
        std::vector<uint32_t> iface;
        for (uint32_t v : c.interfaces)
            if (v && std::find(iface.begin(), iface.end(), v) == iface.end())
                iface.push_back(v);
        e.code.push_back(((3 + words + (uint32_t)iface.size()) << 16) |
                         OPC_ENTRY_POINT);
        e.code.push_back(c.pixel ? 4u : 0u);
        e.code.push_back(c.mainId);
        for (size_t i = 0; i <= n; i += 4) {
            uint32_t w = 0;
            for (size_t k = 0; k < 4 && i + k <= n; ++k)
                w |= (uint32_t)(uint8_t)nm[i + k] << (8 * k);
            e.code.push_back(w);
        }
        for (uint32_t v : iface) e.code.push_back(v);
        if (c.pixel) e.op(OPC_EXECUTION_MODE, {c.mainId, 7u});  // OriginUpperLeft
    }

    uint32_t fnTy = c.id();
    c.glob.op(OPC_TYPE_FUNCTION, {fnTy, c.tVoid});
    c.fn.op(OPC_FUNCTION, {c.tVoid, c.mainId, 0u, fnTy});
    uint32_t entry = c.id();
    c.Label(entry);
    c.DeclareLocals();

    // Pass 2: code.
    {
        uint32_t i = 1;
        while (i < dwordCount) {
            uint32_t head = tokens[i];
            uint32_t opcode = head & 0xFFFF;
            if (opcode == OP_END) break;
            uint32_t len = Sm3InstructionDwords(head);
            if (opcode == OP_COMMENT) {
                i += len;
                continue;
            }
            if (i + len > dwordCount) {
                error = "instruction runs past end of blob";
                return false;
            }
            c.EmitInstr(opcode, head, tokens + i + 1, len - 1);
            i += len;
        }
    }

    if (c.pixel) {
        for (uint32_t reg : c.colorUsed) {
            auto itLocal = c.outVars.find(0x100 + reg);
            auto itReal = c.realOutVars.find(0x100 + reg);
            if (itLocal != c.outVars.end() && itReal != c.realOutVars.end()) {
                uint32_t finalVal = c.LoadVar(itLocal->second, c.tVec4);
                if (reg == 0) {
                    uint32_t alpha = c.id();
                    c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, alpha, finalVal, 3});
                    uint32_t c_lt = c.id();
                    c.fn.op(OPC_FORD_LESS, {c.tBool, c_lt, alpha, c.specAlphaRef});
                    uint32_t c_eq = c.id();
                    c.fn.op(OPC_FORD_EQUAL, {c.tBool, c_eq, alpha, c.specAlphaRef});
                    uint32_t c_gt = c.id();
                    c.fn.op(OPC_FORD_GREATER, {c.tBool, c_gt, alpha, c.specAlphaRef});
                    uint32_t c_le = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, c_le, c_lt, c_eq});
                    uint32_t c_ge = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, c_ge, c_gt, c_eq});
                    uint32_t c_ne = c.id();
                    c.fn.op(OPC_LOGICAL_NOT, {c.tBool, c_ne, c_eq});

                    uint32_t f_2 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_2, c.specAlphaFunc, c.cUInt2});
                    uint32_t f_3 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_3, c.specAlphaFunc, c.cUInt3});
                    uint32_t f_4 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_4, c.specAlphaFunc, c.cUInt4});
                    uint32_t f_5 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_5, c.specAlphaFunc, c.cUInt5});
                    uint32_t f_6 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_6, c.specAlphaFunc, c.cUInt6});
                    uint32_t f_7 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_7, c.specAlphaFunc, c.cUInt7});
                    uint32_t f_8 = c.id();
                    c.fn.op(OPC_I_EQUAL, {c.tBool, f_8, c.specAlphaFunc, c.cUInt8});

                    uint32_t p2 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p2, f_2, c_lt});
                    uint32_t p3 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p3, f_3, c_eq});
                    uint32_t p4 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p4, f_4, c_le});
                    uint32_t p5 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p5, f_5, c_gt});
                    uint32_t p6 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p6, f_6, c_ne});
                    uint32_t p7 = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, p7, f_7, c_ge});

                    uint32_t o1 = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, o1, p2, p3});
                    uint32_t o2 = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, o2, p4, p5});
                    uint32_t o3 = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, o3, p6, p7});
                    uint32_t o4 = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, o4, o1, o2});
                    uint32_t o5 = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, o5, o3, f_8});
                    uint32_t pass = c.id();
                    c.fn.op(OPC_LOGICAL_OR, {c.tBool, pass, o4, o5});

                    uint32_t active = c.id();
                    c.fn.op(OPC_I_NOT_EQUAL, {c.tBool, active, c.specAlphaTest, c.cUInt0});
                    uint32_t fail = c.id();
                    c.fn.op(OPC_LOGICAL_NOT, {c.tBool, fail, pass});
                    uint32_t killCond = c.id();
                    c.fn.op(OPC_LOGICAL_AND, {c.tBool, killCond, active, fail});

                    uint32_t mergeB = c.id();
                    uint32_t killB = c.id();
                    c.fn.op(OPC_SELECTION_MERGE, {mergeB, 0u});
                    c.fn.op(OPC_BRANCH_CONDITIONAL, {killCond, killB, mergeB});
                    c.Label(killB);
                    c.fn.op(OPC_KILL, {});
                    c.Label(mergeB);
                }
                c.fn.op(OPC_STORE, {itReal->second, finalVal});
            }
        }
    } else {
        for (auto& kv : c.realOutVars) {
            if (kv.second == c.realPosVar) continue;
            auto itLocal = c.outVars.find(kv.first);
            if (itLocal != c.outVars.end()) {
                uint32_t rawVal = c.LoadVar(itLocal->second, c.tVec4);
                std::vector<uint32_t> sanLanes = {c.tVec4, c.id()};
                for (uint32_t i = 0; i < 4; ++i) {
                    uint32_t laneVal = c.id();
                    c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, laneVal, rawVal, i});
                    uint32_t bVal = c.id();
                    c.fn.op(OPC_BITCAST, {c.tUint, bVal, laneVal});
                    uint32_t bExp = c.id();
                    c.fn.op(OPC_SHIFT_RIGHT_LOGICAL, {c.tUint, bExp, bVal, c.cUInt23});
                    uint32_t bExp8 = c.id();
                    c.fn.op(OPC_BITWISE_AND, {c.tUint, bExp8, bExp, c.ConstU(0xFF)});
                    uint32_t bExp8S = c.id();
                    c.fn.op(OPC_BITCAST, {c.tSint, bExp8S, bExp8});
                    uint32_t diff = c.id();
                    c.fn.op(OPC_I_SUB, {c.tSint, diff, c.cSint254, bExp8S});
                    uint32_t badFloat = c.id();
                    c.fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {c.tSint, badFloat, diff, c.cUInt31});
                    uint32_t badFloatU = c.id();
                    c.fn.op(OPC_BITCAST, {c.tUint, badFloatU, badFloat});
                    uint32_t notBadFloatU = c.id();
                    c.fn.op(OPC_NOT, {c.tUint, notBadFloatU, badFloatU});
                    uint32_t keptBits = c.id();
                    c.fn.op(OPC_BITWISE_AND, {c.tUint, keptBits, bVal, notBadFloatU});
                    uint32_t cleanF = c.id();
                    c.fn.op(OPC_BITCAST, {c.tFloat, cleanF, keptBits});
                    sanLanes.push_back(cleanF);
                }
                uint32_t sanVec = sanLanes[1];
                c.fn.opv(OPC_COMPOSITE_CONSTRUCT, sanLanes);
                c.fn.op(OPC_STORE, {kv.second, sanVec});
            }
        }
        uint32_t rawPos = c.LoadVar(c.posVar, c.tVec4);
        uint32_t px = c.id(), py = c.id(), pz = c.id(), pw = c.id();
        c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, px, rawPos, 0});
        c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, py, rawPos, 1});
        c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, pz, rawPos, 2});
        c.fn.op(OPC_COMPOSITE_EXTRACT, {c.tFloat, pw, rawPos, 3});

        uint32_t ux = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, ux, px});
        uint32_t expX = c.id(); c.fn.op(OPC_SHIFT_RIGHT_LOGICAL, {c.tUint, expX, ux, c.cUInt23});
        uint32_t expX8 = c.id(); c.fn.op(OPC_BITWISE_AND, {c.tUint, expX8, expX, c.ConstU(0xFF)});
        uint32_t expX8S = c.id(); c.fn.op(OPC_BITCAST, {c.tSint, expX8S, expX8});
        uint32_t diffX = c.id(); c.fn.op(OPC_I_SUB, {c.tSint, diffX, c.cSint254, expX8S});
        uint32_t badX_s = c.id(); c.fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {c.tSint, badX_s, diffX, c.cUInt31});
        uint32_t badX = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, badX, badX_s});

        uint32_t uy = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, uy, py});
        uint32_t expY = c.id(); c.fn.op(OPC_SHIFT_RIGHT_LOGICAL, {c.tUint, expY, uy, c.cUInt23});
        uint32_t expY8 = c.id(); c.fn.op(OPC_BITWISE_AND, {c.tUint, expY8, expY, c.ConstU(0xFF)});
        uint32_t expY8S = c.id(); c.fn.op(OPC_BITCAST, {c.tSint, expY8S, expY8});
        uint32_t diffY = c.id(); c.fn.op(OPC_I_SUB, {c.tSint, diffY, c.cSint254, expY8S});
        uint32_t badY_s = c.id(); c.fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {c.tSint, badY_s, diffY, c.cUInt31});
        uint32_t badY = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, badY, badY_s});

        uint32_t uz = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, uz, pz});
        uint32_t expZ = c.id(); c.fn.op(OPC_SHIFT_RIGHT_LOGICAL, {c.tUint, expZ, uz, c.cUInt23});
        uint32_t expZ8 = c.id(); c.fn.op(OPC_BITWISE_AND, {c.tUint, expZ8, expZ, c.ConstU(0xFF)});
        uint32_t expZ8S = c.id(); c.fn.op(OPC_BITCAST, {c.tSint, expZ8S, expZ8});
        uint32_t diffZ = c.id(); c.fn.op(OPC_I_SUB, {c.tSint, diffZ, c.cSint254, expZ8S});
        uint32_t badZ_s = c.id(); c.fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {c.tSint, badZ_s, diffZ, c.cUInt31});
        uint32_t badZ = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, badZ, badZ_s});

        uint32_t uw = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, uw, pw});
        uint32_t expW = c.id(); c.fn.op(OPC_SHIFT_RIGHT_LOGICAL, {c.tUint, expW, uw, c.cUInt23});
        uint32_t expW8 = c.id(); c.fn.op(OPC_BITWISE_AND, {c.tUint, expW8, expW, c.ConstU(0xFF)});
        uint32_t expW8S = c.id(); c.fn.op(OPC_BITCAST, {c.tSint, expW8S, expW8});
        uint32_t diffW = c.id(); c.fn.op(OPC_I_SUB, {c.tSint, diffW, c.cSint254, expW8S});
        uint32_t badW_s = c.id(); c.fn.op(OPC_SHIFT_RIGHT_ARITHMETIC, {c.tSint, badW_s, diffW, c.cUInt31});
        uint32_t badW = c.id(); c.fn.op(OPC_BITCAST, {c.tUint, badW, badW_s});

        uint32_t badXY = c.id(); c.fn.op(OPC_BITWISE_OR, {c.tUint, badXY, badX, badY});
        uint32_t badZW = c.id(); c.fn.op(OPC_BITWISE_OR, {c.tUint, badZW, badZ, badW});
        uint32_t badAny = c.id(); c.fn.op(OPC_BITWISE_OR, {c.tUint, badAny, badXY, badZW});

        // A non-finite clip coordinate has no D3D9 meaning, and one NaN lane
        // poisons every triangle that shares the vertex. Such lanes are
        // replaced with the identity vertex (0,0,0,1). This is a bit-exact test
        // of the exponent field, so finite coordinates (of any magnitude) pass
        // through unchanged; large but finite geometry is the engine's data to
        // get right, not the translator's to clamp.
        uint32_t notBad = c.id();
        c.fn.op(OPC_NOT, {c.tUint, notBad, badAny});
        uint32_t replZeroX = c.id();
        c.fn.op(OPC_BITWISE_AND, {c.tUint, replZeroX, c.ConstU(0u), badAny});
        uint32_t replOneW = c.id();
        c.fn.op(OPC_BITWISE_AND, {c.tUint, replOneW, c.ConstU(0x3F800000u), badAny});
        auto cleanLane = [&](uint32_t bits, uint32_t repl) {
            uint32_t kept = c.id();
            c.fn.op(OPC_BITWISE_AND, {c.tUint, kept, bits, notBad});
            uint32_t out = c.id();
            c.fn.op(OPC_BITWISE_OR, {c.tUint, out, kept, repl});
            return out;
        };
        uint32_t kx = cleanLane(ux, replZeroX);
        uint32_t ky = cleanLane(uy, replZeroX);
        uint32_t kz = cleanLane(uz, replZeroX);
        uint32_t kw = cleanLane(uw, replOneW);

        uint32_t fx = c.id(); c.fn.op(OPC_BITCAST, {c.tFloat, fx, kx});
        uint32_t fy = c.id(); c.fn.op(OPC_BITCAST, {c.tFloat, fy, ky});
        uint32_t fz = c.id(); c.fn.op(OPC_BITCAST, {c.tFloat, fz, kz});
        uint32_t fw = c.id(); c.fn.op(OPC_BITCAST, {c.tFloat, fw, kw});

        uint32_t finalPos = c.id();
        c.fn.op(OPC_COMPOSITE_CONSTRUCT, {c.tVec4, finalPos, fx, fy, fz, fw});
        c.fn.op(OPC_STORE, {c.realPosVar, finalPos});
    }

    c.fn.op(OPC_RETURN, {});
    c.fn.op(OPC_FUNCTION_END, {});

    std::vector<uint32_t> code;
    code.push_back(0x07230203u);
    code.push_back(0x00010300u);
    code.push_back(0u);
    code.push_back(c.nextId);
    code.push_back(0u);
    auto append = [&](const Sect& s) {
        code.insert(code.end(), s.code.begin(), s.code.end());
    };
    append(c.pre);
    append(c.ep);
    append(c.ann);
    append(c.glob);
    append(c.fn);

    if (c.unsupported)
        LOG("[sm3] %s shader compiled with %u unsupported opcode(s)",
            c.pixel ? "pixel" : "vertex", c.unsupported);

    out.swap(code);
    return true;
}

uint32_t Sm3TexldpSamplerMask(const uint32_t* tokens, uint32_t dwordCount) {
    if (!tokens || dwordCount < 2) return 0;
    if ((tokens[0] & 0xFFFF0000u) != 0xFFFF0000u) return 0;
    uint32_t mask = 0;
    uint32_t i = 1;
    while (i < dwordCount) {
        const uint32_t head = tokens[i];
        if (head == OP_END) break;
        const uint32_t n = Sm3InstructionDwords(head);
        if ((head & 0xFFFFu) == OP_TEX && (head & 0x00010000u) != 0 &&
            n >= 4 && i + n <= dwordCount) {
            for (uint32_t j = 2; j < n; ++j) {
                const uint32_t tok = tokens[i + j];
                if (RegTypeOf(tok) == RT_SAMPLER) {
                    const uint32_t slot = tok & 0x7FFu;
                    if (slot < 16) mask |= (1u << slot);
                }
            }
        }
        i += n;
    }
    return mask;
}
