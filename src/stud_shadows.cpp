#include "stud_shadows.h"

#include "d3d9_internal.h"
#include "resources.h"
#include "vk_context.h"

#include <cmath>
#include <cstring>
#include <vector>

StudShadows g_studShadows;

namespace {

struct alignas(16) ShadowVert {
    float nx, ny;
    float px, py, pz, radius;
    float inv0[4];
    float inv1[4];
    float inv2[4];
    float inv3[4];
    float invW, invH;
};

constexpr uint32_t kSlotBytes = 512u * 6u * sizeof(ShadowVert);

constexpr uint32_t SPV_OP_EXT_INST_IMPORT = 11;
constexpr uint32_t SPV_OP_EXT_INST = 12;
constexpr uint32_t SPV_OP_MEMORY_MODEL = 14;
constexpr uint32_t SPV_OP_ENTRY_POINT = 15;
constexpr uint32_t SPV_OP_EXECUTION_MODE = 16;
constexpr uint32_t SPV_OP_CAPABILITY = 17;
constexpr uint32_t SPV_OP_TYPE_VOID = 19;
constexpr uint32_t SPV_OP_TYPE_FLOAT = 22;
constexpr uint32_t SPV_OP_TYPE_VECTOR = 23;
constexpr uint32_t SPV_OP_TYPE_IMAGE = 25;
constexpr uint32_t SPV_OP_TYPE_SAMPLED_IMAGE = 27;
constexpr uint32_t SPV_OP_TYPE_POINTER = 32;
constexpr uint32_t SPV_OP_TYPE_FUNCTION = 33;
constexpr uint32_t SPV_OP_CONSTANT = 43;
constexpr uint32_t SPV_OP_FUNCTION = 54;
constexpr uint32_t SPV_OP_FUNCTION_END = 56;
constexpr uint32_t SPV_OP_VARIABLE = 59;
constexpr uint32_t SPV_OP_LOAD = 61;
constexpr uint32_t SPV_OP_STORE = 62;
constexpr uint32_t SPV_OP_DECORATE = 71;
constexpr uint32_t SPV_OP_COMPOSITE_CONSTRUCT = 80;
constexpr uint32_t SPV_OP_COMPOSITE_EXTRACT = 81;
constexpr uint32_t SPV_OP_IMAGE_SAMPLE_EXPLICIT_LOD = 88;
constexpr uint32_t SPV_OP_FADD = 129;
constexpr uint32_t SPV_OP_FSUB = 131;
constexpr uint32_t SPV_OP_FMUL = 133;
constexpr uint32_t SPV_OP_FDIV = 136;
constexpr uint32_t SPV_OP_VECTOR_TIMES_SCALAR = 142;
constexpr uint32_t SPV_OP_DOT = 148;
constexpr uint32_t SPV_OP_DPDX = 207;
constexpr uint32_t SPV_OP_DPDY = 208;
constexpr uint32_t SPV_OP_LABEL = 248;
constexpr uint32_t SPV_OP_RETURN = 253;

constexpr uint32_t SC_UNIFORM_CONSTANT = 0;
constexpr uint32_t SC_INPUT = 1;
constexpr uint32_t SC_OUTPUT = 3;

constexpr uint32_t DEC_BUILTIN = 11;
constexpr uint32_t DEC_FLAT = 14;
constexpr uint32_t DEC_LOCATION = 30;
constexpr uint32_t DEC_BINDING = 33;
constexpr uint32_t DEC_DESCRIPTOR_SET = 34;

constexpr uint32_t BI_POSITION = 0;
constexpr uint32_t BI_FRAG_COORD = 15;

constexpr uint32_t GLSL_FABS = 4;
constexpr uint32_t GLSL_INVERSE_SQRT = 32;
constexpr uint32_t GLSL_FMAX = 40;
constexpr uint32_t GLSL_FCLAMP = 43;
constexpr uint32_t GLSL_CROSS = 68;

struct SpvSection {
    std::vector<uint32_t> words;

    void op(uint16_t opcode, std::initializer_list<uint32_t> args) {
        const uint16_t count = (uint16_t)(1u + args.size());
        words.push_back(((uint32_t)count << 16) | opcode);
        for (uint32_t a : args) words.push_back(a);
    }

    void strWords(const char* s) {
        const size_t len = std::strlen(s) + 1;
        const size_t wc = (len + 3) / 4;
        for (size_t i = 0; i < wc; ++i) {
            uint32_t w = 0;
            for (size_t b = 0; b < 4; ++b) {
                const size_t idx = i * 4 + b;
                if (idx < len) w |= ((uint32_t)(uint8_t)s[idx]) << (b * 8);
            }
            words.push_back(w);
        }
    }

    void extImport(uint32_t resId, const char* name) {
        const size_t wc = (std::strlen(name) + 4) / 4;
        const uint16_t count = (uint16_t)(2u + wc);
        words.push_back(((uint32_t)count << 16) | SPV_OP_EXT_INST_IMPORT);
        words.push_back(resId);
        strWords(name);
    }

    void entryPoint(uint32_t execModel, uint32_t fnId, const char* name,
                    std::initializer_list<uint32_t> interfaces) {
        const size_t wc = (std::strlen(name) + 4) / 4;
        const uint16_t count = (uint16_t)(3u + wc + interfaces.size());
        words.push_back(((uint32_t)count << 16) | SPV_OP_ENTRY_POINT);
        words.push_back(execModel);
        words.push_back(fnId);
        strWords(name);
        for (uint32_t i : interfaces) words.push_back(i);
    }
};

uint32_t FloatBits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

std::vector<uint32_t> BuildShadowVs() {
    uint32_t nextId = 1;
    auto id = [&]() { return nextId++; };

    const uint32_t mainId = id();
    const uint32_t tVoid = id(), tFn = id(), tF32 = id(), tVec2 = id(), tVec4 = id();
    const uint32_t pInVec2 = id(), pInVec4 = id(), pOutVec2 = id(), pOutVec4 = id();
    const uint32_t glPos = id();
    const uint32_t inNdc = id(), inStud = id();
    const uint32_t inInv0 = id(), inInv1 = id(), inInv2 = id(), inInv3 = id();
    const uint32_t inInvSize = id();
    const uint32_t outStud = id();
    const uint32_t outInv0 = id(), outInv1 = id(), outInv2 = id(), outInv3 = id();
    const uint32_t outInvSize = id();
    const uint32_t cF0 = id(), cF1 = id();

    SpvSection pre;
    pre.op(SPV_OP_CAPABILITY, {1});
    pre.op(SPV_OP_MEMORY_MODEL, {0, 1});
    pre.entryPoint(0, mainId, "main",
                   {glPos, inNdc, inStud, inInv0, inInv1, inInv2, inInv3,
                    inInvSize, outStud, outInv0, outInv1, outInv2, outInv3,
                    outInvSize});

    SpvSection dec;
    dec.op(SPV_OP_DECORATE, {glPos, DEC_BUILTIN, BI_POSITION});
    dec.op(SPV_OP_DECORATE, {inNdc, DEC_LOCATION, 0});
    dec.op(SPV_OP_DECORATE, {inStud, DEC_LOCATION, 1});
    dec.op(SPV_OP_DECORATE, {inInv0, DEC_LOCATION, 2});
    dec.op(SPV_OP_DECORATE, {inInv1, DEC_LOCATION, 3});
    dec.op(SPV_OP_DECORATE, {inInv2, DEC_LOCATION, 4});
    dec.op(SPV_OP_DECORATE, {inInv3, DEC_LOCATION, 5});
    dec.op(SPV_OP_DECORATE, {inInvSize, DEC_LOCATION, 6});
    dec.op(SPV_OP_DECORATE, {outStud, DEC_LOCATION, 0});
    dec.op(SPV_OP_DECORATE, {outStud, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outInv0, DEC_LOCATION, 1});
    dec.op(SPV_OP_DECORATE, {outInv0, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outInv1, DEC_LOCATION, 2});
    dec.op(SPV_OP_DECORATE, {outInv1, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outInv2, DEC_LOCATION, 3});
    dec.op(SPV_OP_DECORATE, {outInv2, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outInv3, DEC_LOCATION, 4});
    dec.op(SPV_OP_DECORATE, {outInv3, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outInvSize, DEC_LOCATION, 5});
    dec.op(SPV_OP_DECORATE, {outInvSize, DEC_FLAT});

    SpvSection tys;
    tys.op(SPV_OP_TYPE_VOID, {tVoid});
    tys.op(SPV_OP_TYPE_FUNCTION, {tFn, tVoid});
    tys.op(SPV_OP_TYPE_FLOAT, {tF32, 32});
    tys.op(SPV_OP_TYPE_VECTOR, {tVec2, tF32, 2});
    tys.op(SPV_OP_TYPE_VECTOR, {tVec4, tF32, 4});
    tys.op(SPV_OP_TYPE_POINTER, {pInVec2, SC_INPUT, tVec2});
    tys.op(SPV_OP_TYPE_POINTER, {pInVec4, SC_INPUT, tVec4});
    tys.op(SPV_OP_TYPE_POINTER, {pOutVec2, SC_OUTPUT, tVec2});
    tys.op(SPV_OP_TYPE_POINTER, {pOutVec4, SC_OUTPUT, tVec4});
    tys.op(SPV_OP_CONSTANT, {tF32, cF0, FloatBits(0.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cF1, FloatBits(1.0f)});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, glPos, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec2, inNdc, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inStud, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv0, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv1, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv2, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv3, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec2, inInvSize, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outStud, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outInv0, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outInv1, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outInv2, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outInv3, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec2, outInvSize, SC_OUTPUT});

    SpvSection fn;
    fn.op(SPV_OP_FUNCTION, {tVoid, mainId, 0, tFn});
    fn.op(SPV_OP_LABEL, {id()});
    const uint32_t vNdc = id(), nx = id(), ny = id(), pos = id();
    fn.op(SPV_OP_LOAD, {tVec2, vNdc, inNdc});
    fn.op(SPV_OP_COMPOSITE_EXTRACT, {tF32, nx, vNdc, 0});
    fn.op(SPV_OP_COMPOSITE_EXTRACT, {tF32, ny, vNdc, 1});
    fn.op(SPV_OP_COMPOSITE_CONSTRUCT, {tVec4, pos, nx, ny, cF0, cF1});
    fn.op(SPV_OP_STORE, {glPos, pos});

    auto passVec4 = [&](uint32_t src, uint32_t dst) {
        const uint32_t v = id();
        fn.op(SPV_OP_LOAD, {tVec4, v, src});
        fn.op(SPV_OP_STORE, {dst, v});
    };
    passVec4(inStud, outStud);
    passVec4(inInv0, outInv0);
    passVec4(inInv1, outInv1);
    passVec4(inInv2, outInv2);
    passVec4(inInv3, outInv3);
    const uint32_t vSz = id();
    fn.op(SPV_OP_LOAD, {tVec2, vSz, inInvSize});
    fn.op(SPV_OP_STORE, {outInvSize, vSz});
    fn.op(SPV_OP_RETURN, {});
    fn.op(SPV_OP_FUNCTION_END, {});

    std::vector<uint32_t> out = {0x07230203u, 0x00010000u, 0x00080001u, nextId, 0u};
    out.insert(out.end(), pre.words.begin(), pre.words.end());
    out.insert(out.end(), dec.words.begin(), dec.words.end());
    out.insert(out.end(), tys.words.begin(), tys.words.end());
    out.insert(out.end(), fn.words.begin(), fn.words.end());
    return out;
}

std::vector<uint32_t> BuildShadowFs() {
    uint32_t nextId = 1;
    auto id = [&]() { return nextId++; };

    const uint32_t glsl = id(), mainId = id();
    const uint32_t tVoid = id(), tFn = id(), tF32 = id(), tVec2 = id(), tVec3 = id(), tVec4 = id();
    const uint32_t tImg = id(), tSImg = id();
    const uint32_t pInVec2 = id(), pInVec4 = id(), pOutVec4 = id(), pUniSImg = id();
    const uint32_t fragCoord = id();
    const uint32_t inStud = id(), inInv0 = id(), inInv1 = id(), inInv2 = id(), inInv3 = id();
    const uint32_t inInvSize = id();
    const uint32_t outColor = id(), uDepth = id();

    const uint32_t c0 = id(), c1 = id(), c2 = id(), c3 = id();
    const uint32_t cEpsW = id(), cScaleW = id(), cEpsLen2 = id();
    const uint32_t cMinDy = id(), cInvDyRamp = id(), cMaxDy = id(), cInvDyFade = id();
    const uint32_t cRadGrow = id(), cSlopeMin = id(), cInvSlopeRamp = id(), cOpacity = id();

    SpvSection pre;
    pre.op(SPV_OP_CAPABILITY, {1});
    pre.extImport(glsl, "GLSL.std.450");
    pre.op(SPV_OP_MEMORY_MODEL, {0, 1});
    pre.entryPoint(4, mainId, "main",
                   {fragCoord, inStud, inInv0, inInv1, inInv2, inInv3, inInvSize, outColor});
    pre.op(SPV_OP_EXECUTION_MODE, {mainId, 7});

    SpvSection dec;
    dec.op(SPV_OP_DECORATE, {fragCoord, DEC_BUILTIN, BI_FRAG_COORD});
    dec.op(SPV_OP_DECORATE, {inStud, DEC_LOCATION, 0});
    dec.op(SPV_OP_DECORATE, {inStud, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {inInv0, DEC_LOCATION, 1});
    dec.op(SPV_OP_DECORATE, {inInv0, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {inInv1, DEC_LOCATION, 2});
    dec.op(SPV_OP_DECORATE, {inInv1, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {inInv2, DEC_LOCATION, 3});
    dec.op(SPV_OP_DECORATE, {inInv2, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {inInv3, DEC_LOCATION, 4});
    dec.op(SPV_OP_DECORATE, {inInv3, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {inInvSize, DEC_LOCATION, 5});
    dec.op(SPV_OP_DECORATE, {inInvSize, DEC_FLAT});
    dec.op(SPV_OP_DECORATE, {outColor, DEC_LOCATION, 0});
    dec.op(SPV_OP_DECORATE, {uDepth, DEC_DESCRIPTOR_SET, 0});
    dec.op(SPV_OP_DECORATE, {uDepth, DEC_BINDING, 0});

    SpvSection tys;
    tys.op(SPV_OP_TYPE_VOID, {tVoid});
    tys.op(SPV_OP_TYPE_FUNCTION, {tFn, tVoid});
    tys.op(SPV_OP_TYPE_FLOAT, {tF32, 32});
    tys.op(SPV_OP_TYPE_VECTOR, {tVec2, tF32, 2});
    tys.op(SPV_OP_TYPE_VECTOR, {tVec3, tF32, 3});
    tys.op(SPV_OP_TYPE_VECTOR, {tVec4, tF32, 4});
    tys.op(SPV_OP_TYPE_IMAGE, {tImg, tF32, 1, 0, 0, 0, 1, 0});
    tys.op(SPV_OP_TYPE_SAMPLED_IMAGE, {tSImg, tImg});
    tys.op(SPV_OP_TYPE_POINTER, {pInVec2, SC_INPUT, tVec2});
    tys.op(SPV_OP_TYPE_POINTER, {pInVec4, SC_INPUT, tVec4});
    tys.op(SPV_OP_TYPE_POINTER, {pOutVec4, SC_OUTPUT, tVec4});
    tys.op(SPV_OP_TYPE_POINTER, {pUniSImg, SC_UNIFORM_CONSTANT, tSImg});

    tys.op(SPV_OP_CONSTANT, {tF32, c0, FloatBits(0.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, c1, FloatBits(1.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, c2, FloatBits(2.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, c3, FloatBits(3.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cEpsW, FloatBits(1e-6f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cScaleW, FloatBits(1e6f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cEpsLen2, FloatBits(1e-12f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cMinDy, FloatBits(0.005f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cInvDyRamp, FloatBits(40.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cMaxDy, FloatBits(0.55f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cInvDyFade, FloatBits(2.8f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cRadGrow, FloatBits(0.14f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cSlopeMin, FloatBits(0.55f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cInvSlopeRamp, FloatBits(4.0f)});
    tys.op(SPV_OP_CONSTANT, {tF32, cOpacity, FloatBits(0.78f)});

    tys.op(SPV_OP_VARIABLE, {pInVec4, fragCoord, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inStud, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv0, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv1, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv2, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec4, inInv3, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pInVec2, inInvSize, SC_INPUT});
    tys.op(SPV_OP_VARIABLE, {pOutVec4, outColor, SC_OUTPUT});
    tys.op(SPV_OP_VARIABLE, {pUniSImg, uDepth, SC_UNIFORM_CONSTANT});

    SpvSection fn;
    fn.op(SPV_OP_FUNCTION, {tVoid, mainId, 0, tFn});
    fn.op(SPV_OP_LABEL, {id()});

    auto ext = [&](uint32_t rtype, uint32_t op, std::initializer_list<uint32_t> args) {
        const uint32_t r = id();
        fn.words.push_back(((uint32_t)(5u + args.size()) << 16) | SPV_OP_EXT_INST);
        fn.words.push_back(rtype);
        fn.words.push_back(r);
        fn.words.push_back(glsl);
        fn.words.push_back(op);
        for (uint32_t a : args) fn.words.push_back(a);
        return r;
    };
    auto bin = [&](uint16_t op, uint32_t rt, uint32_t a, uint32_t b) {
        const uint32_t r = id();
        fn.op(op, {rt, r, a, b});
        return r;
    };
    auto extr = [&](uint32_t rt, uint32_t v, uint32_t idx) {
        const uint32_t r = id();
        fn.op(SPV_OP_COMPOSITE_EXTRACT, {rt, r, v, idx});
        return r;
    };

    const uint32_t vFc = id(), vInvSz = id(), vTex = id(), vStud = id();
    const uint32_t vI0 = id(), vI1 = id(), vI2 = id(), vI3 = id();
    fn.op(SPV_OP_LOAD, {tVec4, vFc, fragCoord});
    fn.op(SPV_OP_LOAD, {tVec2, vInvSz, inInvSize});
    fn.op(SPV_OP_LOAD, {tSImg, vTex, uDepth});
    fn.op(SPV_OP_LOAD, {tVec4, vStud, inStud});
    fn.op(SPV_OP_LOAD, {tVec4, vI0, inInv0});
    fn.op(SPV_OP_LOAD, {tVec4, vI1, inInv1});
    fn.op(SPV_OP_LOAD, {tVec4, vI2, inInv2});
    fn.op(SPV_OP_LOAD, {tVec4, vI3, inInv3});

    const uint32_t fcX = extr(tF32, vFc, 0);
    const uint32_t fcY = extr(tF32, vFc, 1);
    const uint32_t fcXY = id();
    fn.op(SPV_OP_COMPOSITE_CONSTRUCT, {tVec2, fcXY, fcX, fcY});
    const uint32_t uv = bin(SPV_OP_FMUL, tVec2, fcXY, vInvSz);

    const uint32_t sample = id();
    fn.op(SPV_OP_IMAGE_SAMPLE_EXPLICIT_LOD, {tVec4, sample, vTex, uv, 0x2u, c0});
    const uint32_t z = extr(tF32, sample, 0);

    const uint32_t uvX = extr(tF32, uv, 0);
    const uint32_t uvY = extr(tF32, uv, 1);
    const uint32_t ndcX = bin(SPV_OP_FSUB, tF32, bin(SPV_OP_FMUL, tF32, uvX, c2), c1);
    const uint32_t ndcY = bin(SPV_OP_FSUB, tF32, c1, bin(SPV_OP_FMUL, tF32, uvY, c2));

    const uint32_t t0 = bin(SPV_OP_VECTOR_TIMES_SCALAR, tVec4, vI0, ndcX);
    const uint32_t t1 = bin(SPV_OP_VECTOR_TIMES_SCALAR, tVec4, vI1, ndcY);
    const uint32_t t2 = bin(SPV_OP_VECTOR_TIMES_SCALAR, tVec4, vI2, z);
    const uint32_t wh = bin(SPV_OP_FADD, tVec4, bin(SPV_OP_FADD, tVec4, t0, t1),
                            bin(SPV_OP_FADD, tVec4, t2, vI3));

    const uint32_t whX = extr(tF32, wh, 0);
    const uint32_t whY = extr(tF32, wh, 1);
    const uint32_t whZ = extr(tF32, wh, 2);
    const uint32_t whW = extr(tF32, wh, 3);
    const uint32_t whXYZ = id();
    fn.op(SPV_OP_COMPOSITE_CONSTRUCT, {tVec3, whXYZ, whX, whY, whZ});

    const uint32_t absW = ext(tF32, GLSL_FABS, {whW});
    const uint32_t safeW = ext(tF32, GLSL_FMAX, {absW, cEpsW});
    const uint32_t invW = bin(SPV_OP_FDIV, tF32, c1, safeW);
    const uint32_t W = bin(SPV_OP_VECTOR_TIMES_SCALAR, tVec3, whXYZ, invW);
    const uint32_t validW =
        ext(tF32, GLSL_FCLAMP,
            {bin(SPV_OP_FMUL, tF32, bin(SPV_OP_FSUB, tF32, whW, cEpsW), cScaleW), c0, c1});

    const uint32_t dWdx = id(), dWdy = id();
    fn.op(SPV_OP_DPDX, {tVec3, dWdx, W});
    fn.op(SPV_OP_DPDY, {tVec3, dWdy, W});
    const uint32_t cr = ext(tVec3, GLSL_CROSS, {dWdx, dWdy});
    const uint32_t cr2 = bin(SPV_OP_DOT, tF32, cr, cr);
    const uint32_t safeCr2 = ext(tF32, GLSL_FMAX, {cr2, cEpsLen2});
    const uint32_t invLen = ext(tF32, GLSL_INVERSE_SQRT, {safeCr2});
    const uint32_t absCy = ext(tF32, GLSL_FABS, {extr(tF32, cr, 1)});
    const uint32_t upCos = bin(SPV_OP_FMUL, tF32, absCy, invLen);
    const uint32_t slopeFade = ext(
        tF32, GLSL_FCLAMP,
        {bin(SPV_OP_FMUL, tF32, bin(SPV_OP_FSUB, tF32, upCos, cSlopeMin), cInvSlopeRamp),
         c0, c1});

    const uint32_t wx = extr(tF32, W, 0);
    const uint32_t wy = extr(tF32, W, 1);
    const uint32_t wz = extr(tF32, W, 2);
    const uint32_t sx = extr(tF32, vStud, 0);
    const uint32_t sy = extr(tF32, vStud, 1);
    const uint32_t sz = extr(tF32, vStud, 2);
    const uint32_t sRad = extr(tF32, vStud, 3);

    const uint32_t dy = bin(SPV_OP_FSUB, tF32, sy, wy);
    const uint32_t hIn = ext(
        tF32, GLSL_FCLAMP,
        {bin(SPV_OP_FMUL, tF32, bin(SPV_OP_FSUB, tF32, dy, cMinDy), cInvDyRamp), c0, c1});
    const uint32_t hOut = ext(
        tF32, GLSL_FCLAMP,
        {bin(SPV_OP_FMUL, tF32, bin(SPV_OP_FSUB, tF32, cMaxDy, dy), cInvDyFade), c0, c1});
    const uint32_t heightFade = bin(SPV_OP_FMUL, tF32, hIn, hOut);

    const uint32_t dx = bin(SPV_OP_FSUB, tF32, wx, sx);
    const uint32_t dz = bin(SPV_OP_FSUB, tF32, wz, sz);
    const uint32_t r2 = bin(SPV_OP_FADD, tF32, bin(SPV_OP_FMUL, tF32, dx, dx),
                            bin(SPV_OP_FMUL, tF32, dz, dz));
    const uint32_t clampedDy = ext(tF32, GLSL_FCLAMP, {dy, c0, cMaxDy});
    const uint32_t effR =
        bin(SPV_OP_FADD, tF32, sRad, bin(SPV_OP_FMUL, tF32, clampedDy, cRadGrow));
    const uint32_t effR2 = bin(SPV_OP_FMUL, tF32, effR, effR);
    const uint32_t u2 = bin(SPV_OP_FDIV, tF32, r2, effR2);
    const uint32_t m = ext(tF32, GLSL_FCLAMP, {bin(SPV_OP_FSUB, tF32, c1, u2), c0, c1});
    const uint32_t m2 = bin(SPV_OP_FMUL, tF32, m, m);
    const uint32_t s32 = bin(SPV_OP_FSUB, tF32, c3, bin(SPV_OP_FMUL, tF32, c2, m));
    const uint32_t radial = bin(SPV_OP_FMUL, tF32, m2, s32);

    uint32_t alpha = bin(SPV_OP_FMUL, tF32, radial, heightFade);
    alpha = bin(SPV_OP_FMUL, tF32, alpha, slopeFade);
    alpha = bin(SPV_OP_FMUL, tF32, alpha, validW);
    alpha = bin(SPV_OP_FMUL, tF32, alpha, cOpacity);

    const uint32_t res = id();
    fn.op(SPV_OP_COMPOSITE_CONSTRUCT, {tVec4, res, c0, c0, c0, alpha});
    fn.op(SPV_OP_STORE, {outColor, res});
    fn.op(SPV_OP_RETURN, {});
    fn.op(SPV_OP_FUNCTION_END, {});

    std::vector<uint32_t> out = {0x07230203u, 0x00010000u, 0x00080001u, nextId, 0u};
    out.insert(out.end(), pre.words.begin(), pre.words.end());
    out.insert(out.end(), dec.words.begin(), dec.words.end());
    out.insert(out.end(), tys.words.begin(), tys.words.end());
    out.insert(out.end(), fn.words.begin(), fn.words.end());
    return out;
}

bool InvertMat4(const float* m, float* invOut) {
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    const float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::fabs(det) < 1e-12f) return false;
    const float invDet = 1.0f / det;
    for (int i = 0; i < 16; ++i) invOut[i] = inv[i] * invDet;
    return true;
}

}

bool StudShadows::Init() {
    if (initialized_) return true;
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (!dev || !alloc) return false;

    const std::vector<uint32_t> vsWords = BuildShadowVs();
    const std::vector<uint32_t> fsWords = BuildShadowFs();
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = vsWords.size() * sizeof(uint32_t);
    smci.pCode = vsWords.data();
    if (vkCreateShaderModule(dev, &smci, nullptr, &vs_) != VK_SUCCESS) return false;
    smci.codeSize = fsWords.size() * sizeof(uint32_t);
    smci.pCode = fsWords.data();
    if (vkCreateShaderModule(dev, &smci, nullptr, &fs_) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 1;
    slci.pBindings = &bind;
    if (vkCreateDescriptorSetLayout(dev, &slci, nullptr, &setLayout_) != VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipeLayout_) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize poolSz{};
    poolSz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSz.descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 2;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &descPool_) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayout layouts[2] = {setLayout_, setLayout_};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = 2;
    dsai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(dev, &dsai, descSets_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &depthSampler_) != VK_SUCCESS) return false;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = (VkDeviceSize)kSlotBytes * 2u;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo baci{};
    baci.usage = VMA_MEMORY_USAGE_AUTO;
    baci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo bai{};
    if (vmaCreateBuffer(alloc, &bci, &baci, &vb_, &vbAlloc_, &bai) != VK_SUCCESS)
        return false;
    vbMapped_ = bai.pMappedData;

    boundDepthViews_[0] = VK_NULL_HANDLE;
    boundDepthViews_[1] = VK_NULL_HANDLE;
    initialized_ = true;
    return true;
}

void StudShadows::Shutdown() {
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (!dev) return;

    if (pipeline_) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    pipeFormat_ = VK_FORMAT_UNDEFINED;
    if (vb_) {
        vmaDestroyBuffer(alloc, vb_, vbAlloc_);
        vb_ = VK_NULL_HANDLE;
        vbAlloc_ = VK_NULL_HANDLE;
        vbMapped_ = nullptr;
    }
    if (depthSampler_) {
        vkDestroySampler(dev, depthSampler_, nullptr);
        depthSampler_ = VK_NULL_HANDLE;
    }
    if (descPool_) {
        vkDestroyDescriptorPool(dev, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        descSets_[0] = VK_NULL_HANDLE;
        descSets_[1] = VK_NULL_HANDLE;
    }
    if (pipeLayout_) {
        vkDestroyPipelineLayout(dev, pipeLayout_, nullptr);
        pipeLayout_ = VK_NULL_HANDLE;
    }
    if (setLayout_) {
        vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
        setLayout_ = VK_NULL_HANDLE;
    }
    if (fs_) {
        vkDestroyShaderModule(dev, fs_, nullptr);
        fs_ = VK_NULL_HANDLE;
    }
    if (vs_) {
        vkDestroyShaderModule(dev, vs_, nullptr);
        vs_ = VK_NULL_HANDLE;
    }
    boundDepthViews_[0] = VK_NULL_HANDLE;
    boundDepthViews_[1] = VK_NULL_HANDLE;
    initialized_ = false;
}

void StudShadows::BeginFrame() {
    studCount_ = 0;
    gbufColor_ = nullptr;
    gbufDepth_ = nullptr;
    sceneColor_ = nullptr;
    flushedThisFrame_ = false;
}

void StudShadows::CollectStud(const float* vsConst, FakeSurface* gbufColor,
                              FakeSurface* gbufDepth) {
    if (!initialized_ || !vsConst || !gbufDepth || gbufDepth->isBackbuffer) return;
    if (studCount_ >= kMaxStuds) return;

    StudItem& item = studs_[studCount_++];
    item.px = vsConst[51u * 4u + 0u];
    item.py = vsConst[51u * 4u + 1u];
    item.pz = vsConst[51u * 4u + 2u];
    std::memcpy(item.viewProj, vsConst, sizeof(item.viewProj));

    gbufColor_ = gbufColor;
    gbufDepth_ = gbufDepth;
    flushedThisFrame_ = false;
}

void StudShadows::TrackTargetDraw(FakeSurface* color0, FakeSurface* depth) {
    if (!initialized_ || studCount_ == 0 || flushedThisFrame_ || !gbufDepth_) return;
    if (!color0 || color0->isBackbuffer || color0 == gbufColor_) return;
    if (!depth || depth->image != gbufDepth_->image) return;
    if (color0->width == gbufDepth_->width && color0->height == gbufDepth_->height) {
        sceneColor_ = color0;
    }
}

bool StudShadows::FlushScene() {
    if (!NeedsFlush()) return false;
    return Render();
}

bool StudShadows::FlushBeforeDepthClear(FakeSurface* depth) {
    if (!initialized_ || studCount_ == 0 || flushedThisFrame_ || !sceneColor_ ||
        !gbufDepth_) {
        return false;
    }
    if (!depth || depth->image != gbufDepth_->image) return false;
    const bool drawn = Render();
    studCount_ = 0;
    gbufColor_ = nullptr;
    gbufDepth_ = nullptr;
    sceneColor_ = nullptr;
    flushedThisFrame_ = false;
    return drawn;
}

bool StudShadows::FlushIfReading(FakeSurface* texSurf) {
    if (!initialized_ || studCount_ == 0 || flushedThisFrame_ || !sceneColor_ ||
        !gbufDepth_ || !texSurf) {
        return false;
    }
    if (texSurf->image != sceneColor_->image) return false;
    return Render();
}

bool StudShadows::FlushEndOfFrame() {
    if (!initialized_ || studCount_ == 0 || flushedThisFrame_ || !sceneColor_ ||
        !gbufDepth_) {
        return false;
    }
    return Render();
}

bool StudShadows::EnsurePipeline(VkFormat colorFormat) {
    if (pipeline_ && pipeFormat_ == colorFormat) return true;
    VkDevice dev = g_backend.Device();
    if (!dev || !vs_ || !fs_) return false;
    if (pipeline_) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vib{0, sizeof(ShadowVert), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[7]{
        {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 24},
        {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 40},
        {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 56},
        {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 72},
        {6, 0, VK_FORMAT_R32G32_SFLOAT, 88},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vib;
    vi.vertexAttributeDescriptionCount = 7;
    vi.pVertexAttributeDescriptions = via;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = 0xF;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = 2;
    dynInfo.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.pNext = &rci;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dynInfo;
    pci.layout = pipeLayout_;

    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_) != VK_SUCCESS) {
        pipeline_ = VK_NULL_HANDLE;
        return false;
    }
    pipeFormat_ = colorFormat;
    return true;
}

bool StudShadows::EnsureDepthDescriptor(VkImageView depthView, uint32_t frameSlot) {
    const uint32_t s = frameSlot & 1u;
    if (boundDepthViews_[s] == depthView) return true;
    VkDevice dev = g_backend.Device();
    if (!dev || !descSets_[s] || !depthView || !depthSampler_) return false;

    VkDescriptorImageInfo ii{};
    ii.sampler = depthSampler_;
    ii.imageView = depthView;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = descSets_[s];
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    boundDepthViews_[s] = depthView;
    return true;
}

uint32_t StudShadows::BuildQuads(uint32_t frameSlot, uint32_t width, uint32_t height) {
    if (!vbMapped_ || studCount_ == 0 || !width || !height) return 0;

    const float invW = 1.0f / (float)width;
    const float invH = 1.0f / (float)height;
    constexpr float kShadowRadius = 0.115f;
    constexpr float kBoxXZ = 0.24f;
    constexpr float kBoxDownY = 0.60f;
    constexpr float kBoxUpY = 0.05f;

    ShadowVert* dst = (ShadowVert*)((uint8_t*)vbMapped_ + (size_t)(frameSlot & 1u) * kSlotBytes);
    ShadowVert* begin = dst;

    for (uint32_t i = 0; i < studCount_; ++i) {
        const StudItem& st = studs_[i];
        float invM[16]{};
        if (!InvertMat4(st.viewProj, invM)) continue;

        float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
        int validCorners = 0;
        bool nearPlaneHit = false;
        const float* m = st.viewProj;

        for (int cx = 0; cx < 2; ++cx) {
            const float wx = st.px + (cx ? kBoxXZ : -kBoxXZ);
            for (int cy = 0; cy < 2; ++cy) {
                const float wy = st.py + (cy ? kBoxUpY : -kBoxDownY);
                for (int cz = 0; cz < 2; ++cz) {
                    const float wz = st.pz + (cz ? kBoxXZ : -kBoxXZ);
                    const float cw = wx * m[3] + wy * m[7] + wz * m[11] + m[15];
                    if (cw <= 0.05f) {
                        nearPlaneHit = true;
                        continue;
                    }
                    const float nx = (wx * m[0] + wy * m[4] + wz * m[8] + m[12]) / cw;
                    const float ny = (wx * m[1] + wy * m[5] + wz * m[9] + m[13]) / cw;
                    if (nx < minX) minX = nx;
                    if (nx > maxX) maxX = nx;
                    if (ny < minY) minY = ny;
                    if (ny > maxY) maxY = ny;
                    ++validCorners;
                }
            }
        }

        if (validCorners == 0) continue;
        if (nearPlaneHit) {
            minX = -1.0f;
            maxX = 1.0f;
            minY = -1.0f;
            maxY = 1.0f;
        } else {
            if (minX < -1.0f) minX = -1.0f;
            if (maxX > 1.0f) maxX = 1.0f;
            if (minY < -1.0f) minY = -1.0f;
            if (maxY > 1.0f) maxY = 1.0f;
        }
        if (minX >= maxX || minY >= maxY) continue;

        ShadowVert base{};
        base.px = st.px;
        base.py = st.py;
        base.pz = st.pz;
        base.radius = kShadowRadius;
        std::memcpy(base.inv0, invM, 16);
        std::memcpy(base.inv1, invM + 4, 16);
        std::memcpy(base.inv2, invM + 8, 16);
        std::memcpy(base.inv3, invM + 12, 16);
        base.invW = invW;
        base.invH = invH;

        const float xs[6] = {minX, maxX, maxX, minX, maxX, minX};
        const float ys[6] = {minY, minY, maxY, minY, maxY, maxY};
        for (int v = 0; v < 6; ++v) {
            *dst = base;
            dst->nx = xs[v];
            dst->ny = ys[v];
            ++dst;
        }
    }

    return (uint32_t)(dst - begin);
}

bool StudShadows::Render() {
    if (!initialized_ || !g_backend.InRecording() || !sceneColor_ || !gbufDepth_) {
        flushedThisFrame_ = true;
        return false;
    }
    if (!sceneColor_->image || !sceneColor_->view || !gbufDepth_->image || !gbufDepth_->view) {
        flushedThisFrame_ = true;
        return false;
    }
    flushedThisFrame_ = true;

    const VkFormat colorFmt = ResourceTable::ToVkFormat(sceneColor_->format);
    const VkFormat depthFmt = ResourceTable::ToVkDepthFormat(gbufDepth_->format, nullptr);
    const uint32_t frameSlot = g_backend.FrameSlot() & 1u;

    if (!EnsurePipeline(colorFmt)) return false;
    if (!EnsureDepthDescriptor(gbufDepth_->view, frameSlot)) return false;

    const uint32_t vertCount = BuildQuads(frameSlot, sceneColor_->width, sceneColor_->height);
    if (!vertCount) return false;

    g_backend.EndRendering();

    VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depthFmt == VK_FORMAT_D24_UNORM_S8_UINT ||
        depthFmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    g_backend.TransitionImage(gbufDepth_->image, depthAspect,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    g_backend.TransitionImage(sceneColor_->image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = sceneColor_->view;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    const VkExtent2D ext{sceneColor_->width, sceneColor_->height};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, ext};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;

    VkCommandBuffer cmd = g_backend.Cmd();
    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout_, 0, 1,
                            &descSets_[frameSlot], 0, nullptr);
    const VkDeviceSize vbOff = (VkDeviceSize)frameSlot * kSlotBytes;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb_, &vbOff);
    VkViewport vp{0.0f, (float)ext.height, (float)ext.width, -(float)ext.height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, vertCount, 1, 0, 0);
    vkCmdEndRendering(cmd);
    return true;
}
