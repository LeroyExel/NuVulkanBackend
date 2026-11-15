#include "post_fx.h"

#include "resources.h"
#include "vk_context.h"

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

PostFx g_postFx;

namespace {

struct PostPush {
    float invW;
    float invH;
    float tanFovX;
    float tanFovY;
    float zNear;
    float zFar;
    float aspect;
    float param;
};

constexpr uint32_t kNumDescSets = 16;

uint32_t FBits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

struct SpvSect {
    std::vector<uint32_t> w;

    void op(uint16_t opcode, std::initializer_list<uint32_t> args) {
        const uint16_t count = (uint16_t)(1u + args.size());
        w.push_back(((uint32_t)count << 16) | opcode);
        for (uint32_t a : args) w.push_back(a);
    }

    void opv(uint16_t opcode, const std::vector<uint32_t>& args) {
        const uint16_t count = (uint16_t)(1u + args.size());
        w.push_back(((uint32_t)count << 16) | opcode);
        for (uint32_t a : args) w.push_back(a);
    }

    void strWords(const char* s) {
        const size_t len = std::strlen(s) + 1;
        const size_t wc = (len + 3) / 4;
        for (size_t i = 0; i < wc; ++i) {
            uint32_t v = 0;
            for (size_t b = 0; b < 4; ++b) {
                const size_t idx = i * 4 + b;
                if (idx < len) v |= ((uint32_t)(uint8_t)s[idx]) << (b * 8);
            }
            w.push_back(v);
        }
    }

    void extImport(uint32_t resId, const char* name) {
        const size_t wc = (std::strlen(name) + 4) / 4;
        const uint16_t count = (uint16_t)(2u + wc);
        w.push_back(((uint32_t)count << 16) | 11u);
        w.push_back(resId);
        strWords(name);
    }

    void entryPoint(uint32_t execModel, uint32_t fnId, const char* name,
                    const std::vector<uint32_t>& ifaces) {
        const size_t wc = (std::strlen(name) + 4) / 4;
        const uint16_t count = (uint16_t)(3u + wc + ifaces.size());
        w.push_back(((uint32_t)count << 16) | 15u);
        w.push_back(execModel);
        w.push_back(fnId);
        strWords(name);
        for (uint32_t i : ifaces) w.push_back(i);
    }
};

std::vector<uint32_t> BuildFullscreenVs() {
    uint32_t nid = 1;
    auto id = [&]() { return nid++; };

    const uint32_t mainId = id();
    const uint32_t tVoid = id(), tFn = id(), tS32 = id(), tF32 = id(), tVec4 = id();
    const uint32_t pInS32 = id(), pOutVec4 = id();
    const uint32_t glPos = id(), inVid = id();
    const uint32_t c1s = id(), c2s = id(), c0f = id(), c1f = id();

    SpvSect pre, dec, tys, fn;
    pre.op(17, {1});
    pre.op(14, {0, 1});
    pre.entryPoint(0, mainId, "main", {glPos, inVid});

    dec.op(71, {glPos, 11, 0});
    dec.op(71, {inVid, 11, 42});

    tys.op(19, {tVoid});
    tys.op(33, {tFn, tVoid});
    tys.op(21, {tS32, 32, 1});
    tys.op(22, {tF32, 32});
    tys.op(23, {tVec4, tF32, 4});
    tys.op(32, {pInS32, 1, tS32});
    tys.op(32, {pOutVec4, 3, tVec4});
    tys.op(43, {tS32, c1s, 1u});
    tys.op(43, {tS32, c2s, 2u});
    tys.op(43, {tF32, c0f, FBits(0.0f)});
    tys.op(43, {tF32, c1f, FBits(1.0f)});
    tys.op(59, {pOutVec4, glPos, 3});
    tys.op(59, {pInS32, inVid, 1});

    fn.op(54, {tVoid, mainId, 0, tFn});
    fn.op(248, {id()});
    const uint32_t vid = id(), b0 = id(), sx = id(), fx = id(), px = id();
    const uint32_t b1 = id(), sy = id(), fy = id(), py = id(), pos = id();
    fn.op(61, {tS32, vid, inVid});
    fn.op(199, {tS32, b0, vid, c1s});
    fn.op(196, {tS32, sx, b0, c2s});
    fn.op(111, {tF32, fx, sx});
    fn.op(131, {tF32, px, fx, c1f});
    fn.op(199, {tS32, b1, vid, c2s});
    fn.op(196, {tS32, sy, b1, c1s});
    fn.op(111, {tF32, fy, sy});
    fn.op(131, {tF32, py, fy, c1f});
    fn.op(80, {tVec4, pos, px, py, c0f, c1f});
    fn.op(62, {glPos, pos});
    fn.op(253, {});
    fn.op(56, {});

    std::vector<uint32_t> out = {0x07230203u, 0x00010300u, 0x00080001u, nid, 0u};
    out.insert(out.end(), pre.w.begin(), pre.w.end());
    out.insert(out.end(), dec.w.begin(), dec.w.end());
    out.insert(out.end(), tys.w.begin(), tys.w.end());
    out.insert(out.end(), fn.w.begin(), fn.w.end());
    return out;
}

struct SpvFrag {
    uint32_t nid = 1;
    uint32_t glsl = 0;
    uint32_t mainId = 0;

    uint32_t tVoid = 0, tFn = 0, tBool = 0, tS32 = 0, tF32 = 0;
    uint32_t tVec2 = 0, tVec3 = 0, tVec4 = 0;
    uint32_t tImg = 0, tSImg = 0;
    uint32_t pInV4 = 0, pOutV4 = 0, pUniSImg = 0;
    uint32_t pFnF32 = 0, pFnS32 = 0, pFnV4 = 0;
    uint32_t pPushStruct = 0, pPushV4 = 0, tPushStruct = 0;

    uint32_t vFragCoord = 0, vOutColor = 0, vPush = 0;
    uint32_t vSamplers[3]{};

    uint32_t push0 = 0, push1 = 0;
    uint32_t fc = 0, fcXy = 0, invSz = 0, uv = 0;

    std::unordered_map<uint32_t, uint32_t> fConsts;
    std::unordered_map<int32_t, uint32_t> sConsts;

    SpvSect dec, tys, fnVars, fnBody;

    uint32_t id() { return nid++; }

    void init(uint32_t samplerCount) {
        glsl = id();
        mainId = id();
        tVoid = id();
        tFn = id();
        tBool = id();
        tS32 = id();
        tF32 = id();
        tVec2 = id();
        tVec3 = id();
        tVec4 = id();
        tImg = id();
        tSImg = id();
        pInV4 = id();
        pOutV4 = id();
        pUniSImg = id();
        pFnF32 = id();
        pFnS32 = id();
        pFnV4 = id();
        tPushStruct = id();
        pPushStruct = id();
        pPushV4 = id();

        vFragCoord = id();
        vOutColor = id();
        vPush = id();

        dec.op(71, {vFragCoord, 11, 15});
        dec.op(71, {vOutColor, 30, 0});
        dec.op(71, {tPushStruct, 2});
        dec.op(72, {tPushStruct, 0, 35, 0});
        dec.op(72, {tPushStruct, 1, 35, 16});

        tys.op(19, {tVoid});
        tys.op(33, {tFn, tVoid});
        tys.op(20, {tBool});
        tys.op(21, {tS32, 32, 1});
        tys.op(22, {tF32, 32});
        tys.op(23, {tVec2, tF32, 2});
        tys.op(23, {tVec3, tF32, 3});
        tys.op(23, {tVec4, tF32, 4});
        tys.op(25, {tImg, tF32, 1, 0, 0, 0, 1, 0});
        tys.op(27, {tSImg, tImg});
        tys.op(32, {pInV4, 1, tVec4});
        tys.op(32, {pOutV4, 3, tVec4});
        tys.op(32, {pUniSImg, 0, tSImg});
        tys.op(32, {pFnF32, 7, tF32});
        tys.op(32, {pFnS32, 7, tS32});
        tys.op(32, {pFnV4, 7, tVec4});
        tys.op(30, {tPushStruct, tVec4, tVec4});
        tys.op(32, {pPushStruct, 9, tPushStruct});
        tys.op(32, {pPushV4, 9, tVec4});

        tys.op(59, {pInV4, vFragCoord, 1});
        tys.op(59, {pOutV4, vOutColor, 3});
        tys.op(59, {pPushStruct, vPush, 9});

        for (uint32_t i = 0; i < samplerCount && i < 3; ++i) {
            vSamplers[i] = id();
            dec.op(71, {vSamplers[i], 34, 0});
            dec.op(71, {vSamplers[i], 33, i});
            tys.op(59, {pUniSImg, vSamplers[i], 0});
        }

        const uint32_t ac0 = id(), ac1 = id();
        push0 = id();
        push1 = id();
        fnBody.op(65, {pPushV4, ac0, vPush, sc(0)});
        fnBody.op(61, {tVec4, push0, ac0});
        fnBody.op(65, {pPushV4, ac1, vPush, sc(1)});
        fnBody.op(61, {tVec4, push1, ac1});

        fc = id();
        fnBody.op(61, {tVec4, fc, vFragCoord});
        fcXy = v2(extr(fc, 0), extr(fc, 1));
        invSz = v2(extr(push0, 0), extr(push0, 1));
        uv = mul(tVec2, fcXy, invSz);
    }

    uint32_t fcConst(float v) {
        const uint32_t bits = FBits(v);
        auto it = fConsts.find(bits);
        if (it != fConsts.end()) return it->second;
        const uint32_t r = id();
        tys.op(43, {tF32, r, bits});
        fConsts[bits] = r;
        return r;
    }

    uint32_t sc(int32_t v) {
        auto it = sConsts.find(v);
        if (it != sConsts.end()) return it->second;
        const uint32_t r = id();
        tys.op(43, {tS32, r, (uint32_t)v});
        sConsts[v] = r;
        return r;
    }

    uint32_t v2(uint32_t x, uint32_t y) {
        const uint32_t r = id();
        fnBody.op(80, {tVec2, r, x, y});
        return r;
    }

    uint32_t v3(uint32_t x, uint32_t y, uint32_t z) {
        const uint32_t r = id();
        fnBody.op(80, {tVec3, r, x, y, z});
        return r;
    }

    uint32_t v4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        const uint32_t r = id();
        fnBody.op(80, {tVec4, r, x, y, z, w});
        return r;
    }

    uint32_t v3f(float x, float y, float z) {
        return v3(fcConst(x), fcConst(y), fcConst(z));
    }

    uint32_t extr(uint32_t v, uint32_t idx) {
        const uint32_t r = id();
        fnBody.op(81, {tF32, r, v, idx});
        return r;
    }

    uint32_t xyz(uint32_t v) { return v3(extr(v, 0), extr(v, 1), extr(v, 2)); }

    uint32_t splat3(uint32_t s) { return v3(s, s, s); }

    uint32_t bin(uint16_t op, uint32_t ty, uint32_t a, uint32_t b) {
        const uint32_t r = id();
        fnBody.op(op, {ty, r, a, b});
        return r;
    }

    uint32_t add(uint32_t ty, uint32_t a, uint32_t b) { return bin(129, ty, a, b); }
    uint32_t sub(uint32_t ty, uint32_t a, uint32_t b) { return bin(131, ty, a, b); }
    uint32_t mul(uint32_t ty, uint32_t a, uint32_t b) { return bin(133, ty, a, b); }
    uint32_t div(uint32_t ty, uint32_t a, uint32_t b) { return bin(136, ty, a, b); }
    uint32_t neg(uint32_t ty, uint32_t a) {
        const uint32_t r = id();
        fnBody.op(127, {ty, r, a});
        return r;
    }
    uint32_t vscale(uint32_t ty, uint32_t v, uint32_t s) { return bin(142, ty, v, s); }
    uint32_t dot(uint32_t a, uint32_t b) { return bin(148, tF32, a, b); }

    uint32_t ext(uint32_t ty, uint32_t eop, std::initializer_list<uint32_t> args) {
        const uint32_t r = id();
        std::vector<uint32_t> a = {ty, r, glsl, eop};
        for (uint32_t x : args) a.push_back(x);
        fnBody.opv(12, a);
        return r;
    }

    uint32_t fabs(uint32_t ty, uint32_t a) { return ext(ty, 4, {a}); }
    uint32_t fsign(uint32_t ty, uint32_t a) { return ext(ty, 6, {a}); }
    uint32_t floor(uint32_t ty, uint32_t a) { return ext(ty, 8, {a}); }
    uint32_t fract(uint32_t ty, uint32_t a) { return ext(ty, 10, {a}); }
    uint32_t sin(uint32_t ty, uint32_t a) { return ext(ty, 13, {a}); }
    uint32_t cos(uint32_t ty, uint32_t a) { return ext(ty, 14, {a}); }
    uint32_t acos(uint32_t ty, uint32_t a) { return ext(ty, 17, {a}); }
    uint32_t pow(uint32_t ty, uint32_t a, uint32_t b) { return ext(ty, 26, {a, b}); }
    uint32_t exp(uint32_t ty, uint32_t a) { return ext(ty, 27, {a}); }
    uint32_t sqrt(uint32_t ty, uint32_t a) { return ext(ty, 31, {a}); }
    uint32_t rsqrt(uint32_t ty, uint32_t a) { return ext(ty, 32, {a}); }
    uint32_t fmin(uint32_t ty, uint32_t a, uint32_t b) { return ext(ty, 37, {a, b}); }
    uint32_t fmax(uint32_t ty, uint32_t a, uint32_t b) { return ext(ty, 40, {a, b}); }
    uint32_t clamp(uint32_t ty, uint32_t x, uint32_t a, uint32_t b) {
        return ext(ty, 43, {x, a, b});
    }
    uint32_t mix(uint32_t ty, uint32_t a, uint32_t b, uint32_t t) {
        return ext(ty, 46, {a, b, t});
    }
    uint32_t step(uint32_t ty, uint32_t edge, uint32_t x) {
        return ext(ty, 48, {edge, x});
    }
    uint32_t cross(uint32_t a, uint32_t b) { return ext(tVec3, 68, {a, b}); }

    uint32_t safeNorm(uint32_t v) {
        const uint32_t d = fmax(tF32, dot(v, v), fcConst(1e-12f));
        return vscale(tVec3, v, rsqrt(tF32, d));
    }

    uint32_t sat(uint32_t ty, uint32_t x) {
        const uint32_t z = (ty == tVec3) ? splat3(fcConst(0.0f)) : fcConst(0.0f);
        const uint32_t o = (ty == tVec3) ? splat3(fcConst(1.0f)) : fcConst(1.0f);
        return clamp(ty, x, z, o);
    }

    uint32_t sample2D(uint32_t slot, uint32_t coord) {
        const uint32_t img = id(), r = id();
        fnBody.op(61, {tSImg, img, vSamplers[slot]});
        fnBody.op(88, {tVec4, r, img, coord, 0x2u, fcConst(0.0f)});
        return r;
    }

    uint32_t linDepth(uint32_t slot, uint32_t coord) {
        const uint32_t d = extr(sample2D(slot, coord), 0);
        const uint32_t zn = extr(push1, 0);
        const uint32_t zf = extr(push1, 1);
        const uint32_t num = mul(tF32, zn, zf);
        const uint32_t den = sub(tF32, zf, mul(tF32, d, sub(tF32, zf, zn)));
        return div(tF32, num, fmax(tF32, den, fcConst(1e-4f)));
    }

    uint32_t viewPos(uint32_t coord, uint32_t zv) {
        const uint32_t ndcX =
            sub(tF32, mul(tF32, extr(coord, 0), fcConst(2.0f)), fcConst(1.0f));
        const uint32_t ndcY =
            sub(tF32, fcConst(1.0f), mul(tF32, extr(coord, 1), fcConst(2.0f)));
        const uint32_t vx =
            mul(tF32, mul(tF32, ndcX, extr(push0, 2)), zv);
        const uint32_t vy =
            mul(tF32, mul(tF32, ndcY, extr(push0, 3)), zv);
        return v3(vx, vy, zv);
    }

    uint32_t viewNormal(uint32_t depthSlot, uint32_t coord, uint32_t invDx,
                        uint32_t invDy, uint32_t p0) {
        const uint32_t uvR = add(tVec2, coord, v2(invDx, fcConst(0.0f)));
        const uint32_t uvL = sub(tVec2, coord, v2(invDx, fcConst(0.0f)));
        const uint32_t uvD = add(tVec2, coord, v2(fcConst(0.0f), invDy));
        const uint32_t uvU = sub(tVec2, coord, v2(fcConst(0.0f), invDy));

        const uint32_t z0 = extr(p0, 2);
        const uint32_t zR = linDepth(depthSlot, uvR);
        const uint32_t zL = linDepth(depthSlot, uvL);
        const uint32_t zD = linDepth(depthSlot, uvD);
        const uint32_t zU = linDepth(depthSlot, uvU);

        const uint32_t pR = sub(tVec3, viewPos(uvR, zR), p0);
        const uint32_t pL = sub(tVec3, p0, viewPos(uvL, zL));
        const uint32_t pD = sub(tVec3, viewPos(uvD, zD), p0);
        const uint32_t pU = sub(tVec3, p0, viewPos(uvU, zU));

        const uint32_t useL =
            step(tF32, fabs(tF32, sub(tF32, zL, z0)), fabs(tF32, sub(tF32, zR, z0)));
        const uint32_t useU =
            step(tF32, fabs(tF32, sub(tF32, zU, z0)), fabs(tF32, sub(tF32, zD, z0)));

        const uint32_t dX = mix(tVec3, pR, pL, splat3(useL));
        const uint32_t dY = mix(tVec3, pD, pU, splat3(useU));
        return safeNorm(cross(dX, dY));
    }

    uint32_t varF(float initVal) {
        const uint32_t v = id();
        fnVars.op(59, {pFnF32, v, 7, fcConst(initVal)});
        return v;
    }

    uint32_t loadF(uint32_t var) {
        const uint32_t r = id();
        fnBody.op(61, {tF32, r, var});
        return r;
    }

    void storeF(uint32_t var, uint32_t val) { fnBody.op(62, {var, val}); }

    template <typename Fn>
    void forLoop(int32_t startVal, int32_t endVal, Fn&& body) {
        const uint32_t iv = id();
        fnVars.op(59, {pFnS32, iv, 7, sc(startVal)});
        const uint32_t hdr = id(), bdy = id(), cnt = id(), mrg = id();
        fnBody.op(62, {iv, sc(startVal)});
        fnBody.op(249, {hdr});
        fnBody.op(248, {hdr});
        const uint32_t iVal = id(), cond = id();
        fnBody.op(61, {tS32, iVal, iv});
        fnBody.op(177, {tBool, cond, iVal, sc(endVal)});
        fnBody.op(246, {mrg, cnt, 0});
        fnBody.op(250, {cond, bdy, mrg});
        fnBody.op(248, {bdy});
        body(iVal);
        fnBody.op(249, {cnt});
        fnBody.op(248, {cnt});
        const uint32_t cur = id(), nxt = id();
        fnBody.op(61, {tS32, cur, iv});
        fnBody.op(128, {tS32, nxt, cur, sc(1)});
        fnBody.op(62, {iv, nxt});
        fnBody.op(249, {hdr});
        fnBody.op(248, {mrg});
    }

    std::vector<uint32_t> finish(uint32_t outColorVec4) {
        fnBody.op(62, {vOutColor, outColorVec4});
        fnBody.op(253, {});
        fnBody.op(56, {});

        SpvSect pre;
        pre.op(17, {1});
        pre.extImport(glsl, "GLSL.std.450");
        pre.op(14, {0, 1});
        pre.entryPoint(4, mainId, "main", {vFragCoord, vOutColor});
        pre.op(16, {mainId, 7});

        SpvSect fnHdr;
        fnHdr.op(54, {tVoid, mainId, 0, tFn});
        fnHdr.op(248, {id()});

        std::vector<uint32_t> out = {0x07230203u, 0x00010300u, 0x00080001u, nid, 0u};
        out.insert(out.end(), pre.w.begin(), pre.w.end());
        out.insert(out.end(), dec.w.begin(), dec.w.end());
        out.insert(out.end(), tys.w.begin(), tys.w.end());
        out.insert(out.end(), fnHdr.w.begin(), fnHdr.w.end());
        out.insert(out.end(), fnVars.w.begin(), fnVars.w.end());
        out.insert(out.end(), fnBody.w.begin(), fnBody.w.end());
        return out;
    }
};

std::vector<uint32_t> BuildGtaoRawFs() {
    SpvFrag b;
    b.init(1);

    const uint32_t depthRaw = b.extr(b.sample2D(0, b.uv), 0);
    const uint32_t z0 = b.linDepth(0, b.uv);
    const uint32_t p0 = b.viewPos(b.uv, z0);
    const uint32_t n0 = b.viewNormal(0, b.uv, b.mul(b.tF32, b.extr(b.push0, 0), b.fcConst(0.5f)),
                                     b.mul(b.tF32, b.extr(b.push0, 1), b.fcConst(0.5f)), p0);
    const uint32_t v0 = b.safeNorm(b.neg(b.tVec3, p0));

    const uint32_t ignSeed = b.v2(b.fcConst(0.06711056f), b.fcConst(0.00583715f));
    const uint32_t ign = b.fract(b.tF32, b.mul(b.tF32, b.fcConst(52.9829189f),
                                                b.fract(b.tF32, b.dot(b.fcXy, ignSeed))));
    const uint32_t phi0 = b.mul(b.tF32, ign, b.fcConst(3.14159265f));
    const uint32_t jitter = b.fract(b.tF32, b.mul(b.tF32, ign, b.fcConst(7.31f)));

    const uint32_t worldR = b.fcConst(0.92f);
    const uint32_t invR2 = b.fcConst(1.0f / (0.92f * 0.92f));
    const uint32_t screenR = b.clamp(b.tF32, b.div(b.tF32, worldR,
                                                   b.mul(b.tF32, z0, b.extr(b.push0, 3))),
                                     b.fcConst(0.008f), b.fcConst(0.13f));
    const uint32_t radUV = b.v2(b.div(b.tF32, screenR, b.extr(b.push1, 2)), screenR);

    const uint32_t visAcc = b.varF(0.0f);
    const uint32_t maxCos1 = b.varF(-1.0f);
    const uint32_t maxCos2 = b.varF(-1.0f);

    b.forLoop(0, 6, [&](uint32_t dInt) {
        const uint32_t df = b.id();
        b.fnBody.op(111, {b.tF32, df, dInt});
        const uint32_t ang = b.add(b.tF32, phi0, b.mul(b.tF32, df, b.fcConst(3.14159265f / 6.0f)));
        const uint32_t dirX = b.cos(b.tF32, ang);
        const uint32_t dirY = b.sin(b.tF32, ang);
        const uint32_t dirUV = b.v2(dirX, dirY);

        const uint32_t sliceDir3 = b.v3(dirX, b.neg(b.tF32, dirY), b.fcConst(0.0f));
        const uint32_t planeN = b.safeNorm(b.cross(sliceDir3, v0));
        const uint32_t tangent = b.safeNorm(b.cross(v0, planeN));
        const uint32_t projN = b.sub(b.tVec3, n0, b.vscale(b.tVec3, planeN, b.dot(n0, planeN)));
        const uint32_t projLen = b.fmax(b.tF32, b.sqrt(b.tF32, b.fmax(b.tF32, b.dot(projN, projN), b.fcConst(1e-8f))),
                                        b.fcConst(1e-4f));
        const uint32_t cosN = b.clamp(b.tF32, b.div(b.tF32, b.dot(projN, v0), projLen),
                                      b.fcConst(-0.999f), b.fcConst(0.999f));
        const uint32_t nSign = b.neg(b.tF32, b.fsign(b.tF32, b.dot(projN, tangent)));
        const uint32_t nAng = b.mul(b.tF32, nSign, b.acos(b.tF32, cosN));
        const uint32_t sinN = b.sin(b.tF32, nAng);

        b.storeF(maxCos1, b.fcConst(-1.0f));
        b.storeF(maxCos2, b.fcConst(-1.0f));

        b.forLoop(0, 4, [&](uint32_t sInt) {
            const uint32_t sf = b.id();
            b.fnBody.op(111, {b.tF32, sf, sInt});
            const uint32_t u = b.mul(b.tF32, b.add(b.tF32, sf, jitter), b.fcConst(0.25f));
            const uint32_t sOff = b.vscale(b.tVec2, b.mul(b.tVec2, dirUV, radUV), b.mul(b.tF32, u, u));

            auto accumHorizon = [&](uint32_t scUV, uint32_t varMax) {
                const uint32_t zS = b.linDepth(0, scUV);
                const uint32_t pS = b.viewPos(scUV, zS);
                const uint32_t hVec = b.sub(b.tVec3, pS, p0);
                const uint32_t hLen2 = b.fmax(b.tF32, b.dot(hVec, hVec), b.fcConst(1e-6f));
                const uint32_t hCos = b.mul(b.tF32, b.dot(hVec, v0), b.rsqrt(b.tF32, hLen2));
                const uint32_t fall = b.sat(b.tF32, b.sub(b.tF32, b.fcConst(1.0f), b.mul(b.tF32, hLen2, invR2)));
                const uint32_t eff = b.mix(b.tF32, b.fcConst(-1.0f), hCos, fall);
                b.storeF(varMax, b.fmax(b.tF32, b.loadF(varMax), eff));
            };

            accumHorizon(b.add(b.tVec2, b.uv, sOff), maxCos1);
            accumHorizon(b.sub(b.tVec2, b.uv, sOff), maxCos2);
        });

        const uint32_t halfPi = b.fcConst(1.57079633f);
        const uint32_t h1 = b.clamp(b.tF32, b.acos(b.tF32, b.clamp(b.tF32, b.loadF(maxCos1), b.fcConst(-0.999f), b.fcConst(0.999f))),
                                    b.sub(b.tF32, nAng, halfPi), b.add(b.tF32, nAng, halfPi));
        const uint32_t h2 = b.clamp(b.tF32, b.neg(b.tF32, b.acos(b.tF32, b.clamp(b.tF32, b.loadF(maxCos2), b.fcConst(-0.999f), b.fcConst(0.999f)))),
                                    b.sub(b.tF32, nAng, halfPi), b.add(b.tF32, nAng, halfPi));

        auto horizonInt = [&](uint32_t h) {
            const uint32_t cTerm = b.neg(b.tF32, b.cos(b.tF32, b.sub(b.tF32, b.mul(b.tF32, b.fcConst(2.0f), h), nAng)));
            const uint32_t sTerm = b.mul(b.tF32, b.mul(b.tF32, b.fcConst(2.0f), h), sinN);
            return b.mul(b.tF32, b.fcConst(0.25f), b.add(b.tF32, b.add(b.tF32, cTerm, cosN), sTerm));
        };

        const uint32_t sliceVis = b.mul(b.tF32, projLen,
                                        b.fmax(b.tF32, b.add(b.tF32, horizonInt(h1), horizonInt(h2)), b.fcConst(0.0f)));
        b.storeF(visAcc, b.add(b.tF32, b.loadF(visAcc), sliceVis));
    });

    const uint32_t rawAo = b.pow(b.tF32, b.clamp(b.tF32, b.mul(b.tF32, b.loadF(visAcc), b.fcConst(1.0f / 6.0f)),
                                                 b.fcConst(1e-4f), b.fcConst(1.0f)),
                                 b.fcConst(1.30f));
    const uint32_t skyMask = b.step(b.tF32, b.fcConst(0.99985f), depthRaw);
    const uint32_t finalAo = b.mix(b.tF32, b.clamp(b.tF32, rawAo, b.fcConst(0.18f), b.fcConst(1.0f)),
                                   b.fcConst(1.0f), skyMask);
    return b.finish(b.v4(finalAo, finalAo, finalAo, b.fcConst(1.0f)));
}

std::vector<uint32_t> BuildBilateralBlurFs(bool vertical) {
    SpvFrag b;
    b.init(2);

    const uint32_t z0 = b.linDepth(1, b.uv);
    const uint32_t sharp = b.add(b.tF32, b.mul(b.tF32, z0, b.fcConst(0.035f)), b.fcConst(0.002f));
    const uint32_t invSharp = b.div(b.tF32, b.fcConst(1.0f), sharp);
    const uint32_t stepX = vertical ? b.fcConst(0.0f) : b.extr(b.push0, 0);
    const uint32_t stepY = vertical ? b.extr(b.push0, 1) : b.fcConst(0.0f);
    const uint32_t stepVec = b.v2(stepX, stepY);

    const uint32_t sumAo = b.varF(0.0f);
    const uint32_t sumW = b.varF(0.0f);

    b.forLoop(-3, 4, [&](uint32_t kInt) {
        const uint32_t kf = b.id();
        b.fnBody.op(111, {b.tF32, kf, kInt});
        const uint32_t scUV = b.add(b.tVec2, b.uv, b.vscale(b.tVec2, stepVec, kf));
        const uint32_t sAo = b.extr(b.sample2D(0, scUV), 0);
        const uint32_t sZ = b.linDepth(1, scUV);

        const uint32_t k2 = b.mul(b.tF32, kf, kf);
        const uint32_t wSpace = b.exp(b.tF32, b.mul(b.tF32, k2, b.fcConst(-0.18f)));
        const uint32_t wDepth = b.exp(b.tF32, b.mul(b.tF32, b.fabs(b.tF32, b.sub(b.tF32, sZ, z0)), b.neg(b.tF32, invSharp)));
        const uint32_t w = b.mul(b.tF32, wSpace, wDepth);

        b.storeF(sumAo, b.add(b.tF32, b.loadF(sumAo), b.mul(b.tF32, sAo, w)));
        b.storeF(sumW, b.add(b.tF32, b.loadF(sumW), w));
    });

    const uint32_t outAo = b.div(b.tF32, b.loadF(sumAo), b.fmax(b.tF32, b.loadF(sumW), b.fcConst(1e-5f)));
    return b.finish(b.v4(outAo, outAo, outAo, b.fcConst(1.0f)));
}

std::vector<uint32_t> BuildSceneEnhanceFs() {
    SpvFrag b;
    b.init(3);

    const uint32_t base4 = b.sample2D(0, b.uv);
    const uint32_t baseCol = b.xyz(base4);
    const uint32_t baseAlpha = b.extr(base4, 3);

    const uint32_t dRaw = b.extr(b.sample2D(2, b.uv), 0);
    const uint32_t z0 = b.linDepth(2, b.uv);
    const uint32_t invZTol = b.div(b.tF32, b.fcConst(1.0f),
                                   b.add(b.tF32, b.mul(b.tF32, z0, b.fcConst(0.03f)), b.fcConst(0.002f)));

    const uint32_t invW = b.extr(b.push0, 0);
    const uint32_t invH = b.extr(b.push0, 1);
    const uint32_t halfDx = b.mul(b.tF32, invW, b.fcConst(0.75f));
    const uint32_t halfDy = b.mul(b.tF32, invH, b.fcConst(0.75f));

    const uint32_t aoAcc = b.varF(0.0f);
    const uint32_t wAcc = b.varF(0.0f);

    auto tapAo = [&](uint32_t ox, uint32_t oy) {
        const uint32_t tc = b.add(b.tVec2, b.uv, b.v2(ox, oy));
        const uint32_t a = b.extr(b.sample2D(1, tc), 0);
        const uint32_t zS = b.linDepth(2, tc);
        const uint32_t w = b.exp(b.tF32, b.mul(b.tF32, b.fabs(b.tF32, b.sub(b.tF32, zS, z0)),
                                                b.neg(b.tF32, invZTol)));
        b.storeF(aoAcc, b.add(b.tF32, b.loadF(aoAcc), b.mul(b.tF32, a, w)));
        b.storeF(wAcc, b.add(b.tF32, b.loadF(wAcc), w));
    };

    tapAo(b.neg(b.tF32, halfDx), b.neg(b.tF32, halfDy));
    tapAo(halfDx, b.neg(b.tF32, halfDy));
    tapAo(b.neg(b.tF32, halfDx), halfDy);
    tapAo(halfDx, halfDy);

    const uint32_t aoUp = b.div(b.tF32, b.loadF(aoAcc), b.fmax(b.tF32, b.loadF(wAcc), b.fcConst(1e-5f)));
    const uint32_t skyMask = b.step(b.tF32, b.fcConst(0.99985f), dRaw);
    const uint32_t ao = b.mix(b.tF32, b.sat(b.tF32, aoUp), b.fcConst(1.0f), skyMask);

    const uint32_t creviceTint = b.v3f(0.42f, 0.45f, 0.56f);
    const uint32_t aoColor = b.mix(b.tVec3, creviceTint, b.v3f(1.0f, 1.0f, 1.0f), b.splat3(ao));
    const uint32_t colWithAo = b.mul(b.tVec3, baseCol, b.mix(b.tVec3, b.v3f(1.0f, 1.0f, 1.0f), aoColor, b.splat3(b.fcConst(0.76f))));

    const uint32_t p0 = b.viewPos(b.uv, z0);
    const uint32_t n0 = b.viewNormal(2, b.uv, invW, invH, p0);
    const uint32_t vDir = b.safeNorm(b.neg(b.tVec3, p0));

    const uint32_t grainCell = b.floor(b.tVec2, b.vscale(b.tVec2, b.fcXy, b.fcConst(0.85f)));
    const uint32_t grainRand = b.fract(b.tF32, b.mul(b.tF32, b.sin(b.tF32, b.dot(grainCell, b.v2(b.fcConst(127.1f), b.fcConst(311.7f)))),
                                                     b.fcConst(43758.5453f)));
    const uint32_t grain = b.sub(b.tF32, b.mul(b.tF32, grainRand, b.fcConst(2.0f)), b.fcConst(1.0f));
    const uint32_t nPert = b.v3(b.mul(b.tF32, grain, b.fcConst(0.022f)),
                                b.mul(b.tF32, grain, b.fcConst(-0.022f)), b.fcConst(0.0f));
    const uint32_t nMicro = b.safeNorm(b.add(b.tVec3, n0, nPert));

    const uint32_t ndotv = b.sat(b.tF32, b.dot(nMicro, vDir));
    const uint32_t fresnel = b.add(b.tF32, b.fcConst(0.04f),
                                   b.mul(b.tF32, b.fcConst(0.96f),
                                         b.pow(b.tF32, b.fmax(b.tF32, b.sub(b.tF32, b.fcConst(1.0f), ndotv), b.fcConst(1e-4f)),
                                               b.fcConst(5.0f))));

    const uint32_t lKey = b.safeNorm(b.v3f(0.42f, 0.78f, -0.46f));
    const uint32_t hDir = b.safeNorm(b.add(b.tVec3, lKey, vDir));
    const uint32_t ndoth = b.fmax(b.tF32, b.dot(nMicro, hDir), b.fcConst(1e-4f));
    const uint32_t shininess = b.add(b.tF32, b.fcConst(52.0f), b.mul(b.tF32, grain, b.fcConst(16.0f)));
    const uint32_t absSpec = b.mul(b.tF32, b.pow(b.tF32, ndoth, shininess),
                                   b.add(b.tF32, b.fcConst(0.14f), b.mul(b.tF32, b.fcConst(0.22f), fresnel)));

    const uint32_t luma = b.dot(baseCol, b.v3f(0.299f, 0.587f, 0.114f));
    const uint32_t geoMask = b.sub(b.tF32, b.fcConst(1.0f), skyMask);
    const uint32_t litMask = b.mul(b.tF32, b.mul(b.tF32, b.clamp(b.tF32, b.mul(b.tF32, luma, b.fcConst(2.4f)),
                                                                  b.fcConst(0.06f), b.fcConst(1.0f)), ao), geoMask);

    const uint32_t hemi = b.fmax(b.tF32, b.add(b.tF32, b.mul(b.tF32, b.extr(nMicro, 1), b.fcConst(0.5f)), b.fcConst(0.5f)),
                                 b.fcConst(0.25f));
    const uint32_t skyRim = b.vscale(b.tVec3, b.v3f(0.84f, 0.91f, 1.0f),
                                     b.mul(b.tF32, b.mul(b.tF32, fresnel, hemi), b.mul(b.tF32, litMask, b.fcConst(0.13f))));
    const uint32_t specCol = b.vscale(b.tVec3, b.v3f(1.0f, 0.97f, 0.92f),
                                      b.mul(b.tF32, absSpec, litMask));

    const uint32_t litAbs = b.add(b.tVec3, colWithAo, b.add(b.tVec3, specCol, skyRim));

    const uint32_t distExcess = b.fmax(b.tF32, b.sub(b.tF32, z0, b.fcConst(26.0f)), b.fcConst(0.0f));
    const uint32_t hazeAmt = b.mul(b.tF32, geoMask,
                                   b.clamp(b.tF32, b.sub(b.tF32, b.fcConst(1.0f), b.exp(b.tF32, b.mul(b.tF32, distExcess, b.fcConst(-0.0036f)))),
                                           b.fcConst(0.0f), b.fcConst(0.28f)));
    const uint32_t hazeCol = b.v3f(0.70f, 0.82f, 0.96f);
    const uint32_t hazed = b.mix(b.tVec3, litAbs, hazeCol, b.splat3(b.mul(b.tF32, hazeAmt, b.fcConst(0.72f))));

    const uint32_t sunUv = b.v2(b.fcConst(0.76f), b.fcConst(0.14f));
    const uint32_t toSun = b.sub(b.tVec2, sunUv, b.uv);
    const uint32_t rayLen = b.fmax(b.tF32, b.sqrt(b.tF32, b.fmax(b.tF32, b.dot(toSun, toSun), b.fcConst(1e-6f))), b.fcConst(1e-4f));
    const uint32_t sunFall = b.pow(b.tF32, b.sat(b.tF32, b.sub(b.tF32, b.fcConst(1.0f), b.mul(b.tF32, rayLen, b.fcConst(1.1f)))),
                                   b.fcConst(2.4f));

    const uint32_t shaftAcc = b.varF(0.0f);
    b.forLoop(1, 5, [&](uint32_t sInt) {
        const uint32_t sf = b.id();
        b.fnBody.op(111, {b.tF32, sf, sInt});
        const uint32_t sUv = b.add(b.tVec2, b.uv, b.vscale(b.tVec2, toSun, b.mul(b.tF32, sf, b.fcConst(0.08f))));
        const uint32_t sD = b.extr(b.sample2D(2, sUv), 0);
        const uint32_t isSky = b.step(b.tF32, b.fcConst(0.9996f), sD);
        const uint32_t sLum = b.dot(b.xyz(b.sample2D(0, sUv)), b.v3f(0.299f, 0.587f, 0.114f));
        b.storeF(shaftAcc, b.add(b.tF32, b.loadF(shaftAcc), b.mul(b.tF32, isSky, b.sat(b.tF32, sLum))));
    });

    const uint32_t shaftDist = b.sat(b.tF32, b.mul(b.tF32, b.sub(b.tF32, z0, b.fcConst(10.0f)), b.fcConst(0.04f)));
    const uint32_t shaftWeight = b.mul(b.tF32, b.mul(b.tF32, b.loadF(shaftAcc), b.fcConst(0.25f)),
                                       b.mul(b.tF32, b.mul(b.tF32, sunFall, shaftDist), b.fcConst(0.11f)));
    const uint32_t finalRgb = b.add(b.tVec3, hazed, b.vscale(b.tVec3, b.v3f(1.0f, 0.90f, 0.68f), shaftWeight));

    return b.finish(b.v4(b.extr(finalRgb, 0), b.extr(finalRgb, 1), b.extr(finalRgb, 2), baseAlpha));
}

std::vector<uint32_t> BuildSmaaEdgesFs() {
    SpvFrag b;
    b.init(1);

    const uint32_t invW = b.extr(b.push0, 0);
    const uint32_t invH = b.extr(b.push0, 1);
    const uint32_t lumaW = b.v3f(0.299f, 0.587f, 0.114f);

    auto lumaAt = [&](uint32_t dx, uint32_t dy) {
        const uint32_t tc = b.add(b.tVec2, b.uv, b.v2(dx, dy));
        return b.dot(b.xyz(b.sample2D(0, tc)), lumaW);
    };

    const uint32_t lC = lumaAt(b.fcConst(0.0f), b.fcConst(0.0f));
    const uint32_t lL = lumaAt(b.neg(b.tF32, invW), b.fcConst(0.0f));
    const uint32_t lT = lumaAt(b.fcConst(0.0f), b.neg(b.tF32, invH));
    const uint32_t lR = lumaAt(invW, b.fcConst(0.0f));
    const uint32_t lB = lumaAt(b.fcConst(0.0f), invH);
    const uint32_t lLL = lumaAt(b.mul(b.tF32, invW, b.fcConst(-2.0f)), b.fcConst(0.0f));
    const uint32_t lTT = lumaAt(b.fcConst(0.0f), b.mul(b.tF32, invH, b.fcConst(-2.0f)));

    const uint32_t dL = b.fabs(b.tF32, b.sub(b.tF32, lC, lL));
    const uint32_t dT = b.fabs(b.tF32, b.sub(b.tF32, lC, lT));
    const uint32_t dR = b.fabs(b.tF32, b.sub(b.tF32, lC, lR));
    const uint32_t dB = b.fabs(b.tF32, b.sub(b.tF32, lC, lB));
    const uint32_t dLL = b.fabs(b.tF32, b.sub(b.tF32, lL, lLL));
    const uint32_t dTT = b.fabs(b.tF32, b.sub(b.tF32, lT, lTT));

    const uint32_t maxLR = b.fmax(b.tF32, b.fmax(b.tF32, dL, dR), b.fmax(b.tF32, dT, dB));
    const uint32_t max2 = b.fmax(b.tF32, maxLR, b.fmax(b.tF32, dLL, dTT));
    const uint32_t halfMax = b.mul(b.tF32, max2, b.fcConst(0.42f));
    const uint32_t thresh = b.fcConst(0.045f);

    const uint32_t edgeH = b.mul(b.tF32, b.step(b.tF32, thresh, dT), b.step(b.tF32, halfMax, dT));
    const uint32_t edgeV = b.mul(b.tF32, b.step(b.tF32, thresh, dL), b.step(b.tF32, halfMax, dL));

    return b.finish(b.v4(edgeH, edgeV, b.fcConst(0.0f), b.fcConst(1.0f)));
}

std::vector<uint32_t> BuildSmaaWeightsFs() {
    SpvFrag b;
    b.init(1);

    const uint32_t invW = b.extr(b.push0, 0);
    const uint32_t invH = b.extr(b.push0, 1);
    const uint32_t eC = b.sample2D(0, b.uv);
    const uint32_t eH = b.extr(eC, 0);
    const uint32_t eV = b.extr(eC, 1);

    const uint32_t dL = b.varF(0.0f);
    const uint32_t dR = b.varF(0.0f);
    const uint32_t actL = b.varF(1.0f);
    const uint32_t actR = b.varF(1.0f);

    b.forLoop(0, 12, [&](uint32_t sInt) {
        const uint32_t sf = b.id();
        b.fnBody.op(111, {b.tF32, sf, sInt});
        const uint32_t stepPix = b.add(b.tF32, b.mul(b.tF32, sf, b.fcConst(2.0f)), b.fcConst(1.5f));

        const uint32_t uvL = b.sub(b.tVec2, b.uv, b.v2(b.mul(b.tF32, stepPix, invW), b.fcConst(0.0f)));
        const uint32_t sL = b.extr(b.sample2D(0, uvL), 0);
        const uint32_t curActL = b.loadF(actL);
        b.storeF(dL, b.add(b.tF32, b.loadF(dL), b.mul(b.tF32, b.mul(b.tF32, curActL, sL), b.fcConst(2.0f))));
        b.storeF(actL, b.mul(b.tF32, curActL, b.step(b.tF32, b.fcConst(0.25f), sL)));

        const uint32_t uvR = b.add(b.tVec2, b.uv, b.v2(b.mul(b.tF32, stepPix, invW), b.fcConst(0.0f)));
        const uint32_t sR = b.extr(b.sample2D(0, uvR), 0);
        const uint32_t curActR = b.loadF(actR);
        b.storeF(dR, b.add(b.tF32, b.loadF(dR), b.mul(b.tF32, b.mul(b.tF32, curActR, sR), b.fcConst(2.0f))));
        b.storeF(actR, b.mul(b.tF32, curActR, b.step(b.tF32, b.fcConst(0.25f), sR)));
    });

    const uint32_t hSpan = b.fmax(b.tF32, b.add(b.tF32, b.loadF(dL), b.loadF(dR)), b.fcConst(1.0f));
    const uint32_t hMinD = b.fmin(b.tF32, b.loadF(dL), b.loadF(dR));
    const uint32_t hCov = b.clamp(b.tF32, b.sub(b.tF32, b.fcConst(0.5f), b.div(b.tF32, hMinD, hSpan)),
                                  b.fcConst(0.0f), b.fcConst(0.5f));

    const uint32_t uvEndL = b.sub(b.tVec2, b.uv, b.v2(b.mul(b.tF32, b.loadF(dL), invW), b.fcConst(0.0f)));
    const uint32_t uvEndR = b.add(b.tVec2, b.uv, b.v2(b.mul(b.tF32, b.loadF(dR), invW), b.fcConst(0.0f)));
    const uint32_t cLT = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndL, b.v2(b.fcConst(0.0f), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1),
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndL, b.v2(invW, b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1));
    const uint32_t cLB = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndL, b.v2(b.fcConst(0.0f), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1),
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndL, b.v2(b.neg(b.tF32, invW), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1));
    const uint32_t cRT = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndR, b.v2(b.fcConst(0.0f), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1),
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndR, b.v2(b.neg(b.tF32, invW), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1));
    const uint32_t cRB = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndR, b.v2(b.fcConst(0.0f), b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1),
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndR, b.v2(invW, b.mul(b.tF32, invH, b.fcConst(0.5f))))), 1));

    const uint32_t hNearL = b.step(b.tF32, b.loadF(dL), b.loadF(dR));
    const uint32_t hCrossT = b.fmax(b.tF32, b.sat(b.tF32, b.mix(b.tF32, cRT, cLT, hNearL)), b.fcConst(0.22f));
    const uint32_t hCrossB = b.fmax(b.tF32, b.sat(b.tF32, b.mix(b.tF32, cRB, cLB, hNearL)), b.fcConst(0.22f));
    const uint32_t wTop = b.mul(b.tF32, b.mul(b.tF32, hCov, eH), hCrossT);
    const uint32_t wBot = b.mul(b.tF32, b.mul(b.tF32, hCov, eH), hCrossB);

    const uint32_t dU = b.varF(0.0f);
    const uint32_t dD = b.varF(0.0f);
    const uint32_t actU = b.varF(1.0f);
    const uint32_t actD = b.varF(1.0f);

    b.forLoop(0, 12, [&](uint32_t sInt) {
        const uint32_t sf = b.id();
        b.fnBody.op(111, {b.tF32, sf, sInt});
        const uint32_t stepPix = b.add(b.tF32, b.mul(b.tF32, sf, b.fcConst(2.0f)), b.fcConst(1.5f));

        const uint32_t uvU = b.sub(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, stepPix, invH)));
        const uint32_t sU = b.extr(b.sample2D(0, uvU), 1);
        const uint32_t curActU = b.loadF(actU);
        b.storeF(dU, b.add(b.tF32, b.loadF(dU), b.mul(b.tF32, b.mul(b.tF32, curActU, sU), b.fcConst(2.0f))));
        b.storeF(actU, b.mul(b.tF32, curActU, b.step(b.tF32, b.fcConst(0.25f), sU)));

        const uint32_t uvD = b.add(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, stepPix, invH)));
        const uint32_t sD = b.extr(b.sample2D(0, uvD), 1);
        const uint32_t curActD = b.loadF(actD);
        b.storeF(dD, b.add(b.tF32, b.loadF(dD), b.mul(b.tF32, b.mul(b.tF32, curActD, sD), b.fcConst(2.0f))));
        b.storeF(actD, b.mul(b.tF32, curActD, b.step(b.tF32, b.fcConst(0.25f), sD)));
    });

    const uint32_t vSpan = b.fmax(b.tF32, b.add(b.tF32, b.loadF(dU), b.loadF(dD)), b.fcConst(1.0f));
    const uint32_t vMinD = b.fmin(b.tF32, b.loadF(dU), b.loadF(dD));
    const uint32_t vCov = b.clamp(b.tF32, b.sub(b.tF32, b.fcConst(0.5f), b.div(b.tF32, vMinD, vSpan)),
                                  b.fcConst(0.0f), b.fcConst(0.5f));

    const uint32_t uvEndU = b.sub(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, b.loadF(dU), invH)));
    const uint32_t uvEndD = b.add(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, b.loadF(dD), invH)));
    const uint32_t cUL = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndU, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.fcConst(0.0f)))), 0),
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndU, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), invH))), 0));
    const uint32_t cUR = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndU, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.fcConst(0.0f)))), 0),
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndU, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.neg(b.tF32, invH)))), 0));
    const uint32_t cDL = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndD, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.fcConst(0.0f)))), 0),
                                b.extr(b.sample2D(0, b.sub(b.tVec2, uvEndD, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.neg(b.tF32, invH)))), 0));
    const uint32_t cDR = b.fmax(b.tF32,
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndD, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), b.fcConst(0.0f)))), 0),
                                b.extr(b.sample2D(0, b.add(b.tVec2, uvEndD, b.v2(b.mul(b.tF32, invW, b.fcConst(0.5f)), invH))), 0));

    const uint32_t vNearU = b.step(b.tF32, b.loadF(dU), b.loadF(dD));
    const uint32_t vCrossL = b.fmax(b.tF32, b.sat(b.tF32, b.mix(b.tF32, cDL, cUL, vNearU)), b.fcConst(0.22f));
    const uint32_t vCrossR = b.fmax(b.tF32, b.sat(b.tF32, b.mix(b.tF32, cDR, cUR, vNearU)), b.fcConst(0.22f));
    const uint32_t wLeft = b.mul(b.tF32, b.mul(b.tF32, vCov, eV), vCrossL);
    const uint32_t wRight = b.mul(b.tF32, b.mul(b.tF32, vCov, eV), vCrossR);

    return b.finish(b.v4(wTop, wBot, wLeft, wRight));
}

std::vector<uint32_t> BuildSmaaBlendAndAcesFs() {
    SpvFrag b;
    b.init(2);

    const uint32_t invW = b.extr(b.push0, 0);
    const uint32_t invH = b.extr(b.push0, 1);

    const uint32_t c0 = b.sample2D(0, b.uv);
    const uint32_t alpha0 = b.extr(c0, 3);
    const uint32_t wCenter = b.sample2D(1, b.uv);
    const uint32_t wRight = b.sample2D(1, b.add(b.tVec2, b.uv, b.v2(invW, b.fcConst(0.0f))));
    const uint32_t wBottom = b.sample2D(1, b.add(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), invH)));

    const uint32_t aTop = b.extr(wCenter, 0);
    const uint32_t aBot = b.extr(wBottom, 1);
    const uint32_t aLeft = b.extr(wCenter, 2);
    const uint32_t aRight = b.extr(wRight, 3);

    const uint32_t maxH = b.fmax(b.tF32, aTop, aBot);
    const uint32_t maxV = b.fmax(b.tF32, aLeft, aRight);
    const uint32_t hasAny = b.step(b.tF32, b.fcConst(1e-4f), b.fmax(b.tF32, maxH, maxV));

    const uint32_t useH = b.step(b.tF32, maxV, maxH);
    const uint32_t colT = b.xyz(b.sample2D(0, b.sub(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, aTop, invH)))));
    const uint32_t colB = b.xyz(b.sample2D(0, b.add(b.tVec2, b.uv, b.v2(b.fcConst(0.0f), b.mul(b.tF32, aBot, invH)))));
    const uint32_t hT = b.div(b.tF32, aBot, b.fmax(b.tF32, b.add(b.tF32, aTop, aBot), b.fcConst(1e-5f)));
    const uint32_t hBlend = b.mix(b.tVec3, colT, colB, b.splat3(hT));

    const uint32_t colL = b.xyz(b.sample2D(0, b.sub(b.tVec2, b.uv, b.v2(b.mul(b.tF32, aLeft, invW), b.fcConst(0.0f)))));
    const uint32_t colR = b.xyz(b.sample2D(0, b.add(b.tVec2, b.uv, b.v2(b.mul(b.tF32, aRight, invW), b.fcConst(0.0f)))));
    const uint32_t vT = b.div(b.tF32, aRight, b.fmax(b.tF32, b.add(b.tF32, aLeft, aRight), b.fcConst(1e-5f)));
    const uint32_t vBlend = b.mix(b.tVec3, colL, colR, b.splat3(vT));

    const uint32_t smaaRaw = b.mix(b.tVec3, vBlend, hBlend, b.splat3(useH));
    const uint32_t smaaCol = b.mix(b.tVec3, b.xyz(c0), smaaRaw, b.splat3(hasAny));

    const uint32_t lumaW = b.v3f(0.299f, 0.587f, 0.114f);
    auto lumaAtUV = [&](uint32_t tc) {
        return b.dot(b.xyz(b.sample2D(0, tc)), lumaW);
    };
    auto lumaOff = [&](uint32_t dx, uint32_t dy) {
        return lumaAtUV(b.add(b.tVec2, b.uv, b.v2(dx, dy)));
    };

    const uint32_t negW = b.neg(b.tF32, invW);
    const uint32_t negH = b.neg(b.tF32, invH);
    const uint32_t zero = b.fcConst(0.0f);

    const uint32_t lM = b.dot(b.xyz(c0), lumaW);
    const uint32_t lN = lumaOff(zero, negH);
    const uint32_t lS = lumaOff(zero, invH);
    const uint32_t lW = lumaOff(negW, zero);
    const uint32_t lE = lumaOff(invW, zero);
    const uint32_t lNW = lumaOff(negW, negH);
    const uint32_t lNE = lumaOff(invW, negH);
    const uint32_t lSW = lumaOff(negW, invH);
    const uint32_t lSE = lumaOff(invW, invH);

    const uint32_t lMin = b.fmin(b.tF32, lM, b.fmin(b.tF32, b.fmin(b.tF32, lN, lS), b.fmin(b.tF32, lW, lE)));
    const uint32_t lMax = b.fmax(b.tF32, lM, b.fmax(b.tF32, b.fmax(b.tF32, lN, lS), b.fmax(b.tF32, lW, lE)));
    const uint32_t lRange = b.sub(b.tF32, lMax, lMin);
    const uint32_t edgeMinThresh = b.fmax(b.tF32, b.fcConst(0.0312f), b.mul(b.tF32, lMax, b.fcConst(0.10f)));
    const uint32_t edgeActive = b.step(b.tF32, edgeMinThresh, lRange);

    const uint32_t lNS = b.add(b.tF32, lN, lS);
    const uint32_t lWE = b.add(b.tF32, lW, lE);
    const uint32_t lNWSW = b.add(b.tF32, lNW, lSW);
    const uint32_t lNESE = b.add(b.tF32, lNE, lSE);
    const uint32_t lNWNE = b.add(b.tF32, lNW, lNE);
    const uint32_t lSWSE = b.add(b.tF32, lSW, lSE);

    const uint32_t subpixLuma = b.mul(b.tF32,
                                      b.add(b.tF32, b.mul(b.tF32, b.add(b.tF32, lNS, lWE), b.fcConst(2.0f)),
                                            b.add(b.tF32, lNWSW, lNESE)),
                                      b.fcConst(1.0f / 12.0f));
    const uint32_t subpixB = b.sat(b.tF32, b.div(b.tF32, b.fabs(b.tF32, b.sub(b.tF32, subpixLuma, lM)),
                                                 b.fmax(b.tF32, lRange, b.fcConst(1e-5f))));
    const uint32_t subpixC = b.mul(b.tF32, b.mul(b.tF32, subpixB, subpixB),
                                   b.sub(b.tF32, b.fcConst(3.0f), b.mul(b.tF32, b.fcConst(2.0f), subpixB)));
    const uint32_t subpixFactor = b.mul(b.tF32, b.mul(b.tF32, subpixC, subpixC), b.fcConst(0.68f));

    const uint32_t edgeH1 = b.fabs(b.tF32, b.sub(b.tF32, lNWSW, b.mul(b.tF32, b.fcConst(2.0f), lW)));
    const uint32_t edgeH2 = b.mul(b.tF32, b.fcConst(2.0f), b.fabs(b.tF32, b.sub(b.tF32, lNS, b.mul(b.tF32, b.fcConst(2.0f), lM))));
    const uint32_t edgeH3 = b.fabs(b.tF32, b.sub(b.tF32, lNESE, b.mul(b.tF32, b.fcConst(2.0f), lE)));
    const uint32_t fxEdgeHorz = b.add(b.tF32, b.add(b.tF32, edgeH1, edgeH2), edgeH3);

    const uint32_t edgeV1 = b.fabs(b.tF32, b.sub(b.tF32, lNWNE, b.mul(b.tF32, b.fcConst(2.0f), lN)));
    const uint32_t edgeV2 = b.mul(b.tF32, b.fcConst(2.0f), b.fabs(b.tF32, b.sub(b.tF32, lWE, b.mul(b.tF32, b.fcConst(2.0f), lM))));
    const uint32_t edgeV3 = b.fabs(b.tF32, b.sub(b.tF32, lSWSE, b.mul(b.tF32, b.fcConst(2.0f), lS)));
    const uint32_t fxEdgeVert = b.add(b.tF32, b.add(b.tF32, edgeV1, edgeV2), edgeV3);

    const uint32_t isHorz = b.step(b.tF32, fxEdgeVert, fxEdgeHorz);
    const uint32_t luma1 = b.mix(b.tF32, lW, lN, isHorz);
    const uint32_t luma2 = b.mix(b.tF32, lE, lS, isHorz);
    const uint32_t grad1 = b.fabs(b.tF32, b.sub(b.tF32, luma1, lM));
    const uint32_t grad2 = b.fabs(b.tF32, b.sub(b.tF32, luma2, lM));
    const uint32_t is1Steep = b.step(b.tF32, grad2, grad1);
    const uint32_t gradScaled = b.mul(b.tF32, b.fcConst(0.25f), b.fmax(b.tF32, grad1, grad2));

    const uint32_t stepMag = b.mix(b.tF32, invW, invH, isHorz);
    const uint32_t signedStep = b.mix(b.tF32, stepMag, b.neg(b.tF32, stepMag), is1Steep);
    const uint32_t lumaLocalAvg = b.mul(b.tF32, b.fcConst(0.5f),
                                        b.add(b.tF32, b.mix(b.tF32, luma2, luma1, is1Steep), lM));

    const uint32_t halfStep = b.mul(b.tF32, signedStep, b.fcConst(0.5f));
    const uint32_t posB = b.add(b.tVec2, b.uv,
                                b.v2(b.mix(b.tF32, halfStep, zero, isHorz),
                                     b.mix(b.tF32, zero, halfStep, isHorz)));
    const uint32_t tangStep = b.v2(b.mix(b.tF32, zero, invW, isHorz),
                                   b.mix(b.tF32, invH, zero, isHorz));

    const uint32_t distN = b.varF(0.0f);
    const uint32_t distP = b.varF(0.0f);
    const uint32_t actN = b.varF(1.0f);
    const uint32_t actP = b.varF(1.0f);
    const uint32_t endSignN = b.varF(0.0f);
    const uint32_t endSignP = b.varF(0.0f);

    b.forLoop(0, 10, [&](uint32_t sInt) {
        const uint32_t sf = b.id();
        b.fnBody.op(111, {b.tF32, sf, sInt});
        const uint32_t stepPix = b.add(b.tF32, b.fcConst(1.0f),
                                       b.mul(b.tF32, b.floor(b.tF32, b.mul(b.tF32, sf, b.fcConst(0.5f))), b.fcConst(0.5f)));

        const uint32_t cActN = b.loadF(actN);
        const uint32_t tryDistN = b.add(b.tF32, b.loadF(distN), stepPix);
        const uint32_t uvN = b.sub(b.tVec2, posB, b.vscale(b.tVec2, tangStep, tryDistN));
        const uint32_t lEndN = b.sub(b.tF32, lumaAtUV(uvN), lumaLocalAvg);
        const uint32_t hitN = b.step(b.tF32, gradScaled, b.fabs(b.tF32, lEndN));
        b.storeF(distN, b.mix(b.tF32, b.loadF(distN), tryDistN, cActN));
        b.storeF(endSignN, b.mix(b.tF32, b.loadF(endSignN), b.step(b.tF32, zero, lEndN), cActN));
        b.storeF(actN, b.mul(b.tF32, cActN, b.sub(b.tF32, b.fcConst(1.0f), hitN)));

        const uint32_t cActP = b.loadF(actP);
        const uint32_t tryDistP = b.add(b.tF32, b.loadF(distP), stepPix);
        const uint32_t uvP = b.add(b.tVec2, posB, b.vscale(b.tVec2, tangStep, tryDistP));
        const uint32_t lEndP = b.sub(b.tF32, lumaAtUV(uvP), lumaLocalAvg);
        const uint32_t hitP = b.step(b.tF32, gradScaled, b.fabs(b.tF32, lEndP));
        b.storeF(distP, b.mix(b.tF32, b.loadF(distP), tryDistP, cActP));
        b.storeF(endSignP, b.mix(b.tF32, b.loadF(endSignP), b.step(b.tF32, zero, lEndP), cActP));
        b.storeF(actP, b.mul(b.tF32, cActP, b.sub(b.tF32, b.fcConst(1.0f), hitP)));
    });

    const uint32_t dN = b.loadF(distN);
    const uint32_t dP = b.loadF(distP);
    const uint32_t isDirN = b.step(b.tF32, dN, dP);
    const uint32_t dstMin = b.fmin(b.tF32, dN, dP);
    const uint32_t closerSign = b.mix(b.tF32, b.loadF(endSignP), b.loadF(endSignN), isDirN);
    const uint32_t lMSign = b.step(b.tF32, zero, b.sub(b.tF32, lM, lumaLocalAvg));
    const uint32_t goodSpan = b.fabs(b.tF32, b.sub(b.tF32, closerSign, lMSign));
    const uint32_t spanLen = b.fmax(b.tF32, b.add(b.tF32, dN, dP), b.fcConst(1e-4f));
    const uint32_t edgeOff = b.mul(b.tF32, goodSpan,
                                   b.fmax(b.tF32, zero, b.sub(b.tF32, b.fcConst(0.5f), b.div(b.tF32, dstMin, spanLen))));
    const uint32_t finalOff = b.mul(b.tF32, b.fmax(b.tF32, edgeOff, subpixFactor), edgeActive);

    const uint32_t fxShift = b.mul(b.tF32, finalOff, signedStep);
    const uint32_t uvFX = b.add(b.tVec2, b.uv,
                                b.v2(b.mix(b.tF32, fxShift, zero, isHorz),
                                     b.mix(b.tF32, zero, fxShift, isHorz)));
    const uint32_t fxaaCol = b.xyz(b.sample2D(0, uvFX));
    const uint32_t aaCol = b.mix(b.tVec3, fxaaCol, smaaCol, b.splat3(b.mul(b.tF32, hasAny, b.fcConst(0.30f))));

    const uint32_t safeAA = b.fmax(b.tVec3, aaCol, b.v3f(1e-4f, 1e-4f, 1e-4f));
    const uint32_t lin = b.pow(b.tVec3, safeAA, b.v3f(2.2f, 2.2f, 2.2f));
    const uint32_t x = b.vscale(b.tVec3, lin, b.fcConst(1.12f));

    const uint32_t num = b.mul(b.tVec3, x, b.add(b.tVec3, b.vscale(b.tVec3, x, b.fcConst(2.51f)), b.v3f(0.03f, 0.03f, 0.03f)));
    const uint32_t den = b.add(b.tVec3, b.mul(b.tVec3, x, b.add(b.tVec3, b.vscale(b.tVec3, x, b.fcConst(2.43f)), b.v3f(0.59f, 0.59f, 0.59f))),
                               b.v3f(0.14f, 0.14f, 0.14f));
    const uint32_t aces = b.sat(b.tVec3, b.div(b.tVec3, num, den));

    const uint32_t lum = b.dot(aces, b.v3f(0.2126f, 0.7152f, 0.0722f));
    const uint32_t r = b.extr(aces, 0), g = b.extr(aces, 1), bv = b.extr(aces, 2);
    const uint32_t mx = b.fmax(b.tF32, b.fmax(b.tF32, r, g), bv);
    const uint32_t mn = b.fmin(b.tF32, b.fmin(b.tF32, r, g), bv);
    const uint32_t satC = b.sub(b.tF32, mx, mn);
    const uint32_t vib = b.add(b.tF32, b.fcConst(1.0f),
                               b.mul(b.tF32, b.fcConst(0.24f),
                                     b.sub(b.tF32, b.fcConst(1.0f), b.sat(b.tF32, b.mul(b.tF32, satC, b.fcConst(1.4f))))));
    const uint32_t vibranced = b.sat(b.tVec3, b.mix(b.tVec3, b.splat3(lum), aces, b.splat3(vib)));

    const uint32_t v2t = b.mul(b.tVec3, vibranced, vibranced);
    const uint32_t sCurve = b.mul(b.tVec3, v2t, b.sub(b.tVec3, b.v3f(3.0f, 3.0f, 3.0f), b.vscale(b.tVec3, vibranced, b.fcConst(2.0f))));
    const uint32_t graded = b.sat(b.tVec3, b.mix(b.tVec3, vibranced, sCurve, b.v3f(0.16f, 0.16f, 0.16f)));
    const uint32_t srgb = b.pow(b.tVec3, b.fmax(b.tVec3, graded, b.v3f(1e-4f, 1e-4f, 1e-4f)),
                                b.v3f(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f));

    return b.finish(b.v4(b.extr(srgb, 0), b.extr(srgb, 1), b.extr(srgb, 2), alpha0));
}

VkShaderModule MakeModule(VkDevice dev, const std::vector<uint32_t>& words) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words.size() * sizeof(uint32_t);
    ci.pCode = words.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return mod;
}

}

bool PostFx::Init() {
    if (initialized_) return true;
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (!dev || !alloc) return false;

    vsFull_ = MakeModule(dev, BuildFullscreenVs());
    fsGtao_ = MakeModule(dev, BuildGtaoRawFs());
    fsBlurH_ = MakeModule(dev, BuildBilateralBlurFs(false));
    fsBlurV_ = MakeModule(dev, BuildBilateralBlurFs(true));
    fsSceneEnh_ = MakeModule(dev, BuildSceneEnhanceFs());
    fsSmaaEdge_ = MakeModule(dev, BuildSmaaEdgesFs());
    fsSmaaWeight_ = MakeModule(dev, BuildSmaaWeightsFs());
    fsSmaaBlend_ = MakeModule(dev, BuildSmaaBlendAndAcesFs());
    if (!vsFull_ || !fsGtao_ || !fsBlurH_ || !fsBlurV_ || !fsSceneEnh_ ||
        !fsSmaaEdge_ || !fsSmaaWeight_ || !fsSmaaBlend_) {
        LOG("[postfx] shader module creation failed");
        Shutdown();
        return false;
    }

    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    slci.bindingCount = 3;
    slci.pBindings = binds;
    if (vkCreateDescriptorSetLayout(dev, &slci, nullptr, &setLayout_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PostPush);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipeLayout_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    VkDescriptorPoolSize poolSz{};
    poolSz.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSz.descriptorCount = kNumDescSets * 3u;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = kNumDescSets;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &descPool_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    VkDescriptorSetLayout layouts[kNumDescSets]{};
    for (uint32_t i = 0; i < kNumDescSets; ++i) layouts[i] = setLayout_;
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = kNumDescSets;
    dsai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(dev, &dsai, descSets_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 1.0f;
    if (vkCreateSampler(dev, &sci, nullptr, &nearestSampler_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(dev, &sci, nullptr, &linearSampler_) != VK_SUCCESS) {
        Shutdown();
        return false;
    }

    initialized_ = true;
    LOG("[postfx] GTAO + SMAA + ACES initialized");
    return true;
}

void PostFx::Shutdown() {
    VkDevice dev = g_backend.Device();
    if (!dev) return;

    DestroyTargets();
    auto killPipe = [&](VkPipeline& p, VkFormat& f) {
        if (p) {
            vkDestroyPipeline(dev, p, nullptr);
            p = VK_NULL_HANDLE;
        }
        f = VK_FORMAT_UNDEFINED;
    };
    killPipe(pipeGtao_, fmtGtao_);
    killPipe(pipeBlurH_, fmtBlurH_);
    killPipe(pipeBlurV_, fmtBlurV_);
    killPipe(pipeSceneEnh_, fmtSceneEnh_);
    killPipe(pipeSmaaEdge_, fmtSmaaEdge_);
    killPipe(pipeSmaaWeight_, fmtSmaaWeight_);
    killPipe(pipeSmaaBlend_, fmtSmaaBlend_);

    if (nearestSampler_) {
        vkDestroySampler(dev, nearestSampler_, nullptr);
        nearestSampler_ = VK_NULL_HANDLE;
    }
    if (linearSampler_) {
        vkDestroySampler(dev, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }
    if (descPool_) {
        vkDestroyDescriptorPool(dev, descPool_, nullptr);
        descPool_ = VK_NULL_HANDLE;
        std::memset(descSets_, 0, sizeof(descSets_));
    }
    if (pipeLayout_) {
        vkDestroyPipelineLayout(dev, pipeLayout_, nullptr);
        pipeLayout_ = VK_NULL_HANDLE;
    }
    if (setLayout_) {
        vkDestroyDescriptorSetLayout(dev, setLayout_, nullptr);
        setLayout_ = VK_NULL_HANDLE;
    }
    auto killMod = [&](VkShaderModule& m) {
        if (m) {
            vkDestroyShaderModule(dev, m, nullptr);
            m = VK_NULL_HANDLE;
        }
    };
    killMod(fsSmaaBlend_);
    killMod(fsSmaaWeight_);
    killMod(fsSmaaEdge_);
    killMod(fsSceneEnh_);
    killMod(fsBlurV_);
    killMod(fsBlurH_);
    killMod(fsGtao_);
    killMod(vsFull_);
    initialized_ = false;
}

void PostFx::OnSwapchainRecreated() {
    DestroyTargets();
}

void PostFx::BeginFrame() {
    hasScene3D_ = false;
    gbufSeen_ = false;
    scenePostDone_ = false;
    bbPostDone_ = false;
    bbCompositeSeen_ = false;
    bbDirect3D_ = false;
    gbufDepth_ = nullptr;
    sceneColor_ = nullptr;
    sceneDepth_ = nullptr;
    forwardCandColor_ = nullptr;
    forwardCandDepth_ = nullptr;
    forwardDrawCount_ = 0;
    bb3dDrawCount_ = 0;
    bbDraws_ = 0;
}

void PostFx::MarkBackbufferComposite() {
    bbCompositeSeen_ = true;
}

void PostFx::MarkBackbufferStretch() {
    bbCompositeSeen_ = true;
}

void PostFx::TrackDraw(FakeSurface* color0, FakeSurface* color1,
                       FakeSurface* depth, bool zEnable, bool zWrite,
                       uint32_t colorWrite, bool readsRt, uint32_t topo,
                       uint32_t primCount) {
    if (!initialized_) return;
    const VkExtent2D sw = g_backend.SwapchainExtent();
    if (!sw.width || !sw.height) return;
    const uint32_t minW = sw.width / 2u;
    const uint32_t minH = sw.height / 2u;

    if (!color0 || color0->isBackbuffer) {
        ++bbDraws_;
        if (readsRt || (topo == 4 && primCount == 2 && bbDraws_ <= 4)) {
            bbCompositeSeen_ = true;
        }
        if (zEnable && zWrite && colorWrite != 0 && !readsRt && primCount > 2) {
            ++bb3dDrawCount_;
            if (bb3dDrawCount_ >= 4 && !hasScene3D_) {
                hasScene3D_ = true;
                bbDirect3D_ = true;
            }
        }
        return;
    }

    if (depth && !depth->isBackbuffer && depth->width >= minW &&
        depth->height >= minH && depth->height <= sw.height * 2u &&
        color0->width == depth->width && color0->height == depth->height) {
        if (color1 != nullptr) {
            gbufSeen_ = true;
            gbufDepth_ = depth;
        } else if (colorWrite != 0) {
            if (gbufSeen_ && gbufDepth_ && depth->image == gbufDepth_->image) {
                sceneColor_ = color0;
                sceneDepth_ = depth;
                hasScene3D_ = true;
            } else if (zEnable && zWrite && primCount > 2) {
                if (forwardCandColor_ != color0 || forwardCandDepth_ != depth) {
                    forwardCandColor_ = color0;
                    forwardCandDepth_ = depth;
                    forwardDrawCount_ = 1;
                } else {
                    ++forwardDrawCount_;
                }
                if (forwardDrawCount_ >= 4 && !sceneColor_) {
                    sceneColor_ = color0;
                    sceneDepth_ = depth;
                    hasScene3D_ = true;
                }
            }
        }
    }
}

bool PostFx::CreateTarget(RenderTarget& rt, uint32_t w, uint32_t h, VkFormat fmt,
                          VkImageUsageFlags extraUsage) {
    DestroyTarget(rt);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(g_backend.Allocator(), &ici, &aci, &rt.image, &rt.alloc,
                       nullptr) != VK_SUCCESS) {
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = rt.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g_backend.Device(), &vci, nullptr, &rt.view) !=
        VK_SUCCESS) {
        vmaDestroyImage(g_backend.Allocator(), rt.image, rt.alloc);
        rt.image = VK_NULL_HANDLE;
        rt.alloc = VK_NULL_HANDLE;
        return false;
    }
    rt.format = fmt;
    rt.width = w;
    rt.height = h;
    return true;
}

void PostFx::DestroyTarget(RenderTarget& rt) {
    VkDevice dev = g_backend.Device();
    VmaAllocator alloc = g_backend.Allocator();
    if (rt.view && dev) vkDestroyImageView(dev, rt.view, nullptr);
    if (rt.image) {
        g_backend.ForgetImage(rt.image);
        if (alloc) vmaDestroyImage(alloc, rt.image, rt.alloc);
    }
    rt = RenderTarget{};
}

void PostFx::DestroySceneTargets() {
    DestroyTarget(gtaoRaw_);
    DestroyTarget(gtaoTmp_);
    DestroyTarget(gtaoFinal_);
    DestroyTarget(sceneCopy_);
    allocSceneW_ = 0;
    allocSceneH_ = 0;
    allocSceneFmt_ = VK_FORMAT_UNDEFINED;
}

void PostFx::DestroyBbTargets() {
    DestroyTarget(bbCopy_);
    DestroyTarget(smaaEdges_);
    DestroyTarget(smaaWeights_);
    allocBbW_ = 0;
    allocBbH_ = 0;
    allocBbFmt_ = VK_FORMAT_UNDEFINED;
}

void PostFx::DestroyTargets() {
    DestroySceneTargets();
    DestroyBbTargets();
}

bool PostFx::EnsureSceneTargets(uint32_t sceneW, uint32_t sceneH,
                                VkFormat sceneFmt) {
    if (!sceneW || !sceneH) return false;
    const uint32_t hw = (sceneW + 1u) / 2u;
    const uint32_t hh = (sceneH + 1u) / 2u;

    if (allocSceneW_ == sceneW && allocSceneH_ == sceneH &&
        allocSceneFmt_ == sceneFmt && gtaoRaw_.view && gtaoTmp_.view &&
        gtaoFinal_.view && sceneCopy_.view) {
        return true;
    }

    DestroySceneTargets();
    if (!CreateTarget(gtaoRaw_, hw, hh, VK_FORMAT_R8_UNORM) ||
        !CreateTarget(gtaoTmp_, hw, hh, VK_FORMAT_R8_UNORM) ||
        !CreateTarget(gtaoFinal_, hw, hh, VK_FORMAT_R8_UNORM) ||
        !CreateTarget(sceneCopy_, sceneW, sceneH, sceneFmt,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        DestroySceneTargets();
        return false;
    }

    allocSceneW_ = sceneW;
    allocSceneH_ = sceneH;
    allocSceneFmt_ = sceneFmt;
    return true;
}

bool PostFx::EnsureBbTargets(uint32_t bbW, uint32_t bbH, VkFormat bbFmt) {
    if (!bbW || !bbH) return false;
    if (allocBbW_ == bbW && allocBbH_ == bbH && allocBbFmt_ == bbFmt &&
        bbCopy_.view && smaaEdges_.view && smaaWeights_.view) {
        return true;
    }

    DestroyBbTargets();
    if (!CreateTarget(bbCopy_, bbW, bbH, bbFmt,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !CreateTarget(smaaEdges_, bbW, bbH, VK_FORMAT_R8G8_UNORM) ||
        !CreateTarget(smaaWeights_, bbW, bbH, VK_FORMAT_R8G8B8A8_UNORM)) {
        DestroyBbTargets();
        return false;
    }

    allocBbW_ = bbW;
    allocBbH_ = bbH;
    allocBbFmt_ = bbFmt;
    return true;
}

bool PostFx::EnsureGraphicsPipeline(VkShaderModule fs, VkFormat colorFmt,
                                    VkPipeline& pipe, VkFormat& pipeFmt) {
    if (pipe && pipeFmt == colorFmt) return true;
    VkDevice dev = g_backend.Device();
    if (!dev || !vsFull_ || !fs) return false;
    if (pipe) {
        vkDestroyPipeline(dev, pipe, nullptr);
        pipe = VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vsFull_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
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
    cba.colorWriteMask = 0xFu;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynInfo.dynamicStateCount = 2;
    dynInfo.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &colorFmt;

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

    if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe) !=
        VK_SUCCESS) {
        pipe = VK_NULL_HANDLE;
        return false;
    }
    pipeFmt = colorFmt;
    return true;
}

void PostFx::UpdateSet(uint32_t idx, VkImageView v0, VkSampler s0,
                       VkImageView v1, VkSampler s1, VkImageView v2,
                       VkSampler s2) {
    VkDescriptorImageInfo ii[3]{};
    ii[0].imageView = v0 ? v0 : g_backend.DefaultView();
    ii[0].sampler = s0 ? s0 : nearestSampler_;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[1].imageView = v1 ? v1 : g_backend.DefaultView();
    ii[1].sampler = s1 ? s1 : nearestSampler_;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[2].imageView = v2 ? v2 : g_backend.DefaultView();
    ii[2].sampler = s2 ? s2 : nearestSampler_;
    ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = descSets_[idx];
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(g_backend.Device(), 3, w, 0, nullptr);
}

void PostFx::DrawPass(VkCommandBuffer cmd, VkPipeline pipe, uint32_t setIdx,
                      VkImage dstImg, VkImageView dstView, uint32_t w,
                      uint32_t h, const void* pushData, uint32_t pushBytes) {
    g_backend.TransitionImage(dstImg, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = dstView;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {w, h}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;

    vkCmdBeginRendering(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout_,
                            0, 1, &descSets_[setIdx], 0, nullptr);
    vkCmdPushConstants(cmd, pipeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       pushBytes, pushData);
    VkViewport vp{0.0f, (float)h, (float)w, -(float)h, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, {w, h}};
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
}

bool PostFx::FlushScenePass(FakeSurface* colorOverride,
                            FakeSurface* depthOverride) {
    if (!initialized_ || !g_backend.InRecording() || scenePostDone_) return false;

    FakeSurface* sc = colorOverride ? colorOverride : sceneColor_;
    FakeSurface* sd = depthOverride ? depthOverride : sceneDepth_;
    VkImage scImg = sc ? sc->image : VK_NULL_HANDLE;
    VkImageView scView = sc ? sc->view : VK_NULL_HANDLE;
    VkFormat scFmt =
        sc ? ResourceTable::ToVkFormat(sc->format) : g_backend.SwapchainColorFormat();
    uint32_t fullW = sc ? sc->width : 0u;
    uint32_t fullH = sc ? sc->height : 0u;

    VkImage sdImg = sd ? sd->image : VK_NULL_HANDLE;
    VkImageView sdView = sd ? sd->view : VK_NULL_HANDLE;
    VkFormat sdFmt =
        sd ? ResourceTable::ToVkDepthFormat(sd->format, nullptr) : g_backend.SwapchainDepthFormat();

    if (bbDirect3D_ && (!scImg || !sdImg)) {
        const VkExtent2D sw = g_backend.SwapchainExtent();
        scImg = g_backend.SwapchainImage();
        scView = g_backend.SwapchainView();
        scFmt = g_backend.SwapchainColorFormat();
        fullW = sw.width;
        fullH = sw.height;
        sdImg = g_backend.SwapchainDepthImage();
        sdView = g_backend.SwapchainDepthSampleView();
        sdFmt = g_backend.SwapchainDepthFormat();
    }

    if (!scImg || !scView || !sdImg || !sdView || !fullW || !fullH) return false;
    if (!EnsureSceneTargets(fullW, fullH, scFmt))
        return false;

    if (!EnsureGraphicsPipeline(fsGtao_, VK_FORMAT_R8_UNORM, pipeGtao_, fmtGtao_) ||
        !EnsureGraphicsPipeline(fsBlurH_, VK_FORMAT_R8_UNORM, pipeBlurH_, fmtBlurH_) ||
        !EnsureGraphicsPipeline(fsBlurV_, VK_FORMAT_R8_UNORM, pipeBlurV_, fmtBlurV_) ||
        !EnsureGraphicsPipeline(fsSceneEnh_, scFmt, pipeSceneEnh_, fmtSceneEnh_)) {
        return false;
    }

    scenePostDone_ = true;
    hasScene3D_ = true;
    g_backend.EndRendering();

    const uint32_t slot = (g_backend.FrameSlot() & 1u) * 8u;
    const uint32_t halfW = gtaoRaw_.width;
    const uint32_t halfH = gtaoRaw_.height;
    const float aspect = (float)fullW / (float)fullH;
    const float tanY = 0.45f;
    const float tanX = tanY * aspect;

    VkImageAspectFlags dAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (sdFmt == VK_FORMAT_D24_UNORM_S8_UINT ||
        sdFmt == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        dAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    g_backend.TransitionImage(sdImg, dAspect,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    g_backend.TransitionImage(scImg, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    g_backend.TransitionImage(sceneCopy_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkCommandBuffer cmd = g_backend.Cmd();
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {fullW, fullH, 1};
    vkCmdCopyImage(cmd, scImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   sceneCopy_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &region);

    g_backend.TransitionImage(sceneCopy_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    PostPush push{};
    push.invW = 1.0f / (float)halfW;
    push.invH = 1.0f / (float)halfH;
    push.tanFovX = tanX;
    push.tanFovY = tanY;
    push.zNear = 0.2f;
    push.zFar = 1000.0f;
    push.aspect = aspect;
    push.param = 1.0f;

    UpdateSet(slot + 0u, sdView, nearestSampler_, VK_NULL_HANDLE, VK_NULL_HANDLE,
              VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeGtao_, slot + 0u, gtaoRaw_.image, gtaoRaw_.view, halfW,
             halfH, &push, sizeof(push));

    g_backend.TransitionImage(gtaoRaw_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    UpdateSet(slot + 1u, gtaoRaw_.view, nearestSampler_, sdView, nearestSampler_,
              VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeBlurH_, slot + 1u, gtaoTmp_.image, gtaoTmp_.view, halfW,
             halfH, &push, sizeof(push));

    g_backend.TransitionImage(gtaoTmp_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    UpdateSet(slot + 2u, gtaoTmp_.view, nearestSampler_, sdView, nearestSampler_,
              VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeBlurV_, slot + 2u, gtaoFinal_.image, gtaoFinal_.view,
             halfW, halfH, &push, sizeof(push));

    g_backend.TransitionImage(gtaoFinal_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    push.invW = 1.0f / (float)fullW;
    push.invH = 1.0f / (float)fullH;
    UpdateSet(slot + 3u, sceneCopy_.view, nearestSampler_, gtaoFinal_.view,
              linearSampler_, sdView, nearestSampler_);
    DrawPass(cmd, pipeSceneEnh_, slot + 3u, scImg, scView, fullW, fullH, &push,
             sizeof(push));

    VkImageMemoryBarrier flushB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    flushB.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    flushB.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    flushB.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    flushB.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    flushB.image = scImg;
    flushB.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &flushB);

    return true;
}

bool PostFx::FlushBeforeDepthClear(FakeSurface* depth) {
    if (!NeedsSceneFlush()) return false;
    if (!depth || depth->image != sceneDepth_->image) return false;
    return FlushScenePass();
}

bool PostFx::FlushIfReading(FakeSurface* texSurf) {
    if (!NeedsSceneFlush() || !texSurf) return false;
    if (texSurf->image != sceneColor_->image) return false;
    return FlushScenePass();
}

bool PostFx::FlushBeforeHud() {
    if (!initialized_ || !g_backend.InRecording() || !hasScene3D_ ||
        bbPostDone_) {
        return false;
    }
    if (NeedsSceneFlush()) {
        FlushScenePass();
    }

    VkImage bbImg = g_backend.SwapchainImage();
    VkImageView bbView = g_backend.SwapchainView();
    const VkExtent2D sw = g_backend.SwapchainExtent();
    const VkFormat bbFmt = g_backend.SwapchainColorFormat();
    if (!bbImg || !bbView || !sw.width || !sw.height ||
        !g_backend.SwapchainTransferSrc()) {
        bbPostDone_ = true;
        return false;
    }

    if (!EnsureBbTargets(sw.width, sw.height, bbFmt)) {
        return false;
    }

    if (!EnsureGraphicsPipeline(fsSmaaEdge_, VK_FORMAT_R8G8_UNORM,
                                pipeSmaaEdge_, fmtSmaaEdge_) ||
        !EnsureGraphicsPipeline(fsSmaaWeight_, VK_FORMAT_R8G8B8A8_UNORM,
                                pipeSmaaWeight_, fmtSmaaWeight_) ||
        !EnsureGraphicsPipeline(fsSmaaBlend_, bbFmt, pipeSmaaBlend_,
                                fmtSmaaBlend_)) {
        return false;
    }

    bbPostDone_ = true;
    g_backend.EndRendering();

    g_backend.TransitionImage(bbImg, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    g_backend.TransitionImage(bbCopy_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkCommandBuffer cmd = g_backend.Cmd();
    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {sw.width, sw.height, 1};
    vkCmdCopyImage(cmd, bbImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   bbCopy_.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &region);

    g_backend.TransitionImage(bbCopy_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    const uint32_t slot = (g_backend.FrameSlot() & 1u) * 8u;
    PostPush push{};
    push.invW = 1.0f / (float)sw.width;
    push.invH = 1.0f / (float)sw.height;
    push.tanFovX = 0.45f * ((float)sw.width / (float)sw.height);
    push.tanFovY = 0.45f;
    push.zNear = 0.2f;
    push.zFar = 1000.0f;
    push.aspect = (float)sw.width / (float)sw.height;
    push.param = 1.0f;

    UpdateSet(slot + 4u, bbCopy_.view, nearestSampler_, VK_NULL_HANDLE,
              VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeSmaaEdge_, slot + 4u, smaaEdges_.image, smaaEdges_.view,
             sw.width, sw.height, &push, sizeof(push));

    g_backend.TransitionImage(smaaEdges_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    UpdateSet(slot + 5u, smaaEdges_.view, linearSampler_, VK_NULL_HANDLE,
              VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeSmaaWeight_, slot + 5u, smaaWeights_.image,
             smaaWeights_.view, sw.width, sw.height, &push, sizeof(push));

    g_backend.TransitionImage(smaaWeights_.image, VK_IMAGE_ASPECT_COLOR_BIT,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    UpdateSet(slot + 6u, bbCopy_.view, linearSampler_, smaaWeights_.view,
              nearestSampler_, VK_NULL_HANDLE, VK_NULL_HANDLE);
    DrawPass(cmd, pipeSmaaBlend_, slot + 6u, bbImg, bbView, sw.width, sw.height,
             &push, sizeof(push));

    VkImageMemoryBarrier flushB{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    flushB.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    flushB.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    flushB.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    flushB.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    flushB.image = bbImg;
    flushB.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &flushB);

    return true;
}

bool PostFx::FlushEndOfFrame() {
    bool did = false;
    if (NeedsSceneFlush()) {
        did = FlushScenePass();
    }
    if (hasScene3D_ && !bbPostDone_ && (bbCompositeSeen_ || bbDirect3D_)) {
        did = FlushBeforeHud() || did;
    }
    return did;
}
