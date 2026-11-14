// End-to-end case: the real captured world vertex shader.
//
// The transform is measured, not assumed. The register holding the shader's
// final position write is copied into a spare output ("probe"), and only then
// is o0 forced onto the pixel, so what comes back is the clip position the
// shader itself computed from the real stream and constants.
//
// The transform is affine in the vertex position, so the origin and three basis
// points recover the whole matrix the shader applies; that measured matrix is
// what gets compared against the CPU replay.

static const char* kDefaultShader =
    "../The LEGO Movie - Videogame/shaders/v32_1621ff3e015efbe3_vs.sm3";
static const uint32_t kProbePosConst = 250;  // not read by this shader's maths

struct Dcl {
    uint32_t usage, index, reg;
};

static bool LoadTokens(const char* path, std::vector<uint32_t>& out) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) return false;
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes < 12 || (bytes % 4)) {
        fclose(f);
        return false;
    }
    out.resize((size_t)bytes / 4);
    const size_t got = fread(out.data(), 4, out.size(), f);
    fclose(f);
    return got == out.size();
}

static std::vector<Dcl> ParseDcls(const std::vector<uint32_t>& t) {
    std::vector<Dcl> out;
    for (size_t i = 1; i + 2 < t.size();) {
        const uint32_t wc = Sm3InstructionDwords(t[i]), op = t[i] & 0xFFFFu;
        if (!wc) break;
        if (op == OPC_END) break;
        if (op == OPC_DCL && wc >= 3) {
            const uint32_t decl = t[i + 1];
            out.push_back({decl & 0x1Fu, (decl >> 16) & 0xFu, t[i + 2] & 0x7FFu});
        }
        i += wc;
    }
    return out;
}

// Start index of every instruction that writes the position output o0. Any
// instruction form counts (mov, mad, dp4, a masked partial write), which is why
// the probe retargets these writes instead of reading a temp register.
static std::vector<size_t> PositionWrites(const std::vector<uint32_t>& t) {
    std::vector<size_t> out;
    for (size_t i = 1; i + 1 < t.size();) {
        const uint32_t wc = Sm3InstructionDwords(t[i]), op = t[i] & 0xFFFFu;
        if (!wc) break;
        if (op == OPC_END) break;
        if (wc >= 3) {
            const uint32_t dst = t[i + 1];
            const uint32_t dt = ((dst & 0x70000000u) >> 28) | ((dst & 0x1800u) >> 11);
            if ((dt == RG_OUTPUT || dt == RG_RASTOUT) && (dst & 0x7FFu) == 0)
                out.push_back(i);
        }
        i += wc;
    }
    return out;
}

// Register the shader reads when it writes o0 (used when the final position
// write is a plain register move).
static long long FinalPositionSource(const std::vector<uint32_t>& t) {
    long long reg = -1;
    for (size_t i = 1; i + 2 < t.size();) {
        const uint32_t wc = Sm3InstructionDwords(t[i]), op = t[i] & 0xFFFFu;
        if (!wc) break;
        if (op == OPC_END) break;
        if (wc == 3) {
            const uint32_t dst = t[i + 1];
            const uint32_t dt = ((dst & 0x70000000u) >> 28) | ((dst & 0x1800u) >> 11);
            if (dt == RG_OUTPUT && (dst & 0x7FFu) == 0) {
                const uint32_t src = t[i + 2];
                const uint32_t st = ((src & 0x70000000u) >> 28) | ((src & 0x1800u) >> 11);
                if (st == RG_TEMP) reg = src & 0x7FFu;
            }
        }
        i += wc;
    }
    return reg;
}

static bool MakeVertexBuffer(Gpu& g, const void* data, size_t bytes, VkBuffer& buf,
                             VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (g.CreateBuffer(g.dev, &bci, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    g.GetBufferMemoryRequirements(g.dev, buf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = PickMemory(g, req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX ||
        g.AllocateMemory(g.dev, &mai, nullptr, &mem) != VK_SUCCESS)
        return false;
    g.BindBufferMemory(g.dev, buf, mem, 0);
    void* map = nullptr;
    g.MapMemory(g.dev, mem, 0, bytes, 0, &map);
    memcpy(map, data, bytes);
    g.UnmapMemory(g.dev, mem);
    return true;
}

static bool LoadCapture(const char* path, std::vector<float>& block, uint64_t& vsHash) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) return false;
    uint64_t hdr[4] = {};
    block.assign(1024, 0.0f);
    bool ok = fread(hdr, sizeof(hdr), 1, f) == 1;
    if (ok) {
        // A capture that also carries the draw's real vertex and index bytes
        // (see CaptureDraw) has an eight-word meta block before the constants;
        // the older constants-only fixtures start the constants right away.
        const long here = ftell(f);
        fseek(f, 0, SEEK_END);
        const long size = ftell(f);
        fseek(f, here, SEEK_SET);
        if (size > 32 + 16384) {
            int32_t m[8] = {};
            ok = fread(m, sizeof(m), 1, f) == 1;
            if (ok) {
                printf("  capture draw: baseVertex=%d startIndex=%d indexFmt=%d "
                       "streamOffset=%d indices=%d maxIndex=%d attrs=%d otherStreams=%d",
                       m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
                puts("");
            }
        }
        ok = ok && fread(block.data(), sizeof(float), 1024, f) == 1024;
    }
    fclose(f);
    vsHash = hdr[0];
    return ok;
}

static int RunRealCase(Gpu& g, const char* path, const std::vector<float>* capBlock) {
    std::vector<uint32_t> toks;
    if (!LoadTokens(path, toks)) {
        printf("FAIL real world shader: cannot read %s", path);
        puts("");
        return 1;
    }
    const std::vector<Dcl> dcls = ParseDcls(toks);
    if (dcls.empty()) {
        puts("FAIL real world shader: no declarations");
        return 1;
    }

    uint32_t maxOut = 0, maxTex = 0;
    bool hasPos = false;
    for (const Dcl& d : dcls) {
        if (d.reg > maxOut) maxOut = d.reg;
        if (d.usage == USAGE_TEXCOORD && d.index > maxTex) maxTex = d.index;
        if (d.usage == USAGE_POSITION) hasPos = true;
    }
    if (!hasPos) {
        puts("FAIL real world shader: no position declaration");
        return 1;
    }
    const uint32_t texIndex = maxTex + 1;
    const uint32_t probeReg = maxOut + 1;
    const std::vector<size_t> posWrites = PositionWrites(toks);
    if (texIndex > 7 || probeReg > 11 || posWrites.empty()) {
        printf("FAIL real world shader: cannot instrument tex=%u probe=%u writes=%u",
               texIndex, probeReg, (unsigned)posWrites.size());
        puts("");
        return 1;
    }

    // Split the stream at the first declaration and at END. The probe goes
    // immediately before the shader's own declarations rather than before the
    // def block: a declaration placed ahead of the defs is not picked up.
    size_t declStart = toks.size(), endIdx = toks.size();
    for (size_t i = 1; i < toks.size();) {
        const uint32_t wc = Sm3InstructionDwords(toks[i]), op = toks[i] & 0xFFFFu;
        if (!wc) break;
        if (op == OPC_END) {
            endIdx = i;
            break;
        }
        if (op == OPC_DCL && declStart == toks.size()) declStart = i;
        i += wc;
    }
    if (declStart == toks.size()) declStart = 1;

    std::vector<uint32_t> probe(toks.begin(), toks.begin() + (long)declStart);
    probe.push_back(Inst(OPC_DCL, 2));
    probe.push_back(Decl(USAGE_TEXCOORD, texIndex));
    probe.push_back(Dst(RG_OUTPUT, probeReg));
    probe.insert(probe.end(), toks.begin() + (long)declStart, toks.begin() + (long)endIdx);
    // Two ways to observe the position. When the last position write is a plain
    // register move, copy that register after the fact (some shaders build the
    // position in stages, and retargeting every write disturbs them). Otherwise
    // retarget the writes themselves, keeping each write mask; the register type
    // becomes OUTPUT because a RASTOUT write renumbered alone lands on an
    // unwritten register and reads back as zero.
    const long long posSrc = FinalPositionSource(toks);
    if (posSrc >= 0) {
        probe.push_back(Inst(OPC_MOV, 2));
        probe.push_back(Dst(RG_OUTPUT, probeReg));
        probe.push_back(Src(RG_TEMP, (uint32_t)posSrc));
    } else {
        const size_t kInsertWords = 3;
        for (size_t w : posWrites) {
            if (w >= declStart && w < endIdx) {
                const uint32_t mask = (probe[w + kInsertWords] >> 16) & 0xFu;
                const uint32_t shift = (probe[w + kInsertWords] >> 24) & 0xFu;
                probe[w + kInsertWords] = Dst(RG_OUTPUT, probeReg, mask, shift);
            }
        }
    }
    probe.push_back(Inst(OPC_MOV, 2));  // o0 = forced on-screen position
    probe.push_back(Dst(RG_OUTPUT, 0));
    probe.push_back(Src(RG_CONST, kProbePosConst));
    probe.push_back(OPC_END);

    std::vector<uint32_t> vsSpv, psSpv;
    std::string cerr;
    if (!Sm3CompileToSpirv(false, probe.data(), (uint32_t)probe.size(), vsSpv, cerr)) {
        printf("FAIL real world shader: compile: %s", cerr.c_str());
        puts("");
        return 1;
    }
    std::vector<uint32_t> psTokens = MakePs(texIndex);
    if (!Sm3CompileToSpirv(true, psTokens.data(), (uint32_t)psTokens.size(), psSpv, cerr)) {
        printf("FAIL real world shader: PS compile: %s", cerr.c_str());
        puts("");
        return 1;
    }

    // Constant block in the layout this shader reads (from its CTAB):
    // c0 viewProj, c4 view, c44 worldView, c48 world, c250 the forced position.
    DrawCfg cfg;
    cfg.consts.assign(1024, 0.0f);
    auto setRow = [&](uint32_t r, float a, float b, float c, float d) {
        cfg.consts[r * 4 + 0] = a;
        cfg.consts[r * 4 + 1] = b;
        cfg.consts[r * 4 + 2] = c;
        cfg.consts[r * 4 + 3] = d;
    };
    const float zn = 0.15f, zf = 1000.0f;
    const float sy = 1.0f / tanf(30.0f * 3.14159265358979f / 180.0f);
    const float sx = sy / 1.7777778f;
    setRow(0, sx, 0, 0, 0);
    setRow(1, 0, sy, 0, 0);
    setRow(2, 0, 0, zf / (zf - zn), 1.0f);
    setRow(3, 0, 0, -zn * zf / (zf - zn), 0);
    setRow(4, 1, 0, 0, 0);
    setRow(5, 0, 1, 0, 0);
    setRow(6, 0, 0, 1, 0);
    setRow(7, 0, 0, -15.0f, 1);
    setRow(44, 2, 0, 0, 0);  // world then view, premultiplied
    setRow(45, 0, 2, 0, 0);
    setRow(46, 0, 0, 2, 0);
    setRow(47, 5, -3, 25, 1);
    setRow(48, 2, 0, 0, 0);  // world
    setRow(49, 0, 2, 0, 0);
    setRow(50, 0, 0, 2, 0);
    setRow(51, 5, -3, 40, 1);
    // Everything else this shader reads (fog, projection params, material and
    // misc groups) is given a neutral 1 so no multiply collapses the transform
    // to zero. A register left at 0 here zeroes the position, which is exactly
    // what the first measurement showed.
    for (uint32_t r : {9u, 10u, 17u, 23u, 24u, 52u, 53u, 54u, 73u})
        setRow(r, 1, 1, 1, 1);
    setRow(kProbePosConst, 0, 0, 0.5f, 1);
    if (capBlock) {
        // Real captured block: it replaces every register, then the probe's own
        // position is restored so the point still lands on the pixel.
        cfg.consts = *capBlock;
        setRow(kProbePosConst, 0, 0, 0.5f, 1);
    }

    // One interleaved stream covering every attribute the shader declares, so no
    // fetch reads outside the buffer.
    std::vector<float> vdata(4 * dcls.size(), 0.0f);
    std::vector<uint32_t> usedLoc;
    for (uint32_t a = 0; a < dcls.size(); ++a) {
        float* v = &vdata[a * 4];
        if (dcls[a].usage == USAGE_POSITION) {
            v[0] = 0.0f;
            v[1] = 0.0f;
            v[2] = 0.0f;
            v[3] = 1.0f;
        }
        const uint32_t loc = Sm3LocationFor(dcls[a].usage, dcls[a].index);
        bool dup = false;
        for (uint32_t l : usedLoc) dup |= (l == loc);
        if (dup) continue;
        usedLoc.push_back(loc);
        VkVertexInputAttributeDescription ad{};
        ad.location = loc;
        ad.binding = 0;
        ad.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ad.offset = a * 16;
        cfg.attrs.push_back(ad);
    }
    cfg.vstride = (uint32_t)(16 * dcls.size());

    auto runAt = [&](float ax, float ay, float az, float* out) -> bool {
        std::vector<float> data = vdata;
        for (uint32_t a = 0; a < dcls.size(); ++a)
            if (dcls[a].usage == USAGE_POSITION) {
                data[a * 4 + 0] = ax;
                data[a * 4 + 1] = ay;
                data[a * 4 + 2] = az;
                data[a * 4 + 3] = 1.0f;
            }
        VkBuffer b = VK_NULL_HANDLE;
        VkDeviceMemory m = VK_NULL_HANDLE;
        if (!MakeVertexBuffer(g, data.data(), data.size() * 4, b, m)) return false;
        DrawCfg rc = cfg;
        rc.vbuf = b;
        std::string e;
        const bool ok = RunCase(g, vsSpv, psSpv, rc, out, e);
        g.DestroyBuffer(g.dev, b, nullptr);
        g.FreeMemory(g.dev, m, nullptr);
        return ok;
    };

    float origin[4], axisX[4], axisY[4], axisZ[4];
    if (!runAt(0, 0, 0, origin) || !runAt(1, 0, 0, axisX) || !runAt(0, 1, 0, axisY) ||
        !runAt(0, 0, 1, axisZ)) {
        puts("FAIL real world shader: draw failed");
        return 1;
    }
    float gpuM[4][4] = {};
    for (int k = 0; k < 4; ++k) {
        gpuM[3][k] = origin[k];
        gpuM[0][k] = axisX[k] - origin[k];
        gpuM[1][k] = axisY[k] - origin[k];
        gpuM[2][k] = axisZ[k] - origin[k];
    }

    // Candidate compositions, each built from the captured constant rows:
    // world -> view -> viewProj, world -> viewProj, and premultiplied
    // worldView -> viewProj. Whichever matches the measured matrix identifies
    // the path the shader actually takes.
    auto clipWith = [&](int worldBase, int midBase, float ax, float ay, float az,
                        float* out) {
        const float p[4] = {ax, ay, az, 1.0f};
        auto apply = [&](const float* in, int base, float* o) {
            float m[4][4];
            for (int r = 0; r < 4; ++r)
                for (int k = 0; k < 4; ++k) m[r][k] = cfg.consts[(base + r) * 4 + k];
            for (int k = 0; k < 4; ++k)
                o[k] = in[0] * m[0][k] + in[1] * m[1][k] + in[2] * m[2][k] + in[3] * m[3][k];
        };
        float a[4], b[4];
        if (worldBase >= 0) {
            apply(p, worldBase, a);
        } else {
            for (int k = 0; k < 4; ++k) a[k] = p[k];
        }
        if (midBase >= 0) {
            apply(a, midBase, b);
        } else {
            for (int k = 0; k < 4; ++k) b[k] = a[k];
        }
        apply(b, 0, out);
    };

    struct Cand {
        const char* name;
        int world;
        int mid;
    };
    const Cand cands[3] = {{"world c48 -> view c4 -> viewProj c0", 48, 4},
                           {"world c48 -> viewProj c0", 48, -1},
                           {"worldView c44 -> viewProj c0", 44, -1}};

    float refM[4][4] = {};
    int matched = -1;
    for (int i = 0; i < 3 && matched < 0; ++i) {
        float o0[4], ox[4], oy[4], oz[4];
        clipWith(cands[i].world, cands[i].mid, 0, 0, 0, o0);
        clipWith(cands[i].world, cands[i].mid, 1, 0, 0, ox);
        clipWith(cands[i].world, cands[i].mid, 0, 1, 0, oy);
        clipWith(cands[i].world, cands[i].mid, 0, 0, 1, oz);
        for (int k = 0; k < 4; ++k) {
            refM[3][k] = o0[k];
            refM[0][k] = ox[k] - o0[k];
            refM[1][k] = oy[k] - o0[k];
            refM[2][k] = oz[k] - o0[k];
        }
        bool eq = true;
        for (int r = 0; r < 4; ++r) eq = eq && Same(refM[r], gpuM[r]);
        if (eq) matched = i;
    }

    if (capBlock) {
        puts("    captured instance world (c48..c51) and camera viewProj (c0..c3)");
        for (int r = 0; r < 4; ++r) {
            printf("      world  row%d = (% 9.4f % 9.4f % 9.4f % 9.4f)", r,
                   cfg.consts[(48 + r) * 4 + 0], cfg.consts[(48 + r) * 4 + 1],
                   cfg.consts[(48 + r) * 4 + 2], cfg.consts[(48 + r) * 4 + 3]);
            puts("");
        }
        for (int r = 0; r < 4; ++r) {
            printf("      viewProj row%d = (% 9.4f % 9.4f % 9.4f % 9.4f)", r,
                   cfg.consts[r * 4 + 0], cfg.consts[r * 4 + 1], cfg.consts[r * 4 + 2],
                   cfg.consts[r * 4 + 3]);
            puts("");
        }
    }

    puts("    measured transform (rows: x y z 1 basis)");
    for (int r = 0; r < 4; ++r) {
        printf("      gpu row%d = (% 9.4f % 9.4f % 9.4f % 9.4f)", r, gpuM[r][0], gpuM[r][1],
               gpuM[r][2], gpuM[r][3]);
        puts("");
    }
    if (matched >= 0) {
        printf("    cpu replay matched candidate %d: %s", matched + 1, cands[matched].name);
        puts("");
    } else {
        puts("    cpu replay: no candidate matched (candidate 1 shown)");
        for (int r = 0; r < 4; ++r) {
            printf("      cpu row%d = (% 9.4f % 9.4f % 9.4f % 9.4f)", r, refM[r][0],
                   refM[r][1], refM[r][2], refM[r][3]);
            puts("");
        }
    }

    if (matched < 0) {
        printf("FAIL %-30s no candidate composition matches the measurement",
               "real world shader");
        puts("");
        return 1;
    }
    printf("PASS %-30s %s", "real world shader", cands[matched].name);
    puts("");
    return 0;
}
