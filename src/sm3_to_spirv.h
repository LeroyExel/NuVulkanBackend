#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Compiles a raw D3D9 shader token stream (vs_3_0 / ps_3_0, the DWORD array the
// engine hands to CreateVertexShader/CreatePixelShader) into SPIR-V 1.3.
//
// Descriptor ABI produced by this compiler (must stay in step with
// vk_backend.cpp and pso_cache.cpp):
//   set 0, binding 0      : std430 vec4 c[256]   vertex constant file
//   set 0, binding 1      : std430 vec4 c[256]   pixel constant file
//   set 0, binding 2 + N  : combined image sampler for SM3 sampler sN (2..17)
//
// One binding per sampler instead of a descriptor array: a SPIR-V array is
// homogeneous, and OpTypeImage dimensionality cannot vary across elements, yet
// s0 may be a sampler2D while s1 is a samplerCube in the same shader.
//
// Vertex and varying locations come from Sm3LocationFor(usage, usageIndex),
// which keeps every location below the 32-attribute interface limit;
// resources.cpp maps D3DVERTEXELEMENT9 through the same function.
bool Sm3CompileToSpirv(bool pixel, const uint32_t* tokens, uint32_t dwordCount,
                       std::vector<uint32_t>& out, std::string& error,
                       uint32_t drefMask = 0);

uint32_t Sm3TexldpSamplerMask(const uint32_t* tokens, uint32_t dwordCount);

// Vulkan interface location for one D3D9 declaration (D3DDECLUSAGE, index).
// Compact instead of usage * 4 + index: COLOR0 would otherwise be location 40,
// past the device's 32 attributes, and drawing with it faults the driver.
// Supports usage indices 0..3.
uint32_t Sm3LocationFor(uint32_t usage, uint32_t usageIndex);

// True when the token stream is a D3D9 SM3 blob this compiler understands.
bool Sm3IsSupported(const uint32_t* tokens, uint32_t dwordCount);

// Total DWORDs occupied by the instruction whose opcode token is `head`,
// including the opcode token itself.
//
// Per the D3D9 instruction-token layout (D3DSI_INSTLENGTH_MASK in d3d9types.h
// and the DDK "Instruction Token" page): [15:0] opcode, [23:16] opcode-specific
// controls (the texld project/bias bits live here), [27:24] the number of
// parameter tokens that follow the opcode token, NOT counting the token itself,
// and bit 28 = predicated, which appends one trailing predicate source token.
// D3DSIO_COMMENT is the exception: its payload length sits in [30:16] and is
// likewise exclusive of the token.
inline uint32_t Sm3InstructionDwords(uint32_t head) {
    if ((head & 0xFFFFu) == 0xFFFEu)  // D3DSIO_COMMENT
        return 1u + ((head >> 16) & 0x7FFFu);
    uint32_t n = 1u + ((head >> 24) & 0x0Fu);
    if (head & 0x10000000u) ++n;  // D3DSI_PREDICATED
    return n;
}
