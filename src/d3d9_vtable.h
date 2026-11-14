#pragma once
#include <cstdint>

#include "d3d9_slots.h"  // generated from the Windows SDK d3d9.h

// Cross-checks against the wheel binary's actual indirect calls
// (docs/stage1_recon.md):
//   IDirect3DDevice9 +12  -> slot 3  TestCooperativeLevel
//   IDirect3DDevice9 +20  -> slot 5  EvictManagedResources
//   IDirect3DDevice9 +28  -> slot 7  GetDeviceCaps
//   IDirect3DDevice9 +64  -> slot 16 Reset
//   IDirect3D9       +20  -> slot 5  GetAdapterIdentifier
//   IDirect3D9       +56  -> slot 14 GetDeviceCaps
//   IDirect3D9       +64  -> slot 16 CreateDevice
// The generated indices satisfy every one of these, which is why the vtable the
// shim hands back is layout-compatible with what the engine indexes into.

namespace d3d9 {

constexpr uint32_t kOffDeviceTestCooperativeLevel = kTestCooperativeLevel * 4;
constexpr uint32_t kOffDeviceEvictManagedResources = kEvictManagedResources * 4;
constexpr uint32_t kOffDeviceGetDeviceCaps = kGetDeviceCaps * 4;
constexpr uint32_t kOffDeviceReset = kReset * 4;
constexpr uint32_t kOffDevicePresent = kPresent * 4;
constexpr uint32_t kOffDeviceDrawIndexed = kDrawIndexedPrimitive * 4;

constexpr uint32_t kOffFactoryGetAdapterIdentifier = kD3D9GetAdapterIdentifier * 4;
constexpr uint32_t kOffFactoryGetDeviceCaps = kD3D9GetDeviceCaps * 4;
constexpr uint32_t kOffFactoryCreateDevice = kD3D9CreateDevice * 4;

static_assert(kOffDeviceReset == 64, "Reset must be at byte offset 64");
static_assert(kOffDevicePresent == 68, "Present must be at byte offset 68");
static_assert(kBeginScene == 41, "BeginScene must be slot 41");
static_assert(kDrawIndexedPrimitive == 82, "DrawIndexedPrimitive must be slot 82");
static_assert(kSetVertexShaderConstantF == 94,
              "SetVertexShaderConstantF must be slot 94");
static_assert(kOffDeviceGetDeviceCaps == 28, "GetDeviceCaps must be at offset 28");
static_assert(kOffFactoryCreateDevice == 64, "CreateDevice must be at offset 64");
static_assert(kOffFactoryGetAdapterIdentifier == 20, "GetAdapterIdentifier offset");

// Render states used by the pipeline key.
enum : uint32_t {
    RS_ZENABLE = 7,
    RS_FILLMODE = 8,
    RS_SHADEMODE = 9,
    RS_ZWRITEENABLE = 14,
    RS_ALPHATESTENABLE = 15,
    RS_SRCBLEND = 19,
    RS_DESTBLEND = 20,
    RS_CULLMODE = 22,
    RS_ZFUNC = 23,
    RS_ALPHAREF = 24,
    RS_ALPHAFUNC = 25,
    RS_ALPHABLENDENABLE = 27,
    RS_FOGENABLE = 28,
    RS_SPECULARENABLE = 29,
    RS_STENCILENABLE = 52,
    RS_STENCILFAIL = 53,
    RS_STENCILZFAIL = 54,
    RS_STENCILPASS = 55,
    RS_STENCILFUNC = 56,
    RS_STENCILREF = 57,
    RS_STENCILMASK = 58,
    RS_STENCILWRITEMASK = 59,
    RS_COLORWRITEENABLE = 168,
    RS_BLENDOP = 171,
    RS_SCISSORTESTENABLE = 174,
    RS_TWOSIDEDSTENCILMODE = 185,
    RS_CCW_STENCILFAIL = 186,
    RS_CCW_STENCILZFAIL = 187,
    RS_CCW_STENCILPASS = 188,
    RS_CCW_STENCILFUNC = 189,
    RS_SEPARATEALPHABLENDENABLE = 206,
    RS_SRCBLENDALPHA = 207,
    RS_DESTBLENDALPHA = 208,
    RS_BLENDOPALPHA = 209,
    RS_SRGBWRITEENABLE = 212,
    RS_COUNT = 256,
};

}  // namespace d3d9
