# NuVulkanBackend

A native Vulkan 1.3 replacement for the Direct3D 9 device inside
*The LEGO Movie - Videogame* (`LEGOEMMET.exe`, Traveller's Tales NuEngine).
No DXVK, no WineD3D, no Gallium Nine — the D3D9 device is never created against
the driver at all.

## How the interception works

`Direct3D9` is statically imported by the executable, and the import table has
exactly one D3D9 entry point (`Direct3DCreate9`, IAT RVA `0x00C49360`). The DLL:

1. patches that IAT slot to `Hook_Direct3DCreate9`;
2. returns a fake `IDirect3D9` whose `CreateDevice` (byte offset 64, verified)
   initialises Vulkan on the `hFocusWindow` the engine passes in;
3. returns a fake `IDirect3DDevice9` whose vtable the DLL owns, so **every**
   D3D9 call the engine makes lands in this code. The engine's renderer
   singleton (`0x01980F78`, `docs/stage1_recon.md`) stores our pointer at
   `+0x004` and never notices.

Because the vtable is ours, coverage is exact by construction — no call-site
scanning, no function detours that can miss an inlined path.

## Vtable safety

Slot indices and argument counts are **generated** from the Windows SDK
`d3d9.h` by `tools/gen_d3d9_slots.py` into `src/d3d9_slots.h`.

This is not ceremony. Hand-transcribing the table produced `DrawIndexedPrimitive
= 62`; the real slot is **82**. A wrong index with a wrong arity corrupts the
stack on the first draw. `src/d3d9_vtable.h` `static_assert`s the four offsets
observed inside the binary (`+12`, `+20`, `+28`, `+64`) against the generated
table, so a drift in either direction fails the build instead of the game.

## Subsystems

| File | Responsibility |
|---|---|
| `src/dllmain.cpp` | IAT hook, log, `Direct3DCreate9` proxy exports |
| `src/d3d9_factory.cpp` | fake `IDirect3D9`, device creation, swap chain/query objects |
| `src/d3d9_device.cpp` | `IDirect3DDevice9` vtable, state, constants, resources, state blocks |
| `src/d3d9_draw.cpp` | pipeline key, draw translation, render-target selection |
| `src/d3d9_internal.h` | shared device state and internal declarations |
| `src/vk_context.cpp` | instance, device, Win32 surface, swapchain, 2 frames in flight, dynamic rendering |
| `src/constants.cpp` | 257-register VS/PS blocks in a persistently mapped dynamic uniform ring |
| `src/resources.cpp` | engine handle → `VkBuffer`/`VkImage`/`VmaAllocation`, COM vtables, lock/unlock upload |
| `src/shaders.cpp` | XXH64 capture and `.sm3` dump of every engine shader, `.spv` cache |
| `src/sm3_to_spirv.cpp` | SM3 token stream → SPIR-V 1.3 |
| `src/pso_cache.cpp` | XXH64 state hash → `VkPipeline` via dynamic rendering |

Constant registers are exact: `c0..c255` is 256 × 16 B = **4096 bytes**, plus a
zeroed `c[256]` padding register for out-of-range relative addressing. A draw
takes two consecutive aligned blocks with dynamic offsets `{vs, ps}`.

## Build

```sh
cmake -S . -B build -A Win32
cmake --build build --config Release
```

Requires MSVC (x86) and CMake. Vulkan headers, VMA, MinHook and xxHash are
vendored under `vendor/`; the DLL loads `vulkan-1.dll` dynamically and needs no
SDK import library.

## Run

```powershell
# start the game, then:
tools\inject.ps1 -Process LEGOEMMET
```

`NuVulkanBackend.log` is written next to the DLL. Shader bytecode lands in
`shaders\*.dxbc` beside the game executable.

## Status

Verified:

* builds as 32-bit PE (x86) against Vulkan 1.3 headers;
* generated vtable satisfies all four offsets the binary actually calls;
* `Direct3DCreate9` IAT RVA and the renderer singleton offsets are read from
  the database, not assumed.

Implemented but **not yet exercised in-game**:

* swapchain takeover, dynamic-rendering frame loop, clears, present;
* buffer/texture upload, constant ring, PSO cache, draw translation.

Not implemented:

* `SetFVF` vertex-declaration synthesis (shader draws use explicit
  declarations);
* two-sided stencil and fixed-function clip planes;
* SM1/SM2 token streams (the engine compiles everything to SM3).
