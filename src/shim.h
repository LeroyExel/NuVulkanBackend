#pragma once
#include "vk_common.h"

bool ShimInit();
void ShimShutdown();

// Replacement for d3d9!Direct3DCreate9.
void* __stdcall ShimDirect3DCreate9(uint32_t sdkVersion);
