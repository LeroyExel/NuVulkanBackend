cmake_minimum_required(VERSION 3.20)
project(NuVulkanBackend CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

set(VK_INC  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/Vulkan-Headers/include")
set(VMA_INC "${CMAKE_CURRENT_SOURCE_DIR}/vendor/VulkanMemoryAllocator/include")
set(XXH_INC "${CMAKE_CURRENT_SOURCE_DIR}/vendor/xxHash")

add_library(NuVulkanBackend SHARED
    src/dllmain.cpp
    src/vk_common.cpp
    src/vk_context.cpp
    src/constants.cpp
    src/resources.cpp
    src/shaders.cpp
    src/sm3_to_spirv.cpp
    src/pso_cache.cpp
    src/fps_overlay.cpp
    src/stud_shadows.cpp
    src/post_fx.cpp
    src/d3d9_device.cpp
    src/d3d9_draw.cpp
    src/d3d9_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NuVulkanBackend.def)

target_include_directories(NuVulkanBackend PRIVATE
    "${VK_INC}" "${VMA_INC}" "${XXH_INC}" src)

target_compile_definitions(NuVulkanBackend PRIVATE
    VK_USE_PLATFORM_WIN32_KHR
    VK_NO_PROTOTYPES
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    UNICODE
    _UNICODE)

target_compile_options(NuVulkanBackend PRIVATE /W3)

target_link_libraries(NuVulkanBackend PRIVATE user32)

# 32-bit build: configure with -A Win32
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "NuVulkanBackend must be built as 32-bit (use -A Win32)")
endif()

set_target_properties(NuVulkanBackend PROPERTIES
    PREFIX ""
    OUTPUT_NAME "NuVulkanBackend")
