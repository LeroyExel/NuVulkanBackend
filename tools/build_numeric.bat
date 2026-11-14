@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /std:c++20 /EHsc /W3 /I src /I vendor\Vulkan-Headers\include /D VK_USE_PLATFORM_WIN32_KHR /D VK_NO_PROTOTYPES /D NOMINMAX /D WIN32_LEAN_AND_MEAN tools\vs_numeric.cpp src\sm3_to_spirv.cpp /Fo:build\ /Fe:build\vs_numeric.exe
if errorlevel 1 exit /b 1
build\vs_numeric.exe
