# Injects NuVulkanBackend.dll into a running game process via CreateRemoteThread.
# Usage:  .\inject.ps1 -Process LEGOEMMET
param(
    [string]$Process = "LEGOEMMET",
    [string]$Dll = (Join-Path (Split-Path -Parent $PSScriptRoot) "build\Release\NuVulkanBackend.dll")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Dll)) {
    throw "DLL not found: $Dll`nBuild it first: cmake --build build --config Release"
}
$Dll = (Resolve-Path $Dll).Path

$target = Get-Process -Name $Process -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $target) { throw "Process '$Process' is not running." }

Add-Type -Namespace NuVk -Name Native -MemberDefinition @"
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr OpenProcess(uint dwDesiredAccess, bool bInheritHandle, uint dwProcessId);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr VirtualAllocEx(IntPtr hProcess, IntPtr lpAddress, uint dwSize, uint flAllocationType, uint flProtect);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool WriteProcessMemory(IntPtr hProcess, IntPtr lpBaseAddress, byte[] lpBuffer, uint nSize, out UIntPtr lpNumberOfBytesWritten);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr GetModuleHandleW([MarshalAs(UnmanagedType.LPWStr)] string lpModuleName);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr CreateRemoteThread(IntPtr hProcess, IntPtr lpThreadAttributes, uint dwStackSize, IntPtr lpStartAddress, IntPtr lpParameter, uint dwCreationFlags, IntPtr lpThreadId);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern uint WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool CloseHandle(IntPtr hObject);
"@

$PROCESS_ALL_ACCESS = 0x1F0FFF
$MEM_COMMIT = 0x1000
$MEM_RESERVE = 0x2000
$PAGE_READWRITE = 0x04

$hProcess = [NuVk.Native]::OpenProcess($PROCESS_ALL_ACCESS, $false, [uint32]$target.Id)
if ($hProcess -eq [IntPtr]::Zero) { throw "OpenProcess failed: $([ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error()).Message)" }

$bytes = [Text.Encoding]::Unicode.GetBytes($Dll + [char]0)
$remote = [NuVk.Native]::VirtualAllocEx($hProcess, [IntPtr]::Zero, [uint32]$bytes.Length, ($MEM_COMMIT -bor $MEM_RESERVE), $PAGE_READWRITE)
if ($remote -eq [IntPtr]::Zero) { throw "VirtualAllocEx failed" }

$written = [UIntPtr]::Zero
if (-not [NuVk.Native]::WriteProcessMemory($hProcess, $remote, $bytes, [uint32]$bytes.Length, [ref]$written)) { throw "WriteProcessMemory failed" }

$kernel32 = [NuVk.Native]::GetModuleHandleW("kernel32.dll")
$loadLibrary = [NuVk.Native]::GetProcAddress($kernel32, "LoadLibraryW")

$thread = [NuVk.Native]::CreateRemoteThread($hProcess, [IntPtr]::Zero, 0, $loadLibrary, $remote, 0, [IntPtr]::Zero)
if ($thread -eq [IntPtr]::Zero) { throw "CreateRemoteThread failed" }

[void][NuVk.Native]::WaitForSingleObject($thread, 5000)
[void][NuVk.Native]::CloseHandle($thread)
[void][NuVk.Native]::CloseHandle($hProcess)

Write-Host "Injected $Dll into $($target.ProcessName) (pid $($target.Id))"
Write-Host "Check NuVulkanBackend.log next to the DLL and shaders\*.dxbc next to the game."
