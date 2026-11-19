<#
.SYNOPSIS
  Fetch the pinned WARP (d3d10warp.dll) from NuGet into a directory.

.DESCRIPTION
  Downloads Microsoft.Direct3D.WARP 1.0.22 from NuGet's flat container,
  verifies the package's SHA-256 against the pin below (stops on a
  mismatch), and copies build/native/bin/<arch>/d3d10warp.dll into
  Destination.  A process finds that copy before the system's when the
  DLL sits beside the executable, which is how the D3D12 tests use it.
  Works on Windows PowerShell 5.1 and PowerShell 7+.

.EXAMPLE
  scripts\fetch_warp.ps1 target\release\deps
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Destination,
    [ValidateSet('x64', 'arm64')][string]$Arch = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Version = '1.0.22'
$Sha256 = 'afc058f4d1066edc4231a38f3ab3592b1930f7287d4ac700e6a056e0874bb0be'
$Url = "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp/$Version/microsoft.direct3d.warp.$Version.nupkg"

if (-not $Arch) {
    $hostArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    switch ($hostArch) {
        'X64'   { $Arch = 'x64' }
        'Arm64' { $Arch = 'arm64' }
        default { throw "unsupported host architecture $hostArch; pass -Arch x64 or arm64" }
    }
}

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("warp-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    $pkg = Join-Path $work 'warp.zip'
    Invoke-WebRequest -Uri $Url -OutFile $pkg -UseBasicParsing

    $actual = (Get-FileHash -Algorithm SHA256 -Path $pkg).Hash.ToLowerInvariant()
    if ($actual -ne $Sha256) {
        throw "SHA-256 mismatch for Microsoft.Direct3D.WARP $Version`: expected $Sha256, got $actual"
    }

    Expand-Archive -Path $pkg -DestinationPath (Join-Path $work 'x') -Force
    $dll = Join-Path $work "x\build\native\bin\$Arch\d3d10warp.dll"
    if (-not (Test-Path -LiteralPath $dll)) {
        throw "d3d10warp.dll for $Arch is not in the package"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -LiteralPath $dll -Destination (Join-Path $Destination 'd3d10warp.dll') -Force
    $out = Join-Path (Resolve-Path -LiteralPath $Destination).Path 'd3d10warp.dll'
    Write-Host "WARP $Version ($Arch): $out, file version $((Get-Item -LiteralPath $out).VersionInfo.FileVersion)"
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $work -ErrorAction SilentlyContinue
}
