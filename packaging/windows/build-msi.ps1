# build-msi.ps1 — build a Windows MSI for kimi-k3-lean using WiX 3.x.
#
# Usage:
#   .\build-msi.ps1 -Version "0.6.8" -SourceDir "C:\dist\kimi-k3-lean-0.6.8"
#
# Output:
#   .\kimi-k3-lean-0.6.8.msi
#
# Prerequisites:
#   - PowerShell 5.1+ (Windows 10/11/Server 2016+)
#   - WiX 3.14+ installed and on PATH: https://wixtoolset.org/
#       Or install via: choco install wixtoolset
#       Or: dotnet tool install --global wix
#
# What it does:
#   1. Generates a WiX .wxs file describing the install layout.
#   2. Compiles it with `candle.exe`.
#   3. Links it into an MSI with `light.exe`.
#   4. The MSI installs to C:\Program Files\kimi-k3-lean\ and adds
#      bin\ to the system PATH.

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Version,

    [Parameter(Mandatory)]
    [string]$SourceDir
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------------------- verify
if (-not (Test-Path $SourceDir)) {
    throw "SourceDir not found: $SourceDir"
}
if (-not (Test-Path "$SourceDir\bin\k3.exe")) {
    throw "SourceDir is missing bin\k3.exe — is this the Windows release zip?"
}
if (-not (Test-Path "$SourceDir\lib\libk3.dll")) {
    throw "SourceDir is missing lib\libk3.dll — is this the Windows release zip?"
}

# Find WiX. Try the .NET tool first, fall back to native install.
$wixPath = $null
$dotnetTool = dotnet tool list -g 2>&1 | Select-String "wix"
if ($dotnetTool) {
    $wixPath = (dotnet tool list -g | ConvertFrom-String -PropertyNames Name, Version) |
        Where-Object { $_.Name -match "wix" } | Select-Object -First 1
    if ($wixPath) {
        $wixBin = "$env:USERPROFILE\.dotnet\tools"
    }
}
if (-not $wixBin -or -not (Test-Path "$wixBin\candle.exe")) {
    # Look on PATH.
    $candle = Get-Command "candle.exe" -ErrorAction SilentlyContinue
    if ($candle) {
        $wixBin = Split-Path $candle.Source -Parent
    } else {
        throw "WiX 3.x not found. Install via 'choco install wixtoolset' " +
              "or 'dotnet tool install --global wix'."
    }
}

Write-Host "==> Building kimi-k3-lean $Version MSI"
Write-Host "    Source:  $SourceDir"
Write-Host "    WiX:     $wixBin"

# --------------------------------------------------------------- generate .wxs
$wxsFile = Join-Path $PSScriptRoot "kimi-k3-lean-$Version.wxs"
$productCode = (New-Guid).Guid.ToString().ToUpper()
$upgradeCode = "{B6E5F8A1-3D2C-4F7E-9A1B-5C8D2E4F6A9B}"

@"
<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Name="kimi-k3-lean"
           Version="$Version"
           Manufacturer="chazhyseni"
           Id="{$productCode}"
           UpgradeCode="$upgradeCode"
           Language="1033">

    <Package InstallerVersion="500" Compressed="yes" InstallScope="perMachine" />
    <MajorUpgrade DowngradeErrorMessage="A newer version of kimi-k3-lean is already installed." />
    <MediaTemplate EmbedCab="yes" />

    <!-- Where to install. -->
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFilesFolder">
        <Directory Id="INSTALLDIR" Name="kimi-k3-lean">
          <Directory Id="BinDir" Name="bin" />
          <Directory Id="LibDir" Name="lib" />
          <Directory Id="IncludeDir" Name="include">
            <Directory Id="Libk3Dir" Name="libk3" />
          </Directory>
          <Directory Id="ShareDir" Name="share">
            <Directory Id="DocDir" Name="doc">
              <Directory Id="KimiDir" Name="kimi-k3-lean" />
            </Directory>
          </Directory>
        </Directory>
      </Directory>
      <Directory Id="ProgramMenuFolder" />
    </Directory>

    <Feature Id="ProductFeature" Title="kimi-k3-lean" Level="1">
      <ComponentGroupRef Id="BinComponents" />
      <ComponentGroupRef Id="LibComponents" />
      <ComponentGroupRef Id="IncludeComponents" />
      <ComponentGroupRef Id="DocComponents" />
      <ComponentRef Id="PathComponent" />
    </Feature>

    <!-- bin/ -->
    <ComponentGroup Id="BinComponents" Directory="BinDir">
      <Component Id="Bin_k3" Guid="(NEWGUID1)">
        <File Id="k3_exe" Source="SourceDir\bin\k3.exe" KeyPath="yes" />
      </Component>
      <Component Id="Bin_libk3_dll" Guid="(NEWGUID2)">
        <File Id="libk3_dll" Source="SourceDir\lib\libk3.dll" />
      </Component>
    </ComponentGroup>

    <!-- lib/ -->
    <ComponentGroup Id="LibComponents" Directory="LibDir">
      <Component Id="Lib_libk3_static" Guid="(NEWGUID3)">
        <File Id="libk3_static" Source="SourceDir\lib\libk3_static.lib" />
      </Component>
    </ComponentGroup>

    <!-- include/libk3/ -->
    <ComponentGroup Id="IncludeComponents" Directory="Libk3Dir">
      <Component Id="Include_libk3_h" Guid="(NEWGUID4)">
        <File Id="libk3_h" Source="SourceDir\include\libk3\libk3.h" KeyPath="yes" />
      </Component>
    </ComponentGroup>

    <!-- share/doc/kimi-k3-lean/ -->
    <ComponentGroup Id="DocComponents" Directory="KimiDir">
      <Component Id="Doc_README" Guid="(NEWGUID5)">
        <File Id="doc_README"   Source="SourceDir\share\doc\kimi-k3-lean\README.md" />
      </Component>
      <Component Id="Doc_LICENSE" Guid="(NEWGUID6)">
        <File Id="doc_LICENSE" Source="SourceDir\share\doc\kimi-k3-lean\LICENSE" />
      </Component>
      <Component Id="Doc_INSTALL" Guid="(NEWGUID7)">
        <File Id="doc_INSTALL" Source="SourceDir\share\doc\kimi-k3-lean\INSTALL.md" />
      </Component>
    </ComponentGroup>

    <!-- Add bin\ to the system PATH. -->
    <Component Id="PathComponent" Guid="(NEWGUID8)" Directory="INSTALLDIR">
      <Environment Id="PATH"
                   Name="PATH"
                   Value="[BinDir]"
                   Permanent="no"
                   Part="last"
                   Action="set"
                   System="yes" />
    </Component>

    <!-- UI: minimal. -->
    <UI Id="WixUI_Minimal" />
    <Property Id="WIXUI_INSTALLDIR" Value="INSTALLDIR" />
  </Product>
</Wix>
"@ | Out-File -FilePath $wxsFile -Encoding utf8

# Substitute (NEWGUIDN) with fresh GUIDs.
$guidMap = @{
    '(NEWGUID1)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID2)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID3)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID4)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID5)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID6)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID7)' = (New-Guid).Guid.ToString().ToUpper()
    '(NEWGUID8)' = (New-Guid).Guid.ToString().ToUpper()
}
$content = Get-Content $wxsFile -Raw
foreach ($key in $guidMap.Keys) {
    $content = $content.Replace($key, $guidMap[$key])
}
Set-Content -Path $wxsFile -Value $content -Encoding utf8

# --------------------------------------------------------------- compile + link
Write-Host "==> candle.exe $wxsFile"
& "$wixBin\candle.exe" -out "$PSScriptRoot\" $wxsFile
if ($LASTEXITCODE -ne 0) { throw "candle.exe failed" }

$wixObjFile = Join-Path $PSScriptRoot "kimi-k3-lean-$Version.wixobj"
$msiFile    = Join-Path $PSScriptRoot "kimi-k3-lean-$Version.msi"

Write-Host "==> light.exe (linking MSI)"
& "$wixBin\light.exe" `
    -out $msiFile `
    -ext WixUtilExtension `
    -cultures:en-US `
    $wixObjFile
if ($LASTEXITCODE -ne 0) { throw "light.exe failed" }

Write-Host ""
Write-Host "==> Done."
Write-Host "    $msiFile"
Write-Host ""
Write-Host "To install on a target Windows host:"
Write-Host "    msiexec /i kimi-k3-lean-$Version.msi /qb"
Write-Host ""
Write-Host "To uninstall:"
Write-Host "    msiexec /x kimi-k3-lean-$Version.msi /qb"