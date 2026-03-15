param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ForwardArgs = @()
)


$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# Ensure release artifacts exist so torch-sys has produced/downladed libtorch.
cargo build --release
if ($LASTEXITCODE -ne 0) {
  throw "cargo build --release failed."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseBuildDir = Join-Path $projectRoot "target\release\build"
$debugBuildDir = Join-Path $projectRoot "target\debug\build"

function Find-TorchLibDir([string]$buildDir) {
  if (-not (Test-Path $buildDir)) {
    return $null
  }

  $candidates = Get-ChildItem -Path $buildDir -Directory -Filter "torch-sys-*" |
    Sort-Object LastWriteTime -Descending

  foreach ($candidate in $candidates) {
    $libDir = Join-Path $candidate.FullName "out\libtorch\libtorch\lib"
    if (Test-Path $libDir) {
      return $libDir
    }
  }

  return $null
}

$torchLib = Find-TorchLibDir $releaseBuildDir
if (-not $torchLib) {
  $torchLib = Find-TorchLibDir $debugBuildDir
}

if (-not $torchLib) {
  throw "Could not find libtorch under target\\{release,debug}\\build\\torch-sys-*\\out\\libtorch\\libtorch\\lib"
}

$torchLib = $torchLib.Trim()
$env:PATH = "$torchLib;$env:PATH"

# Make sure torch-sys does not try to use Python's torch runtime.
if (Test-Path Env:\LIBTORCH_USE_PYTORCH) {
  Remove-Item Env:\LIBTORCH_USE_PYTORCH
}

Write-Host "Using libtorch runtime from: $torchLib"

cargo run --release -- -v @ForwardArgs
