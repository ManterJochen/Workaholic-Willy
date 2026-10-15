<#
.SYNOPSIS
    Install the external native dependencies of Workaholic-Willy, Coal and cuRobo, into ext_deps/.

.DESCRIPTION
    A shim for scripts\ext_deps\install.py, the one installer for Windows and Linux (P0h h3). It runs that file with
    a Python 3.10 or newer and hands it these switches under their Python names; everything the install does, and
    why, is written there. On Windows it runs exactly the commands this script ran before it became a shim
    (tests\data\install_plan\win-64.txt), and the exit code is still the doctor's: "this box can plan", not "the
    downloads finished". Nothing outside ext_deps\ is touched, no administrator rights are needed, and deleting
    ext_deps\ undoes the whole thing.

    The one thing this shim needs that it cannot bootstrap is a Python 3.10 or newer, which Willy's own .venv needs
    anyway: the py launcher, else python on PATH, else the repository's .venv.

.PARAMETER Component
    'all' (default), 'coal', or 'curobo'.

.PARAMETER Clean
    Delete the target environments first, which is the honest way to test that the install really
    does reproduce the stack from nothing.

.PARAMETER SkipCompiledBackend
    Skip the compiled (pybind) kernel backend of cuRobo. It needs MSVC and takes several minutes.
    Not recommended: it leaves the planner with a single kernel backend and therefore with a
    single point of failure. Either backend can be taken out on its own by an application-control
    verdict, a broken wheel or a CUDA upgrade, and with only one installed that takes the planner
    down and every motion on a cell configured for cuRobo is refused.

.PARAMETER SslNoRevoke
    Skip TLS certificate-revocation checks while micromamba downloads. Corporate networks that
    intercept TLS often break the revocation lookup of schannel, which surfaces as an
    unrelated-looking download failure. Off by default, because it does weaken verification.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1 -Component coal -Clean
#>
[CmdletBinding()]
param(
    [ValidateSet('all', 'coal', 'curobo')] [string] $Component = 'all',
    [switch] $Clean,
    [switch] $SkipCompiledBackend,
    [switch] $SslNoRevoke
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$installer = Join-Path $PSScriptRoot 'install.py'

# A Python 3.10 or newer, in the order a Windows box most likely has one.
$candidates = @(@('py', '-3'), @('python'), @((Join-Path $repo '.venv\Scripts\python.exe')))
$python = $null
foreach ($candidate in $candidates) {
    $exe = $candidate[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $rest = @($candidate | Select-Object -Skip 1)
    & $exe @rest -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
}
if ($null -eq $python) {
    Write-Host 'missing prerequisite: Python 3.10 or newer, which runs scripts\ext_deps\install.py.' -ForegroundColor Red
    Write-Host '  Install Python for this user from python.org ("Install for all users" unticked needs no' -ForegroundColor Red
    Write-Host '  administrator rights), or create the repository''s .venv first.' -ForegroundColor Red
    exit 2
}

$arguments = @('--component', $Component)
if ($Clean) { $arguments += '--clean' }
if ($SkipCompiledBackend) { $arguments += '--skip-compiled-backend' }
if ($SslNoRevoke) { $arguments += '--ssl-no-revoke' }

$exe = $python[0]
$rest = @($python | Select-Object -Skip 1)
& $exe @rest $installer @arguments
exit $LASTEXITCODE
