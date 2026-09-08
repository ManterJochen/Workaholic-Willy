<#
.SYNOPSIS
    Install the external native dependencies of Workaholic-Willy, Coal and cuRobo, into ext_deps/.

.DESCRIPTION
    One command, from nothing to a verified motion stack. It bootstraps its own micromamba,
    creates both environments from the committed lockfiles, clones and builds cuRobo with both
    kernel backends, generates the robot descriptors, and finishes by running the doctor. Where
    the repository venv exists the exit code is the verdict of the doctor, so it means "this box
    can plan" rather than "the downloads finished"; without a venv the install still completes and
    the script prints the command to run by hand.

    Nothing outside ext_deps/ is touched: no system conda, no PATH or registry changes, no admin
    rights. Deleting ext_deps/ undoes the whole thing.

    Why lockfiles. Every conda package is pinned to an exact build, by URL and hash. That is
    ordinary reproducibility, and on Windows it is also a load-bearing safety property: an
    application-control policy such as Smart App Control refuses unsigned native code that its
    reputation service does not vouch for, and a build published days ago typically has no
    reputation yet. Pinning to builds that are known to load is what keeps a fresh install
    working. ext_deps/README.md states the rule in full.

.PARAMETER Component
    'all' (default), 'coal', or 'curobo'.

.PARAMETER Clean
    Delete the target environments first, which is the honest way to test that this script really
    does reproduce the stack from nothing.

.PARAMETER SkipCompiledBackend
    Skip the compiled (pybind) kernel backend of cuRobo. It needs MSVC and takes several minutes.
    Not recommended: it leaves the planner with a single kernel backend and therefore with a
    single point of failure. Either backend can be taken out on its own by an application-control
    verdict, a broken wheel or a CUDA upgrade, and with only one installed that takes the planner
    down and every motion on a cell configured for cuRobo is refused.

.PARAMETER SslNoRevoke
    Skip TLS certificate-revocation checks while downloading. Corporate networks that intercept
    TLS often break the revocation lookup of schannel, which surfaces as an unrelated-looking
    download failure. Off by default, because it does weaken verification.

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

# This file lives at <repo>\scripts\ext_deps\, so the repository root is two levels up. The root
# is what every path below is anchored on, and ext_deps/ has to land there because that is where
# the library looks for it: curobo_python_path and inject_coal_prefix in
# src/robot/safety/planning/environment.py resolve their defaults from the repository root.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$extDeps = Join-Path $repo 'ext_deps'
$locks = Join-Path $extDeps 'locks'

# Keep the package cache inside ext_deps too, so "delete ext_deps/" really does undo everything and
# a rebuild does not silently inherit a cache from some earlier install elsewhere on the machine.
# That matters for the -Clean test: a cache outside the folder would make a rebuild that claims to
# start from nothing a lie.
$env:MAMBA_ROOT_PREFIX = Join-Path $extDeps 'micromamba\root'
# pip caches wheels too, and by default in the user profile, where deleting ext_deps/ does not
# reach it. Measured on the reference workstation: 607 MB at AppData\Local\pip\Cache.
# Redirecting it here is what makes the promise above true rather than nearly true.
$env:PIP_CACHE_DIR = Join-Path $extDeps 'pip-cache'
$mambaArgs = @('-c', 'conda-forge')
if ($SslNoRevoke) { $mambaArgs += '--ssl-no-revoke' }

# Pinned cuRobo revision. 8e734f3 carries two commits this repository depends on:
#   82ea96e  repairs the fallback from the cuda.core backend to pybind itself.
#            _try_load_cuda_core_backend used to raise instead of returning None, so a missing
#            cuda.core took the planner down rather than falling through to the compiled backend.
#   8e734f3  fixes the mesh SDF gradient sign for query points outside the surface.
# The newest tag, v0.8.0, is older than this commit, so the pin tracks the branch, not the tag.
$CUROBO_REV = '8e734f3'
$TORCH_SPEC = 'torch==2.7.0'
$TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'   # cu128 is the Blackwell / sm_120 line

function Write-Step { param([string] $Message) Write-Host "`n=== $Message ===" -ForegroundColor Cyan }
function Write-Note { param([string] $Message) Write-Host "    $Message" -ForegroundColor DarkGray }

function Get-Micromamba {
    <# Bootstrap a private micromamba into ext_deps/. Deliberately not a system install: the
       package manager is an implementation detail of this folder, and a customer must not have to
       adopt conda in order to run Willy. #>
    $exe = Join-Path $extDeps 'micromamba\Library\bin\micromamba.exe'
    if (Test-Path $exe) { return $exe }
    Write-Step 'bootstrapping micromamba (private to ext_deps/)'
    $root = Join-Path $extDeps 'micromamba'
    New-Item -ItemType Directory -Force $root | Out-Null
    $archive = Join-Path $root 'micromamba.tar.bz2'
    Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile $archive
    tar -xf $archive -C $root | Out-Host
    if (-not (Test-Path $exe)) { throw "micromamba bootstrap failed: no exe at $exe" }
    Write-Note "micromamba at $exe"
    return $exe
}

# Every native call in this script pipes to Out-Host, and no function that runs one returns a
# value. A bare `& exe` writes the console output of the command to the success stream, which
# PowerShell folds into the return value of the enclosing function, so a caller assigning the
# result of such a function receives an array of log lines with the intended value at the end, and
# the first empty line in that array reaches Join-Path as an empty -Path. Callers derive the
# prefix for themselves instead.
function New-EnvFromLock {
    param([string] $Mamba, [string] $Name)
    $prefix = Join-Path $extDeps $Name
    $lock = Join-Path $locks "$Name.win-64.lock"
    if (-not (Test-Path $lock)) { throw "no lockfile at $lock" }
    if ($Clean -and (Test-Path $prefix)) {
        Write-Note "removing $prefix"
        Remove-Item -Recurse -Force $prefix
    }
    Write-Step "creating $Name from its lockfile"
    & $Mamba create -y -p $prefix --file $lock @mambaArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "micromamba create failed for $Name (exit $LASTEXITCODE). " +
              'If the failure looks like a TLS or download error on a corporate network, retry with -SslNoRevoke.'
    }
    if (-not (Test-Path $prefix)) { throw "micromamba reported success but $prefix does not exist" }
}

# --------------------------------------------------------------------------------- Coal
function Install-Coal {
    param([string] $Mamba)
    New-EnvFromLock -Mamba $Mamba -Name 'coal_env'
    Write-Note 'Coal is reached by injecting this env into the host interpreter: its DLL'
    Write-Note 'directory plus its site-packages, appended so the host numpy still wins. The'
    Write-Note 'python.exe of this env is never invoked, so what it can import does not matter.'
}

# --------------------------------------------------------------------------------- cuRobo
function Install-Curobo {
    param([string] $Mamba)
    New-EnvFromLock -Mamba $Mamba -Name 'curobo_env'
    $prefix = Join-Path $extDeps 'curobo_env'
    $python = Join-Path $prefix 'python.exe'
    if (-not (Test-Path $python)) { throw "no interpreter at $python after creating the env" }
    $source = Join-Path $extDeps 'curobo'

    Write-Step "torch ($TORCH_SPEC, cu128)"
    & $python -m pip install --quiet $TORCH_SPEC --index-url $TORCH_INDEX | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit $LASTEXITCODE)" }

    Write-Step "cuRobo source at $CUROBO_REV"
    if ($Clean -and (Test-Path $source)) { Remove-Item -Recurse -Force $source }
    if (-not (Test-Path $source)) {
        # No `git lfs install` here. It writes filter.lfs.* into the user's global .gitconfig,
        # which outlives ext_deps/ and belongs to every other repository on the machine. It also
        # buys nothing: the pinned cuRobo revision ships an empty .gitattributes, so it declares
        # no LFS filters at all, and a clone of it produces no .git/lfs directory.
        git clone https://github.com/NVlabs/curobo.git $source | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
    }
    Push-Location $source
    try {
        git fetch --quiet origin | Out-Host
        git checkout --quiet $CUROBO_REV | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git checkout $CUROBO_REV failed" }
    } finally { Pop-Location }

    Write-Step 'cuRobo, JIT kernel backend (cuda.core)'
    & $python -m pip install --quiet -e "$source[cu12]" --no-build-isolation | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "cuRobo [cu12] install failed (exit $LASTEXITCODE)" }

    # Pinned down, not up. `[cu12]` pulls cuda-core at whatever is newest, and the newest build of
    # a package is the one least likely to be admitted: an application-control policy decides on
    # reach rather than recency, so a freshly published build has no reputation yet and can be
    # refused at load time. A refused cuda-core leaves the JIT backend dead, and where the
    # compiled backend below is also absent the planner then has no kernel backend at all and
    # every motion on a cell configured for cuRobo is refused. If this pin is ever refused in
    # turn, step back to the build before it rather than reaching for the policy switch.
    & $python -m pip install --quiet 'cuda-core==1.1.0' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "cuda-core pin failed (exit $LASTEXITCODE)" }

    # The compiled backend is the normal case. It is built unless the operator opts out, unless it
    # is already there, or unless this machine has no C++ toolchain at all. Only the last of those
    # is a fallback, and it is an emergency one: see the warning it prints.
    $pybind = Join-Path $source 'curobo\_src\curobolib\backends\pybind'
    $built = @('geometry', 'kinematics', 'optimization', 'trajectory') |
        ForEach-Object { Join-Path $pybind "$_.cp310-win_amd64.pyd" }
    $haveAll = ($built | Where-Object { Test-Path $_ }).Count -eq 4

    if ($SkipCompiledBackend) {
        Write-Warning 'skipping the compiled backend on request: the planner then has one kernel'
        Write-Warning 'backend, so one policy verdict, one broken wheel or one CUDA upgrade takes'
        Write-Warning 'it down. Run scripts\curobo\build_compiled_backend.bat when a C++ toolchain'
        Write-Warning 'is available.'
    } elseif ($haveAll -and -not $Clean) {
        Write-Step 'cuRobo, compiled kernel backend (pybind): already built, skipping'
    } else {
        Write-Step 'cuRobo, compiled kernel backend (pybind): several minutes'
        $bat = Join-Path $repo 'scripts\curobo\build_compiled_backend.bat'
        & cmd /c "`"$bat`"" | Out-Host
        $rc = $LASTEXITCODE
        if ($rc -eq 2) {
            # Exit 2 from that script means it never started: no active x64 environment and no
            # toolchain to activate. Installing one needs administrator rights, and refusing the
            # whole install over that would leave a machine without them unable to run this stack
            # at all. So the install continues and the doctor below decides whether the result is
            # usable: one backend is a warning and exits 0, no backend is broken and exits
            # non-zero.
            Write-Warning ''
            Write-Warning 'EMERGENCY FALLBACK: no C++ toolchain, so the compiled kernel backend was'
            Write-Warning 'not built. The planner will run on cuda.core alone, with no spare.'
            Write-Warning 'This is not the intended configuration. On 2026-08-30 an application'
            Write-Warning 'control policy refused cuda.core on this very machine, and the compiled'
            Write-Warning 'backend was the only reason the planner kept working.'
            Write-Warning 'Build it as soon as a toolchain exists, from a 64-bit developer prompt:'
            Write-Warning '  scripts\curobo\build_compiled_backend.bat'
        } elseif ($rc -ne 0) {
            # Any other code means a toolchain was found and the build itself failed, which is a
            # fault rather than a missing prerequisite.
            throw "compiled backend build failed (exit $rc)"
        }
    }

    Write-Step 'robot descriptors (ur5e, ur3e)'
    # These live inside the content directory of the clone, so a fresh clone always needs this
    # step; it is not something the environment carries. The builder runs under the interpreter of
    # the cuRobo env rather than the host one, because it imports curobo.sphere_fit.
    foreach ($model in @('ur5e', 'ur3e')) {
        & $python (Join-Path $repo 'scripts\curobo\build_ur_config.py') $model | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "building the $model descriptor failed" }
    }
}

# --------------------------------------------------------------------------------- run
# What this script cannot bootstrap, checked before it downloads anything.
#
# micromamba, both environments, torch and cuRobo all arrive under ext_deps/ without
# administrator rights. Two programs do not, and only one of them ships with Windows. git is used
# three times to clone and pin the cuRobo source, and on a fresh machine it is simply absent, so
# the install spent several minutes bootstrapping micromamba and building an environment before
# failing with "the term 'git' is not recognized" at a point that says nothing about what to do.
#
# Checked here rather than at the call site so the answer arrives in seconds, and so a machine
# missing both is told about both.
$missing = @()
foreach ($tool in @('git', 'tar')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $missing += $tool }
}
if ($missing.Count) {
    Write-Host "missing prerequisite(s): $($missing -join ', ')" -ForegroundColor Red
    if ($missing -contains 'git') {
        Write-Host '  git: needed to clone the pinned cuRobo revision. Git for Windows installs' -ForegroundColor Red
        Write-Host '       for a single user without administrator rights: choose "Only for me"' -ForegroundColor Red
        Write-Host '       in the installer, or use the portable build.' -ForegroundColor Red
    }
    if ($missing -contains 'tar') {
        Write-Host '  tar: ships with Windows 10 1803 and later; on an older build, install it or' -ForegroundColor Red
        Write-Host '       unpack ext_deps/micromamba by hand.' -ForegroundColor Red
    }
    exit 2
}

$mamba = Get-Micromamba
if ($Component -in @('all', 'coal')) { Install-Coal -Mamba $mamba }
if ($Component -in @('all', 'curobo')) { Install-Curobo -Mamba $mamba }

Write-Step 'verifying (this is what the exit code means)'
$hostPython = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $hostPython)) {
    Write-Warning "no .venv at $hostPython, so the doctor did not run. Run it yourself:"
    Write-Warning '  python -m src.robot.safety.planning --doctor'
    exit 0
}
Push-Location $repo
try {
    & $hostPython -m src.robot.safety.planning --doctor
    $doctor = $LASTEXITCODE
} finally { Pop-Location }

switch ($doctor) {
    0 { Write-Host "`ndone: the motion stack loads on this box." -ForegroundColor Green }
    2 { Write-Host "`nan OS application-control policy is blocking a dependency." -ForegroundColor Red
        Write-Host  "The fix is to pin that package to a build the policy admits, normally the" -ForegroundColor Red
        Write-Host  "one before it. Re-downloading or moving the install does not help: the" -ForegroundColor Red
        Write-Host  "verdict follows the contents of the file. See ext_deps/README.md." -ForegroundColor Red }
    default { Write-Host "`nthe stack installed but something is degraded; read the doctor above." -ForegroundColor Yellow }
}
exit $doctor
