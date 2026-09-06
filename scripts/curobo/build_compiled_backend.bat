@echo off
REM ---------------------------------------------------------------------------
REM Build cuRobo's compiled (pybind) kernel backend on Windows.
REM
REM Why this is a script rather than three lines of setup notes: doing it by hand
REM on Windows walks into five traps. The first four fail quietly, meaning the
REM build reports success and produces no extensions, while the fifth makes a
REM good build look broken.
REM
REM   1. `set CUROBO_USE_PYBIND=1 && ...` puts the space in the value. setup.py
REM      compares against "1", gets "1 ", and silently skips the build.
REM   2. `set "PATH=...;%PATH%"` inside a && chain expands %PATH% when the line is
REM      parsed, before vcvars64 has run, so it reverts the compiler PATH and
REM      cl.exe is not found.
REM   3. torch's cpp_extension refuses to build when the VC environment is active
REM      and DISTUTILS_USE_SDK is not set.
REM   4. a conda CUDA toolkit puts its import libraries flat in "Library\lib",
REM      not in the "%CUDA_HOME%\lib\x64" that torch looks in, so the compile
REM      succeeds and the link fails. Defused at the LIB assignment below.
REM   5. the verification step has to import torch before it touches a cuRobo
REM      backend, or a good build reports as absent. Defused at step 3 below.
REM
REM Traps 1 to 3 are defused together after vcvars; traps 4 and 5 are defused at
REM their own steps, where the comment explains what each one costs.
REM
REM Usage, from the repository root:
REM     scripts\curobo\build_compiled_backend.bat
REM
REM Optional: set CUROBO_ENV / CUROBO_SRC first to override the in-repo defaults.
REM
REM This answers the one-off question "does a compiled backend build here". The
REM ongoing question, which backends resolve on this box and what a missing one
REM costs, is `python -m src.robot.safety.planning --doctor`, whose remedy text
REM names this script.
REM ---------------------------------------------------------------------------
setlocal

if "%CUROBO_ENV%"=="" set "CUROBO_ENV=%~dp0..\..\ext_deps\curobo_env"
if "%CUROBO_SRC%"=="" set "CUROBO_SRC=%~dp0..\..\ext_deps\curobo"

if not exist "%CUROBO_ENV%\python.exe" (
    echo [error] no python at "%CUROBO_ENV%\python.exe"
    echo         create the env first; see ext_deps\README.md section 1.
    exit /b 2
)
if not exist "%CUROBO_SRC%\setup.py" (
    echo [error] no cuRobo source at "%CUROBO_SRC%"
    echo         clone it first; see ext_deps\README.md section 2.
    exit /b 2
)

REM -- locate the MSVC developer environment without hardcoding a path ---------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [error] vswhere.exe not found. Install the Visual Studio Build Tools:
    echo         winget install --id Microsoft.VisualStudio.2022.BuildTools
    exit /b 2
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
        -property installationPath`) do set "VSPATH=%%i"
if "%VSPATH%"=="" (
    echo [error] no Visual Studio installation with the C++ toolset.
    echo         Install the "Desktop development with C++" workload.
    exit /b 2
)

echo [1/3] activating MSVC from "%VSPATH%"
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

REM -- traps 1 to 3, defused, in this order -----------------------------------
REM  * quoted set: no trailing space in the value
REM  * PATH prepended AFTER vcvars, on its own line, so %PATH% expands to the
REM    already-activated value
REM  * DISTUTILS_USE_SDK: torch requires it once the VC env is active
set "CUROBO_USE_PYBIND=1"
set "DISTUTILS_USE_SDK=1"
set "PATH=%CUROBO_ENV%\Library\bin;%PATH%"

REM Trap 4: the conda CUDA layout. torch's cpp_extension looks for import libraries
REM in "%CUDA_HOME%\lib\x64", but a conda cuda-toolkit puts them flat in
REM "Library\lib", so the compile succeeds and the LINK fails with
REM "LNK1181: cannot open input file 'cudart.lib'". Adding that directory to LIB,
REM which link.exe honours, fixes it without moving anything around.
set "CUDA_HOME=%CUROBO_ENV%\Library"
set "CUDA_PATH=%CUROBO_ENV%\Library"
set "LIB=%CUROBO_ENV%\Library\lib;%CUROBO_ENV%\Library\lib\x64;%LIB%"

where cl.exe >nul 2>&1 || (echo [error] cl.exe still not on PATH after vcvars & exit /b 1)
where nvcc.exe >nul 2>&1 || (echo [error] nvcc.exe not found; is cuda-toolkit in the env? & exit /b 1)

echo [2/3] compiling CUDA kernels (this takes several minutes)
pushd "%CUROBO_SRC%"
"%CUROBO_ENV%\python.exe" setup.py build_ext --inplace
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo [error] build failed with exit code %RC%
    exit /b %RC%
)

echo [3/3] verifying that the compiled backend is selectable
REM Trap 5: `import torch` first, or this check fails on a perfectly good build. The compiled
REM extensions link against torch's DLLs (c10, torch_cpu, torch_cuda), and since Python 3.8 an
REM extension module does not find dependency DLLs via PATH; importing torch is what adds its
REM lib directory to the search path. Without it the loader reports the module as simply absent,
REM and cuRobo turns that into "PyBind backend not available. Compile with: pip install .[compiled]",
REM which reads as "your build failed" when the build was fine. Real usage is unaffected, because
REM cuRobo imports torch itself long before it touches a backend.
REM
REM This asks whether pybind is SELECTABLE once it has been explicitly set, which is what a fresh
REM build has to prove. The doctor's probe asks the weaker question, which backends import at all,
REM and it asks it from the project interpreter by spawning this environment. Here the answer is
REM one interpreter away, so it is taken directly.
"%CUROBO_ENV%\python.exe" -c "import torch; import curobo._src.runtime as rt; rt.kernel_backend='pybind'; from curobo._src.curobolib.backends import get_backend_name; print('   backend ->', get_backend_name())" || exit /b 1

echo.
echo done. cuRobo now has two independent kernel backends.
endlocal
