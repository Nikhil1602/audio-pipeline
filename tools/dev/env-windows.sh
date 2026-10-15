# Source in Git Bash before building sd-asr (whisper.cpp) on Windows:  . tools/dev/env-windows.sh
#
# 1. CMake: Visual Studio's bundled CMake if none is on PATH.
# 2. libclang for bindgen: set LIBCLANG_PATH to a directory containing libclang.dll (LLVM, or
#    `pip install --target <dir> libclang` and use <dir>/clang/native).
# 3. Optimization: the `cmake` crate replaces CMake's release flags on MSVC with flags that lack
#    /O2, which made whisper.cpp ~9x slower (measured, docs/m3-report.md). whisper-rs-sys
#    forwards CMAKE_* variables, so the release flags are set explicitly. Dash-style flags are
#    used because Git Bash rewrites /O2 into a path.
if ! command -v cmake >/dev/null 2>&1; then
    for v in 18 2022; do
        d="/c/Program Files/Microsoft Visual Studio/$v/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin"
        [ -x "$d/cmake.exe" ] && export PATH="$d:$PATH" && break
    done
fi
: "${LIBCLANG_PATH:=C:\Program Files\LLVM\bin}"
export LIBCLANG_PATH
export CMAKE_C_FLAGS_RELEASE="-O2 -Ob2 -DNDEBUG -nologo -MD -Brepro -W0"
export CMAKE_CXX_FLAGS_RELEASE="-O2 -Ob2 -DNDEBUG -utf-8 -nologo -MD -Brepro -W0"
