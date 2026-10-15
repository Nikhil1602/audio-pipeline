@echo off
rem Builds apm_cli.exe against a webrtc-audio-processing 2.1 build tree (see README.md).
rem usage: build.cmd <path-to-sys-crate-out-dir>
setlocal
set OUT=%~1
set B=%OUT%\webrtc-audio-processing-build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /O2 /MD /EHsc /DWEBRTC_WIN /D_WIN32 /DNOMINMAX /D_USE_MATH_DEFINES /D__STDC_FORMAT_MACROS=1 ^
  /I"%OUT%\include\webrtc-audio-processing-2" /I"%OUT%\include" ^
  "%~dp0apm_cli.cpp" /Fe"%~dp0apm_cli.exe" /Fo"%TEMP%\apm_cli.obj" ^
  /link "%B%\webrtc\modules\audio_processing\libwebrtc-audio-processing-2.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_strings.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_base.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_numeric.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_types.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_synchronization.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_time.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_debugging.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_container.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_hash.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_log.a" ^
  "%B%\subprojects\abseil-cpp-20240722.0\libabsl_profiling.a" "%B%\subprojects\abseil-cpp-20240722.0\libabsl_crc.a" ^
  winmm.lib advapi32.lib
