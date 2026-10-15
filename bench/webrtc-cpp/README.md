# WebRTC C++ reference harness

`apm_cli.cpp` runs the original WebRTC AudioProcessing (from `webrtc-audio-processing` 2.1) over raw
float files so its output can be compared with `sonora` on identical input
(`crates/sd-dsp/src/tests.rs::compare_sonora_with_webrtc_cpp`). It is a benchmarking tool only and is
not part of the product.

Building on Windows (what worked, 2026-10-13):

1. Build the library through the `webrtc-audio-processing-sys` crate with the `bundled` feature,
   inside a VS 2022+ developer environment, with `CXXFLAGS=/std:c++20` and a short target directory
   (`subst W: <dir>`; the abseil sources otherwise exceed MSVC's 260-character path limit). The
   crate's own C++ wrapper then fails to build on MSVC, but meson has already built the library.
2. `build.cmd <sys-crate out dir>` compiles `apm_cli.exe`, linking meson's *build-tree* library
   (the installed copy has symbols renamed with a `v2_` prefix).

`apm_cli.exe` is not committed.
