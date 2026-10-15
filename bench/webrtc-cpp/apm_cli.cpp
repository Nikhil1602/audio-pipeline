// Reference harness: runs the original WebRTC C++ AudioProcessing (webrtc-audio-processing 2.1)
// over raw float files so its output can be compared with the sonora (Rust) port on identical
// input. Not part of the product; see bench/webrtc-cpp/README.md.
//
// usage: apm_cli <mic_48k_mono.f32> <ref_48k_stereo.f32> <out_16k_mono.f32> <aec 0|1> <ns -1..3> <agc2 0|1> <hpf 0|1>
// Files are raw little-endian float32. Prints the backend statistics at the end.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "api/audio/audio_processing.h"

static std::vector<float> read_f32(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f) / 4;
  std::fseek(f, 0, SEEK_SET);
  std::vector<float> v(n);
  if (std::fread(v.data(), 4, n, f) != static_cast<size_t>(n)) { std::exit(2); }
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 8) { std::fprintf(stderr, "usage: see source\n"); return 2; }
  auto mic = read_f32(argv[1]);
  auto ref = read_f32(argv[2]);
  bool aec = std::atoi(argv[4]) != 0;
  int ns = std::atoi(argv[5]);
  bool agc = std::atoi(argv[6]) != 0;
  bool hpf = std::atoi(argv[7]) != 0;

  webrtc::AudioProcessing::Config cfg;
  cfg.echo_canceller.enabled = aec;
  cfg.noise_suppression.enabled = ns >= 0;
  if (ns >= 0) cfg.noise_suppression.level = static_cast<webrtc::AudioProcessing::Config::NoiseSuppression::Level>(ns);
  cfg.gain_controller2.enabled = agc;
  cfg.gain_controller2.adaptive_digital.enabled = agc;
  cfg.high_pass_filter.enabled = hpf;

  rtc::scoped_refptr<webrtc::AudioProcessing> apm = webrtc::AudioProcessingBuilder().SetConfig(cfg).Create();
  webrtc::StreamConfig in(48000, 1), out(16000, 1), rcfg(48000, 2);

  const size_t frames = mic.size() / 480;
  std::vector<float> outbuf(frames * 160);
  std::vector<float> rl(480), rr(480), dl(480), dr(480);
  for (size_t k = 0; k < frames; ++k) {
    for (int i = 0; i < 480; ++i) {
      size_t j = (k * 480 + i) * 2;
      rl[i] = j < ref.size() ? ref[j] : 0.f;
      rr[i] = j + 1 < ref.size() ? ref[j + 1] : 0.f;
    }
    const float* rsrc[2] = {rl.data(), rr.data()};
    float* rdst[2] = {dl.data(), dr.data()};
    apm->ProcessReverseStream(rsrc, rcfg, rcfg, rdst);
    apm->set_stream_delay_ms(0);
    const float* csrc[1] = {mic.data() + k * 480};
    float* cdst[1] = {outbuf.data() + k * 160};
    apm->ProcessStream(csrc, in, out, cdst);
  }
  FILE* f = std::fopen(argv[3], "wb");
  std::fwrite(outbuf.data(), 4, outbuf.size(), f);
  std::fclose(f);
  auto st = apm->GetStatistics();
  std::printf("erl=%.2f erle=%.2f delay_ms=%d\n", st.echo_return_loss.value_or(-1),
              st.echo_return_loss_enhancement.value_or(-1), st.delay_ms.value_or(-1));
  return 0;
}
