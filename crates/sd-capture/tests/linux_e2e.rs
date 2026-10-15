//! End-to-end capture tests on a real Linux sound server (PulseAudio, or PipeWire through
//! pipewire-pulse) with virtual devices. CI sets them up with `tools/ci/linux-audio-e2e.sh`:
//!
//! * `e2e_speaker`: a null sink, the default sink (system audio is its monitor);
//! * `e2e_mic_feed`: a mono null sink whose monitor is remapped to the source `e2e_mic`, the
//!   default source, so audio played into `e2e_mic_feed` is what the "microphone" hears.
//!
//! The tests change server defaults, so they run one at a time, and only when
//! `SD_CAPTURE_E2E=1` (otherwise they pass without doing anything):
//!
//! ```text
//! SD_CAPTURE_E2E=1 cargo test --release -p sd-capture --test linux_e2e -- --test-threads=1 --nocapture
//! ```
#![cfg(target_os = "linux")]

mod common;

use std::process::Command;
use std::time::Duration;

use common::*;
use sd_capture::{CaptureEvent, DeviceSelector, SwitchReason, play};
use sd_core::{Source, StreamFormat};

fn pactl(args: &[&str]) -> String {
    let out = Command::new("pactl")
        .args(args)
        .output()
        .expect("run pactl");
    assert!(
        out.status.success(),
        "pactl {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn system_audio_captures_what_the_default_sink_plays() {
    if !enabled() {
        return;
    }
    pactl(&["set-default-sink", "e2e_speaker"]);
    let cap = capture(Source::System, DeviceSelector::Default);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_speaker"));
    std::thread::sleep(Duration::from_millis(300));
    let fmt = StreamFormat::new(RATE, 2).unwrap();
    let sig = signal(3.0, 2);
    let report = play(&DeviceSelector::Default, fmt, &sig).expect("play");
    std::thread::sleep(Duration::from_millis(500));
    let (c, _) = cap.finish();
    verify("system", &mono_of(&sig), &report, &c);
}

#[test]
fn microphone_captures_the_virtual_mic() {
    if !enabled() {
        return;
    }
    pactl(&["set-default-source", "e2e_mic"]);
    let cap = capture(Source::Mic, DeviceSelector::Default);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_mic"));
    std::thread::sleep(Duration::from_millis(300));
    let fmt = StreamFormat::new(RATE, 2).unwrap();
    let sig = signal(3.0, 2);
    let report = play(&DeviceSelector::Id("e2e_mic_feed".into()), fmt, &sig).expect("play");
    std::thread::sleep(Duration::from_millis(500));
    let (c, _) = cap.finish();
    verify("mic", &mono_of(&sig), &report, &c);
}

#[test]
fn long_capture_has_no_loss_and_no_drift() {
    if !enabled() {
        return;
    }
    pactl(&["set-default-sink", "e2e_speaker"]);
    let cap = capture(Source::System, DeviceSelector::Default);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_speaker"));
    std::thread::sleep(Duration::from_millis(300));
    let fmt = StreamFormat::new(RATE, 2).unwrap();
    let sig = signal(30.0, 2);
    let report = play(&DeviceSelector::Default, fmt, &sig).expect("play");
    std::thread::sleep(Duration::from_millis(500));
    let (c, elapsed) = cap.finish();
    verify("30 s", &mono_of(&sig), &report, &c);
    // Drift: host time per timeline frame against the nominal rate, by least squares.
    let n = c.stamps.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    let (f0, t0) = c.stamps[0];
    for &(f, t) in &c.stamps {
        let x = (f - f0) as f64;
        let y = (t.as_ns() as f64 - t0.as_ns() as f64) * RATE as f64 / 1e9;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let ppm = (slope - 1.0) * 1e6;
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    println!(
        "30 s: drift {ppm:+.0} ppm, timeline {timeline_s:.2} s for {:.2} s wall time",
        elapsed.as_secs_f64()
    );
    assert!(ppm.abs() < 500.0, "drift {ppm:+.0} ppm");
    assert_eq!(c.jumps, 0, "timeline jumped");
}

#[test]
fn follows_a_default_sink_change_without_breaking_the_timeline() {
    if !enabled() {
        return;
    }
    let module = pactl(&[
        "load-module",
        "module-null-sink",
        "sink_name=e2e_alt",
        "rate=48000",
        "channels=2",
    ]);
    pactl(&["set-default-sink", "e2e_speaker"]);
    let cap = capture(Source::System, DeviceSelector::Default);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_speaker"));
    std::thread::sleep(Duration::from_secs(1));
    pactl(&["set-default-sink", "e2e_alt"]);
    let events = cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_alt"));
    assert!(
        events.iter().any(|e| matches!(
            e,
            CaptureEvent::Switching {
                reason: SwitchReason::DefaultDeviceChanged
            }
        )),
        "{events:?}"
    );
    std::thread::sleep(Duration::from_millis(300));
    let fmt = StreamFormat::new(RATE, 2).unwrap();
    let sig = signal(2.0, 2);
    let report = play(&DeviceSelector::Default, fmt, &sig).expect("play");
    std::thread::sleep(Duration::from_millis(500));
    let (c, elapsed) = cap.finish();
    pactl(&["set-default-sink", "e2e_speaker"]);
    pactl(&["unload-module", &module]);
    verify("after switch", &mono_of(&sig), &report, &c);
    // The timeline kept running through the switch (silence filled the gap).
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    let wall_s = elapsed.as_secs_f64();
    println!("switch: timeline {timeline_s:.2} s for {wall_s:.2} s wall time");
    assert!(
        (timeline_s - wall_s).abs() < 0.5,
        "timeline {timeline_s:.2} s vs wall {wall_s:.2} s"
    );
}

#[test]
fn a_removed_device_falls_back_to_the_default() {
    if !enabled() {
        return;
    }
    let module = pactl(&[
        "load-module",
        "module-null-sink",
        "sink_name=e2e_tmp",
        "rate=48000",
        "channels=2",
    ]);
    pactl(&["set-default-sink", "e2e_speaker"]);
    let cap = capture(Source::System, DeviceSelector::Id("e2e_tmp".into()));
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_tmp"));
    std::thread::sleep(Duration::from_secs(1));
    pactl(&["unload-module", &module]);
    let events = cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_speaker"));
    assert!(
        events.iter().any(|e| matches!(
            e,
            CaptureEvent::Switching {
                reason: SwitchReason::DeviceRemoved
            }
        )),
        "{events:?}"
    );
    let (c, _) = cap.finish();
    println!(
        "removal: {} blocks, timeline jumps {}, gap fill {} frames",
        c.blocks, c.jumps, c.gap_fill_frames
    );
}

#[test]
fn lists_the_virtual_devices() {
    if !enabled() {
        return;
    }
    let sinks = sd_capture::list_devices(sd_capture::DeviceKind::Output).unwrap();
    let mics = sd_capture::list_devices(sd_capture::DeviceKind::Input).unwrap();
    println!("sinks: {sinks:?}\nmics: {mics:?}");
    assert!(sinks.iter().any(|d| d.id == "e2e_speaker"));
    assert!(mics.iter().any(|d| d.id == "e2e_mic"));
    assert!(
        !mics.iter().any(|d| d.id.ends_with(".monitor")),
        "monitors are not microphones"
    );
}

#[test]
fn returns_to_the_chosen_device_when_it_reappears() {
    if !enabled() {
        return;
    }
    let load = || {
        pactl(&[
            "load-module",
            "module-null-sink",
            "sink_name=e2e_back",
            "rate=48000",
            "channels=2",
        ])
    };
    let module = load();
    pactl(&["set-default-sink", "e2e_speaker"]);
    let cap = capture(Source::System, DeviceSelector::Id("e2e_back".into()));
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_back"));
    std::thread::sleep(Duration::from_millis(500));
    pactl(&["unload-module", &module]);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_speaker"));
    std::thread::sleep(Duration::from_millis(500));
    let module = load();
    let events = cap.wait_for(Duration::from_secs(5), |e| started_on(e, "e2e_back"));
    println!("reappear: back on e2e_back after {} events", events.len());
    let (c, elapsed) = cap.finish();
    pactl(&["unload-module", &module]);
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    println!(
        "reappear: timeline {timeline_s:.2} s for {:.2} s wall time, jumps {}",
        elapsed.as_secs_f64(),
        c.jumps
    );
}
