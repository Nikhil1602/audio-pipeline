//! End-to-end capture tests on macOS with the BlackHole virtual audio driver (MIT), which
//! loops what is played into it back to its input. CI installs it with
//! `tools/ci/mac-audio-e2e.sh`:
//!
//! * "BlackHole 2ch" and "BlackHole 16ch": two virtual devices, each an output and an input;
//! * `SwitchAudioSource` (switchaudio-osx) changes the default devices.
//!
//! The tests change system defaults, so they run one at a time, and only when
//! `SD_CAPTURE_E2E=1` (otherwise they pass without doing anything):
//!
//! ```text
//! SD_CAPTURE_E2E=1 cargo test --release -p sd-capture --test mac_e2e -- --test-threads=1 --nocapture
//! ```
#![cfg(target_os = "macos")]

mod common;

use std::process::Command;
use std::time::Duration;

use common::*;
use sd_capture::{
    CaptureEvent, DeviceKind, DeviceSelector, PlaybackReport, SwitchReason, list_devices, play,
};
use sd_core::{Source, StreamFormat};

const DEV_A: &str = "BlackHole 2ch";
const DEV_B: &str = "BlackHole 16ch";

/// Sets the default device of `kind` by name.
fn set_default(kind: &str, name: &str) {
    let out = Command::new("SwitchAudioSource")
        .args(["-t", kind, "-s", name])
        .output()
        .expect("run SwitchAudioSource");
    assert!(
        out.status.success(),
        "SwitchAudioSource -t {kind} -s {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::thread::sleep(Duration::from_millis(300));
}

/// The device id (Core Audio UID) of the device called `name`.
fn uid(kind: DeviceKind, name: &str) -> String {
    let devices = list_devices(kind).expect("list devices");
    devices
        .iter()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("{name} not found among {devices:?}"))
        .id
        .clone()
}

/// Prints a hint when the OS withheld audio for lack of a permission.
fn permission_events(events: &[CaptureEvent]) -> Vec<&CaptureEvent> {
    events
        .iter()
        .filter(|e| matches!(e, CaptureEvent::PermissionNeeded { .. }))
        .collect()
}

/// Runs `once` up to three times while the test player reports skipped output cycles: the CI
/// virtual machine stalled the audio system, so the played signal itself has a gap and the run
/// says nothing about capture.
fn retry_on_player_skips<T>(
    name: &str,
    mut once: impl FnMut() -> (PlaybackReport, T),
) -> (PlaybackReport, T) {
    let mut attempt = 1;
    loop {
        let (report, t) = once();
        if report.underruns == 0 || attempt == 3 {
            return (report, t);
        }
        println!(
            "{name}: the output device skipped {} cycles (machine stall), retrying",
            report.underruns
        );
        attempt += 1;
    }
}

/// Waits until system audio started on `id`; panics with the reason if it could not.
fn expect_system_started(cap: &Capture, id: &str) {
    let events = cap.wait_for(Duration::from_secs(10), |e| {
        started_on(e, id)
            || matches!(
                e,
                CaptureEvent::Retrying { .. } | CaptureEvent::PermissionNeeded { .. }
            )
    });
    match events.last() {
        Some(CaptureEvent::Retrying { error, .. }) => {
            panic!("system audio could not start: {error}");
        }
        Some(CaptureEvent::PermissionNeeded { what, .. }) => {
            panic!("system audio could not start, permission needed: {what}");
        }
        _ => {}
    }
}

#[test]
fn lists_the_virtual_devices() {
    if !enabled() {
        return;
    }
    let outputs = list_devices(DeviceKind::Output).unwrap();
    let inputs = list_devices(DeviceKind::Input).unwrap();
    println!("outputs: {outputs:?}\ninputs: {inputs:?}");
    for name in [DEV_A, DEV_B] {
        assert!(outputs.iter().any(|d| d.name == name), "{name} output");
        assert!(inputs.iter().any(|d| d.name == name), "{name} input");
    }
}

#[test]
fn microphone_hears_the_virtual_device() {
    if !enabled() {
        return;
    }
    set_default("input", DEV_A);
    let id = uid(DeviceKind::Input, DEV_A);
    let sig = signal(3.0, 2);
    let (report, c) = retry_on_player_skips("mic", || {
        let cap = capture(Source::Mic, DeviceSelector::Default);
        let events = cap.wait_for(Duration::from_secs(5), |e| started_on(e, &id));
        std::thread::sleep(Duration::from_millis(300));
        let fmt = StreamFormat::new(RATE, 2).unwrap();
        let report = play(
            &DeviceSelector::Id(uid(DeviceKind::Output, DEV_A)),
            fmt,
            &sig,
        )
        .expect("play");
        std::thread::sleep(Duration::from_millis(500));
        let mut later = Vec::new();
        while let Ok(e) = cap.handle.events().try_recv() {
            later.push(e);
        }
        let (c, _) = cap.finish();
        println!(
            "mic: permission events {:?}",
            permission_events(&[events, later].concat())
        );
        (report, c)
    });
    verify("mic", &mono_of(&sig), &report, &c);
}

#[test]
fn system_audio_tap_hears_what_plays() {
    if !enabled() {
        return;
    }
    set_default("output", DEV_A);
    let id = uid(DeviceKind::Output, DEV_A);
    let sig = signal(3.0, 2);
    let (report, c) = retry_on_player_skips("tap", || {
        let cap = capture(Source::System, DeviceSelector::Default);
        expect_system_started(&cap, &id);
        std::thread::sleep(Duration::from_millis(300));
        let fmt = StreamFormat::new(RATE, 2).unwrap();
        let report = play(&DeviceSelector::Default, fmt, &sig).expect("play");
        std::thread::sleep(Duration::from_millis(500));
        let (c, _) = cap.finish();
        (report, c)
    });
    let peak = c.mono.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("tap: peak {peak:.4} over {} frames", c.mono.len());
    assert!(
        peak > 0.0,
        "the tap delivered only silence: System Audio Recording permission is probably missing"
    );
    verify("tap", &mono_of(&sig), &report, &c);
}

#[test]
fn long_capture_has_no_loss_and_no_drift() {
    if !enabled() {
        return;
    }
    set_default("input", DEV_A);
    let id = uid(DeviceKind::Input, DEV_A);
    let sig = signal(30.0, 2);
    let (report, (c, elapsed)) = retry_on_player_skips("30 s", || {
        let cap = capture(Source::Mic, DeviceSelector::Default);
        cap.wait_for(Duration::from_secs(5), |e| started_on(e, &id));
        std::thread::sleep(Duration::from_millis(300));
        let fmt = StreamFormat::new(RATE, 2).unwrap();
        let report = play(
            &DeviceSelector::Id(uid(DeviceKind::Output, DEV_A)),
            fmt,
            &sig,
        )
        .expect("play");
        std::thread::sleep(Duration::from_millis(500));
        (report, cap.finish())
    });
    verify("30 s", &mono_of(&sig), &report, &c);
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
    let ppm = ((n * sxy - sx * sy) / (n * sxx - sx * sx) - 1.0) * 1e6;
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    println!(
        "30 s: drift {ppm:+.0} ppm, timeline {timeline_s:.2} s for {:.2} s wall time",
        elapsed.as_secs_f64()
    );
    assert!(ppm.abs() < 500.0, "drift {ppm:+.0} ppm");
    assert_eq!(c.jumps, 0, "timeline jumped");
}

#[test]
fn follows_a_default_input_change() {
    if !enabled() {
        return;
    }
    set_default("input", DEV_A);
    let (a, b) = (uid(DeviceKind::Input, DEV_A), uid(DeviceKind::Input, DEV_B));
    let cap = capture(Source::Mic, DeviceSelector::Default);
    cap.wait_for(Duration::from_secs(5), |e| started_on(e, &a));
    std::thread::sleep(Duration::from_secs(1));
    set_default("input", DEV_B);
    let events = cap.wait_for(Duration::from_secs(5), |e| started_on(e, &b));
    assert!(
        events.iter().any(|e| matches!(
            e,
            CaptureEvent::Switching {
                reason: SwitchReason::DefaultDeviceChanged
            }
        )),
        "{events:?}"
    );
    let (c, elapsed) = cap.finish();
    set_default("input", DEV_A);
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    let wall_s = elapsed.as_secs_f64();
    println!("switch: timeline {timeline_s:.2} s for {wall_s:.2} s wall time");
    assert!(
        (timeline_s - wall_s).abs() < 0.5,
        "timeline {timeline_s:.2} s vs wall {wall_s:.2} s"
    );
}

#[test]
fn system_audio_follows_a_default_output_change() {
    if !enabled() {
        return;
    }
    set_default("output", DEV_A);
    let (a, b) = (
        uid(DeviceKind::Output, DEV_A),
        uid(DeviceKind::Output, DEV_B),
    );
    let cap = capture(Source::System, DeviceSelector::Default);
    expect_system_started(&cap, &a);
    std::thread::sleep(Duration::from_secs(1));
    set_default("output", DEV_B);
    let events = cap.wait_for(Duration::from_secs(10), |e| started_on(e, &b));
    assert!(
        events.iter().any(|e| matches!(
            e,
            CaptureEvent::Switching {
                reason: SwitchReason::DefaultDeviceChanged
            }
        )),
        "{events:?}"
    );
    // The tap on the new device hears what plays there.
    std::thread::sleep(Duration::from_millis(300));
    let sig = signal(2.0, 2);
    let report = play(
        &DeviceSelector::Default,
        StreamFormat::new(RATE, 2).unwrap(),
        &sig,
    )
    .expect("play");
    std::thread::sleep(Duration::from_millis(500));
    let (c, elapsed) = cap.finish();
    set_default("output", DEV_A);
    let timeline_s = c.mono.len() as f64 / RATE as f64;
    let wall_s = elapsed.as_secs_f64();
    println!("output switch: timeline {timeline_s:.2} s for {wall_s:.2} s wall time");
    if report.underruns == 0 {
        verify("tap after switch", &mono_of(&sig), &report, &c);
    }
    assert!(
        (timeline_s - wall_s).abs() < 0.5,
        "timeline {timeline_s:.2} s vs wall {wall_s:.2} s"
    );
}
