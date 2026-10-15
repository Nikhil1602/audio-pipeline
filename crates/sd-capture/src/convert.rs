//! Converts device audio (any rate, any channel count, interleaved f32) to the capture format
//! (the configured rate, mono or stereo), for APIs that deliver the device's own format
//! (Core Audio). Windows and PulseAudio convert in the OS instead.
//!
//! * Channels: mono output averages all input channels; stereo output takes the first two
//!   (a mono input is duplicated).
//! * Rate: WebRTC's sinc resampler (as in the processing stage), in fixed chunks of
//!   [`Converter::chunk_ms`] ms; equal rates pass straight through.
//! * Timing: every output chunk carries the host time of its first input frame, interpolated
//!   from the packets' host times, and output frames are counted continuously so their count is
//!   the stream position.

use std::collections::VecDeque;

use sonora_common_audio::push_sinc_resampler::PushSincResampler;

/// Output frames with the host time (ns) of the first one; `data` is interleaved.
pub(crate) struct Chunk<'a> {
    pub host_ns: u64,
    /// Output frames emitted before this chunk (the stream position of its first frame).
    pub position: u64,
    pub frames: usize,
    pub data: &'a [f32],
}

pub(crate) struct Converter {
    in_rate: u32,
    in_channels: usize,
    out_channels: usize,
    /// Per output channel, when rates differ.
    resamplers: Vec<PushSincResampler>,
    chunk_in: usize,
    chunk_out: usize,
    /// Mapped input frames not yet resampled, per output channel.
    pending: Vec<Vec<f32>>,
    /// Input frame index of `pending[..][0]`.
    pending_start: u64,
    /// Input frames received so far.
    in_index: u64,
    /// (input frame index, host time ns) of recent packets.
    anchors: VecDeque<(u64, f64)>,
    out_position: u64,
    mapped: Vec<f32>,
    out: Vec<f32>,
    out_planar: Vec<Vec<f32>>,
}

impl Converter {
    pub fn new(in_rate: u32, in_channels: usize, out_rate: u32, out_channels: usize) -> Self {
        assert!(in_channels > 0 && (1..=2).contains(&out_channels));
        let (chunk_in, chunk_out, resamplers) = if in_rate == out_rate {
            (0, 0, Vec::new())
        } else {
            let ms = Self::chunk_ms(in_rate, out_rate);
            let ci = (in_rate as u64 * ms as u64 / 1000) as usize;
            let co = (out_rate as u64 * ms as u64 / 1000) as usize;
            (
                ci,
                co,
                (0..out_channels)
                    .map(|_| PushSincResampler::new(ci, co))
                    .collect(),
            )
        };
        Self {
            in_rate,
            in_channels,
            out_channels,
            resamplers,
            chunk_in,
            chunk_out,
            pending: vec![Vec::new(); out_channels],
            pending_start: 0,
            in_index: 0,
            anchors: VecDeque::new(),
            out_position: 0,
            mapped: Vec::new(),
            out: Vec::new(),
            out_planar: vec![vec![0.0; chunk_out]; out_channels],
        }
    }

    /// The shortest chunk (10, 20, ... ms) holding a whole number of frames at both rates.
    pub fn chunk_ms(in_rate: u32, out_rate: u32) -> u32 {
        (1..=100)
            .map(|k| k * 10)
            .find(|&ms| (in_rate * ms).is_multiple_of(1000) && (out_rate * ms).is_multiple_of(1000))
            .unwrap_or(1000)
    }

    /// Host time of input frame `index`, from the latest packet that started at or before it.
    fn host_of(&self, index: u64) -> u64 {
        let anchor = self
            .anchors
            .iter()
            .rev()
            .find(|a| a.0 <= index)
            .or(self.anchors.front())
            .copied()
            .unwrap_or((index, 0.0));
        let t = anchor.1 + (index as f64 - anchor.0 as f64) * 1e9 / self.in_rate as f64;
        t.max(0.0) as u64
    }

    /// Adds a packet of `data` (interleaved, `in_channels`) whose first frame was captured at
    /// `host_ns`, and calls `emit` for every output chunk that became ready.
    pub fn push(&mut self, host_ns: u64, data: &[f32], mut emit: impl FnMut(Chunk<'_>)) {
        let frames = data.len() / self.in_channels;
        if frames == 0 {
            return;
        }
        self.anchors.push_back((self.in_index, host_ns as f64));
        // Map channels.
        let (ic, oc) = (self.in_channels, self.out_channels);
        self.mapped.clear();
        for f in data.chunks_exact(ic) {
            match oc {
                1 => self.mapped.push(f.iter().sum::<f32>() / ic as f32),
                _ => {
                    self.mapped.push(f[0]);
                    self.mapped.push(if ic > 1 { f[1] } else { f[0] });
                }
            }
        }
        if self.resamplers.is_empty() {
            let position = self.out_position;
            self.out_position += frames as u64;
            emit(Chunk {
                host_ns,
                position,
                frames,
                data: &self.mapped,
            });
            self.in_index += frames as u64;
            self.anchors.pop_front();
            return;
        }
        for (k, p) in self.pending.iter_mut().enumerate() {
            p.extend(self.mapped.iter().skip(k).step_by(oc));
        }
        self.in_index += frames as u64;
        while self.pending[0].len() >= self.chunk_in {
            for (k, r) in self.resamplers.iter_mut().enumerate() {
                r.resample(&self.pending[k][..self.chunk_in], &mut self.out_planar[k]);
                self.pending[k].drain(..self.chunk_in);
            }
            self.out.clear();
            for i in 0..self.chunk_out {
                for k in 0..oc {
                    self.out.push(self.out_planar[k][i]);
                }
            }
            let host_ns = self.host_of(self.pending_start);
            self.pending_start += self.chunk_in as u64;
            let position = self.out_position;
            self.out_position += self.chunk_out as u64;
            emit(Chunk {
                host_ns,
                position,
                frames: self.chunk_out,
                data: &self.out,
            });
        }
        // Anchors before the pending input are no longer needed (keep the one covering it).
        while self.anchors.len() > 1 && self.anchors[1].0 <= self.pending_start {
            self.anchors.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_sizes_hold_whole_frames() {
        assert_eq!(Converter::chunk_ms(48_000, 48_000), 10);
        assert_eq!(Converter::chunk_ms(44_100, 48_000), 10);
        assert_eq!(Converter::chunk_ms(22_050, 48_000), 20);
        assert_eq!(Converter::chunk_ms(11_025, 48_000), 40);
    }

    #[test]
    fn same_rate_maps_channels_and_passes_through() {
        let mut c = Converter::new(48_000, 4, 48_000, 1);
        let data: Vec<f32> = (0..480 * 4).map(|i| (i % 4) as f32).collect();
        let mut got = Vec::new();
        c.push(1_000, &data, |ch| {
            got.push((ch.host_ns, ch.position, ch.data.to_vec()))
        });
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 1_000);
        assert!(
            got[0].2.iter().all(|&v| (v - 1.5).abs() < 1e-6),
            "mean of 0,1,2,3"
        );
        let mut s = Converter::new(48_000, 1, 48_000, 2);
        s.push(0, &[0.25; 10], |ch| assert_eq!(ch.data, &[0.25; 20][..]));
    }

    #[test]
    fn resampling_keeps_rate_timing_and_a_tone() {
        // 44.1 kHz stereo, 1 kHz tone, in odd packet sizes; out 48 kHz mono.
        let mut c = Converter::new(44_100, 2, 48_000, 1);
        let mut out: Vec<f32> = Vec::new();
        let mut chunks: Vec<(u64, u64, usize)> = Vec::new();
        let mut n = 0usize;
        let mut packet = 0;
        while n < 44_100 {
            let len = [512usize, 470, 333][packet % 3].min(44_100 - n);
            let host = (n as f64 * 1e9 / 44_100.0) as u64 + 5_000_000_000;
            let data: Vec<f32> = (n..n + len)
                .flat_map(|i| {
                    let v = (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 44_100.0).sin();
                    [v, v]
                })
                .collect();
            c.push(host, &data, |ch| {
                out.extend_from_slice(ch.data);
                chunks.push((ch.host_ns, ch.position, ch.frames));
            });
            n += len;
            packet += 1;
        }
        // 1 s in -> 1 s out (whole 10 ms chunks).
        assert_eq!(out.len(), 48_000);
        // Positions are continuous and host times follow the input clock.
        for w in chunks.windows(2) {
            assert_eq!(w[1].1, w[0].1 + w[0].2 as u64);
            let dt = w[1].0 as f64 - w[0].0 as f64;
            assert!((dt - 10e6).abs() < 1e3, "chunk spacing {dt} ns");
        }
        assert_eq!(chunks[0].0, 5_000_000_000);
        // The tone survives: correlate the second half with a 1 kHz sine at 48 kHz.
        let (mut best, half) = (0.0f64, &out[24_000..]);
        for phase in 0..48 {
            let mut c = 0.0f64;
            for (i, &v) in half.iter().enumerate() {
                let r =
                    (2.0 * std::f64::consts::PI * 1_000.0 * (i + phase) as f64 / 48_000.0).sin();
                c += v as f64 * r;
            }
            best = best.max(c / (half.len() as f64 / 2.0));
        }
        assert!(best > 0.95, "tone correlation {best}");
    }
}
