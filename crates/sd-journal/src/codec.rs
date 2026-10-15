//! Lossless per-block audio codec for journal segments.
//!
//! Every block is coded on its own, so a torn write or a corrupt block never affects another
//! block, and the existing per-block CRC still covers exactly the stored bytes.
//!
//! The scheme is FLAC's "fixed" subframe: per channel, the best of the polynomial predictors
//! of order 0–3 is chosen and its residual is Rice-coded in partitions of [`PARTITION`]
//! samples, each with its own Rice parameter. Stereo blocks may store `left, right - left`
//! instead of `left, right` when that is smaller, which makes mono content in a stereo stream
//! (common for system audio) nearly free. Digital silence costs one byte per channel, and a
//! channel that does not compress is stored verbatim at the bit width it needs.
//!
//! Block layout:
//!
//! ```text
//! u8 stereo mode (0 = independent, 1 = left/side)
//! per channel, byte-aligned:
//!   u8 0                      all zero
//!   u8 1, u8 width, bits      verbatim zigzag values, `width` bits each
//!   u8 2 + order, bits        per partition: 5-bit Rice k, then Rice-coded zigzag residuals
//! ```
//!
//! [`decode`] treats its input as untrusted: any malformed block returns `None`, never panics
//! and never allocates more than the caller's expected size.

const PARTITION: usize = 128;
const MAX_ORDER: usize = 3;
/// Order-3 residuals of side (17-bit) signals stay below 2^20 after zigzag.
const MAX_K: u32 = 24;
/// Bound on decoded zigzag values; anything larger cannot come from the encoder.
const MAX_ZIGZAG: u64 = 1 << 22;

const STEREO_INDEPENDENT: u8 = 0;
const STEREO_LEFT_SIDE: u8 = 1;
const CH_ZERO: u8 = 0;
const CH_VERBATIM: u8 = 1;
const CH_FIXED: u8 = 2;

/// Encodes interleaved `samples` with `channels` channels into `out` (cleared first).
pub fn encode(samples: &[i16], channels: usize, out: &mut Vec<u8>) {
    out.clear();
    debug_assert!(channels > 0 && samples.len().is_multiple_of(channels));
    let frames = samples.len() / channels;
    let chan = |c: usize| -> Vec<i32> {
        (0..frames)
            .map(|f| samples[f * channels + c] as i32)
            .collect()
    };
    let mut scratch = Vec::with_capacity(frames);

    if channels == 2 {
        let (l, r) = (chan(0), chan(1));
        let side: Vec<i32> = l.iter().zip(&r).map(|(a, b)| b - a).collect();
        let mut enc_r = Vec::new();
        let mut enc_s = Vec::new();
        encode_channel(&r, &mut enc_r, &mut scratch);
        encode_channel(&side, &mut enc_s, &mut scratch);
        let (mode, second) = if enc_s.len() < enc_r.len() {
            (STEREO_LEFT_SIDE, enc_s)
        } else {
            (STEREO_INDEPENDENT, enc_r)
        };
        out.push(mode);
        encode_channel(&l, out, &mut scratch);
        out.extend_from_slice(&second);
        return;
    }
    out.push(STEREO_INDEPENDENT);
    for c in 0..channels {
        encode_channel(&chan(c), out, &mut scratch);
    }
}

/// Decodes a block produced by [`encode`]. Returns `None` if `bytes` is not a valid block of
/// exactly `frames` frames of `channels` channels.
pub fn decode(bytes: &[u8], channels: usize, frames: usize) -> Option<Vec<i16>> {
    if channels == 0 {
        return None;
    }
    let (&mode, mut rest) = bytes.split_first()?;
    if mode != STEREO_INDEPENDENT && !(mode == STEREO_LEFT_SIDE && channels == 2) {
        return None;
    }
    let mut chans = Vec::with_capacity(channels);
    for _ in 0..channels {
        let (x, used) = decode_channel(rest, frames)?;
        rest = &rest[used..];
        chans.push(x);
    }
    if !rest.is_empty() {
        return None;
    }
    if mode == STEREO_LEFT_SIDE {
        let (l, s) = chans.split_at_mut(1);
        let (l, s) = (&l[0], &mut s[0]);
        for (si, &li) in s.iter_mut().zip(l) {
            *si += li;
        }
    }
    let mut out = Vec::with_capacity(frames * channels);
    for f in 0..frames {
        for ch in &chans {
            out.push(i16::try_from(ch[f]).ok()?);
        }
    }
    Some(out)
}

fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

/// Prediction of sample `n` from the previous `order` samples (FLAC fixed predictors).
fn predict(x: &[i32], n: usize, order: usize) -> i64 {
    let at = |k: usize| x[n - k] as i64;
    match order {
        0 => 0,
        1 => at(1),
        2 => 2 * at(1) - at(2),
        _ => 3 * at(1) - 3 * at(2) + at(3),
    }
}

/// Zigzag residuals; sample `n < order` uses order `n`, so no warm-up samples are stored.
fn residuals(x: &[i32], order: usize, out: &mut Vec<u32>) {
    out.clear();
    out.extend((0..x.len()).map(|n| zigzag((x[n] as i64 - predict(x, n, order.min(n))) as i32)));
}

fn rice_bits(u: &[u32], k: u32) -> u64 {
    u.iter().map(|&v| (v >> k) as u64).sum::<u64>() + u.len() as u64 * (k as u64 + 1)
}

/// Best Rice parameter for one partition and its cost in bits (including the 5-bit k).
fn best_k(u: &[u32]) -> (u32, u64) {
    let mean = u.iter().map(|&v| v as u64).sum::<u64>() / u.len().max(1) as u64;
    let guess = (64 - mean.leading_zeros()).min(MAX_K);
    (guess.saturating_sub(2)..=(guess + 1).min(MAX_K))
        .map(|k| (k, rice_bits(u, k) + 5))
        .min_by_key(|&(_, b)| b)
        .expect("non-empty range")
}

fn encode_channel(x: &[i32], out: &mut Vec<u8>, scratch: &mut Vec<u32>) {
    if x.iter().all(|&v| v == 0) {
        out.push(CH_ZERO);
        return;
    }
    let mut best: Option<(usize, u64)> = None;
    for order in 0..=MAX_ORDER.min(x.len().saturating_sub(1)) {
        residuals(x, order, scratch);
        let bits: u64 = scratch.chunks(PARTITION).map(|p| best_k(p).1).sum();
        if best.is_none_or(|(_, b)| bits < b) {
            best = Some((order, bits));
        }
    }
    let (order, bits) = best.expect("at least order 0");
    let width = x
        .iter()
        .map(|&v| 32 - zigzag(v).leading_zeros())
        .max()
        .unwrap_or(1);
    let verbatim_bits = 8 + x.len() as u64 * width as u64;

    let mut w = BitWriter::new(out);
    if verbatim_bits <= bits {
        w.out.push(CH_VERBATIM);
        w.out.push(width as u8);
        for &v in x {
            w.put(zigzag(v), width);
        }
    } else {
        w.out.push(CH_FIXED + order as u8);
        residuals(x, order, scratch);
        for p in scratch.chunks(PARTITION) {
            let (k, _) = best_k(p);
            w.put(k, 5);
            for &v in p {
                w.unary(v >> k);
                w.put(v & ((1u32 << k) - 1), k);
            }
        }
    }
    w.finish();
}

/// Decodes one channel; returns the samples and the number of bytes consumed.
fn decode_channel(b: &[u8], frames: usize) -> Option<(Vec<i32>, usize)> {
    let (&tag, body) = b.split_first()?;
    match tag {
        CH_ZERO => Some((vec![0; frames], 1)),
        CH_VERBATIM => {
            let (&width, body) = body.split_first()?;
            let width = width as u32;
            if width == 0 || width > 18 {
                return None;
            }
            if (body.len() as u64) * 8 < frames as u64 * width as u64 {
                return None;
            }
            let mut r = BitReader::new(body);
            let x = (0..frames)
                .map(|_| r.bits(width).map(unzigzag))
                .collect::<Option<Vec<_>>>()?;
            Some((x, 2 + r.bytes_used()))
        }
        t if (CH_FIXED..=CH_FIXED + MAX_ORDER as u8).contains(&t) => {
            let order = (t - CH_FIXED) as usize;
            let mut r = BitReader::new(body);
            let mut x: Vec<i32> = Vec::with_capacity(frames);
            let mut n = 0;
            while n < frames {
                let k = r.bits(5)?;
                if k > MAX_K {
                    return None;
                }
                for _ in 0..PARTITION.min(frames - n) {
                    let q = r.unary()? as u64;
                    let u = (q << k) | r.bits(k)? as u64;
                    if u >= MAX_ZIGZAG {
                        return None;
                    }
                    let v = unzigzag(u as u32) as i64 + predict(&x, n, order.min(n));
                    // Channels hold i16 samples or 17-bit side signals.
                    if !(-(1 << 17)..(1 << 17)).contains(&v) {
                        return None;
                    }
                    x.push(v as i32);
                    n += 1;
                }
            }
            Some((x, 1 + r.bytes_used()))
        }
        _ => None,
    }
}

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    n: u32,
}

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out, acc: 0, n: 0 }
    }

    /// Appends the low `bits` (≤ 32) bits of `v`, most significant first.
    fn put(&mut self, v: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.acc = (self.acc << bits) | (v as u64 & ((1u64 << bits) - 1));
        self.n += bits;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
        self.acc &= (1u64 << self.n) - 1;
    }

    /// `q` zero bits followed by a one.
    fn unary(&mut self, mut q: u32) {
        while q >= 32 {
            self.put(0, 32);
            q -= 32;
        }
        self.put(1, q + 1);
    }

    fn finish(mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put(0, pad);
        }
    }
}

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn bit(&mut self) -> Option<u32> {
        let byte = *self.b.get(self.pos >> 3)?;
        let v = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Some(v as u32)
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Some(v)
    }

    /// Counts zero bits up to the terminating one. Bounded by the input length.
    fn unary(&mut self) -> Option<u32> {
        let mut q = 0u32;
        while self.bit()? == 0 {
            q += 1;
        }
        Some(q)
    }

    fn bytes_used(&self) -> usize {
        self.pos.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG (xorshift) so tests need no extra dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn i16(&mut self) -> i16 {
            self.next() as i16
        }
    }

    fn round_trip(samples: &[i16], channels: usize) -> usize {
        let mut enc = Vec::new();
        encode(samples, channels, &mut enc);
        let dec = decode(&enc, channels, samples.len() / channels).expect("decodes");
        assert_eq!(dec, samples, "lossless round trip ({channels} ch)");
        enc.len()
    }

    #[test]
    fn round_trips_edge_signals() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let full_noise: Vec<i16> = (0..960).map(|_| rng.i16()).collect();
        let extremes: Vec<i16> = (0..960)
            .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
            .collect();
        let sine: Vec<i16> = (0..960)
            .map(|i| ((i as f32 * 0.05).sin() * 20_000.0) as i16)
            .collect();
        let mut opposite = Vec::new();
        for i in 0..480 {
            let v = if i % 3 == 0 { i16::MAX } else { i16::MIN };
            opposite.extend([v, v.saturating_neg()]);
        }
        for s in [&full_noise, &extremes, &sine, &opposite] {
            round_trip(s, 1);
            round_trip(s, 2);
        }
        round_trip(&sine[..957], 3);
        round_trip(&[], 1);
        round_trip(&[5], 1);
        round_trip(&[i16::MIN, i16::MAX], 2);
        // Full-scale noise never expands by more than the per-channel header.
        let n = round_trip(&full_noise, 1);
        assert!(n <= 960 * 2 + 3, "noise expanded to {n} bytes");
    }

    #[test]
    fn compresses_silence_mono_in_stereo_and_tones() {
        assert_eq!(
            round_trip(&[0; 960], 2),
            3,
            "digital silence: header bytes only"
        );
        let tone: Vec<i16> = (0..480)
            .map(|i| ((i as f32 * 0.07).sin() * 3_000.0) as i16)
            .collect();
        let mono_in_stereo: Vec<i16> = tone.iter().flat_map(|&v| [v, v]).collect();
        let mono = round_trip(&tone, 1);
        let stereo = round_trip(&mono_in_stereo, 2);
        assert!(
            stereo <= mono + 2,
            "identical channels: {stereo} vs mono {mono}"
        );
        assert!(
            mono < 480 * 2 / 2,
            "a quiet tone should compress >2x, got {mono} bytes"
        );
    }

    #[test]
    fn random_round_trips() {
        let mut rng = Rng(42);
        for _ in 0..300 {
            let channels = 1 + (rng.next() % 3) as usize;
            let frames = (rng.next() % 600) as usize;
            let amp = 1i32 << (rng.next() % 16);
            let mut prev = 0i32;
            let samples: Vec<i16> = (0..frames * channels)
                .map(|_| {
                    prev = (prev + (rng.next() as i32 % amp)).clamp(-32768, 32767);
                    prev as i16
                })
                .collect();
            round_trip(&samples, channels);
        }
    }

    /// Corrupted or random input must be rejected or decode to *some* block, never panic.
    #[test]
    fn decode_survives_garbage() {
        let mut rng = Rng(7);
        let tone: Vec<i16> = (0..960)
            .map(|i| ((i as f32 * 0.03).sin() * 9_000.0) as i16)
            .collect();
        let mut enc = Vec::new();
        encode(&tone, 2, &mut enc);
        for _ in 0..3_000 {
            let mut b = enc.clone();
            for _ in 0..1 + rng.next() % 4 {
                let i = (rng.next() as usize) % b.len();
                b[i] ^= 1 << (rng.next() % 8);
            }
            b.truncate(b.len() - (rng.next() % 3) as usize);
            let _ = decode(&b, 2, 480);
        }
        for _ in 0..3_000 {
            let len = (rng.next() % 64) as usize;
            let b: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let _ = decode(
                &b,
                1 + (rng.next() % 2) as usize,
                (rng.next() % 500) as usize,
            );
        }
        assert_eq!(decode(&enc, 2, 479), None, "wrong frame count is rejected");
        assert_eq!(
            decode(&enc, 1, 960),
            None,
            "wrong channel count is rejected"
        );
    }
}
