//! CTC prefix beam search with optional word n-gram language-model fusion.
//!
//! Hypotheses are prefixes of collapsed token sequences, each with the probability of ending
//! in blank and in non-blank (the standard CTC prefix search). Tokens are SentencePiece
//! pieces; a piece starting with `▁` begins a new word, and when a word is complete it is
//! scored by the language model: `alpha · ln P(word | previous two words) + beta`, plus
//! `unk_penalty` if the word is not in the model's vocabulary. The final word and the end of
//! the sentence are scored when decoding ends.
//!
//! Because a word is scored only when it ends, a hypothesis that keeps a word open (gluing
//! the next word on) would postpone its cost and push correct hypotheses out of the beam.
//! The unknown-word penalty is therefore applied as soon as the open word can no longer
//! become a vocabulary word (prefix check), not when it ends.

use std::collections::HashMap;

use crate::lm::{EOS, LmState, NgramLm, UNK};
use crate::metrics::normalize;

const LN10: f32 = std::f32::consts::LN_10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamConfig {
    pub beam: usize,
    /// Per frame, tokens more than this many nats below the best are not expanded.
    pub token_prune: f32,
    /// At most this many tokens are expanded per frame.
    pub max_tokens: usize,
    /// Language-model weight.
    pub alpha: f32,
    /// Bonus per word (counters the LM's preference for fewer words).
    pub beta: f32,
    /// Added (nats) when a word is outside the LM vocabulary.
    pub unk_penalty: f32,
}

/// Defaults tuned for IndicConformer with the IndicCorp trigram models on FLEURS dev (11
/// languages, 100 clips each): weight 0.2, bonus 0.5 and penalty −5 were at or near the best
/// setting for almost every language. The test set was used only to report the result.
impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam: 32,
            token_prune: 8.0,
            max_tokens: 12,
            alpha: 0.2,
            beta: 0.5,
            unk_penalty: -5.0,
        }
    }
}

#[derive(Clone)]
struct Beam {
    tokens: Vec<(u16, u32)>,
    hash: u64,
    last: Option<u16>,
    pb: f32,
    pnb: f32,
    lm: f32,
    state: LmState,
    partial: String,
    /// The open word is already known not to be a vocabulary word (penalty applied).
    oov: bool,
}

fn lse(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn extend_hash(h: u64, tok: u16) -> u64 {
    (h ^ (tok as u64 + 1)).wrapping_mul(0x0000_0100_0000_01b3)
}

/// Scores completed words of `text` (normalized like the LM's training text).
fn score_words(
    lm: Option<&NgramLm>,
    cfg: &BeamConfig,
    state: LmState,
    text: &str,
    unk_already_penalized: bool,
) -> (f32, LmState) {
    let Some(lm) = lm else {
        let n = normalize(text).split(' ').filter(|w| !w.is_empty()).count();
        return (cfg.beta * n as f32, state);
    };
    let mut st = state;
    let mut total = 0.0;
    for w in normalize(text).split(' ').filter(|w| !w.is_empty()) {
        let id = lm.id(w);
        let (p, s) = lm.score(st, id);
        total += cfg.alpha * p * LN10 + cfg.beta;
        if id == UNK && !unk_already_penalized {
            total += cfg.unk_penalty;
        }
        st = s;
    }
    (total, st)
}

/// Decodes `frames` frames of per-token log-probabilities (natural log, one row of
/// `pieces.len()` values per frame, from `logp(t)`). Returns the path as (token, frame,
/// frame + 1).
pub fn decode<'a>(
    frames: usize,
    logp: impl Fn(usize) -> &'a [f32],
    blank: usize,
    pieces: &[String],
    lm: Option<&NgramLm>,
    cfg: &BeamConfig,
) -> Vec<(usize, usize, usize)> {
    let mut beams = vec![Beam {
        tokens: Vec::new(),
        hash: 0xcbf2_9ce4_8422_2325,
        last: None,
        pb: 0.0,
        pnb: f32::NEG_INFINITY,
        lm: 0.0,
        state: LmState::START,
        partial: String::new(),
        oov: false,
    }];
    let mut cands: Vec<(usize, f32)> = Vec::with_capacity(pieces.len());
    for t in 0..frames {
        let lp = logp(t);
        let best = lp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        cands.clear();
        cands.extend(
            lp.iter()
                .enumerate()
                .filter(|&(i, &v)| i != blank && v >= best - cfg.token_prune)
                .map(|(i, &v)| (i, v)),
        );
        cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        cands.truncate(cfg.max_tokens);

        let mut next: HashMap<u64, Beam> = HashMap::with_capacity(beams.len() * 4);
        let same = |b: &Beam| Beam {
            pb: f32::NEG_INFINITY,
            pnb: f32::NEG_INFINITY,
            ..b.clone()
        };
        for b in &beams {
            let total = lse(b.pb, b.pnb);
            let e = next.entry(b.hash).or_insert_with(|| same(b));
            e.pb = lse(e.pb, total + lp[blank]);
            for &(c, lpc) in &cands {
                let c16 = c as u16;
                let from = if b.last == Some(c16) {
                    // A repeated token without a blank in between collapses into the prefix.
                    let e = next.entry(b.hash).or_insert_with(|| same(b));
                    e.pnb = lse(e.pnb, b.pnb + lpc);
                    b.pb
                } else {
                    total
                };
                if from == f32::NEG_INFINITY {
                    continue;
                }
                let score = from + lpc;
                let h = extend_hash(b.hash, c16);
                if let Some(e) = next.get_mut(&h) {
                    e.pnb = lse(e.pnb, score);
                    continue;
                }
                let piece = &pieces[c];
                let mut nb = Beam {
                    tokens: b.tokens.clone(),
                    hash: h,
                    last: Some(c16),
                    pb: f32::NEG_INFINITY,
                    pnb: score,
                    lm: b.lm,
                    state: b.state,
                    partial: b.partial.clone(),
                    oov: b.oov,
                };
                nb.tokens.push((c16, t as u32));
                if piece == "<unk>" {
                    // Kept in the path for alignment, never in the text.
                } else {
                    if let Some(rest) = piece.strip_prefix('▁') {
                        if !nb.partial.is_empty() {
                            let (s, st) = score_words(lm, cfg, nb.state, &nb.partial, nb.oov);
                            nb.lm += s;
                            nb.state = st;
                        }
                        nb.partial = rest.to_string();
                        nb.oov = false;
                    } else {
                        nb.partial.push_str(piece);
                    }
                    if let Some(lm) = lm
                        && !nb.oov
                        && !nb.partial.is_empty()
                        && !lm.is_prefix(&normalize(&nb.partial))
                    {
                        nb.lm += cfg.unk_penalty;
                        nb.oov = true;
                    }
                }
                next.insert(h, nb);
            }
        }
        let mut v: Vec<Beam> = next.into_values().collect();
        v.sort_unstable_by(|a, b| (lse(b.pb, b.pnb) + b.lm).total_cmp(&(lse(a.pb, a.pnb) + a.lm)));
        v.truncate(cfg.beam);
        beams = v;
    }
    // Score the last word and the end of the sentence.
    let finish = |b: &Beam| -> f32 {
        let (s, st) = score_words(lm, cfg, b.state, &b.partial, b.oov);
        let eos = lm.map_or(0.0, |lm| cfg.alpha * lm.score(st, EOS).0 * LN10);
        lse(b.pb, b.pnb) + b.lm + s + eos
    };
    let best = beams
        .iter()
        .max_by(|a, b| finish(a).total_cmp(&finish(b)))
        .expect("at least one beam");
    best.tokens
        .iter()
        .map(|&(tok, f)| (tok as usize, f as usize, f as usize + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces() -> Vec<String> {
        ["▁a", "b", "▁c", "<blank>"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Frames as probabilities over [▁a, b, ▁c, blank].
    fn run(frames: &[[f32; 4]], cfg: &BeamConfig) -> Vec<usize> {
        let rows: Vec<Vec<f32>> = frames
            .iter()
            .map(|f| f.iter().map(|p| p.max(1e-9).ln()).collect())
            .collect();
        decode(rows.len(), |t| &rows[t], 3, &pieces(), None, cfg)
            .into_iter()
            .map(|p| p.0)
            .collect()
    }

    #[test]
    fn collapses_repeats_and_keeps_tokens_separated_by_blank() {
        let cfg = BeamConfig {
            beta: 0.0,
            ..Default::default()
        };
        // a a blank a -> "a a"
        let f = [
            [0.9, 0.0, 0.0, 0.1],
            [0.9, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
            [0.9, 0.0, 0.0, 0.1],
        ];
        assert_eq!(run(&f, &cfg), vec![0, 0]);
        // a a -> "a"
        assert_eq!(run(&f[..2], &cfg), vec![0]);
    }

    #[test]
    fn sums_paths_that_greedy_decoding_misses() {
        // Greedy picks blank in every frame (0.4 > 0.3), but "b" via its three paths
        // (b-, -b, bb) has probability 0.3*0.4 + 0.4*0.3 + 0.3*0.3 = 0.33 > blank-blank 0.16.
        let cfg = BeamConfig {
            beta: 0.0,
            ..Default::default()
        };
        let f = [[0.0, 0.3, 0.3, 0.4], [0.0, 0.3, 0.3, 0.4]];
        let out = run(&f, &cfg);
        assert_eq!(out.len(), 1, "{out:?}");
    }
}
