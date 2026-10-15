//! Word n-gram language models (orders 1–3) for beam-search decoding.
//!
//! * [`build`]: interpolated modified Kneser-Ney (Chen & Goodman, 1998) from a text file, one
//!   sentence per line, with a vocabulary cap and count pruning of bigrams and trigrams.
//! * [`NgramLm`]: a backoff model as in the ARPA format (each stored probability is the full
//!   interpolated estimate; the backoff weight of a context is its interpolation weight),
//!   kept in sorted arrays and queried by binary search.
//!
//! Words are compared after [`crate::metrics::normalize`], so the model and the decoder agree
//! on case, punctuation and whitespace. Implemented from the published method; no code from
//! other toolkits.

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::path::Path;

use crate::metrics::normalize;

pub const UNK: u32 = 0;
pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
const BITS: u32 = 21;
const MAGIC: &[u8; 8] = b"APLM\x01\0\0\0";

#[derive(Debug, thiserror::Error)]
pub enum LmError {
    #[error("{0}: {1}")]
    Io(String, std::io::Error),
    #[error("{0}: not a language model file ({1})")]
    Format(String, String),
    #[error("{0}")]
    Invalid(String),
}

/// Language-model state: the last two word ids (BOS at sentence start).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LmState(pub u32, pub u32);

impl LmState {
    pub const START: LmState = LmState(BOS, BOS);
}

#[derive(Debug, Default)]
struct Level {
    keys: Vec<u64>,
    logp: Vec<f32>,
    bow: Vec<f32>,
}

impl Level {
    fn find(&self, key: u64) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }
}

/// A word n-gram backoff model. Probabilities are log10, as in ARPA files.
#[derive(Debug)]
pub struct NgramLm {
    order: usize,
    words: Vec<String>,
    ids: HashMap<String, u32>,
    /// Word ids (specials excluded) in byte order of the words, for prefix queries.
    sorted: Vec<u32>,
    /// levels[0]: unigrams indexed by id (keys unused); levels[1]: bigrams; levels[2]: trigrams.
    levels: Vec<Level>,
}

fn key(ws: &[u32]) -> u64 {
    ws.iter().fold(0u64, |k, &w| (k << BITS) | w as u64)
}

impl NgramLm {
    pub fn order(&self) -> usize {
        self.order
    }

    pub fn vocab_size(&self) -> usize {
        self.words.len()
    }

    /// Number of stored n-grams per order.
    pub fn sizes(&self) -> Vec<usize> {
        let mut v = vec![self.words.len()];
        v.extend(self.levels[1..].iter().map(|l| l.keys.len()));
        v
    }

    /// The id of an already normalized word, or [`UNK`].
    pub fn id(&self, word: &str) -> u32 {
        self.ids.get(word).copied().unwrap_or(UNK)
    }

    /// True if some vocabulary word starts with `prefix` (already normalized).
    pub fn is_prefix(&self, prefix: &str) -> bool {
        let i = self
            .sorted
            .partition_point(|&id| self.words[id as usize].as_str() < prefix);
        self.sorted
            .get(i)
            .is_some_and(|&id| self.words[id as usize].starts_with(prefix))
    }

    /// log10 P(w | state) with backoff, and the next state.
    pub fn score(&self, state: LmState, w: u32) -> (f32, LmState) {
        let next = LmState(state.1, w);
        (self.logp(&[state.0, state.1, w][3 - self.order..]), next)
    }

    fn logp(&self, ngram: &[u32]) -> f32 {
        let n = ngram.len();
        if n == 1 {
            return self.levels[0].logp[ngram[0] as usize];
        }
        if let Some(i) = self.levels[n - 1].find(key(ngram)) {
            return self.levels[n - 1].logp[i];
        }
        let ctx = &ngram[..n - 1];
        let bow = if ctx.len() == 1 {
            self.levels[0].bow[ctx[0] as usize]
        } else {
            self.levels[ctx.len() - 1]
                .find(key(ctx))
                .map_or(0.0, |i| self.levels[ctx.len() - 1].bow[i])
        };
        bow + self.logp(&ngram[1..])
    }

    /// log10 probability of a whole sentence (already normalized words), with </s>.
    pub fn sentence_logp(&self, sentence: &str) -> f32 {
        let mut st = LmState::START;
        let mut total = 0.0;
        for w in sentence.split(' ').filter(|w| !w.is_empty()) {
            let (p, s) = self.score(st, self.id(w));
            total += p;
            st = s;
        }
        total + self.score(st, EOS).0
    }

    pub fn save(&self, path: &Path) -> Result<(), LmError> {
        let io = |e| LmError::Io(path.display().to_string(), e);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
        f.write_all(MAGIC).map_err(io)?;
        f.write_all(&(self.order as u32).to_le_bytes())
            .map_err(io)?;
        f.write_all(&(self.words.len() as u32).to_le_bytes())
            .map_err(io)?;
        for w in &self.words {
            f.write_all(&(w.len() as u32).to_le_bytes()).map_err(io)?;
            f.write_all(w.as_bytes()).map_err(io)?;
        }
        for (n, l) in self.levels.iter().enumerate() {
            f.write_all(&(l.logp.len() as u64).to_le_bytes())
                .map_err(io)?;
            if n > 0 {
                for k in &l.keys {
                    f.write_all(&k.to_le_bytes()).map_err(io)?;
                }
            }
            for v in l.logp.iter().chain(&l.bow) {
                f.write_all(&v.to_le_bytes()).map_err(io)?;
            }
        }
        f.flush().map_err(io)
    }

    pub fn load(path: &Path) -> Result<Self, LmError> {
        let name = path.display().to_string();
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| LmError::Io(name.clone(), e))?;
        let bad = |why: &str| LmError::Format(name.clone(), why.to_string());
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8], LmError> {
            let s = bytes.get(pos..pos + n).ok_or_else(|| bad("truncated"))?;
            pos += n;
            Ok(s)
        };
        if take(8)? != MAGIC {
            return Err(bad("bad magic"));
        }
        let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
        let order = u32_at(take(4)?) as usize;
        if !(1..=3).contains(&order) {
            return Err(bad("order must be 1-3"));
        }
        let nwords = u32_at(take(4)?) as usize;
        let mut words = Vec::with_capacity(nwords);
        for _ in 0..nwords {
            let len = u32_at(take(4)?) as usize;
            let w = std::str::from_utf8(take(len)?).map_err(|_| bad("word is not UTF-8"))?;
            words.push(w.to_string());
        }
        let mut levels = Vec::with_capacity(order);
        for n in 0..order {
            let count = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes")) as usize;
            let keys = if n > 0 {
                take(count * 8)?
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
                    .collect()
            } else {
                Vec::new()
            };
            let mut floats = |count: usize| -> Result<Vec<f32>, LmError> {
                Ok(take(count * 4)?
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                    .collect())
            };
            let logp = floats(count)?;
            let bow = floats(count)?;
            levels.push(Level { keys, logp, bow });
        }
        if levels[0].logp.len() != nwords {
            return Err(bad("unigram count does not match the vocabulary"));
        }
        let ids = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Ok(Self {
            order,
            sorted: sorted_ids(&words),
            words,
            ids,
            levels,
        })
    }
}

/// Model-building options.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    pub order: usize,
    /// Most frequent words kept; the rest become `<unk>`.
    pub vocab: usize,
    /// Bigrams and trigrams seen fewer times than this are dropped from the stored model
    /// (they still shape the smoothing statistics). 1 keeps everything.
    pub min_count: u32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            order: 3,
            vocab: 200_000,
            min_count: 2,
        }
    }
}

/// Summary of a built model.
#[derive(Debug, Clone)]
pub struct BuildReport {
    pub sentences: u64,
    pub tokens: u64,
    pub distinct_words: usize,
    pub oov_rate: f64,
    pub discounts: Vec<[f64; 3]>,
    pub sizes: Vec<usize>,
}

fn for_each_sentence(path: &Path, mut f: impl FnMut(Vec<&str>)) -> Result<(), LmError> {
    let io = |e| LmError::Io(path.display().to_string(), e);
    let file = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(path).map_err(io)?);
    for line in file.lines() {
        let line = line.map_err(io)?;
        let norm = normalize(&line);
        if norm.is_empty() {
            continue;
        }
        f(norm.split(' ').collect());
    }
    Ok(())
}

/// Sorted keys -> (key, count) runs.
fn runs(mut keys: Vec<u64>) -> (Vec<u64>, Vec<u32>) {
    keys.sort_unstable();
    runs_sorted(keys)
}

fn runs_sorted(keys: Vec<u64>) -> (Vec<u64>, Vec<u32>) {
    let mut k = Vec::new();
    let mut c: Vec<u32> = Vec::new();
    for x in keys {
        if k.last() == Some(&x) {
            *c.last_mut().expect("parallel") += 1;
        } else {
            k.push(x);
            c.push(1);
        }
    }
    (k, c)
}

/// Merges two sorted (key, count) lists, adding the counts of equal keys.
fn merge_counts(a: (Vec<u64>, Vec<u32>), b: (Vec<u64>, Vec<u32>)) -> (Vec<u64>, Vec<u32>) {
    if a.0.is_empty() {
        return b;
    }
    if b.0.is_empty() {
        return a;
    }
    let mut k = Vec::with_capacity(a.0.len() + b.0.len());
    let mut c = Vec::with_capacity(a.0.len() + b.0.len());
    let (mut i, mut j) = (0, 0);
    while i < a.0.len() || j < b.0.len() {
        if j == b.0.len() || (i < a.0.len() && a.0[i] < b.0[j]) {
            k.push(a.0[i]);
            c.push(a.1[i]);
            i += 1;
        } else if i == a.0.len() || b.0[j] < a.0[i] {
            k.push(b.0[j]);
            c.push(b.1[j]);
            j += 1;
        } else {
            k.push(a.0[i]);
            c.push(a.1[i].saturating_add(b.1[j]));
            i += 1;
            j += 1;
        }
    }
    k.shrink_to_fit();
    c.shrink_to_fit();
    (k, c)
}

/// Counts the n-grams of order `n`: keys and raw counts, sorted by key. Keys are buffered,
/// sorted and merged in batches, so memory follows the number of distinct n-grams rather
/// than the size of the corpus.
fn count_order(
    corpus: &Path,
    ids: &HashMap<String, u32>,
    n: usize,
) -> Result<(Vec<u64>, Vec<u32>), LmError> {
    const BATCH: usize = 1 << 25;
    let mut buf: Vec<u64> = Vec::with_capacity(BATCH);
    let mut acc: (Vec<u64>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut seq: Vec<u32> = Vec::new();
    for_each_sentence(corpus, |ws| {
        seq.clear();
        seq.push(BOS);
        seq.extend(ws.iter().map(|w| ids.get(*w).copied().unwrap_or(UNK)));
        seq.push(EOS);
        for i in 1..seq.len() {
            if i + 1 >= n {
                buf.push(key(&seq[i + 1 - n..=i]));
            }
        }
        if buf.len() >= BATCH {
            let batch = runs(std::mem::replace(&mut buf, Vec::with_capacity(BATCH)));
            acc = merge_counts(std::mem::take(&mut acc), batch);
        }
    })?;
    Ok(merge_counts(acc, runs(buf)))
}

/// Modified Kneser-Ney discounts D1, D2, D3+ from counts-of-counts.
fn discounts(counts: &[u32]) -> [f64; 3] {
    let mut n = [0f64; 5];
    for &c in counts {
        if (1..=4).contains(&c) {
            n[c as usize] += 1.0;
        }
    }
    if n[1] == 0.0 || n[2] == 0.0 || n[3] == 0.0 {
        return [0.5, 1.0, 1.5];
    }
    let y = n[1] / (n[1] + 2.0 * n[2]);
    let d = [
        1.0 - 2.0 * y * n[2] / n[1],
        2.0 - 3.0 * y * n[3] / n[2],
        3.0 - 4.0 * y * n[4] / n[3],
    ];
    [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)]
}

fn disc(d: &[f64; 3], c: u32) -> f64 {
    match c {
        0 => 0.0,
        1 => d[0],
        2 => d[1],
        _ => d[2],
    }
}

/// For one order's n-grams (sorted, so each context's n-grams are contiguous): per context,
/// the sum of counts and the interpolation weight γ = (D1·N1 + D2·N2 + D3·N3+) / sum.
/// `out[ctx_index(ctx)]` receives (sum, γ); contexts without n-grams keep (0, 0).
fn context_weights(
    keys: &[u64],
    counts: &[u32],
    d: &[f64; 3],
    ctx_len: usize,
    ctx_index: impl Fn(u64) -> Option<usize>,
) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0, 0.0); ctx_len];
    let mut i = 0;
    while i < keys.len() {
        let ctx = keys[i] >> BITS;
        let (mut total, mut nk) = (0.0f64, [0f64; 3]);
        while i < keys.len() && keys[i] >> BITS == ctx {
            let c = counts[i];
            if c > 0 {
                total += c as f64;
                nk[(c.min(3) - 1) as usize] += 1.0;
            }
            i += 1;
        }
        if total > 0.0
            && let Some(j) = ctx_index(ctx)
        {
            out[j] = (total, (d[0] * nk[0] + d[1] * nk[1] + d[2] * nk[2]) / total);
        }
    }
    out
}

fn retain_by<T>(v: &mut Vec<T>, keep: &[bool]) {
    let mut i = 0;
    v.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}

/// Builds a model from `corpus` (one sentence per line).
pub fn build(corpus: &Path, cfg: &BuildConfig) -> Result<(NgramLm, BuildReport), LmError> {
    let order = cfg.order;
    if !(1..=3).contains(&order) {
        return Err(LmError::Invalid("order must be 1, 2 or 3".into()));
    }
    // Pass 1: vocabulary.
    let mut freq: HashMap<String, u64> = HashMap::new();
    let (mut sentences, mut tokens) = (0u64, 0u64);
    for_each_sentence(corpus, |ws| {
        sentences += 1;
        tokens += ws.len() as u64;
        for w in ws {
            match freq.get_mut(w) {
                Some(c) => *c += 1,
                None => {
                    freq.insert(w.to_string(), 1);
                }
            }
        }
    })?;
    let distinct_words = freq.len();
    let mut by_freq: Vec<(String, u64)> = freq.into_iter().collect();
    by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let cap = cfg.vocab.min((1usize << BITS) - 3);
    let in_vocab: u64 = by_freq.iter().take(cap).map(|x| x.1).sum();
    by_freq.truncate(cap);
    let mut words: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    words.extend(by_freq.into_iter().map(|x| x.0));
    let ids: HashMap<String, u32> = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i as u32))
        .collect();
    let v = words.len();

    // Passes 2..: raw counts, one order at a time. keys[l] / raw[l] hold order l + 1.
    let mut keys: Vec<Vec<u64>> = Vec::with_capacity(order);
    let mut raw: Vec<Vec<u32>> = Vec::with_capacity(order);
    for n in 1..=order {
        let (k, c) = count_order(corpus, &ids, n)?;
        keys.push(k);
        raw.push(c);
    }

    // Adjusted counts: the highest order and n-grams starting with <s> keep raw counts; lower
    // orders use continuation counts (number of distinct left extensions).
    let mut adj: Vec<Option<Vec<u32>>> = vec![None; order];
    for l in (0..order - 1).rev() {
        let n = l as u32 + 1;
        let mask = (1u64 << (BITS * n)) - 1;
        let mut suffix: Vec<u64> = keys[l + 1].iter().map(|&k| k & mask).collect();
        suffix.sort_unstable();
        let (sk, sc) = runs_sorted(suffix);
        let mut a = raw[l].clone();
        let mut j = 0;
        for (i, &k) in keys[l].iter().enumerate() {
            if (k >> (BITS * (n - 1))) == BOS as u64 {
                continue;
            }
            while j < sk.len() && sk[j] < k {
                j += 1;
            }
            a[i] = if j < sk.len() && sk[j] == k { sc[j] } else { 0 };
        }
        adj[l] = Some(a);
    }
    let counts = |adj: &[Option<Vec<u32>>], raw: &[Vec<u32>], l: usize| -> Vec<u32> {
        adj[l].clone().unwrap_or_else(|| raw[l].clone())
    };
    let ds: Vec<[f64; 3]> = (0..order)
        .map(|l| discounts(adj[l].as_deref().unwrap_or(&raw[l])))
        .collect();

    // Context weights: ctxw[l] is aligned with order l + 1 and describes the order l + 2
    // n-grams that extend each entry.
    let mut ctxw: Vec<Vec<(f64, f64)>> = Vec::with_capacity(order.saturating_sub(1));
    for l in 1..order {
        let c = adj[l].as_deref().unwrap_or(&raw[l]);
        let w = if l == 1 {
            context_weights(&keys[l], c, &ds[l], v, |ctx| Some(ctx as usize))
        } else {
            let lower = &keys[l - 1];
            context_weights(&keys[l], c, &ds[l], lower.len(), |ctx| {
                lower.binary_search(&ctx).ok()
            })
        };
        ctxw.push(w);
    }

    // Pruning (before probabilities are stored): drop bigrams/trigrams seen fewer than
    // `min_count` times, unless a kept trigram needs the bigram as its context or suffix.
    if cfg.min_count > 1 && order >= 2 {
        for l in (1..order).rev() {
            let mut keep: Vec<bool> = raw[l].iter().map(|&c| c >= cfg.min_count).collect();
            if l + 1 < order {
                let mask = (1u64 << (BITS * (l as u32 + 1))) - 1;
                let mut needed: Vec<u64> = keys[l + 1]
                    .iter()
                    .flat_map(|&k| [k >> BITS, k & mask])
                    .collect();
                needed.sort_unstable();
                needed.dedup();
                for k in needed {
                    if let Ok(i) = keys[l].binary_search(&k) {
                        keep[i] = true;
                    }
                }
            }
            retain_by(&mut keys[l], &keep);
            retain_by(&mut raw[l], &keep);
            if let Some(a) = adj[l].as_mut() {
                retain_by(a, &keep);
            }
            if l < ctxw.len() {
                retain_by(&mut ctxw[l], &keep);
            }
        }
    }

    // Probabilities, lowest order first.
    let mut levels: Vec<Level> = Vec::with_capacity(order);
    {
        // Unigrams: interpolate with the uniform distribution over every word but <s>.
        let mut a = vec![0u32; v];
        for (&k, &c) in keys[0].iter().zip(&counts(&adj, &raw, 0)) {
            a[k as usize] = c;
        }
        a[BOS as usize] = 0;
        let total: f64 = a.iter().map(|&c| c as f64).sum();
        let mut nk = [0f64; 3];
        for &c in &a {
            if c > 0 {
                nk[(c.min(3) - 1) as usize] += 1.0;
            }
        }
        let d = &ds[0];
        let gamma = (d[0] * nk[0] + d[1] * nk[1] + d[2] * nk[2]) / total.max(1.0);
        let uniform = 1.0 / (v - 1) as f64;
        let logp: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if i == BOS as usize {
                    -99.0
                } else {
                    let p = ((c as f64 - disc(d, c)).max(0.0)) / total.max(1.0) + gamma * uniform;
                    p.log10() as f32
                }
            })
            .collect();
        levels.push(Level {
            keys: Vec::new(),
            logp,
            bow: vec![0.0; v],
        });
    }
    for l in 1..order {
        let c = counts(&adj, &raw, l);
        let d = ds[l];
        let suffix_mask = (1u64 << (BITS * l as u32)) - 1;
        let mut logp = Vec::with_capacity(keys[l].len());
        for (&k, &c) in keys[l].iter().zip(&c) {
            let ctx = k >> BITS;
            let ci = if l == 1 {
                Some(ctx as usize)
            } else {
                levels[l - 1].find(ctx)
            };
            let (total, gamma) = ci.map_or((0.0, 0.0), |i| ctxw[l - 1][i]);
            let suffix = k & suffix_mask;
            let lower = &levels[l - 1];
            let lower_logp = if l == 1 {
                lower.logp[suffix as usize]
            } else {
                lower
                    .find(suffix)
                    .map_or(f32::NEG_INFINITY, |i| lower.logp[i])
            };
            let lower_p = 10f64.powf(lower_logp as f64);
            let p = if total > 0.0 {
                ((c as f64 - disc(&d, c)).max(0.0)) / total + gamma * lower_p
            } else {
                lower_p
            };
            logp.push(p.log10() as f32);
        }
        // Backoff weights of this order's contexts, on the order below.
        for (b, &(total, gamma)) in levels[l - 1].bow.iter_mut().zip(&ctxw[l - 1]) {
            if total > 0.0 {
                *b = gamma.max(1e-30).log10() as f32;
            }
        }
        let n = keys[l].len();
        levels.push(Level {
            keys: std::mem::take(&mut keys[l]),
            logp,
            bow: vec![0.0; n],
        });
    }

    let lm = NgramLm {
        order,
        sorted: sorted_ids(&words),
        words,
        ids,
        levels,
    };
    let report = BuildReport {
        sentences,
        tokens,
        distinct_words,
        oov_rate: 1.0 - in_vocab as f64 / tokens.max(1) as f64,
        discounts: ds,
        sizes: lm.sizes(),
    };
    Ok((lm, report))
}

fn sorted_ids(words: &[String]) -> Vec<u32> {
    let mut v: Vec<u32> = (EOS + 1..words.len() as u32).collect();
    v.sort_unstable_by(|&a, &b| words[a as usize].cmp(&words[b as usize]));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(lines: &[&str]) -> tempfile_path::TempPath {
        let p = tempfile_path::TempPath::new();
        std::fs::write(&p.0, lines.join("\n")).unwrap();
        p
    }

    mod tempfile_path {
        pub struct TempPath(pub std::path::PathBuf);
        impl TempPath {
            pub fn new() -> Self {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                let p = std::env::temp_dir().join(format!(
                    "sd-lm-test-{}-{}.txt",
                    std::process::id(),
                    N.fetch_add(1, Ordering::Relaxed)
                ));
                Self(p)
            }
        }
        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    fn sample() -> Vec<&'static str> {
        let mut v = Vec::new();
        for _ in 0..20 {
            v.extend_from_slice(&[
                "the cat sat on the mat",
                "the dog sat on the log",
                "a cat ran to the dog",
                "the cat ate the fish",
            ]);
        }
        v
    }

    #[test]
    fn probabilities_sum_to_one_over_the_vocabulary() {
        let c = corpus(&sample());
        let cfg = BuildConfig {
            order: 3,
            vocab: 1000,
            min_count: 1,
        };
        let (lm, _) = build(&c.0, &cfg).unwrap();
        for ctx in [
            LmState::START,
            LmState(lm.id("the"), lm.id("cat")),
            LmState(lm.id("sat"), lm.id("on")),
            LmState(lm.id("fish"), lm.id("dog")), // unseen context
        ] {
            let total: f64 = (0..lm.vocab_size() as u32)
                .filter(|&w| w != BOS)
                .map(|w| 10f64.powf(lm.score(ctx, w).0 as f64))
                .sum();
            assert!((total - 1.0).abs() < 1e-3, "{ctx:?}: {total}");
        }
    }

    #[test]
    fn seen_word_orders_score_higher_and_roundtrip_through_a_file() {
        let c = corpus(&sample());
        let (lm, rep) = build(&c.0, &BuildConfig::default()).unwrap();
        assert_eq!(rep.sentences, 80);
        assert!(
            lm.sentence_logp("the cat sat on the mat") > lm.sentence_logp("mat the on sat cat the")
        );
        let out = tempfile_path::TempPath::new();
        lm.save(&out.0).unwrap();
        let back = NgramLm::load(&out.0).unwrap();
        assert_eq!(back.sizes(), lm.sizes());
        for s in ["the cat sat on the mat", "a dog ate the mat", "zebra"] {
            assert_eq!(back.sentence_logp(s), lm.sentence_logp(s));
        }
        assert_eq!(back.id("zebra"), UNK);
        assert!(back.is_prefix("ca") && back.is_prefix("cat") && back.is_prefix("m"));
        assert!(!back.is_prefix("cax") && !back.is_prefix("zebra"));
    }
}
