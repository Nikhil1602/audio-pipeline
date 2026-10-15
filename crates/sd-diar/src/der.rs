//! Diarization error rate, scored on 10 ms frames as in NIST md-eval:
//! DER = (missed speech + false alarm + speaker confusion) / scored reference speech, after
//! the one-to-one speaker mapping that maximises agreement (Hungarian algorithm). Frames
//! within `collar` seconds of a reference boundary are not scored; with `skip_overlap`, frames
//! where the reference has two or more speakers are not scored either.

use std::collections::HashMap;

const STEP: f64 = 0.01;

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

/// Reads the SPEAKER lines of an RTTM file.
pub fn read_rttm(text: &str) -> Vec<Turn> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            if f.first() != Some(&"SPEAKER") || f.len() < 8 {
                return None;
            }
            let start: f64 = f[3].parse().ok()?;
            let dur: f64 = f[4].parse().ok()?;
            Some(Turn {
                start,
                end: start + dur,
                speaker: f[7].to_string(),
            })
        })
        .collect()
}

pub fn write_rttm(file_id: &str, turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| {
            format!(
                "SPEAKER {file_id} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
                t.start,
                t.end - t.start,
                t.speaker
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DerResult {
    /// Scored reference speech (seconds, counting each speaker).
    pub total: f64,
    pub miss: f64,
    pub false_alarm: f64,
    pub confusion: f64,
}

impl DerResult {
    pub fn der(&self) -> f64 {
        (self.miss + self.false_alarm + self.confusion) / self.total.max(1e-9)
    }
    pub fn add(&mut self, o: DerResult) {
        self.total += o.total;
        self.miss += o.miss;
        self.false_alarm += o.false_alarm;
        self.confusion += o.confusion;
    }
}

fn frames(turns: &[Turn], n: usize) -> (Vec<String>, Vec<Vec<usize>>) {
    let mut ids: HashMap<&str, usize> = HashMap::new();
    let mut names = Vec::new();
    let mut active = vec![Vec::new(); n];
    for t in turns {
        let id = *ids.entry(&t.speaker).or_insert_with(|| {
            names.push(t.speaker.clone());
            names.len() - 1
        });
        let a = ((t.start / STEP).round() as usize).min(n);
        let b = ((t.end / STEP).round() as usize).clamp(a, n);
        for f in &mut active[a..b] {
            if !f.contains(&id) {
                f.push(id);
            }
        }
    }
    (names, active)
}

/// Maximum-weight one-to-one assignment of rows to columns (Hungarian algorithm on the
/// negated matrix); unassigned rows get None.
pub fn assign_max(w: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = w.len();
    let cols = w.first().map_or(0, |r| r.len());
    let n = rows.max(cols);
    if n == 0 {
        return Vec::new();
    }
    let max = w.iter().flatten().copied().fold(0.0, f64::max);
    let cost = |i: usize, j: usize| {
        if i < rows && j < cols {
            max - w[i][j]
        } else {
            max
        }
    };
    // Shortest augmenting path formulation (1-indexed potentials).
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for j in 1..=n {
        if p[j] >= 1 && p[j] <= rows && j <= cols {
            out[p[j] - 1] = Some(j - 1);
        }
    }
    out
}

pub fn der(reference: &[Turn], hypothesis: &[Turn], collar: f64, skip_overlap: bool) -> DerResult {
    let end = reference
        .iter()
        .chain(hypothesis)
        .map(|t| t.end)
        .fold(0.0, f64::max);
    let n = (end / STEP).ceil() as usize + 1;
    let (ref_names, r) = frames(reference, n);
    let (hyp_names, h) = frames(hypothesis, n);
    let mut scored = vec![true; n];
    let c = (collar / STEP).round() as usize;
    for t in reference {
        for edge in [t.start, t.end] {
            let e = (edge / STEP).round() as usize;
            for f in &mut scored[e.saturating_sub(c)..(e + c).min(n)] {
                *f = false;
            }
        }
    }
    if skip_overlap {
        for (s, rf) in scored.iter_mut().zip(&r) {
            if rf.len() > 1 {
                *s = false;
            }
        }
    }
    let mut w = vec![vec![0.0; hyp_names.len()]; ref_names.len()];
    for f in 0..n {
        if scored[f] {
            for &a in &r[f] {
                for &b in &h[f] {
                    w[a][b] += 1.0;
                }
            }
        }
    }
    let map = assign_max(&w);
    let mut res = DerResult::default();
    for f in 0..n {
        if !scored[f] {
            continue;
        }
        let (nr, nh) = (r[f].len() as f64, h[f].len() as f64);
        let correct = r[f]
            .iter()
            .filter(|&&a| map[a].is_some_and(|b| h[f].contains(&b)))
            .count() as f64;
        res.total += nr * STEP;
        res.miss += (nr - nh).max(0.0) * STEP;
        res.false_alarm += (nh - nr).max(0.0) * STEP;
        res.confusion += (nr.min(nh) - correct) * STEP;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(start: f64, end: f64, s: &str) -> Turn {
        Turn {
            start,
            end,
            speaker: s.into(),
        }
    }

    #[test]
    fn perfect_and_relabelled_hypotheses_score_zero() {
        let r = vec![t(0.0, 5.0, "A"), t(5.0, 10.0, "B")];
        let h = vec![t(0.0, 5.0, "x"), t(5.0, 10.0, "y")];
        assert!(der(&r, &h, 0.0, false).der() < 1e-9);
    }

    #[test]
    fn components_are_counted() {
        let r = vec![t(0.0, 10.0, "A"), t(10.0, 20.0, "B")];
        // 2 s missed at the start, B mislabelled as A for 5 s, 3 s false alarm at the end.
        let h = vec![t(2.0, 15.0, "x"), t(15.0, 20.0, "y"), t(20.0, 23.0, "y")];
        let d = der(&r, &h, 0.0, false);
        assert!((d.total - 20.0).abs() < 0.02);
        assert!((d.miss - 2.0).abs() < 0.02, "{d:?}");
        assert!((d.confusion - 5.0).abs() < 0.02, "{d:?}");
        assert!((d.false_alarm - 3.0).abs() < 0.02, "{d:?}");
    }

    #[test]
    fn collar_and_overlap_exclusion() {
        let r = vec![t(0.0, 10.0, "A"), t(5.0, 10.0, "B")];
        let h = vec![t(0.0, 10.0, "x")];
        let all = der(&r, &h, 0.0, false);
        assert!((all.miss - 5.0).abs() < 0.02);
        let no_overlap = der(&r, &h, 0.0, true);
        assert!(no_overlap.miss < 0.02 && (no_overlap.total - 5.0).abs() < 0.02);
        let collared = der(&r, &h, 0.25, false);
        assert!(collared.total < all.total);
        assert_eq!(read_rttm(&write_rttm("f", &r)), r);
    }
}
