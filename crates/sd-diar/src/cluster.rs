//! Average-linkage agglomerative clustering (UPGMA) with the nearest-neighbour-chain
//! algorithm: O(n²) time and memory for n items. Distances are Euclidean between the given
//! vectors (unit vectors in practice: WeSpeaker embeddings).

use crate::embed::distance;

/// One merge of the dendrogram: the two clusters' representative items and the height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Merge {
    pub a: usize,
    pub b: usize,
    pub height: f32,
}

/// The dendrogram of `items` (n − 1 merges, sorted by height).
pub fn average_linkage(items: &[Vec<f32>]) -> Vec<Merge> {
    let n = items.len();
    if n < 2 {
        return Vec::new();
    }
    // Condensed upper-triangular distance matrix, updated in place (Lance-Williams).
    let idx = |i: usize, j: usize| {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * n - i * (i + 1) / 2 + (j - i - 1)
    };
    let mut d = vec![0f32; n * (n - 1) / 2];
    for i in 0..n {
        for j in i + 1..n {
            d[idx(i, j)] = distance(&items[i], &items[j]);
        }
    }
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut merges = Vec::with_capacity(n - 1);
    let mut chain: Vec<usize> = Vec::new();
    let mut remaining = n;
    while remaining > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&a| a).expect("an active cluster"));
        }
        loop {
            let x = *chain.last().expect("non-empty chain");
            let prev = chain.len().checked_sub(2).map(|k| chain[k]);
            // Nearest active neighbour of x; ties prefer the previous chain element.
            let mut best: Option<(usize, f32)> = prev.map(|p| (p, d[idx(x, p)]));
            for y in 0..n {
                if y != x && active[y] {
                    let dy = d[idx(x, y)];
                    if best.is_none_or(|(_, bd)| dy < bd) {
                        best = Some((y, dy));
                    }
                }
            }
            let (y, dxy) = best.expect("another active cluster");
            if Some(y) == prev {
                chain.pop();
                chain.pop();
                // Merge y into x (x keeps the merged cluster).
                let (sx, sy) = (size[x] as f32, size[y] as f32);
                for k in 0..n {
                    if active[k] && k != x && k != y {
                        let v = (sx * d[idx(x, k)] + sy * d[idx(y, k)]) / (sx + sy);
                        d[idx(x, k)] = v;
                    }
                }
                size[x] += size[y];
                active[y] = false;
                merges.push(Merge {
                    a: x,
                    b: y,
                    height: dxy,
                });
                remaining -= 1;
                break;
            }
            chain.push(y);
        }
    }
    merges.sort_by(|p, q| p.height.total_cmp(&q.height));
    merges
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Flat clusters: apply merges below `threshold`, or (if `k` is given) until `k` clusters
/// remain. Labels are 0.. in order of first appearance.
pub fn cut(n: usize, merges: &[Merge], threshold: f32, k: Option<usize>) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    let mut clusters = n;
    for m in merges {
        let stop = match k {
            Some(k) => clusters <= k.max(1),
            None => m.height >= threshold,
        };
        if stop {
            break;
        }
        let (a, b) = (find(&mut parent, m.a), find(&mut parent, m.b));
        if a != b {
            parent[b] = a;
            clusters -= 1;
        }
    }
    let mut label_of = std::collections::HashMap::new();
    (0..n)
        .map(|i| {
            let r = find(&mut parent, i);
            let next = label_of.len();
            *label_of.entry(r).or_insert(next)
        })
        .collect()
}

/// The minimum cluster size: min(12, ⌈n / 10⌉) members.
pub fn min_cluster_size(n: usize) -> usize {
    12.min(n.div_ceil(10)).max(1)
}

/// Absorbs clusters with fewer than `min_size` members into the nearest large cluster
/// (centroid distance between unit vectors), but only when that distance is below
/// `max_distance`: a small cluster far from every large one is a brief but distinct speaker
/// and is kept. Labels are renumbered 0.. in order of first appearance. With no large
/// cluster, the labels are returned unchanged.
pub fn absorb_small(
    items: &[Vec<f32>],
    labels: &[usize],
    min_size: usize,
    max_distance: f32,
) -> Vec<usize> {
    let k = labels.iter().copied().max().map_or(0, |m| m + 1);
    let mut count = vec![0usize; k];
    for &l in labels {
        count[l] += 1;
    }
    let dim = items.first().map_or(0, |v| v.len());
    let centroid = |c: usize| -> Vec<f32> {
        let mut m = vec![0f32; dim];
        for (v, _) in items.iter().zip(labels).filter(|(_, l)| **l == c) {
            for (a, b) in m.iter_mut().zip(v) {
                *a += b;
            }
        }
        let n = m.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        m.iter_mut().for_each(|x| *x /= n);
        m
    };
    let big: Vec<usize> = (0..k).filter(|&c| count[c] >= min_size).collect();
    if big.is_empty() {
        return labels.to_vec();
    }
    let big_centroids: Vec<Vec<f32>> = big.iter().map(|&c| centroid(c)).collect();
    let mut target: Vec<usize> = (0..k).collect();
    for c in (0..k).filter(|&c| count[c] < min_size) {
        let cc = centroid(c);
        if let Some((i, d)) = big_centroids
            .iter()
            .map(|b| distance(b, &cc))
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            && d < max_distance
        {
            target[c] = big[i];
        }
    }
    let mut renumber = std::collections::HashMap::new();
    labels
        .iter()
        .map(|&l| {
            let next = renumber.len();
            *renumber.entry(target[l]).or_insert(next)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(items: &[Vec<f32>]) -> Vec<f32> {
        // Textbook UPGMA: repeatedly merge the closest pair by average pairwise distance.
        let mut clusters: Vec<Vec<usize>> = (0..items.len()).map(|i| vec![i]).collect();
        let mut heights = Vec::new();
        while clusters.len() > 1 {
            let mut best = (0, 1, f32::INFINITY);
            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let mut s = 0.0;
                    for &a in &clusters[i] {
                        for &b in &clusters[j] {
                            s += distance(&items[a], &items[b]);
                        }
                    }
                    let avg = s / (clusters[i].len() * clusters[j].len()) as f32;
                    if avg < best.2 {
                        best = (i, j, avg);
                    }
                }
            }
            let merged = clusters.remove(best.1);
            clusters[best.0].extend(merged);
            heights.push(best.2);
        }
        heights.sort_by(|a, b| a.total_cmp(b));
        heights
    }

    #[test]
    fn nn_chain_matches_textbook_upgma() {
        // Deterministic pseudo-random points.
        let mut x = 12345u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 10_000) as f32 / 10_000.0
        };
        let items: Vec<Vec<f32>> = (0..40).map(|_| vec![rnd(), rnd(), rnd()]).collect();
        let fast: Vec<f32> = average_linkage(&items).iter().map(|m| m.height).collect();
        let slow = brute_force(&items);
        assert_eq!(fast.len(), slow.len());
        for (a, b) in fast.iter().zip(&slow) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn small_clusters_join_the_nearest_large_one() {
        let items = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.9, 0.1],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1, 2];
        assert_eq!(
            absorb_small(&items, &labels, 2, f32::INFINITY),
            vec![0, 0, 0, 1, 1, 1, 0]
        );
        // A small cluster far from both large ones is kept as its own speaker.
        let far = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
        ];
        assert_eq!(
            absorb_small(&far, &[0, 0, 1, 1, 2], 2, 1.0),
            vec![0, 0, 1, 1, 2]
        );
        // Allowed further away, it joins the nearer large cluster, (0, 1) at 1.41.
        assert_eq!(absorb_small(&far, &[0, 0, 1, 1, 2], 2, 3.0)[4], 1);
        assert_eq!(min_cluster_size(1000), 12);
        assert_eq!(min_cluster_size(55), 6);
    }

    #[test]
    fn cut_by_threshold_and_by_count() {
        let items = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![10.0, 0.0],
        ];
        let m = average_linkage(&items);
        assert_eq!(cut(5, &m, 1.0, None), vec![0, 0, 1, 1, 2]);
        assert_eq!(
            cut(5, &m, 0.0, Some(2))
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            2
        );
        assert_eq!(cut(5, &m, 100.0, None), vec![0; 5]);
    }
}
