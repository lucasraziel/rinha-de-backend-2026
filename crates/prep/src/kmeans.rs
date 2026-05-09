// Lloyd's k-means with deterministic random init.
// We compute distances using only the first 14 lanes of each STRIDE-16 vector
// (the last 2 are zero-pad for SIMD alignment in the api crate).

use rayon::prelude::*;

pub const STRIDE: usize = 16;
pub const DIMS: usize = 14;

pub struct Kmeans {
    pub centroids: Vec<f32>,   // nlist × STRIDE
    pub assignments: Vec<u32>, // count
    pub iters: usize,
    pub final_max_change: f32,
}

/// Splitmix64 — one-line PRNG with no dependencies. Deterministic given a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[inline(always)]
fn dist_sq_first_14(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for d in 0..DIMS {
        let diff = unsafe { *a.get_unchecked(d) - *b.get_unchecked(d) };
        s += diff * diff;
    }
    s
}

pub fn run(
    vectors: &[f32],
    count: usize,
    nlist: usize,
    max_iter: usize,
    tol: f32,
    seed: u64,
) -> Kmeans {
    assert!(vectors.len() == count * STRIDE);
    let mut centroids = init_random(vectors, count, nlist, seed);
    let mut assignments = vec![0u32; count];

    let mut iter = 0usize;
    let mut final_change = f32::INFINITY;
    for it in 0..max_iter {
        // 1) assignment step (parallel over points)
        assignments
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| {
                let v = &vectors[i * STRIDE..i * STRIDE + STRIDE];
                let mut best = (f32::INFINITY, 0u32);
                for c in 0..nlist {
                    let cent = &centroids[c * STRIDE..c * STRIDE + STRIDE];
                    let d = dist_sq_first_14(v, cent);
                    if d < best.0 {
                        best = (d, c as u32);
                    }
                }
                *slot = best.1;
            });

        // 2) recompute step (parallel reduction over points → per-cluster sums)
        let zero = (vec![0f64; nlist * STRIDE], vec![0u32; nlist]);
        let (sums, counts) = (0..count)
            .into_par_iter()
            .fold(
                || (vec![0f64; nlist * STRIDE], vec![0u32; nlist]),
                |mut acc, i| {
                    let c = assignments[i] as usize;
                    let base = c * STRIDE;
                    let v = &vectors[i * STRIDE..i * STRIDE + STRIDE];
                    for d in 0..STRIDE {
                        acc.0[base + d] += v[d] as f64;
                    }
                    acc.1[c] += 1;
                    acc
                },
            )
            .reduce(
                || zero.clone(),
                |mut a, b| {
                    for i in 0..a.0.len() { a.0[i] += b.0[i]; }
                    for i in 0..a.1.len() { a.1[i] += b.1[i]; }
                    a
                },
            );

        // 3) update centroids
        let mut max_change = 0f32;
        for c in 0..nlist {
            if counts[c] == 0 {
                continue;
            }
            let n = counts[c] as f64;
            for d in 0..STRIDE {
                let new_val = (sums[c * STRIDE + d] / n) as f32;
                let cur = centroids[c * STRIDE + d];
                let diff = (cur - new_val).abs();
                if diff > max_change { max_change = diff; }
                centroids[c * STRIDE + d] = new_val;
            }
        }

        iter = it + 1;
        final_change = max_change;
        eprintln!("kmeans: iter {} max_change {:.6}", iter, max_change);
        if max_change < tol { break; }
    }

    Kmeans { centroids, assignments, iters: iter, final_max_change: final_change }
}

fn init_random(vectors: &[f32], count: usize, nlist: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut chosen = vec![false; count];
    let mut centroids = vec![0f32; nlist * STRIDE];
    let mut placed = 0;
    while placed < nlist {
        let idx = (rng.next() % count as u64) as usize;
        if !chosen[idx] {
            chosen[idx] = true;
            centroids[placed * STRIDE..(placed + 1) * STRIDE]
                .copy_from_slice(&vectors[idx * STRIDE..(idx + 1) * STRIDE]);
            placed += 1;
        }
    }
    centroids
}
