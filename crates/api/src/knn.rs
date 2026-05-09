// Brute-force KNN, k = 5. Picks the fastest implementation at runtime
// for the host CPU; on the evaluator's Haswell box that means AVX2 + F16C.
//
// Top-k tracking is a sorted-descending fixed-size array — the worst
// neighbour sits at index 0 so the early-exit comparison is one load.

use crate::dataset::{Dataset, VectorStore, STRIDE};

pub const K: usize = 5;

#[derive(Clone, Copy)]
pub struct Top5 {
    pub dist: [f32; K],
    pub idx: [u32; K],
}

impl Top5 {
    #[inline(always)]
    pub fn new() -> Self {
        Self { dist: [f32::INFINITY; K], idx: [u32::MAX; K] }
    }

    #[inline(always)]
    pub fn worst(&self) -> f32 { self.dist[0] }

    #[inline(always)]
    pub fn admit(&mut self, d: f32, i: u32) {
        if d >= self.dist[0] { return; }
        let mut pos = 0usize;
        while pos + 1 < K && self.dist[pos + 1] > d {
            self.dist[pos] = self.dist[pos + 1];
            self.idx[pos] = self.idx[pos + 1];
            pos += 1;
        }
        self.dist[pos] = d;
        self.idx[pos] = i;
    }
}

#[inline]
pub fn fraud_count_in_top_k(query: &[f32; STRIDE], dataset: &Dataset) -> u8 {
    let top = match &dataset.vectors {
        VectorStore::F16(refs) => search_f16(query, refs, dataset.count()),
        VectorStore::F32(refs) => search_f32(query, refs, dataset.count()),
    };
    let mut frauds: u8 = 0;
    for k in 0..K {
        let id = top.idx[k] as usize;
        if dataset.is_fraud(id) { frauds += 1; }
    }
    frauds
}

#[inline]
fn search_f16(query: &[f32; STRIDE], refs: &[u16], count: usize) -> Top5 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c"))]
    unsafe { return search_f16_avx2(query, refs, count); }

    #[allow(unreachable_code)]
    search_f16_scalar(query, refs, count)
}

#[inline]
fn search_f32(query: &[f32; STRIDE], refs: &[f32], count: usize) -> Top5 {
    let mut top = Top5::new();
    let mut i = 0;
    while i < count {
        let row = unsafe { refs.get_unchecked(i * STRIDE..i * STRIDE + STRIDE) };
        let mut s = 0f32;
        for d in 0..STRIDE {
            let diff = query[d] - row[d];
            s += diff * diff;
        }
        top.admit(s, i as u32);
        i += 1;
    }
    top
}

#[inline]
fn search_f16_scalar(query: &[f32; STRIDE], refs: &[u16], count: usize) -> Top5 {
    let mut top = Top5::new();
    let mut i = 0;
    while i < count {
        let base = i * STRIDE;
        let row = unsafe { refs.get_unchecked(base..base + STRIDE) };
        let mut s = 0f32;
        for d in 0..STRIDE {
            let r = half::f16::from_bits(row[d]).to_f32();
            let diff = query[d] - r;
            s += diff * diff;
        }
        top.admit(s, i as u32);
        i += 1;
    }
    top
}

/// AVX2 + F16C path: each row is 16 f16 lanes (32 bytes). We load the row
/// as two 128-bit registers, expand each half to 8 f32 with `_mm256_cvtph_ps`,
/// subtract the corresponding query halves, fma-square them, then horizontally
/// reduce the two 8-lane accumulators into a scalar squared distance.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c"))]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn search_f16_avx2(query: &[f32; STRIDE], refs: &[u16], count: usize) -> Top5 {
    use std::arch::x86_64::*;

    // Split the 16-lane query into two 8-lane f32 vectors.
    let q_lo = _mm256_loadu_ps(query.as_ptr());
    let q_hi = _mm256_loadu_ps(query.as_ptr().add(8));

    let mut top = Top5::new();
    let mut i = 0;
    let base_ptr = refs.as_ptr();
    while i < count {
        let row_ptr = base_ptr.add(i * STRIDE);
        // 16 × u16 = 256 bits → load as a 128 + 128 (lo + hi halves).
        let lo_h = _mm_loadu_si128(row_ptr as *const __m128i);
        let hi_h = _mm_loadu_si128(row_ptr.add(8) as *const __m128i);
        let lo_f = _mm256_cvtph_ps(lo_h);
        let hi_f = _mm256_cvtph_ps(hi_h);

        let dlo = _mm256_sub_ps(q_lo, lo_f);
        let dhi = _mm256_sub_ps(q_hi, hi_f);

        // squared distance = sum_lo(dlo² ) + sum_hi(dhi²)
        let acc = _mm256_fmadd_ps(dhi, dhi, _mm256_mul_ps(dlo, dlo));

        // horizontal sum of acc (8 lanes → scalar)
        let lo128 = _mm256_castps256_ps128(acc);
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let s = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

        top.admit(s, i as u32);
        i += 1;
    }
    top
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_keeps_sorted_descending() {
        let mut t = Top5::new();
        for (i, &d) in [9.0f32, 5.0, 7.0, 1.0, 3.0, 8.0, 2.0, 6.0].iter().enumerate() {
            t.admit(d, i as u32);
        }
        let mut got = t.dist;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, [1.0, 2.0, 3.0, 5.0, 6.0]);
    }
}
