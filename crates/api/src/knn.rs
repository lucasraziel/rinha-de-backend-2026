// Brute-force and IVF KNN, k = 5. Picks AVX2 + F16C on Haswell+, falls back
// to a half→f32 scalar loop everywhere else.
//
// Top-k tracking is a sorted-descending fixed-size array — the worst neighbour
// sits at index 0 so the early-exit comparison is one load.

use crate::dataset::{Dataset, IvfIndex, VectorStore, STRIDE};

pub const K: usize = 5;
pub const NPROBE_MAX: usize = 64;

#[derive(Clone, Copy)]
pub struct TopK<const N: usize> {
    pub dist: [f32; N],
    pub idx: [u32; N],
}

impl<const N: usize> TopK<N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self { dist: [f32::INFINITY; N], idx: [u32::MAX; N] }
    }

    #[inline(always)]
    pub fn worst(&self) -> f32 { self.dist[0] }

    #[inline(always)]
    pub fn admit(&mut self, d: f32, i: u32) {
        if d >= self.dist[0] { return; }
        let mut pos = 0usize;
        while pos + 1 < N && self.dist[pos + 1] > d {
            self.dist[pos] = self.dist[pos + 1];
            self.idx[pos] = self.idx[pos + 1];
            pos += 1;
        }
        self.dist[pos] = d;
        self.idx[pos] = i;
    }
}

#[inline]
pub fn fraud_count_in_top_k(query: &[f32; STRIDE], dataset: &Dataset, nprobe: usize) -> u8 {
    let top: TopK<K> = match (&dataset.ivf, &dataset.vectors) {
        (Some(ivf), VectorStore::F16(refs)) => search_ivf_f16(query, refs, ivf, nprobe),
        (Some(ivf), VectorStore::F32(refs)) => search_ivf_f32(query, refs, ivf, nprobe),
        (None, VectorStore::F16(refs)) => search_brute_f16(query, refs, dataset.count()),
        (None, VectorStore::F32(refs)) => search_brute_f32(query, refs, dataset.count()),
    };
    let mut frauds: u8 = 0;
    for k in 0..K {
        let id = top.idx[k] as usize;
        if id < dataset.count() && dataset.is_fraud(id) {
            frauds += 1;
        }
    }
    frauds
}

// ─── Brute force over the whole dataset ────────────────────────────────────

#[inline]
fn search_brute_f16(query: &[f32; STRIDE], refs: &[u16], count: usize) -> TopK<K> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c"))]
    unsafe { return search_range_f16_avx2(query, refs, 0, count); }

    #[allow(unreachable_code)]
    search_range_f16_scalar(query, refs, 0, count)
}

#[inline]
fn search_brute_f32(query: &[f32; STRIDE], refs: &[f32], count: usize) -> TopK<K> {
    let mut top = TopK::<K>::new();
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

// ─── IVF 2-stage (centroids → cluster scan) ────────────────────────────────

#[inline]
fn search_ivf_f16(
    query: &[f32; STRIDE],
    refs: &[u16],
    ivf: &IvfIndex,
    nprobe: usize,
) -> TopK<K> {
    let probes = pick_probes(query, ivf, nprobe);
    let n = probes.len.min(NPROBE_MAX);
    let mut top = TopK::<K>::new();
    for p in 0..n {
        let cluster = probes.idx[p] as usize;
        let lo = ivf.offsets[cluster] as usize;
        let hi = ivf.offsets[cluster + 1] as usize;
        if lo >= hi { continue; }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c"))]
        let cluster_top = unsafe { search_range_f16_avx2(query, refs, lo, hi) };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c")))]
        let cluster_top = search_range_f16_scalar(query, refs, lo, hi);

        for k in 0..K {
            top.admit(cluster_top.dist[k], cluster_top.idx[k]);
        }
    }
    top
}

#[inline]
fn search_ivf_f32(
    query: &[f32; STRIDE],
    refs: &[f32],
    ivf: &IvfIndex,
    nprobe: usize,
) -> TopK<K> {
    let probes = pick_probes(query, ivf, nprobe);
    let n = probes.len.min(NPROBE_MAX);
    let mut top = TopK::<K>::new();
    for p in 0..n {
        let cluster = probes.idx[p] as usize;
        let lo = ivf.offsets[cluster] as usize;
        let hi = ivf.offsets[cluster + 1] as usize;
        if lo >= hi { continue; }
        for i in lo..hi {
            let row = unsafe { refs.get_unchecked(i * STRIDE..i * STRIDE + STRIDE) };
            let mut s = 0f32;
            for d in 0..STRIDE { let diff = query[d] - row[d]; s += diff * diff; }
            top.admit(s, i as u32);
        }
    }
    top
}

// Top-`nprobe` centroids in scalar f32 (centroids are pre-decoded to f32).
struct ProbeList { idx: [u32; NPROBE_MAX], len: usize }

#[inline]
fn pick_probes(query: &[f32; STRIDE], ivf: &IvfIndex, nprobe: usize) -> ProbeList {
    let n = nprobe.min(NPROBE_MAX).max(1).min(ivf.nlist);
    let mut top = TopK::<NPROBE_MAX>::new();
    let cents = &ivf.centroids_f32;
    for c in 0..ivf.nlist {
        let row = unsafe { cents.get_unchecked(c * STRIDE..c * STRIDE + STRIDE) };
        let mut s = 0f32;
        for d in 0..STRIDE { let diff = query[d] - row[d]; s += diff * diff; }
        top.admit(s, c as u32);
    }
    // top is sorted descending: worst at 0, best at NPROBE_MAX-1.
    // We want the `n` best — last `n` entries — copied in any order.
    let mut out = ProbeList { idx: [0u32; NPROBE_MAX], len: n };
    for i in 0..n {
        out.idx[i] = top.idx[NPROBE_MAX - 1 - i];
    }
    out
}

// ─── Range scan helpers (used by both brute and IVF stage 2) ───────────────

#[inline]
fn search_range_f16_scalar(query: &[f32; STRIDE], refs: &[u16], lo: usize, hi: usize) -> TopK<K> {
    let mut top = TopK::<K>::new();
    let mut i = lo;
    while i < hi {
        let row = unsafe { refs.get_unchecked(i * STRIDE..i * STRIDE + STRIDE) };
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "f16c"))]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn search_range_f16_avx2(
    query: &[f32; STRIDE],
    refs: &[u16],
    lo: usize,
    hi: usize,
) -> TopK<K> {
    use std::arch::x86_64::*;

    let q_lo = _mm256_loadu_ps(query.as_ptr());
    let q_hi = _mm256_loadu_ps(query.as_ptr().add(8));

    let mut top = TopK::<K>::new();
    let base_ptr = refs.as_ptr();
    let mut i = lo;
    while i < hi {
        let row_ptr = base_ptr.add(i * STRIDE);
        let lo_h = _mm_loadu_si128(row_ptr as *const __m128i);
        let hi_h = _mm_loadu_si128(row_ptr.add(8) as *const __m128i);
        let lo_f = _mm256_cvtph_ps(lo_h);
        let hi_f = _mm256_cvtph_ps(hi_h);

        let dlo = _mm256_sub_ps(q_lo, lo_f);
        let dhi = _mm256_sub_ps(q_hi, hi_f);

        let acc = _mm256_fmadd_ps(dhi, dhi, _mm256_mul_ps(dlo, dlo));

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
        let mut t = TopK::<5>::new();
        for (i, &d) in [9.0f32, 5.0, 7.0, 1.0, 3.0, 8.0, 2.0, 6.0].iter().enumerate() {
            t.admit(d, i as u32);
        }
        let mut got = t.dist;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, [1.0, 2.0, 3.0, 5.0, 6.0]);
    }
}
