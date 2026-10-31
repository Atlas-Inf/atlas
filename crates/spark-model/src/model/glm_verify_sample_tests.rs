// SPDX-License-Identifier: AGPL-3.0-only
//! Sampled verify on the split head: the records `argmax_bf16_candidates`
//! writes (a host model of the kernel, which the GPU test holds the kernel
//! to), their merge, and the claim that the merged rows sample exactly what
//! the full rows sample.
//!
//!   cargo test --release -p spark-model --lib glm_verify_sample -- --include-ignored
use super::*;
use spark_runtime::sampler::{SamplingParams, sample_with_absent, sample_with_params_seeded};

/// GLM-5.3's vocabulary.
const VOCAB: usize = 154_880;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A BF16 logit, coarse enough that exact ties are common.
    fn logit(&mut self, spread: f32) -> u16 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        let x = (u - 0.5) * spread;
        (x.to_bits() >> 16) as u16
    }
}

fn value(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The kernel's record of one row part (`part`: this rank's logits, its
/// first id `start`), computed the obvious way.
fn host_record(part: &[u16], start: usize, inv_t: f32, specials: &[u32]) -> Vec<u32> {
    let key = |i: usize| {
        let v = value(part[i]);
        if v == 0.0 { 0.0 } else { v }
    };
    let mut ids: Vec<usize> = (0..part.len()).collect();
    ids.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap().then(a.cmp(&b)));
    let c = CANDIDATES.min(part.len());
    let best = key(ids[0]);
    let mut rec = vec![0u32; REC_WORDS];
    rec[0] = c as u32;
    rec[1] = best.to_bits();
    let is_special = |i: usize| specials.contains(&((start + i) as u32));
    let tail: f64 = ids[c..]
        .iter()
        .filter(|&&i| !is_special(i))
        .map(|&i| (((key(i) - best) * inv_t) as f64).exp())
        .sum();
    rec[2] = (tail as f32).to_bits();
    for s in 0..SPECIALS {
        rec[4 + s] = specials
            .get(s)
            .and_then(|&id| (id as usize).checked_sub(start))
            .filter(|&i| i < part.len())
            .map_or(NONE, |i| value(part[i]).to_bits());
    }
    for (j, &i) in ids[..c].iter().enumerate() {
        rec[8 + 2 * j] = key(i).to_bits();
        rec[9 + 2 * j] = (start + i) as u32;
    }
    rec
}

fn row_of(rng: &mut Rng, vocab: usize, spread: f32) -> Vec<u16> {
    (0..vocab).map(|_| rng.logit(spread)).collect()
}

/// The merge over the pair's halves and the chain's quarters keeps every
/// rank's candidates and the specials present, bounds the absent ones and
/// adds up their mass; the merged row samples what the full row samples.
#[test]
fn merged_records_sample_like_the_full_row() {
    let specials = [7u32, 100_000, 154_879];
    let mut params = SamplingParams::greedy(1);
    (params.temperature, params.top_p, params.top_k, params.min_p) = (0.7, 0.95, 20, 0.0);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for ranks in [2usize, 4] {
        let shard = VOCAB / ranks;
        for trial in 0..8u64 {
            let row = row_of(&mut rng, VOCAB, 30.0);
            let inv_t = 1.0 / params.temperature;
            let recs: Vec<Vec<u32>> = (0..ranks)
                .map(|q| {
                    host_record(
                        &row[q * shard..(q + 1) * shard],
                        q * shard,
                        inv_t,
                        &specials,
                    )
                })
                .collect();
            let per_rank: Vec<&[u32]> = recs.iter().map(Vec::as_slice).collect();
            let merged = merge_records(&per_rank, 1, params.temperature, &specials).unwrap();
            let split = &merged[0];
            // Every rank's candidates and the three specials, once each.
            let mut ids: Vec<u32> = split.present.iter().map(|&(id, _)| id).collect();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), split.present.len());
            assert!(specials.iter().all(|s| ids.contains(s)));
            for &(id, v) in &split.present {
                assert_eq!(
                    v.to_bits(),
                    {
                        let x = value(row[id as usize]);
                        if x == 0.0 { 0.0f32 } else { x }
                    }
                    .to_bits()
                );
            }
            // The absent ones: after the bound, their mass as summed directly.
            let absent = split.absent.expect("ranks fill their candidates");
            let mut mass = 0.0f64;
            for (i, &b) in row.iter().enumerate() {
                if !ids.contains(&(i as u32)) {
                    let v = value(b);
                    assert!(v < absent.bound || (v == absent.bound && i as u32 > absent.bound_id));
                    mass += ((value(b) * inv_t) as f64).exp();
                }
            }
            let rel = ((absent.log_mass as f64) - mass.ln()).abs();
            assert!(rel < 1e-4, "log mass {} vs {}", absent.log_mass, mass.ln());
            let full: Vec<u8> = row.iter().flat_map(|&b| value(b).to_le_bytes()).collect();
            for s in 0..16 {
                let seed = Some(trial * 100 + s);
                assert_eq!(
                    sample_with_absent(&split.present, VOCAB, &params, seed, absent),
                    Some(sample_with_params_seeded(&full, &params, &[], seed)),
                    "ranks {ranks} trial {trial} seed {s}"
                );
            }
        }
    }
}

/// GPU (`#[ignore]`; a GB10 and a glm-5.3-flash kernel build): the kernel's
/// records are the host model's, at the pair's and the chain's shard widths,
/// with ties, a -0 and the specials on either side of a shard edge.
#[test]
#[ignore = "GPU"]
fn candidates_kernel_matches_the_host_records() -> Result<()> {
    use spark_runtime::cuda_backend::AtlasCudaBackend;
    use spark_runtime::gpu::GpuBackend;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    gpu.bind_to_thread()?;
    let kernel = gpu.kernel("argmax", "argmax_bf16_candidates")?;
    let rows = 9usize;
    let mut rng = Rng(42);
    for ranks in [2usize, 4] {
        let shard = VOCAB / ranks;
        for (rank, spread) in [(0usize, 30.0f32), (ranks - 1, 4.0)] {
            let start = rank * shard;
            let mut logits: Vec<u16> = (0..rows)
                .flat_map(|_| row_of(&mut rng, VOCAB, spread))
                .collect();
            logits[start + 3] = 0x8000; // -0
            let specials = [
                start as u32,
                (start + shard - 1) as u32,
                (start + shard) as u32,
            ];
            let inv_t = 1.0 / 0.7f32;
            let bytes: Vec<u8> = logits.iter().flat_map(|h| h.to_le_bytes()).collect();
            let d_logits = gpu.alloc(bytes.len())?;
            gpu.copy_h2d(&bytes, d_logits)?;
            let d_out = gpu.alloc(rows * REC_WORDS * 4)?;
            let mut launch = KernelLaunch::new(&gpu, kernel)
                .grid([rows as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(d_logits.offset(start * 2))
                .arg_ptr(d_out)
                .arg_u32(shard as u32)
                .arg_u32(VOCAB as u32)
                .arg_u32(start as u32)
                .arg_f32(inv_t);
            for s in 0..SPECIALS {
                launch = launch.arg_u32(specials.get(s).copied().unwrap_or(NONE));
            }
            launch.launch(gpu.default_stream())?;
            gpu.synchronize(gpu.default_stream())?;
            let mut out = vec![0u8; rows * REC_WORDS * 4];
            gpu.copy_d2h(d_out, &mut out)?;
            let got: Vec<u32> = out
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            for r in 0..rows {
                let part = &logits[r * VOCAB + start..r * VOCAB + start + shard];
                let want = host_record(part, start, inv_t, &specials);
                let g = &got[r * REC_WORDS..(r + 1) * REC_WORDS];
                assert_eq!(
                    (g[0], g[1]),
                    (want[0], want[1]),
                    "ranks {ranks} row {r} head"
                );
                assert_eq!(&g[4..], &want[4..], "ranks {ranks} row {r} candidates");
                let (gt, wt) = (f32::from_bits(g[2]), f32::from_bits(want[2]));
                assert!((gt - wt).abs() <= 1e-4 * wt.max(1.0), "tail {gt} vs {wt}");
            }
            gpu.free(d_logits)?;
            gpu.free(d_out)?;
        }
    }
    Ok(())
}
