// SPDX-License-Identifier: AGPL-3.0-only
//! Draft equality and timing of the DFlash2 on-device candidate selector at
//! the GLM-5.3 drafter geometry (gamma 8, vocab 154880, rank 256, top_k 16).
//!
//! `dflash2_selector_parity.rs` checks the kernel against a host reference
//! with a near-tie tolerance on the scores. This test pins the kernel's
//! exact output instead: every case hashes the drafted tokens (and, through
//! the confidence twin, the confidence words) and compares the hash with the
//! one the reference kernel produced on the same inputs (the selector of
//! Enntity/atlas fea1ef6c, recorded on ennspark03). A change to the selector
//! that moves one token or one confidence bit fails here. The logits are
//! drawn from small palettes so that exact ties straddle every top-k cut,
//! with -inf, signed zeros and NaN among them, and the bans are aimed at the
//! rows' leaders.
//!
//! Run on the GB10 (all tests are `#[ignore]`d):
//!   cargo test --release -p spark-model --test dflash2_selector_exact -- --ignored --nocapture
#[path = "arm2_common/support.rs"]
mod support;

use anyhow::{Context, Result, ensure};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl, pdl_enabled};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use support::{Rng, bf16_bits_to_f32, f32_to_bf16_bits, rd_u32, up_u16, up_u32};

#[path = "dflash2_selector_exact/geometry.rs"]
mod geometry;
#[path = "dflash2_selector_exact/timing.rs"]
mod timing;

fn setup() -> Result<(AtlasCudaBackend, u64)> {
    configure_pdl("glm-5.3-flash");
    let set = atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("selector tests require the glm-5.3-flash/nvfp4 kernel target")?;
    let gpu = AtlasCudaBackend::new(0, &set.modules).context("selector test CUDA backend")?;
    ensure!(
        pdl_enabled() == (std::env::var("ATLAS_PDL").as_deref() == Ok("1")),
        "PDL policy was latched before GLM target selection; run in a fresh process"
    );
    let stream = gpu.create_stream()?;
    Ok((gpu, stream))
}

const GAMMA: usize = 8;
const VOCAB: usize = 154_880;
const RANK: usize = 256;
const TOP_K: usize = 16;

/// The logits of one case.
#[derive(Clone, Copy, Debug)]
enum Logits {
    /// Independent uniform values: few exact ties.
    Uniform,
    /// Every value from a palette of `n` bf16 values: exact ties at every
    /// top-k cut (n = 1 makes the whole row one tie).
    Palette(usize),
    /// A palette of 24 with -inf, -0.0, +0.0 and NaN in it.
    Specials,
}

struct Case {
    name: &'static str,
    logits: Logits,
    /// Rows `1..=ban_depth` may not pick an end token; the end ids are the
    /// leaders of rows 1 and 2.
    ban_depth: u32,
    seed: u64,
}

const CASES: &[Case] = &[
    Case {
        name: "uniform",
        logits: Logits::Uniform,
        ban_depth: 0,
        seed: 0xD5E1_0001,
    },
    Case {
        name: "palette64",
        logits: Logits::Palette(64),
        ban_depth: 0,
        seed: 0xD5E1_0002,
    },
    Case {
        name: "palette3-banned",
        logits: Logits::Palette(3),
        ban_depth: 3,
        seed: 0xD5E1_0003,
    },
    Case {
        name: "one-tie",
        logits: Logits::Palette(1),
        ban_depth: 7,
        seed: 0xD5E1_0004,
    },
    Case {
        name: "specials",
        logits: Logits::Specials,
        ban_depth: 2,
        seed: 0xD5E1_0005,
    },
    Case {
        name: "uniform-banned",
        logits: Logits::Uniform,
        ban_depth: 7,
        seed: 0xD5E1_0006,
    },
];

/// FNV-1a over the output words.
fn fnv(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Device inputs of one case.
struct Inputs {
    logits: Vec<u16>,
    d_logits: DevicePtr,
    d_proj: DevicePtr,
    d_pred: DevicePtr,
    d_succ: DevicePtr,
    d_anchor: DevicePtr,
    d_ban: DevicePtr,
    end_ids: [u32; 4],
}

fn palette(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| f32_to_bf16_bits(-8.0 + 16.0 * rng.unit()))
        .collect()
}

fn inputs(gpu: &dyn GpuBackend, case: &Case) -> Result<Inputs> {
    let mut rng = Rng(case.seed);
    let colors = match case.logits {
        Logits::Uniform => Vec::new(),
        Logits::Palette(n) => palette(&mut rng, n),
        Logits::Specials => {
            let mut p = palette(&mut rng, 20);
            // -inf, -0.0, +0.0, NaN.
            p.extend([0xFF80, 0x8000, 0x0000, 0x7FC0]);
            p
        }
    };
    let logits: Vec<u16> = (0..GAMMA * VOCAB)
        .map(|_| {
            if colors.is_empty() {
                f32_to_bf16_bits(-8.0 + 16.0 * rng.unit())
            } else {
                colors[(rng.next_u64() % colors.len() as u64) as usize]
            }
        })
        .collect();
    let codebook = |rng: &mut Rng| -> Vec<u16> {
        (0..VOCAB * RANK)
            .map(|_| f32_to_bf16_bits(-0.05 + 0.1 * rng.unit()))
            .collect()
    };
    let pred = codebook(&mut rng);
    let succ = codebook(&mut rng);
    let proj: Vec<u16> = (0..GAMMA * RANK)
        .map(|_| f32_to_bf16_bits(-1.0 + 2.0 * rng.unit()))
        .collect();
    let anchor = (rng.next_u64() % VOCAB as u64) as u32;
    // Ban the first-index leaders of rows 1 and 2.
    let leader = |row: usize| -> u32 {
        let r = &logits[row * VOCAB..(row + 1) * VOCAB];
        let mut best = 0usize;
        for (i, &b) in r.iter().enumerate() {
            if bf16_bits_to_f32(b) > bf16_bits_to_f32(r[best]) {
                best = i;
            }
        }
        best as u32
    };
    let end_ids = [leader(1), leader(2), u32::MAX, u32::MAX];
    Ok(Inputs {
        d_logits: up_u16(gpu, &logits)?,
        d_proj: up_u16(gpu, &proj)?,
        d_pred: up_u16(gpu, &pred)?,
        d_succ: up_u16(gpu, &succ)?,
        d_anchor: up_u32(gpu, &[anchor])?,
        d_ban: up_u32(gpu, &[case.ban_depth])?,
        logits,
        end_ids,
    })
}

/// The fused selector `kernel` over `inp`, twice over one scratch (the
/// second launch must find its tickets reset): the output words (`words`
/// per launch: tokens, then confidences for the twin).
fn fused(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernel: KernelHandle,
    inp: &Inputs,
    words: usize,
) -> Result<Vec<u32>> {
    let scratch_bytes = ops::dflash2_selector_scratch_bytes(GAMMA);
    let d_scratch = gpu.alloc(scratch_bytes)?;
    gpu.memset(d_scratch, 0, scratch_bytes)?;
    let d_out = gpu.alloc(words * 4)?;
    let mut runs = Vec::new();
    for _ in 0..2 {
        gpu.memset(d_out, 0xff, words * 4)?;
        ops::dflash2_candidate_selector(
            gpu,
            kernel,
            inp.d_logits,
            inp.d_proj,
            inp.d_pred,
            inp.d_succ,
            d_out,
            inp.d_anchor,
            inp.d_ban,
            inp.end_ids,
            GAMMA as u32,
            VOCAB as u32,
            RANK as u32,
            TOP_K as u32,
            d_scratch,
            stream,
        )?;
        gpu.synchronize(stream)?;
        runs.push(rd_u32(gpu, d_out, words)?);
    }
    assert_eq!(
        runs[0], runs[1],
        "a second launch over one scratch diverged"
    );
    gpu.free(d_scratch)?;
    gpu.free(d_out)?;
    Ok(runs.pop().unwrap())
}

/// The rank split's selector over `inp`: `dflash2_selector_topk` over the
/// logits columns below `cut` and from `cut` (each share in its own buffer,
/// as each rank holds its share), then `walk` over the two lists, listed in
/// `order`.
fn split(
    gpu: &dyn GpuBackend,
    stream: u64,
    topk: KernelHandle,
    walk: KernelHandle,
    inp: &Inputs,
    cut: usize,
    order: [usize; 2],
    words: usize,
) -> Result<Vec<u32>> {
    let shares = [(0, cut), (cut, VOCAB - cut)];
    let mut lists = Vec::new();
    for (first, columns) in shares {
        let share: Vec<u16> = (0..GAMMA)
            .flat_map(|r| inp.logits[r * VOCAB + first..r * VOCAB + first + columns].to_vec())
            .collect();
        let d_share = up_u16(gpu, &share)?;
        let scratch_bytes = ops::dflash2_selector_scratch_bytes(GAMMA);
        let d_scratch = gpu.alloc(scratch_bytes)?;
        gpu.memset(d_scratch, 0, scratch_bytes)?;
        let d_lists = gpu.alloc(ops::dflash2_selector_lists_bytes(GAMMA))?;
        // Twice over one scratch: the row tickets must be reset.
        for _ in 0..2 {
            ops::dflash2_selector_topk(
                gpu,
                topk,
                d_share,
                columns as u32,
                columns as u32,
                first as u32,
                GAMMA as u32,
                d_lists,
                d_scratch,
                stream,
            )?;
        }
        gpu.synchronize(stream)?;
        gpu.free(d_share)?;
        gpu.free(d_scratch)?;
        lists.push(d_lists);
    }
    let d_out = gpu.alloc(words * 4)?;
    gpu.memset(d_out, 0xff, words * 4)?;
    ops::dflash2_selector_walk(
        gpu,
        walk,
        [lists[order[0]], lists[order[1]]],
        inp.d_proj,
        inp.d_pred,
        inp.d_succ,
        d_out,
        inp.d_anchor,
        inp.d_ban,
        inp.end_ids,
        GAMMA as u32,
        VOCAB as u32,
        RANK as u32,
        TOP_K as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let out = rd_u32(gpu, d_out, words)?;
    for p in lists.into_iter().chain([d_out]) {
        gpu.free(p)?;
    }
    Ok(out)
}

/// `(topk, walk, walk with confidences)`, when this build ships them.
fn split_kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle, KernelHandle)> {
    Ok((
        gpu.kernel("dflash2_candidate_selector", "dflash2_selector_topk")?,
        gpu.kernel("dflash2_candidate_selector", "dflash2_selector_walk")?,
        gpu.kernel("glm_dflash2_selector_conf", "dflash2_selector_walk_conf")?,
    ))
}

fn kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    Ok((
        gpu.kernel("dflash2_candidate_selector", "dflash2_candidate_selector")?,
        gpu.kernel(
            "glm_dflash2_selector_conf",
            "dflash2_candidate_selector_conf",
        )?,
    ))
}

/// `(case, production hash, confidence-twin hash)` of the reference kernel.
const REFERENCE: &[(&str, u64, u64)] = &[
    ("uniform", 0xed158af2dd6dfe04, 0xedbd8d2ee37d8c7e),
    ("palette64", 0x2370b65a1b7a7852, 0x4fa3df391c751545),
    ("palette3-banned", 0xf70c46125564b045, 0x72984a7b2b2a7ffe),
    ("one-tie", 0xbdc8d1fc3e2d8a9b, 0xd5179e66e2e6cf37),
    ("specials", 0xa489a4fd4fb2080f, 0xdf3d3596bdc7cafb),
    ("uniform-banned", 0xbd204c1e601940e0, 0x1ad8cccfd1145b94),
];

#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn selector_drafts_match_the_reference_kernel() -> Result<()> {
    let (gpu, stream) = setup()?;
    let (prod, conf) = kernels(&gpu)?;
    let mut mismatches = Vec::new();
    for case in CASES {
        let inp = inputs(&gpu, case)?;
        let tokens = fused(&gpu, stream, prod, &inp, GAMMA)?;
        let twin = fused(&gpu, stream, conf, &inp, 2 * GAMMA)?;
        // These fixtures have valid unbanned candidates in every row.
        for (row, word) in twin[GAMMA..].iter().enumerate() {
            let confidence = f32::from_bits(*word);
            ensure!(
                confidence.is_finite() && confidence <= 0.0,
                "{} row {row}: invalid log-confidence {confidence}",
                case.name
            );
        }
        // The twin makes the production picks.
        assert_eq!(
            &twin[..GAMMA],
            &tokens[..],
            "{}: twin picks differ",
            case.name
        );
        let (h_prod, h_conf) = (fnv(&tokens), fnv(&twin));
        println!(
            "HASH {} prod={h_prod:#018x} conf={h_conf:#018x} tokens={tokens:?}",
            case.name
        );
        match REFERENCE.iter().find(|r| r.0 == case.name) {
            Some(&(_, p, c)) if (p, c) != (h_prod, h_conf) => mismatches.push(case.name),
            Some(_) => {}
            None => anyhow::bail!("no reference output recorded for {}", case.name),
        }
        let _ = &inp.logits;
        for p in [
            inp.d_logits,
            inp.d_proj,
            inp.d_pred,
            inp.d_succ,
            inp.d_anchor,
            inp.d_ban,
        ] {
            gpu.free(p)?;
        }
    }
    assert!(mismatches.is_empty(), "drafts moved on {mismatches:?}");
    Ok(())
}

/// The rank split's two launches pick the fused kernel's tokens and
/// confidences, bit for bit, at the split's cut (the largest 16-row boundary
/// at or under half the vocabulary), at an uneven cut and with the lists in
/// either order.
#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn split_selector_matches_the_fused_kernel() -> Result<()> {
    let (gpu, stream) = setup()?;
    let (prod, conf) = kernels(&gpu)?;
    let (topk, walk, walk_conf) = split_kernels(&gpu)?;
    for case in CASES {
        let inp = inputs(&gpu, case)?;
        let tokens = fused(&gpu, stream, prod, &inp, GAMMA)?;
        let twin = fused(&gpu, stream, conf, &inp, 2 * GAMMA)?;
        for (cut, order) in [
            (VOCAB / 2 / 16 * 16, [0, 1]),
            (VOCAB / 2 / 16 * 16, [1, 0]),
            (1000, [0, 1]),
        ] {
            let got = split(&gpu, stream, topk, walk, &inp, cut, order, GAMMA)?;
            assert_eq!(
                got, tokens,
                "{} cut {cut} {order:?}: split picks differ",
                case.name
            );
            let got = split(&gpu, stream, topk, walk_conf, &inp, cut, order, 2 * GAMMA)?;
            assert_eq!(
                got, twin,
                "{} cut {cut} {order:?}: split confidences differ",
                case.name
            );
        }
        println!("SPLIT {} matches the fused kernel", case.name);
        for p in [
            inp.d_logits,
            inp.d_proj,
            inp.d_pred,
            inp.d_succ,
            inp.d_anchor,
            inp.d_ban,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}
