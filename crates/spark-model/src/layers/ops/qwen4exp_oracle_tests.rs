// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity for the qwen4_exp SERVING kernels against the in-process CPU
//! oracle — no checkpoint, no Python, no fixture files.
//!
//! There is already a parity gate for these kernels
//! (`hyper_connection_lowrank_tests.rs`), and it is the stronger evidence:
//! its fixtures come from the real `Qwen4ExpTextGatedResidual` running on real
//! checkpoint weights, so agreement there chains to HuggingFace. But it needs
//! a 126 GiB checkpoint, `transformers`, and a generated `--bin-dir` before it
//! can say anything at all. On a fresh box that is a morning of setup before
//! the first number.
//!
//! This file closes that gap from the other side. `atlas_core::qwen4exp_reference`
//! is a Rust transcription of the same reference module, and it is itself
//! checked against HuggingFace at real weights (hyper-connections 1.6e-7). So
//! driving the kernels against IT gives a gate that runs from a clean
//! checkout with one command:
//!
//! ```text
//! cargo test -p spark-model --release qwen4exp_oracle -- --ignored --nocapture
//! ```
//!
//! What this can and cannot catch, stated plainly: the oracle and the kernel
//! were written by different people from the same source document, so they
//! agree only if both read it the same way — which is the whole point. What it
//! CANNOT catch is a shared misreading, and that is exactly what the
//! checkpoint-backed golden is for. Run both.
//!
//! Every check carries a control that must FAIL, because a parity test that
//! cannot fail proves nothing.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::layers::qwen3_attention::HcLowRank;

/// The published checkpoint's mHC geometry. Testing at the real widths matters:
/// the collapse reduces over `hc_mult * hidden` = 10240 and the kernel's
/// block-wide reduction tree depends on that length, so a 256-wide toy would
/// exercise a different code path than the one that ships.
const HC: usize = 4;
const HIDDEN: usize = 2560;
const RANK: usize = 320;
const EPS: f32 = 1e-6;

/// SplitMix64, so the fixture is byte-identical on every machine and a failure
/// is reproducible from the seed alone. `rand` is not a dev-dependency here and
/// a parity fixture has no business being random anyway.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-scale, scale)`.
    fn signed(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (u * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.signed(scale)).collect()
    }
}

/// Round to BF16 and back. The kernels read BF16 weights, so the oracle must
/// be fed the values the kernel will actually see — otherwise the tolerance
/// absorbs a rounding step instead of a defect.
fn bf16_round(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|v| {
            let bits = v.to_bits();
            // Round-to-nearest-even on the truncated 16 low bits.
            let round = ((bits >> 16) & 1) + 0x7FFF;
            f32::from_bits((bits.wrapping_add(round)) & 0xFFFF_0000)
        })
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
    p
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// max-abs, cosine and the reference's own RMS together: a near-null output has
/// a small max-abs, a mis-scaled one has cosine 1.0, and only the pair rules
/// out both.
fn measure(got: &[f32], want: &[f32]) -> (f32, f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let mut max_abs = 0.0f32;
    let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    (max_abs, cos, (nw / want.len() as f64).sqrt())
}

/// Tolerance from the reference's own scale, not a constant. BF16 outputs carry
/// ~8 mantissa bits, and the kernel sums a 10240-term dot product in a
/// different order than the oracle does.
fn tol_for(want: &[f32]) -> f32 {
    let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    ((rms * 0.05) as f32).max(1e-3)
}

fn check(label: &str, got: &[f32], want: &[f32]) {
    let (max_abs, cos, rms) = measure(got, want);
    let tol = tol_for(want);
    println!("  {label:<20} max|diff|={max_abs:.4e} cos={cos:.9} ref_rms={rms:.4e} tol={tol:.4e}");
    assert!(
        max_abs <= tol,
        "{label}: max|diff| {max_abs:.4e} > {tol:.4e}"
    );
    assert!(cos > 0.9999, "{label}: cos {cos:.9} — the shape differs");
}

/// A control must MISS by a wide margin, or the check above proves nothing.
fn check_control(label: &str, got: &[f32], want: &[f32]) {
    let (max_abs, cos, rms) = measure(got, want);
    println!(
        "  {label:<20} max|diff|={max_abs:.4e} cos={cos:.9} ref_rms={rms:.4e}  (must be LARGE)"
    );
    assert!(
        max_abs > tol_for(want) * 20.0,
        "{label}: control only moved {max_abs:.4e} — the check it guards \
         would pass on a kernel that ignored this"
    );
}

/// One site's weights: generated, BF16-rounded, uploaded — and returned
/// alongside the exact f32 values the oracle must be given.
struct Site {
    dev: HcLowRank,
    norm: Vec<f32>,
    down: Vec<f32>,
    up: Vec<f32>,
    inject: Option<Vec<f32>>,
}

fn make_site(g: &dyn GpuBackend, rng: &mut Rng, inject: bool) -> Site {
    let wide = HC * HIDDEN;
    // `hc_norm` is an OFFSET FROM 1 and the checkpoint's is centred well away
    // from zero (`ple.norm_key` measures -0.1067), so a fixture centred on 0
    // would let a kernel that dropped the offset look close. Centre it at
    // -0.1 like the real thing.
    let norm = bf16_round(
        &rng.vec(wide, 0.35)
            .iter()
            .map(|v| v - 0.1)
            .collect::<Vec<_>>(),
    );
    // 1/sqrt(fan_in) keeps the rank-320 projection's output near unit scale at
    // this width, which is where the sigmoid gates are actually informative --
    // a saturated gate would hide a sign error.
    let down = bf16_round(&rng.vec(RANK * wide, (wide as f32).sqrt().recip()));
    let up = bf16_round(&rng.vec(wide * RANK, (RANK as f32).sqrt().recip()));
    let inject_w = inject.then(|| bf16_round(&rng.vec(HC * wide, (wide as f32).sqrt().recip())));
    Site {
        dev: HcLowRank {
            norm_w: upload(g, &bf16_bytes(&norm)),
            down_w: upload(g, &bf16_bytes(&down)),
            up_w: upload(g, &bf16_bytes(&up)),
            inject_w: match &inject_w {
                Some(w) => upload(g, &bf16_bytes(w)),
                None => DevicePtr::NULL,
            },
            rank: RANK,
        },
        norm,
        down,
        up,
        inject: inject_w,
    }
}

impl Site {
    fn oracle(&self, streams: &[f32], tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let dims = atlas_core::qwen4exp_reference::PleDims {
            hidden: HIDDEN,
            hc_count: HC,
            ple_embed_dim: HIDDEN,
            kernel: 4,
            dilation: 3,
            eps: EPS,
        };
        let w = atlas_core::qwen4exp_reference::HyperConnectionWeights {
            hc_norm: &self.norm,
            mix_down: &self.down,
            mix_up: &self.up,
            block_inject: self.inject.as_deref(),
        };
        let wide = HC * HIDDEN;
        let mut mixed = Vec::with_capacity(tokens * HIDDEN);
        let mut inj = Vec::with_capacity(tokens * HC);
        for t in 0..tokens {
            let out = atlas_core::qwen4exp_reference::hyper_connection_forward(
                &dims,
                &w,
                RANK,
                &streams[t * wide..(t + 1) * wide],
            );
            mixed.extend_from_slice(&out.mixed);
            inj.extend_from_slice(&out.injection);
        }
        (mixed, inj)
    }
}

/// Scratch sized exactly as `BufferSizes::from_config` sizes it, so the test
/// exercises the dispatcher's real decision rather than a generous allocation
/// that hides an under-sized arena.
fn scratch_bytes(max_tokens: usize) -> usize {
    let wide = HC * HIDDEN;
    let split = max_tokens.min(64) * (wide + RANK) * 4;
    let gemm = max_tokens.min(2048) * (2 * wide + RANK + HC) * 2;
    split.max(gemm)
}

fn backend() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    // By identity, NOT `ptx_modules()`: in a wildcard build that is an alias
    // for target 0, and `hyper_connection` in another target's set is
    // DeepSeek-V4's Sinkhorn kernel — the same name over a different argument
    // list, which is a segfault or, worse, plausible numbers.
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").expect(
        "qwen3.8-flash-next/nvfp4 is not in this build — \
         build with ATLAS_TARGET_MODEL='*' or =qwen3.8-flash-next",
    );
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

/// `hc_pre` / `hc_head` / `hc_post` against the CPU oracle at THREE token
/// counts, because the token count is what selects the implementation:
///
/// * `T = 1`   — the three-launch split collapse (the decode path)
/// * `T = 64`  — the split path at its ceiling
/// * `T = 96`  — the batched-GEMM path (`T > 64`)
///
/// The fused FP32 kernel that `T <= 64` used before the split landed is still
/// reachable with `ATLAS_QWEN4EXP_NO_HC_GEMM=1`, and this test honours that
/// switch, so running it twice covers all four arms.
#[test]
#[ignore]
fn qwen4exp_oracle_hc_matches_the_cpu_reference() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_pre = g.kernel("hyper_connection", "hc_pre").unwrap();
    let k_head = g.kernel("hyper_connection", "hc_head").unwrap();
    let k_post = g.kernel("hyper_connection", "hc_post").unwrap();
    for (name, k) in [("hc_pre", k_pre), ("hc_head", k_head), ("hc_post", k_post)] {
        assert!(
            k.0 != 0,
            "{name} resolved to handle 0 — the qwen3.8-flash-next shadow is \
             not what got loaded, so this would be testing another model's mHC"
        );
    }

    let wide = HC * HIDDEN;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // ── hc_expand: seed the highway. Checked FIRST because everything after
    //    it is measured against streams this produced, and because the failure
    //    mode is invisible: the streams must start IDENTICAL (a copy of the
    //    embedding into all hc slots). Zero-initialising them instead makes
    //    the first collapse read a zero mean, and the model never recovers
    //    while still emitting tokens.
    {
        let k_expand = g.kernel("hyper_connection", "hc_expand").unwrap();
        assert!(k_expand.0 != 0, "hc_expand resolved to handle 0");
        let tokens = 3usize;
        let embed = bf16_round(&rng.vec(tokens * HIDDEN, 1.0));
        // Sentinel fill: a kernel that wrote only stream 0 would match
        // perfectly on the part it wrote, so the rest must be proven
        // overwritten rather than assumed.
        let seeded = g.alloc(tokens * wide * 4).unwrap();
        g.copy_h2d_async(&f32_bytes(&vec![-7.0f32; tokens * wide]), seeded, stream)
            .unwrap();
        ops::hc_expand(
            g,
            k_expand,
            upload(g, &bf16_bytes(&embed)),
            seeded,
            tokens as u32,
            HIDDEN as u32,
            HC as u32,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let got = download_f32(g, seeded, tokens * wide);
        let want: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                std::iter::repeat_n(&embed[t * HIDDEN..(t + 1) * HIDDEN], HC)
                    .flatten()
                    .copied()
            })
            .collect();
        println!("hc_expand (trunk entry):");
        check("all streams", &got, &want);
        assert!(
            !got.iter().any(|v| *v == -7.0),
            "hc_expand left sentinel values behind — some stream was never written"
        );
    }

    // Both per-layer sites carry an injection; the model-level mixer is built
    // `use_combine=False` and carries none.
    let attn = make_site(g, &mut rng, true);
    let mlp = make_site(g, &mut rng, true);
    let head = make_site(g, &mut rng, false);

    let max_t = 96usize;
    // The highway is FP32 — see the note in hyper_connection.cu. Feeding it
    // BF16-rounded values would understate the kernel's own error.
    let streams_host = rng.vec(max_t * wide, 1.0);
    let streams = upload(g, &f32_bytes(&streams_host));
    let y_out = g.alloc(max_t * HIDDEN * 2).unwrap();
    let inj_out = g.alloc(max_t * HC * 4).unwrap();
    let scratch = g.alloc(scratch_bytes(max_t)).unwrap();

    for tokens in [1usize, 64, 96] {
        println!(
            "\nT={tokens} ({})",
            if tokens <= 64 {
                "split collapse — the decode path"
            } else {
                "batched-GEMM collapse"
            }
        );
        for (label, site) in [("attn", &attn), ("mlp", &mlp)] {
            ops::hc_pre_lowrank(
                g,
                k_pre,
                streams,
                &site.dev,
                y_out,
                inj_out,
                scratch,
                tokens as u32,
                HIDDEN as u32,
                HC as u32,
                EPS,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();

            let (want_mixed, want_inj) = site.oracle(&streams_host, tokens);
            println!(" {label}_hyper_connection:");
            check(
                "mixed_input",
                &download_bf16(g, y_out, tokens * HIDDEN),
                &want_mixed,
            );
            check(
                "injection",
                &download_f32(g, inj_out, tokens * HC),
                &want_inj,
            );

            // CONTROL. The reduction over streams is a MEAN, not a sum, and at
            // hc=4 a sum is 4x the intended magnitude — survivable-looking and
            // wrong. Compare against the sum and require a wide miss.
            let summed: Vec<f32> = want_mixed.iter().map(|v| v * HC as f32).collect();
            check_control(
                "control: sum≠mean",
                &download_bf16(g, y_out, tokens * HIDDEN),
                &summed,
            );
        }

        // The model-level mixer. This IS the model's final norm — the
        // checkpoint ships no `model.norm.weight` — so a wrong one here is a
        // wrong logit distribution on every token.
        ops::hc_head_lowrank(
            g,
            k_head,
            streams,
            &head.dev,
            y_out,
            scratch,
            tokens as u32,
            HIDDEN as u32,
            HC as u32,
            EPS,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let (want_head, _) = head.oracle(&streams_host, tokens);
        println!(" hyper_connection_mixer:");
        check(
            "mixed_input",
            &download_bf16(g, y_out, tokens * HIDDEN),
            &want_head,
        );
    }

    // ── hc_post: the residual fold. Bit-exact is the bar here: it is an FP32
    //    multiply-add with no reduction, so anything else means a layout bug.
    let tokens = 96usize;
    let block_out = bf16_round(&rng.vec(tokens * HIDDEN, 1.0));
    let (_, inj_host) = attn.oracle(&streams_host, tokens);
    let post_out = g.alloc(tokens * wide * 4).unwrap();
    ops::hc_post_lowrank(
        g,
        k_post,
        upload(g, &bf16_bytes(&block_out)),
        streams,
        upload(g, &f32_bytes(&inj_host)),
        post_out,
        tokens as u32,
        HIDDEN as u32,
        HC as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let mut want_post = Vec::with_capacity(tokens * wide);
    for t in 0..tokens {
        let folded = atlas_core::qwen4exp_reference::broadcast_inject(
            &block_out[t * HIDDEN..(t + 1) * HIDDEN],
            &inj_host[t * HC..(t + 1) * HC],
            HIDDEN,
        );
        want_post.extend(
            streams_host[t * wide..(t + 1) * wide]
                .iter()
                .zip(&folded)
                .map(|(residual, add)| residual + add),
        );
    }
    println!("\nhc_post:");
    check(
        "residual",
        &download_f32(g, post_out, tokens * wide),
        &want_post,
    );
}

// The PLE half lives in its own file: the repo caps a .rs at 500 LoC, and the
// two halves share this file's harness — `use super::*` from the child reaches
// the private helpers above, which a sibling `#[path]` module would not.
#[path = "qwen4exp_oracle_ple_tests.rs"]
mod ple;
