// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle gate for `ATLAS_GDN_DEFERRED_COMMIT` (`gated_delta_rule_wy{K}_defer`
//! verify + `gated_delta_rule_commit` replay). Asserts, for every K in the
//! wyN band and every accept count a ∈ [1, K]:
//!
//! 1. `_defer` verify: per-token outputs byte-identical to the storing
//!    `gated_delta_rule_wy{K}`, and `h_state` left holding H0 (no stores).
//! 2. `gated_delta_rule_commit` with `accepted_count = a`: `h_state` ends
//!    byte-identical to the oracle's `intermediates[a-1]` (a < K) or final
//!    H (a == K) — the replay runs the same `gated_delta_rule_wyn_impl`
//!    arithmetic, so this must hold bit-for-bit, not approximately.
//!
//! GPU test: `#[ignore]` per repo convention. Run with
//! ```text
//! cargo test -p spark-model --release --features cuda \
//!   gdn_deferred_commit -- --ignored --nocapture
//! ```

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const CONV_DIM: usize = NK * KD * 2 + NV * VD;
const KEY_DIM: usize = NK * KD;
const GB_STRIDE: usize = NV * 2;
const HV: usize = NV * KD * VD;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> DevicePtr {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1)).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> DevicePtr {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1)).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b).unwrap();
    b
}

fn check_deferred_commit(k: usize) {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-27b", "nvfp4")
        .expect("qwen3.8-27b/nvfp4 not in this build");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let store_k = g
        .kernel("gated_delta_rule_wyn", &format!("gated_delta_rule_wy{k}"))
        .unwrap();
    let defer_k = g
        .kernel(
            "gated_delta_rule_wyn",
            &format!("gated_delta_rule_wy{k}_defer"),
        )
        .unwrap();
    let commit_k = g
        .kernel("gated_delta_rule_wyn", "gated_delta_rule_commit")
        .unwrap();
    assert!(
        store_k.0 != 0 && defer_k.0 != 0 && commit_k.0 != 0,
        "wy{k}/defer/commit handles must all resolve (got {:x}/{:x}/{:x})",
        store_k.0,
        defer_k.0,
        commit_k.0
    );

    let mut rng = Lcg(0xde17_5eed ^ (k as u64) << 20);
    let qkv: Vec<bf16> = (0..k * CONV_DIM)
        .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
        .collect();
    let gate: Vec<f32> = (0..k * GB_STRIDE).map(|_| rng.r(0.90, 0.999)).collect();
    let beta: Vec<f32> = (0..k * GB_STRIDE).map(|_| rng.r(0.1, 0.9)).collect();
    let h_init: Vec<f32> = (0..HV).map(|_| rng.r(-0.5, 0.5)).collect();

    let d_qkv = up_bf16(g, &qkv);
    let d_gate = up_f32(g, &gate);
    let d_beta = up_f32(g, &beta);
    let q = d_qkv;
    let kk = d_qkv.offset(KEY_DIM * 2);
    let vv = d_qkv.offset(KEY_DIM * 2 * 2);
    let out_bytes = k * NV * VD * 2;

    // Oracle: storing wyK — final H in `h_state`, Hi_0..K-2 in the pool.
    let ni = k - 1;
    let oracle_h = up_f32(g, &h_init);
    let oracle_hi = g.alloc(ni * HV * 4).unwrap();
    let oracle_out = g.alloc(out_bytes).unwrap();
    ops::gdn_decode_wyn(
        g,
        store_k,
        oracle_h,
        q,
        kk,
        vv,
        d_gate,
        d_beta,
        oracle_out,
        oracle_hi,
        HV as u32,
        1,
        NK as u32,
        NV as u32,
        KD as u32,
        VD as u32,
        CONV_DIM as u32,
        CONV_DIM as u32,
        GB_STRIDE as u32,
        false,
        0,
    )
    .unwrap();
    g.synchronize(0).unwrap();

    // Defer verify: same outputs, h_state untouched.
    let defer_h = up_f32(g, &h_init);
    let defer_out = g.alloc(out_bytes).unwrap();
    ops::gdn_decode_wyn(
        g,
        defer_k,
        defer_h,
        q,
        kk,
        vv,
        d_gate,
        d_beta,
        defer_out,
        DevicePtr(0), // intermediates arg unused under MODE 1
        0,
        1,
        NK as u32,
        NV as u32,
        KD as u32,
        VD as u32,
        CONV_DIM as u32,
        CONV_DIM as u32,
        GB_STRIDE as u32,
        false,
        0,
    )
    .unwrap();
    g.synchronize(0).unwrap();
    assert_eq!(
        down(g, oracle_out, out_bytes),
        down(g, defer_out, out_bytes),
        "k={k}: defer verify outputs differ"
    );
    assert_eq!(
        down(g, defer_h, HV * 4),
        h_init
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        "k={k}: defer verify wrote h_state (must stay H0)"
    );

    for a in 1..=k {
        // Commit arm: H0 + replay of the first `a` rows -> h_state.
        let tst_h = up_f32(g, &h_init);
        ops::gdn_commit_accepted(
            g,
            commit_k,
            tst_h,
            q,
            kk,
            vv,
            d_gate,
            d_beta,
            a as u32,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            CONV_DIM as u32,
            CONV_DIM as u32,
            GB_STRIDE as u32,
            0,
        )
        .unwrap();
        g.synchronize(0).unwrap();

        // Oracle for state after token a-1: intermediates[a-1] when a<K,
        // else the storing kernel's final h_state.
        let want = if a == k {
            down(g, oracle_h, HV * 4)
        } else {
            down(g, oracle_hi.offset((a - 1) * HV * 4), HV * 4)
        };
        assert_eq!(
            down(g, tst_h, HV * 4),
            want,
            "k={k} a={a}: committed state differs from oracle"
        );
    }
    eprintln!("k={k}: outputs identical; commit matches oracle for every a in 1..={k}");
}

/// wyN band: K = 5..16. Covers 5/8/9/12/16 per the v1 brief — the DFlash
/// γ=8 hot path (K=9) and both band edges.
#[test]
#[ignore]
fn gdn_deferred_commit_byte_identical() {
    for k in [5usize, 8, 9, 12, 16] {
        check_deferred_commit(k);
    }
}

/// The `gdn_commit_pending` flag is the accept path's record that a verify
/// ran deferred — it MUST be assigned at the dispatch entry (`mod.rs`
/// `decode_verify*` fns, outside `begin_capture`), never inside the layer
/// forward, because replayed graphs skip host code entirely. Job-621
/// corruption: pending set inside the captured forward stayed false on
/// replay → accept restored from intermediates MODE 1 never wrote.
#[test]
fn gdn_commit_pending_is_set_at_dispatch_not_in_the_layer() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/src/layers/qwen3_ssm/");
    for e in std::fs::read_dir(root).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "rs") {
            continue;
        }
        let src = std::fs::read_to_string(&p).unwrap();
        assert!(
            !src.contains("gdn_commit_pending = true"),
            "{}: gdn_commit_pending is assigned inside the (capturable)              layer forward — replayed graphs never see it; set it at the              decode_verify* dispatch entry instead",
            p.display()
        );
    }
    // …and the dispatch entries must actually call the mark. Every public
    // verify path pins `mark_gdn_deferred_commit` at its entry.
    let mod_rs = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/model/trait_impl/mod.rs"
    ))
    .unwrap();
    for entry in [
        "decode_verify_dispatch(tokens",
        "decode_verify_graphed_dispatch(tokens",
        "decode_verify_graphed_k3_dispatch(tokens",
        "decode_verify_graphed_k4_dispatch(tokens",
        "decode_verify_batched_dispatch(tokens",
        "decode_verify_graphed_kgamma_dispatch(tokens",
        "decode_and_verify_fused_dispatch(tokens",
    ] {
        let pos = mod_rs
            .find(entry)
            .unwrap_or_else(|| panic!("mod.rs: `{entry}` not found"));
        let window = &mod_rs[..pos];
        // The mark precedes the dispatch call within the same fn body —
        // look back a bounded window (the rollback guard + mark are the
        // only statements between the signature and the dispatch).
        let tail = window.rsplit_once("fn ").map(|(_, t)| t).unwrap_or(window);
        assert!(
            tail.contains("mark_gdn_deferred_commit"),
            "`{entry}` is invoked without a preceding mark_gdn_deferred_commit              in its fn body — replays would run the deferred kernel with the              pending flag unset (stale-intermediates corruption)"
        );
    }
}
