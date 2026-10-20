// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`: a GB10, a glm-5.3-flash kernel build and the real
//! nvidia/GLM-5.3-Flash-FP4-Local checkpoint) of the FP4-Local loader paths
//! against host references computed from the checkpoint's own bytes:
//!
//! - every expansion `dense_auto` makes (MXFP8 KDA / dense MLP, FP8_PB_WO MLA
//!   and shared expert) is bit-identical to the host reference;
//! - a whole KDA layer (layer 0) loads on both TP2 ranks, its NVFP4 decode
//!   twins sit within NVFP4's error of the rank's slice of the exact BF16,
//!   and its MXFP8 sources are then released;
//! - a shared expert (layer 3) requantizes within NVFP4's error;
//! - the lm_head expands exactly and each rank's split-head shard is the
//!   checkpoint's rows;
//! - ViT block 0: MXFP8 qkv exact, W4A16 NVFP4 MLP within one BF16 ulp, the
//!   gate/up concatenation in order.
//!
//! ```sh
//! ATLAS_GLM_FP4L_CKPT=<checkpoint-dir> cargo test --release -p spark-model --lib \
//!     glm5::fp4l::tests::gpu -- --ignored --nocapture --test-threads=1
//! ```

use super::*;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;
use crate::weight_map::{QuantizeCtx, QuantizedWeight, mxfp8_to_bf16_host};
use anyhow::{Context, Result, ensure};
use atlas_core::numeric::{f32_to_bf16, fp8_e4m3_to_f32, nvfp4_dequant};
use spark_runtime::cuda_backend::AtlasCudaBackend;

fn ckpt() -> Result<std::path::PathBuf> {
    std::env::var("ATLAS_GLM_FP4L_CKPT")
        .map(Into::into)
        .context("ATLAS_GLM_FP4L_CKPT=<nvidia/GLM-5.3-Flash-FP4-Local dir>")
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

/// Host bytes of the tensors a test reads, by name.
type Host = HashMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>;

/// `names` read from the checkpoint and uploaded into a fresh store, with the
/// host bytes kept for references.
fn upload(gpu: &dyn GpuBackend, names: &[String]) -> Result<(WeightStore, Host)> {
    let dir = ckpt()?;
    let (mut map, mut host) = (HashMap::new(), Host::new());
    for name in names {
        let (dtype, shape, bytes) = read_checkpoint_tensor(&dir, name)?;
        let dtype = WeightDtype::from_safetensors_str(&dtype)?;
        let ptr = gpu.alloc(bytes.len().max(1))?;
        gpu.copy_h2d(&bytes, ptr)?;
        map.insert(
            name.clone(),
            WeightTensor {
                ptr,
                shape: shape.clone(),
                dtype,
            },
        );
        host.insert(name.clone(), (dtype, shape, bytes));
    }
    Ok((WeightStore::from_map(map), host))
}

fn with_parts(stems: &[String], parts: &[&str]) -> Vec<String> {
    stems
        .iter()
        .flat_map(|s| parts.iter().map(move |p| format!("{s}.{p}")))
        .collect()
}

fn bf16s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b)?;
    let values: Vec<_> = b
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    ensure!(
        values.iter().all(|&v| f(v).is_finite()),
        "non-finite BF16 GPU output"
    );
    Ok(values)
}

fn f(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// The host BF16 of checkpoint projection `stem`: MXFP8 (exact) or FP8 block
/// (`e4m3 * scale` in f32, rounded once to BF16, as the kernel does).
fn host_bf16(host: &Host, stem: &str) -> Result<(Vec<u16>, usize, usize)> {
    let (_, ws, w) = &host[&format!("{stem}.weight")];
    let (sd, ss, s) = &host[&format!("{stem}.weight_scale")];
    let [n, k] = ws[..] else {
        anyhow::bail!("{stem}: {ws:?}")
    };
    if *sd == WeightDtype::UInt8 {
        return Ok((mxfp8_to_bf16_host(w, s, n, k)?, n, k));
    }
    let (sn, sk) = crate::weight_map::block_scale_grid(ss).context("block grid")?;
    let scale: Vec<f32> = s
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (bn, bk) = (n / sn, k / sk);
    let out = (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            f32_to_bf16(fp8_e4m3_to_f32(w[i]) * scale[(r / bn) * sk + c / bk])
        })
        .collect();
    Ok((out, n, k))
}

fn l(i: usize, rest: &str) -> String {
    format!("model.language_model.layers.{i}.{rest}")
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn expansions_are_bit_identical_to_the_host_reference() -> Result<()> {
    let gpu = backend()?;
    let stems = [
        l(0, "self_attn.q_proj"),
        l(0, "self_attn.o_proj"),
        l(0, "mlp.down_proj"),
        l(3, "self_attn.q_a_proj"),
        l(3, "self_attn.q_b_proj"),
        l(3, "self_attn.kv_a_proj_with_mqa"),
        l(3, "self_attn.o_proj"),
        l(3, "mlp.shared_experts.gate_proj"),
        l(3, "mlp.shared_experts.down_proj"),
    ];
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    for stem in &stems {
        let (want, n, k) = host_bf16(&host, stem)?;
        let out = crate::weight_map::dense_auto(&store, &format!("{stem}.weight"), gpu)?;
        let got = bf16s(gpu, out.weight, n * k)?;
        gpu.free(out.weight)?;
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        ensure!(
            bad == 0,
            "{stem} [{n}, {k}]: {bad} elements differ from the host"
        );
        eprintln!("{stem} [{n}, {k}]: kernel == host");
    }
    Ok(())
}

/// Per 16-value block, every NVFP4 element within a quarter of the block's
/// absolute max of the reference (E2M1's widest step is 2 of 6, plus the
/// E4M3 rounding of the block scale), and the whole tensor within 12% RMS.
fn assert_nvfp4_close(
    gpu: &dyn GpuBackend,
    what: &str,
    q: &QuantizedWeight,
    reference: &[u16],
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(!q.has_per_row_scale2(), "{what}: per-row scale2");
    let kernel = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;
    let out = gpu.alloc(n * k * 2)?;
    let stream = gpu.default_stream();
    crate::layers::ops::dequant_nvfp4_to_bf16(
        gpu,
        kernel,
        q.weight,
        q.weight_scale,
        out,
        q.weight_scale_2,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let got = bf16s(gpu, out, n * k)?;
    gpu.free(out)?;
    let (mut err2, mut ref2) = (0f64, 0f64);
    for (b, (g, r)) in got.chunks(16).zip(reference.chunks(16)).enumerate() {
        let amax = r.iter().map(|&x| f(x).abs()).fold(0f32, f32::max);
        for (&g, &r) in g.iter().zip(r) {
            let e = (f(g) - f(r)).abs();
            ensure!(
                e <= 0.25 * amax + 1e-30,
                "{what}: block {b} error {e} > amax/4 ({amax})"
            );
            err2 += (e as f64).powi(2);
            ref2 += (f(r) as f64).powi(2);
        }
    }
    let rel = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
    ensure!(rel < 0.12, "{what}: relative RMS error {rel}");
    eprintln!("{what} [{n}, {k}]: NVFP4 rel RMS {rel:.4}");
    Ok(())
}

fn qctx(gpu: &dyn GpuBackend) -> Result<QuantizeCtx> {
    Ok(QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    })
}

/// The config a TP2 rank loads with (`serve_phases::topology` halves the
/// head counts).
fn tp2_rank(rank: usize) -> ModelConfig {
    let mut c = fp4l_config();
    c.num_attention_heads /= 2;
    c.num_key_value_heads /= 2;
    c.linear_num_key_heads /= 2;
    c.linear_num_value_heads /= 2;
    (c.tp_rank, c.tp_world_size, c.ep_rank, c.ep_world_size) = (rank, 2, rank, 2);
    c
}

/// Rows `[lo, hi)` and columns `[c0, c1)` of a row-major `[_, k]` matrix.
fn block(m: &[u16], k: usize, (lo, hi): (usize, usize), (c0, c1): (usize, usize)) -> Vec<u16> {
    (lo..hi)
        .flat_map(|r| m[r * k + c0..r * k + c1].to_vec())
        .collect()
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn kda_layer_0_loads_its_tp2_nvfp4_twins_from_mxfp8() -> Result<()> {
    let gpu = backend()?;
    let p = |x: &str| l(0, &format!("self_attn.{x}"));
    let mxfp8 = ["q_proj", "k_proj", "v_proj", "o_proj"].map(p);
    let mut names = with_parts(&mxfp8, &["weight", "weight_scale"]);
    for x in [
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "b_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
    ] {
        names.push(p(x));
    }
    for rank in 0..2 {
        let config = tp2_rank(rank);
        let (store, host) = upload(gpu, &names)?;
        let lp = config.layer_prefix(0);
        let w = crate::weight_loader::glm5::components::load_kda_weights(
            &store,
            &lp,
            &config,
            gpu,
            qctx(gpu)?,
            None,
        )?;
        for (stem, q, column) in [
            (&mxfp8[0], &w.q_proj.nvfp4, true),
            (&mxfp8[3], &w.o_proj.nvfp4, false),
        ] {
            let (full, n, k) = host_bf16(&host, stem)?;
            let (rows, cols) = if column {
                ((rank * n / 2, (rank + 1) * n / 2), (0, k))
            } else {
                ((0, n), (rank * k / 2, (rank + 1) * k / 2))
            };
            let want = block(&full, k, rows, cols);
            let what = format!("rank {rank} {stem}");
            assert_nvfp4_close(gpu, &what, q, &want, rows.1 - rows.0, cols.1 - cols.0)?;
        }
        // The MXFP8 sources outlive the layer build, then go in one place.
        let mut released = 0;
        for stem in decoder_expanded_stems(&lp, LayerType::LinearAttention, true) {
            released += release_expanded(&store, gpu, &stem, false)?;
        }
        let want: usize = with_parts(&mxfp8, &["weight", "weight_scale"])
            .iter()
            .map(|n| host[n].2.len())
            .sum();
        ensure!(
            released == want,
            "released {released} of {want} source bytes"
        );
    }
    Ok(())
}

/// MLA layer 3 builds on both TP2 ranks from FP8_PB_WO q_a/q_b/kv_a/o and
/// BF16 kv_b/indexer (the expansions themselves are checked bit for bit
/// above), and its FP8 sources are then released, nothing else.
#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn mla_layer_3_builds_on_both_tp2_ranks() -> Result<()> {
    let gpu = backend()?;
    let a = |x: &str| l(3, &format!("self_attn.{x}"));
    let fp8 = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"].map(a);
    let mut names = with_parts(&fp8, &["weight", "weight_scale"]);
    for x in [
        "kv_b_proj.weight",
        "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.weights_proj.weight",
        "indexer.index_kpool_compress_gate",
        "indexer.index_kpool_compress_ape",
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
    ] {
        names.push(a(x));
    }
    names.push(l(3, "input_layernorm.weight"));
    names.push(l(3, "post_attention_layernorm.weight"));
    for rank in 0..2 {
        let config = tp2_rank(rank);
        let (store, host) = upload(gpu, &names)?;
        let lp = config.layer_prefix(3);
        let norm = |x: &str| norm_bf16(&store, &format!("{lp}.{x}"), gpu);
        crate::weight_loader::glm5::layers::load_mla_layer(
            &store,
            &lp,
            3,
            0,
            norm("input_layernorm.weight")?,
            norm("post_attention_layernorm.weight")?,
            crate::layers::FfnComponent::None,
            None,
            &config,
            gpu,
            spark_runtime::kv_cache::KvCacheDtype::Bf16,
            false,
            None,
        )?;
        let mut released = 0;
        for stem in decoder_expanded_stems(&lp, LayerType::FullAttention, false) {
            released += release_expanded(&store, gpu, &stem, false)?;
        }
        let want: usize = with_parts(&fp8, &["weight", "weight_scale"])
            .iter()
            .map(|n| host[n].2.len())
            .sum();
        ensure!(
            released == want,
            "rank {rank}: released {released} of {want} bytes"
        );
        ensure!(
            !store.was_reclaimed(&a("kv_b_proj.weight")),
            "BF16 kv_b released"
        );
        eprintln!("rank {rank}: MLA layer 3 built, {released} FP8 source bytes released");
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn shared_expert_requantizes_from_fp8_blocks() -> Result<()> {
    let gpu = backend()?;
    let stems = ["gate_proj", "down_proj"].map(|x| l(3, &format!("mlp.shared_experts.{x}")));
    let (store, host) = upload(gpu, &with_parts(&stems, &["weight", "weight_scale"]))?;
    for stem in &stems {
        let (want, n, k) = host_bf16(&host, stem)?;
        let q = crate::weight_map::quantized_any(
            &store,
            stem,
            n,
            k,
            gpu,
            crate::weight_map::Nvfp4Variant::Standard,
            qctx(gpu)?,
        )?;
        assert_nvfp4_close(gpu, stem, &q, &want, n, k)?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn lm_head_expands_exactly_and_each_rank_shard_is_the_checkpoint_rows() -> Result<()> {
    let gpu = backend()?;
    let (store, host) = upload(
        gpu,
        &with_parts(&["lm_head".into()], &["weight", "weight_scale"]),
    )?;
    let (_, ws, w) = &host["lm_head.weight"];
    let (_, _, s) = &host["lm_head.weight_scale"];
    let (n, k) = (ws[0], ws[1]);
    let head = lm_head_bf16(&store, gpu)?;
    ensure!(head.dtype == WeightDtype::BF16 && head.shape == [n, k]);
    // Sampled rows (the host reference is per element): both ends, the
    // vocabulary split and the padding.
    let vocab = fp4l_config().vocab_size;
    for lo in [0, vocab / 2 - 8, vocab / 2, n - 64] {
        let hi = (lo + 64).min(n);
        let want =
            mxfp8_to_bf16_host(&w[lo * k..hi * k], &s[lo * k / 32..hi * k / 32], hi - lo, k)?;
        let got = bf16s(gpu, head.ptr.offset(lo * k * 2), (hi - lo) * k)?;
        ensure!(
            got == want,
            "lm_head rows [{lo}, {hi}) differ from the host"
        );
    }
    let raw = crate::weight_map::raw_mxfp8(&store, "lm_head")?.context("MXFP8 lm_head")?;
    let shard = vocab / 2;
    for rank in 0..2 {
        let twin = crate::weight_map::mxfp8_row_range_twin(
            gpu,
            raw,
            rank * shard,
            shard,
            gpu.default_stream(),
        )?;
        let (mut tw, mut ts) = (vec![0u8; shard * k], vec![0u8; shard * k / 32]);
        gpu.copy_d2h(twin.data, &mut tw)?;
        gpu.copy_d2h(twin.scales, &mut ts)?;
        let (r0, r1) = (rank * shard, (rank + 1) * shard);
        ensure!(
            tw == w[r0 * k..r1 * k] && ts == s[r0 * k / 32..r1 * k / 32],
            "rank {rank} shard"
        );
        gpu.free(twin.data)?;
        gpu.free(twin.scales)?;
    }
    crate::weight_map::release_raw_mxfp8(&store, gpu, "lm_head")?;
    gpu.free(head.ptr)?;
    Ok(())
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and ATLAS_GLM_FP4L_CKPT"]
fn vit_block_0_linears_expand_to_the_host_reference() -> Result<()> {
    let gpu = backend()?;
    let b = |x: &str| format!("model.visual.blocks.0.{x}");
    let (qkv, gate, up) = (b("attn.qkv"), b("mlp.gate_proj"), b("mlp.up_proj"));
    let mut names = with_parts(std::slice::from_ref(&qkv), &["weight", "weight_scale"]);
    names.extend(with_parts(
        &[gate.clone(), up.clone()],
        &["weight", "weight_scale", "weight_scale_2"],
    ));
    let (store, host) = upload(gpu, &names)?;
    let (want, n, k) = host_bf16(&host, &qkv)?;
    let (ptr, shape) = crate::weight_loader::glm5::vision::linear(&store, gpu, &qkv)?;
    ensure!(
        shape == [n, k] && bf16s(gpu, ptr, n * k)? == want,
        "qkv not exact"
    );
    let nvfp4_host = |stem: &str| -> Result<(Vec<u16>, usize, usize)> {
        let (_, ws, w) = &host[&format!("{stem}.weight")];
        let (_, _, s) = &host[&format!("{stem}.weight_scale")];
        let (_, _, g) = &host[&format!("{stem}.weight_scale_2")];
        let (n, k) = (ws[0], ws[1] * 2);
        let global = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
        let v = nvfp4_dequant(w, s, global, n, k, 16).map_err(anyhow::Error::msg)?;
        Ok((v.into_iter().map(f32_to_bf16).collect(), n, k))
    };
    let mut concat = Vec::new();
    for stem in [&gate, &up] {
        let (want, n, k) = nvfp4_host(stem)?;
        let (ptr, shape) = crate::weight_loader::glm5::vision::linear(&store, gpu, stem)?;
        ensure!(shape == [n, k], "{stem}: shape {shape:?}");
        let got = bf16s(gpu, ptr, n * k)?;
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        ensure!(worst <= 1, "{stem}: {worst} BF16 ulps from the host");
        concat.extend(want);
        gpu.free(ptr)?;
    }
    let parts = [gate.clone(), up.clone()];
    let fused =
        crate::weight_loader::glm5::vision::concat_weights(&store, gpu, "absent.gate_up", &parts)?;
    let got = bf16s(gpu, fused, concat.len())?;
    let worst = got
        .iter()
        .zip(&concat)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max();
    ensure!(worst <= Some(1), "gate/up concatenation: {worst:?} ulps");
    Ok(())
}
