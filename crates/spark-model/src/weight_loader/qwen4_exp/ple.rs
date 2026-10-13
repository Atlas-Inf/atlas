// SPDX-License-Identifier: AGPL-3.0-only

//! PLE weights: the projections, the three norms, the dilated conv, and the
//! 320M-row n-gram table served off NVMe.
//!
//! ```text
//! {lp}.ple.key_proj.weight                       [hc*H, ple_embed_dim]
//! {lp}.ple.value_proj.weight                     [H,    ple_embed_dim]
//! {lp}.ple.norm_key/norm_query/norm_conv.weight  [hc*H]
//! {lp}.ple.conv1d.weight                         [hc*H, 1, K]
//! {lp}.ple.ple_embedding.ngram_embedding.layer_multipliers       [ngram_size]   I64
//! {lp}.ple.ple_embedding.ngram_embedding.head_offsets            [ngram_heads]  I64
//! {lp}.ple.ple_embedding.ngram_embedding.head_vocab_sizes        [ngram_heads]  I64
//! {lp}.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight   [R, 160] BF16
//!
//! NVFP4 packs ship the three lookup tensors under the old names
//! (`ple_embedding.layer_multipliers`, `.ngram_heads_offsets`,
//! `.ngram_heads_vocab_sizes`). Probe the EXL3 names first, then those.
//! A single unsharded `ngram_embedding.trellis` (4.05 bpw) is one contiguous
//! row-major table. The row cache opens it at that tensor's byte offset,
//! the same way a safetensors shard is opened. Equal-sized `shard_{i}`
//! tensors stay on the segmented path.
//! ```
//!
//! The 128 shards are ONE logical table of `128 * R` rows. They live in a
//! single safetensors file but are NOT laid out consecutively — other weights
//! interleave — so the row cache is opened SEGMENTED, with each shard's own
//! base offset. A single-offset open would read the wrong rows for every
//! shard past the first and, since the rows are all valid embeddings, would
//! do it silently.

#[cfg(feature = "cuda")]
use anyhow::Context;
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

#[cfg(feature = "cuda")]
use crate::layers::ngram_embed::NgramTable;
use crate::layers::ple::PleLayer;
#[cfg(feature = "cuda")]
use crate::layers::ple::{PleIdDims, PleWeights};
#[cfg(feature = "cuda")]
use crate::weight_map::dense;

/// Resident rows in the pinned arena. A forward's gather pins at most one
/// scratch span of rows at once — `scratch_tokens * ngram_heads` — because
/// the chunked pipeline resolves and releases a span at a time, so the
/// default is DERIVED from the span width: `(scratch + warm_ahead) *
/// ngram_heads`, rounded up to a power of two, floored at 65536. The
/// `warm_ahead` term is the prefill prefetch window (`warm.rs`): without
/// slack beside the span's pins, the worker's rows would have to evict
/// each other — or the span's — before the gather reads them. The old
/// `max_batch_tokens` derivation predates the span loop and would
/// over-provision the arena 4x at a 32K --max-prefill-tokens.
#[cfg(feature = "cuda")]
fn derived_slots(scratch_tokens: usize, ngram_heads: usize) -> usize {
    (scratch_tokens
        .saturating_add(crate::layers::ple::warm_ahead_tokens(scratch_tokens))
        .saturating_mul(ngram_heads)
        .next_power_of_two())
    .max(65536)
}

/// The forward pipeline's scratch width. The six `[span, hc*H]` buffers cost
/// `scratch * 10240 * 14` bytes — ~1.18 GB at the 8192 default — whatever the
/// forward width: the llama.cpp ubatch discipline, applied to the one
/// subsystem whose scratch used to scale with `--max-prefill-tokens`.
#[cfg(feature = "cuda")]
fn chunk_from_env() -> usize {
    std::env::var("ATLAS_PLE_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(8192)
}

#[cfg(feature = "cuda")]
fn slots_from_env(scratch_tokens: usize, ngram_heads: usize) -> (usize, &'static str) {
    match std::env::var("ATLAS_PLE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(n) if n > 0 => (n, "ATLAS_PLE_CACHE_SLOTS"),
        _ => (
            derived_slots(scratch_tokens, ngram_heads),
            "(span + warm_ahead)*heads rounded up",
        ),
    }
}

/// First uploaded name that exists. EXL3 checkpoints use the nested
/// `ngram_embedding.*` names; RadixArk NVFP4 still ships the old
/// `ple_embedding.*` names. Missing both is the caller's error.
#[cfg(feature = "cuda")]
fn first_present<'a>(store: &WeightStore, names: &[&'a str]) -> Option<&'a str> {
    names.iter().copied().find(|n| store.get(n).is_ok())
}

/// EXL3 name first (`ngram_embedding.{exl3}`), then the NVFP4 name
/// (`{nvfp4}` directly under `ple_embedding`). If neither is uploaded, return
/// the EXL3 name so `i64_host` reports that miss.
#[cfg(feature = "cuda")]
fn ple_i64_name(store: &WeightStore, lp: &str, exl3: &str, nvfp4: &str) -> String {
    let nested = format!("{lp}.ple_embedding.ngram_embedding.{exl3}");
    let flat = format!("{lp}.ple_embedding.{nvfp4}");
    first_present(store, &[&nested, &flat])
        .unwrap_or(nested.as_str())
        .to_string()
}

/// Read a small I64 device tensor back to the host.
///
/// `layer_multipliers` and the two per-head tables are 3 and 16 elements —
/// they are uploaded like any other weight, and the id hash needs them on the
/// host. Reading them back beats adding a host-side path to `WeightStore` for
/// 280 bytes.
#[cfg(feature = "cuda")]
fn i64_host(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<Vec<u64>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    let n = t.num_elements();
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// A one-element BF16 tensor, read back to the host as f32.
///
/// The FP8 n-gram table's dequant scale is stored this way — `shape: [1]`,
/// dtype BF16 — rather than as a per-row scale file.
#[cfg(feature = "cuda")]
fn bf16_scalar(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == 1,
        "PLE: {name} has {} elements, expected 1 (a per-tensor scale)",
        t.num_elements()
    );
    let mut raw = [0u8; 2];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    let bits = u16::from_le_bytes(raw);
    let v = f32::from_bits((bits as u32) << 16);
    anyhow::ensure!(
        v.is_finite() && v > 0.0,
        "PLE: {name} is {v}, which cannot be a dequant scale"
    );
    Ok(v)
}

/// A resident FP16 vector, read back as raw fp16 bits.
///
/// EXL3 `head_bias` stays FP16 (`keeps_raw_f16`) because the n-gram dequant
/// adds it in the same units as `decode_mul1` * scale.
#[cfg(feature = "cuda")]
fn fp16_host(store: &WeightStore, name: &str, n: usize, gpu: &dyn GpuBackend) -> Result<Vec<u16>> {
    let t = store.get(name).with_context(|| format!("PLE: {name}"))?;
    anyhow::ensure!(
        t.num_elements() == n,
        "PLE: {name} has {} elements, expected {n}",
        t.num_elements()
    );
    anyhow::ensure!(
        matches!(t.dtype, spark_runtime::weights::WeightDtype::FP16),
        "PLE: {name} is {:?}, expected FP16 (raw head_bias)",
        t.dtype
    );
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(t.ptr, &mut raw)
        .with_context(|| format!("PLE: reading {name} back to host"))?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Build the PLE layer for `layer_idx`, or `None` if this model has none.
#[cfg(feature = "cuda")]
pub(super) fn load(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    max_tokens: usize,
    gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    // `ple_layer_ids` is 1-INDEXED — the reference selects with
    // `ple_layer_ids.index(layer_idx + 1)` — so `[2]` means MODEL LAYER 1.
    if !config.ple_layer_ids.contains(&(layer_idx + 1)) {
        return Ok(None);
    }
    let lp = format!("{}.ple", config.layer_prefix(layer_idx));
    let h = config.hidden_size;
    let hc = config.hc_mult;
    let eos = config.eos_token_id;

    let dims = PleIdDims {
        ngram_size: config.emb_neighbor_num,
        heads_per_ngram: config.emb_split_num,
        multipliers: i64_host(
            store,
            &ple_i64_name(store, &lp, "layer_multipliers", "layer_multipliers"),
            gpu,
        )?,
        head_vocab_sizes: i64_host(
            store,
            &ple_i64_name(store, &lp, "head_vocab_sizes", "ngram_heads_vocab_sizes"),
            gpu,
        )?,
        head_offsets: i64_host(
            store,
            &ple_i64_name(store, &lp, "head_offsets", "ngram_heads_offsets"),
            gpu,
        )?,
        eos_token_id: eos,
    };
    dims.validate().context("PLE: checkpoint id geometry")?;
    let heads = dims.ngram_heads();

    // ── the segmented table ──
    // (backing file, byte offset) per shard. The shards are NOT confined to
    // one file: the RadixArk NVFP4 conversion spreads its 128 shards over 10
    // `model-plefp8-*.safetensors`, interleaved (shards 0 and 1 in the first,
    // shard 2 in the fourth), so the cache opens one descriptor per distinct
    // file and resolves a row's file from its shard.
    let mut shards: Vec<(std::path::PathBuf, u64)> = Vec::new();
    let mut rows_per = 0usize;
    let mut head_dim = 0usize;
    let mut dtype = None;
    for i in 0.. {
        // Two naming families: RadixArk NVFP4 ships `shard_{i}.weight`,
        // EXL3 ships `shard_{i}.trellis` (I16). Probe `.weight` first so the
        // NVFP4 path is byte-for-byte unchanged; fall back to `.trellis`.
        let base = format!("{lp}.ple_embedding.ngram_embedding.shard_{i}");
        let d = match store
            .deferred(&format!("{base}.weight"))
            .or_else(|| store.deferred(&format!("{base}.trellis")))
        {
            Some(d) => d,
            None => break,
        };
        anyhow::ensure!(
            d.shape.len() == 2,
            "PLE: shard {i} has shape {:?}, expected 2-D",
            d.shape
        );
        if i == 0 {
            rows_per = d.shape[0];
            head_dim = d.shape[1];
            dtype = Some(d.dtype);
        } else {
            anyhow::ensure!(
                Some(d.dtype) == dtype,
                "PLE: shard {i} is {:?} but shard 0 is {:?}; one table cannot \
                 mix element types",
                d.dtype,
                dtype
            );
            anyhow::ensure!(
                d.shape[0] == rows_per && d.shape[1] == head_dim,
                "PLE: shard {i} is {:?} but shard 0 is [{rows_per}, {head_dim}]. \
                 The segmented row cache maps a global id with one divide, which \
                 requires every shard to hold the same number of rows.",
                d.shape
            );
        }
        shards.push((d.path.clone(), d.offset));
    }
    // 4.05 bpw stores the whole table as one contiguous tensor, not shards.
    // open_at reads it at the safetensors data offset. Sharded tables stay
    // on open_segmented below.
    let single_name = format!("{lp}.ple_embedding.ngram_embedding.trellis");
    let single = if shards.is_empty() {
        store.deferred(&single_name)
    } else {
        None
    };
    if shards.is_empty() && single.is_none() {
        anyhow::bail!(
            "PLE: no `{lp}.ple_embedding.ngram_embedding.shard_*` was deferred, \
             and `{single_name}` is not deferred either. Either the checkpoint \
             has none, or they were UPLOADED whole — which for this table is \
             102 GB of BF16 and would not have fit."
        );
    }
    if let Some(d) = single {
        anyhow::ensure!(
            d.shape.len() == 2,
            "PLE: `{single_name}` shape {:?}, expected 2-D",
            d.shape
        );
        rows_per = d.shape[0];
        head_dim = d.shape[1];
        dtype = Some(d.dtype);
        shards.push((d.path.clone(), d.offset));
    }
    let distinct_files = {
        let mut seen: Vec<&std::path::Path> = Vec::new();
        for (path, _) in &shards {
            if !seen.contains(&path.as_path()) {
                seen.push(path.as_path());
            }
        }
        seen.len()
    };
    // The element type decides the row stride, and getting it wrong is
    // invisible until a read runs off the end of a shard. RadixArk's NVFP4
    // conversion ships this table as F8_E4M3 (1 byte/element, 51.2 GB) with
    // ONE BF16 scalar scale; the announced BF16 form would be 102.4 GB, which
    // does not fit in the 126 GB checkpoint alongside 73 GB of other weights.
    let dtype = dtype.context("PLE: no shard dtype")?;
    let elem = match dtype {
        spark_runtime::weights::WeightDtype::BF16 => 2,
        spark_runtime::weights::WeightDtype::FP8E4M3 => 1,
        // EXL3 trellis shards are packed I16 (2 bytes/element). The row
        // cache copies those bytes; host dequant expands each row to 160
        // BF16 before the GEMM. See `layers/ple/ngram_trellis.rs`.
        spark_runtime::weights::WeightDtype::Int16 => 2,
        other => anyhow::bail!(
            "PLE: n-gram table is {other:?}; the row cache gathers BF16, \
             F8_E4M3, or I16 rows (`batched_embed` / `batched_embed_fp8`)"
        ),
    };
    // The gather pins at most one span's rows at once, so the arena sizes
    // off the effective span — `bounded_scratch` is the same clamp
    // `PleLayer::new` applies, keeping the two derivations from drifting.
    let chunk = chunk_from_env();
    let span = crate::layers::ple::bounded_scratch(chunk, max_tokens);
    let (slots, slots_from) = slots_from_env(span, heads);
    let mut cache = if single.is_some() {
        let (path, off) = &shards[0];
        spark_storage::NgramRowCache::open_at(
            path,
            *off,
            None,
            rows_per as u64,
            head_dim * elem,
            slots,
        )
    } else {
        spark_storage::NgramRowCache::open_segmented(
            &shards,
            rows_per as u64,
            None, // no per-row scale FILE; FP8 uses the per-tensor scalar below
            head_dim * elem,
            slots,
        )
    }
    .context("PLE: n-gram row cache")?;
    if let Some((period, idle)) = spark_storage::ngram_cache::keepalive_from_env() {
        cache
            .start_keepalive(period, idle)
            .context("PLE: NVMe keepalive")?;
        tracing::info!(
            "PLE NVMe keepalive: one 4 KiB read every {period:?} while rows were \
             resolved within {idle:?} (ATLAS_PLE_NVME_KEEPALIVE_MS)"
        );
    }

    // FP8 rows need their dequant scale, or the gather returns raw E4M3
    // magnitudes and the whole n-gram contribution is off by a constant
    // factor — fluent output, wrong logits.
    if elem == 1 {
        let name = format!("{lp}.ple_embedding.ngram_embedding.weight_scale");
        let scale = bf16_scalar(store, &name, gpu)?;
        cache
            .set_constant_scale(scale)
            .context("PLE: FP8 per-tensor scale")?;
        tracing::info!("PLE n-gram table: F8_E4M3, per-tensor scale {scale:.6} from {name}");
    }

    let weights = PleWeights {
        key_proj: dense(store, &format!("{lp}.key_proj.weight"))?,
        value_proj: dense(store, &format!("{lp}.value_proj.weight"))?,
        norm_key: dense(store, &format!("{lp}.norm_key.weight"))?,
        norm_query: dense(store, &format!("{lp}.norm_query.weight"))?,
        norm_conv: dense(store, &format!("{lp}.norm_conv.weight"))?,
        conv1d: dense(store, &format!("{lp}.conv1d.weight"))?,
    };

    let dilation = config.emb_neighbor_num; // conv dilation IS ngram_size
    tracing::info!(
        "PLE at MODEL LAYER {layer_idx} (ple_layer_ids={:?}, 1-indexed): \
         {} shards over {} file(s) x {rows_per} rows x {head_dim} dims = {} rows \
         ({:.1} GB {dtype:?}) \
         served off NVMe with {slots} cached slots ({:.1} MB, {slots_from}: \
         span {span} x {heads} heads, floored at 65536); \
         scratch {span} tokens = {:.2} GB (ATLAS_PLE_CHUNK={chunk}); \
         conv k={} dilation={dilation} (state {} steps)",
        config.ple_layer_ids,
        shards.len(),
        distinct_files,
        shards.len() * rows_per,
        (shards.len() * rows_per * head_dim * elem) as f64 / 1e9,
        (slots * head_dim * 2) as f64 / 1e6,
        (span * 10240 * 14) as f64 / 1e9,
        config.ple_conv_kernel_size,
        (config.ple_conv_kernel_size - 1) * dilation,
    );

    let (embed_dim, trellis) = if matches!(dtype, spark_runtime::weights::WeightDtype::Int16) {
        let k = crate::layers::ple::ngram_trellis::k_from_packed_words(head_dim).with_context(|| {
            format!(
                "PLE: I16 n-gram row width {head_dim} is not 1+10*K (exllamav3 ngram_codec.words_per_row); cannot dequant the trellis"
            )
        })?;
        let decoded = crate::layers::ple::ngram_trellis::ROW_DIM;
        let bias_name = format!("{lp}.ple_embedding.ngram_embedding.head_bias");
        let bias = fp16_host(store, &bias_name, heads * decoded, gpu)?;
        tracing::info!(
            "PLE EXL3 ngram trellis: {head_dim} i16 words/row, K={k}, decode to {decoded}-dim mul1 * scale + head_bias before the GEMM (geometry uses {decoded}, not {head_dim})"
        );
        (
            decoded,
            Some(crate::layers::ple::ngram_trellis::NgramTrellis {
                k,
                packed_words: head_dim,
                heads,
                bias_fp16: bias,
                head_offsets: dims.head_offsets.clone(),
                head_vocab_sizes: dims.head_vocab_sizes.clone(),
            }),
        )
    } else {
        (head_dim, None)
    };

    PleLayer::new(
        dims,
        embed_dim,
        h,
        hc,
        config.ple_conv_kernel_size,
        dilation,
        config.rms_norm_eps as f32,
        weights,
        NgramTable::Cached(Box::new(cache)),
        max_tokens,
        span,
        trellis,
        gpu,
    )
    .map(Some)
    .context("PLE: layer construction")
}

/// Non-CUDA builds have no NVMe row cache — it serves rows out of a pinned,
/// GPU-addressable arena — so a PLE model cannot be served here. REFUSE
/// rather than return `None` (same rationale as `longcat/ngram.rs`): `None`
/// means "this model has no PLE", and quietly answering that for a model
/// that does have one silently drops the n-gram injection.
#[cfg(not(feature = "cuda"))]
pub(super) fn load(
    _store: &WeightStore,
    config: &ModelConfig,
    _layer_idx: usize,
    _max_tokens: usize,
    _gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    if config.ple_layer_ids.is_empty() {
        return Ok(None);
    }
    anyhow::bail!(
        "qwen4_exp PLE: this checkpoint has n-gram embeddings, but the row \
         cache that serves them needs the `cuda` feature; this build cannot \
         serve it"
    )
}

#[cfg(all(test, feature = "cuda"))]
mod slots_tests {
    use super::derived_slots;

    /// The fixed 65536 this replaced assumed a 2048-token chunk; the default
    /// serve config presents 8193, and a prefill pins tokens x heads rows.
    /// Default warm lookahead is 2 spans (`ATLAS_PLE_WARM_AHEAD`), so the
    /// derived count covers span + lookahead.
    #[test]
    fn derived_slots_cover_the_default_chunk() {
        assert_eq!(derived_slots(8193, 16), 524_288); // (8193 + 16386) x 16, rounded up
        assert_eq!(derived_slots(2048, 16), 131_072); // (2048 + 4096) x 16, exact
        assert_eq!(derived_slots(4096, 16), 262_144); // (4096 + 8192) x 16, exact
        assert!(derived_slots(20481, 16) >= 20481 * 16);
    }
}
