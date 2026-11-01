// SPDX-License-Identifier: AGPL-3.0-only
//
//! `ATLAS_PLE_RESIDENT=1`: the whole NVFP4 U8 n-gram table on device — no row
//! cache, no keepalive/fault-pool threads, no host-side row->slot resolve.
//! Refuses (PCND) on any table that is not the NVFP4 U8 pack, on a single
//! `trellis` table, and on a missing scale sidecar.

use super::*;

/// Resident load: fill `rows`/`scales` device buffers from the segmented
/// shard lists, run the boot self-check, and build the layer.
#[allow(clippy::too_many_arguments)]
pub(super) fn load(
    store: &WeightStore,
    config: &ModelConfig,
    layer_idx: usize,
    max_tokens: usize,
    lp: &str,
    dims: &PleIdDims,
    shards: Vec<(std::path::PathBuf, u64)>,
    rows_per: usize,
    dtype: spark_runtime::weights::WeightDtype,
    single: bool,
    nvfp4: Option<crate::layers::ple::ngram_nvfp4::NgramNvfp4>,
    gpu: &dyn GpuBackend,
) -> Result<Option<PleLayer>> {
    anyhow::ensure!(
        dtype == spark_runtime::weights::WeightDtype::UInt8 && !single,
        "PLE resident: ATLAS_PLE_RESIDENT=1 requires the NVFP4 U8 n-gram \
         pack (shard_{{i}}.weight + .weight_scale); this table is {dtype:?}"
    );
    let nv = nvfp4
        .as_ref()
        .context("PLE resident: NVFP4 table without its block scales")?;
    for env in [
        "ATLAS_PLE_CACHE_GB",
        "ATLAS_PLE_CACHE_SLOTS",
        "ATLAS_PLE_NVME_KEEPALIVE_MS",
        "ATLAS_PLE_PACKED_DIR",
    ] {
        if std::env::var_os(env).is_some() {
            tracing::warn!("PLE resident: {env} is set and ignored (no row cache)");
        }
    }
    let n_shards = shards.len();
    let rows_total = n_shards as u64 * rows_per as u64;
    let (row_bytes, scale_bytes) = (nv.geom.packed_stride(), nv.geom.scale_stride());
    let t0 = std::time::Instant::now();
    let rows = gpu.alloc(rows_total as usize * row_bytes)?;
    let scales = gpu.alloc(rows_total as usize * scale_bytes)?;
    spark_storage::ngram_cache::resident::load_segments_to_device(
        &shards,
        (rows_per * row_bytes) as u64,
        rows,
        gpu,
        gpu.default_stream(),
    )
    .context("PLE resident: weight table")?;
    spark_storage::ngram_cache::resident::load_segments_to_device(
        &nv.scale_shards,
        (rows_per * scale_bytes) as u64,
        scales,
        gpu,
        gpu.default_stream(),
    )
    .context("PLE resident: block-scale table")?;
    let load_s = t0.elapsed().as_secs_f64();
    self_check(
        &shards,
        &nv.scale_shards,
        rows,
        scales,
        &nv.geom,
        rows_total,
        rows_per as u64,
        dims,
        gpu,
    )
    .context("PLE resident: boot self-check")?;
    tracing::info!(
        "PLE resident: {rows_total} rows, {:.1} GB on device, loaded in {load_s:.1}s, \
         self-check PASS",
        (rows_total as usize * (row_bytes + scale_bytes)) as f64 / 1e9,
    );
    let weights = ple_weights(store, lp, gpu)?;
    let dilation = config.emb_neighbor_num;
    PleLayer::new(
        dims.clone(),
        nv.geom.dim,
        config.hidden_size,
        config.hc_mult,
        config.ple_conv_kernel_size,
        dilation,
        config.rms_norm_eps as f32,
        weights,
        NgramTable::ResidentNvfp4 {
            rows,
            scales,
            row_base: 0,
            rows_total,
        },
        max_tokens,
        crate::layers::ple::bounded_scratch(chunk_from_env(), max_tokens),
        None,
        nvfp4,
        gpu,
    )
    .map(Some)
    .with_context(|| format!("PLE: layer construction at {layer_idx}"))
}

/// Boot self-check for `ATLAS_PLE_RESIDENT` — fail closed. Gathers row 0,
/// the last row, each head's first and last row, and 64 deterministic
/// pseudo-random rows through `batched_embed_nvfp4` on the resident buffers
/// (slot = row id) and compares bitwise against the host NVFP4 oracle
/// decoding the same bytes read straight from the backing files.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn self_check(
    shards: &[(std::path::PathBuf, u64)],
    scale_shards: &[(std::path::PathBuf, u64)],
    rows: spark_runtime::gpu::DevicePtr,
    scales: spark_runtime::gpu::DevicePtr,
    geom: &crate::layers::ple::ngram_nvfp4::Nvfp4RowGeom,
    rows_total: u64,
    rows_per: u64,
    dims: &PleIdDims,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let mut check: Vec<u64> = vec![0, rows_total - 1];
    for h in 0..dims.head_offsets.len() {
        check.push(dims.head_offsets[h]);
        check.push(dims.head_offsets[h] + dims.head_vocab_sizes[h] - 1);
    }
    // xorshift64* over the row space — deterministic across ranks/runs.
    let mut s = 0x9e3779b97f4a7c15u64;
    for _ in 0..64 {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        check.push((s.wrapping_mul(0x2545f4914f6cdd1d) >> 11) % rows_total);
    }
    check.sort_unstable();
    check.dedup();

    let stream = gpu.default_stream();
    let ids_dev = gpu.alloc(check.len() * 4)?;
    let emb = gpu.alloc(check.len() * geom.dim * 2)?;
    let id_bytes: Vec<u8> = check
        .iter()
        .flat_map(|r| (*r as u32).to_le_bytes())
        .collect();
    gpu.copy_h2d_async(&id_bytes, ids_dev, stream)?;
    let k = gpu.kernel("embed_from_argmax", "batched_embed_nvfp4")?;
    crate::layers::ops::batched_embed_nvfp4(
        gpu,
        k,
        ids_dev,
        rows,
        scales,
        geom.global,
        emb,
        check.len() as u32,
        geom.dim as u32,
        geom.group as u32,
        geom.packed_stride() as u32,
        geom.scale_stride() as u32,
        stream,
    )?;
    let mut got = vec![0u8; check.len() * geom.dim * 2];
    gpu.copy_d2h(emb, &mut got)?;

    let (pw, sw) = (geom.packed_stride(), geom.scale_stride());
    for (i, &r) in check.iter().enumerate() {
        let (si, ri) = ((r / rows_per) as usize, r % rows_per);
        let packed = spark_storage::ngram_cache::resident::read_shard_bytes(
            &shards[si].0,
            shards[si].1 + ri * pw as u64,
            pw,
        )?;
        let scale = spark_storage::ngram_cache::resident::read_shard_bytes(
            &scale_shards[si].0,
            scale_shards[si].1 + ri * sw as u64,
            sw,
        )?;
        let want = crate::layers::ple::ngram_nvfp4::decode_rows_bf16(geom, &[packed], &[scale])?;
        let off = i * geom.dim * 2;
        anyhow::ensure!(
            got[off..off + geom.dim * 2] == want[..],
            "resident self-check: row {r} mismatch (shard {si}, row {ri})"
        );
    }
    Ok(())
}
