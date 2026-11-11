// SPDX-License-Identifier: AGPL-3.0-only
//
// `PagedKvCache` debug probes (env-gated KV fingerprints). Split from
// `paged_impl.rs` (500-LoC cap).

use super::PagedKvCache;

impl PagedKvCache {
    /// DEBUG: decode a BF16 KV block buffer into (sum, ssq, sabs) reductions.
    /// Each element is 2 bytes (BF16): top 16 bits of an f32. Used by
    /// `debug_kv_checksum` to fingerprint K/V without cancellation hiding a
    /// localized per-element divergence.
    fn bf16_reductions(buf: &[u8]) -> (f64, f64, f64) {
        let (mut sum, mut ssq, mut sabs) = (0f64, 0f64, 0f64);
        for c in buf.chunks_exact(2) {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let v = f32::from_bits((bits as u32) << 16) as f64;
            sum += v;
            ssq += v * v;
            sabs += v.abs();
        }
        (sum, ssq, sabs)
    }

    /// DEBUG (env-gated): PER-LAYER K and V fingerprint over `blocks`, emitting
    /// (sum, ssq, sabs) for each attention layer so a localized divergence
    /// can't cancel in a global sum. Splits the block list at `boundary_idx`:
    /// blocks `[0, boundary_idx)` are the REUSED-PREFIX region (carried over
    /// from a prior turn's prefill) and `[boundary_idx, end)` are the
    /// RECOMPUTED-SUFFIX region. Each region gets its own per-layer line so we
    /// can localize the FIRST layer/region where chained (ON) differs from cold
    /// (OFF). Only valid for BF16 KV (the experiment uses `--kv-cache-dtype
    /// bf16`); non-BF16 layers are skipped with a one-shot warning.
    pub fn debug_kv_checksum_per_layer(
        &self,
        blocks: &[u32],
        boundary_idx: usize,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        if self.latent_shard.is_some() {
            tracing::warn!("ATLAS_KV debug probe [{tag}] skipped: latents are sharded");
            return;
        }
        gpu.synchronize(stream).ok();
        let boundary = boundary_idx.min(blocks.len());
        let regions: [(&str, &[u32]); 2] = [
            ("prefix", &blocks[..boundary]),
            ("suffix", &blocks[boundary..]),
        ];
        for (li, layer) in self.layers.iter().enumerate() {
            if layer.dtype != super::KvCacheDtype::Bf16 {
                if li == 0 {
                    tracing::warn!(
                        "ATLAS_KV_CKSUM[{tag}] layer 0 dtype={:?} != bf16 — probe \
                         only decodes BF16; skipping",
                        layer.dtype
                    );
                }
                continue;
            }
            // BF16-only probe: K and V strides are equal for symmetric dtypes.
            let nbytes = layer.k_block_stride;
            for (rname, rblocks) in &regions {
                let (mut k_sum, mut k_ssq, mut k_sabs) = (0f64, 0f64, 0f64);
                let (mut v_sum, mut v_ssq, mut v_sabs) = (0f64, 0f64, 0f64);
                for &blk in *rblocks {
                    let mut kb = vec![0u8; nbytes];
                    let mut vb = vec![0u8; nbytes];
                    if gpu.copy_d2h(self.k_cache_ptr(li, blk), &mut kb).is_err()
                        || gpu.copy_d2h(self.v_cache_ptr(li, blk), &mut vb).is_err()
                    {
                        continue;
                    }
                    let (ks, kq, ka) = Self::bf16_reductions(&kb);
                    let (vs, vq, va) = Self::bf16_reductions(&vb);
                    k_sum += ks;
                    k_ssq += kq;
                    k_sabs += ka;
                    v_sum += vs;
                    v_ssq += vq;
                    v_sabs += va;
                }
                tracing::warn!(
                    "ATLAS_KV_CKSUM[{tag}] L{li} {rname} nblk={} \
                     k_sum={k_sum:.4} k_ssq={k_ssq:.4} k_sabs={k_sabs:.4} \
                     v_sum={v_sum:.4} v_ssq={v_ssq:.4} v_sabs={v_sabs:.4}",
                    rblocks.len(),
                );
            }
        }
    }

    /// DEBUG (env-gated): per-LOGICAL-BLOCK K/V fingerprint for ONE layer,
    /// walking `blocks` in block_table order. Emits (logical_idx,
    /// physical_block, k_ssq, v_ssq) per block so a per-position aliasing /
    /// reordering bug (identical region SUM but wrong block→position mapping)
    /// is visible. BF16 only.
    pub fn debug_kv_per_block(
        &self,
        layer_idx: usize,
        blocks: &[u32],
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
        tag: &str,
    ) {
        if self.latent_shard.is_some() {
            tracing::warn!("ATLAS_KV debug probe [{tag}] skipped: latents are sharded");
            return;
        }
        gpu.synchronize(stream).ok();
        let layer = &self.layers[layer_idx];
        if layer.dtype != super::KvCacheDtype::Bf16 {
            return;
        }
        // BF16-only probe: K and V strides are equal for symmetric dtypes.
        let nbytes = layer.k_block_stride;
        for (li, &blk) in blocks.iter().enumerate() {
            let mut kb = vec![0u8; nbytes];
            let mut vb = vec![0u8; nbytes];
            if gpu
                .copy_d2h(self.k_cache_ptr(layer_idx, blk), &mut kb)
                .is_err()
                || gpu
                    .copy_d2h(self.v_cache_ptr(layer_idx, blk), &mut vb)
                    .is_err()
            {
                continue;
            }
            let (_, k_ssq, _) = Self::bf16_reductions(&kb);
            let (_, v_ssq, _) = Self::bf16_reductions(&vb);
            tracing::warn!(
                "ATLAS_KVBLK[{tag}] L{layer_idx} logical={li} phys={blk} \
                 k_ssq={k_ssq:.4} v_ssq={v_ssq:.4}"
            );
        }
    }
}
