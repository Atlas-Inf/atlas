// SPDX-License-Identifier: AGPL-3.0-only

//! Immutable native predictor prefixes: share sealed blocks, copy the open tail.
use super::*;

fn prefix_blocks(rows: usize, block_size: usize) -> (usize, bool) {
    (rows / block_size, !rows.is_multiple_of(block_size))
}

fn copy_prefix_blocks(
    cache: &mut PagedKvCache,
    source: &[u32],
    rows: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
    out: &mut Vec<u32>,
) -> Result<()> {
    let (sealed, partial) = prefix_blocks(rows, cache.block_size());
    anyhow::ensure!(
        source.len() >= sealed + usize::from(partial),
        "native MTP prefix block table is short"
    );
    for &block in &source[..sealed] {
        cache.inc_ref(block);
        out.push(block);
    }
    if partial {
        let src = source[sealed] as usize;
        let dst = cache.alloc_block()?;
        out.push(dst);
        let copy = |pool: DevicePtr, stride: usize| -> Result<()> {
            if !pool.is_null() && stride > 0 {
                gpu.copy_d2d_async(
                    pool.offset(src * stride),
                    pool.offset(dst as usize * stride),
                    stride,
                    stream,
                )?;
            }
            Ok(())
        };
        copy(cache.k_pool_ptr(0), cache.k_block_stride_bytes_for_layer(0))?;
        if cache.v_pool_ptr(0) != cache.k_pool_ptr(0) {
            copy(cache.v_pool_ptr(0), cache.v_block_stride_bytes_for_layer(0))?;
        }
        anyhow::ensure!(
            cache.sparse_index_scale_pool_ptr(0).is_null(),
            "native prefix clone requires publisher BF16 semantic index"
        );
        copy(
            cache.sparse_index_pool_ptr(0),
            cache.sparse_index_block_stride_bytes(0),
        )?;
        copy(
            cache.sparse_index_tail_pool_ptr(0),
            cache.sparse_index_tail_block_stride_bytes(0),
        )?;
    }
    Ok(())
}

impl Glm5MtpHead {
    pub(super) fn clone_prefix_state_inner(
        &self,
        state: &Glm5MtpProposerState,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Glm5MtpProposerState> {
        anyhow::ensure!(
            rows == state.seq_len
                && state.last_num_drafted == 0
                && state.target_end == rows.checked_add(1).context("native prefix overflow")?,
            "native MTP prefix must end at a canonical target boundary"
        );
        let mut clone = self.alloc_state_inner(gpu)?;
        let result = (|| -> Result<()> {
            let mut cache = self.cache()?;
            copy_prefix_blocks(
                &mut cache,
                &state.block_table,
                rows,
                gpu,
                stream,
                &mut clone.block_table,
            )?;
            gpu.copy_d2d_async(
                state.target_hidden,
                clone.target_hidden,
                self.draft_config.hidden_size * 2,
                stream,
            )?;
            clone.seq_len = rows;
            clone.target_end = state.target_end;
            Ok(())
        })();
        if let Err(error) = result {
            // A failed copy may still have queued earlier work against the new blocks.
            let _ = gpu.synchronize(stream);
            let _ = self.free_state(gpu, None, &mut clone);
            return Err(error);
        }
        Ok(clone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::SparseIndexCacheConfig;

    #[test]
    fn prefix_survives_source_mutation_and_request_release() {
        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut cache = PagedKvCache::new_with_v_alias(config, 4, &gpu, true).unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 8), &gpu)
            .unwrap();
        let source = vec![cache.alloc_block().unwrap(), cache.alloc_block().unwrap()];
        let pools = [
            (cache.k_pool_ptr(0), cache.k_block_stride_bytes_for_layer(0)),
            (
                cache.sparse_index_pool_ptr(0),
                cache.sparse_index_block_stride_bytes(0),
            ),
            (
                cache.sparse_index_tail_pool_ptr(0),
                cache.sparse_index_tail_block_stride_bytes(0),
            ),
        ];
        for (pool, stride) in pools {
            gpu.copy_h2d(
                &vec![0x5a; stride],
                pool.offset(source[1] as usize * stride),
            )
            .unwrap();
        }
        let mut cloned = Vec::new();
        copy_prefix_blocks(&mut cache, &source, 31, &gpu, 0, &mut cloned).unwrap();
        assert_eq!(cloned[0], source[0]);
        assert_ne!(cloned[1], source[1]);
        assert_eq!(cache.ref_count(source[0]), 2);
        for (pool, stride) in pools {
            gpu.copy_h2d(
                &vec![0xe7; stride],
                pool.offset(source[1] as usize * stride),
            )
            .unwrap();
            let mut bytes = vec![0; stride];
            gpu.copy_d2h(pool.offset(cloned[1] as usize * stride), &mut bytes)
                .unwrap();
            assert_eq!(
                bytes,
                vec![0x5a; stride],
                "snapshot tail must not alias writer"
            );
        }
        cache.free_blocks(&source);
        assert_eq!(cache.ref_count(cloned[0]), 1);
        assert_eq!(cache.ref_count(cloned[1]), 1);
        cache.free_blocks(&cloned);
        assert_eq!(
            cache.num_free_blocks(),
            4,
            "eviction releases exactly snapshot refs"
        );
    }

    #[test]
    fn sealed_blocks_are_shared_and_only_an_open_tail_is_copied() {
        assert_eq!(prefix_blocks(0, 16), (0, false));
        assert_eq!(prefix_blocks(15, 16), (0, true));
        assert_eq!(prefix_blocks(16, 16), (1, false));
        assert_eq!(prefix_blocks(17, 16), (1, true));
        assert_eq!(prefix_blocks(1_048_575, 16), (65_535, true));
    }
}
