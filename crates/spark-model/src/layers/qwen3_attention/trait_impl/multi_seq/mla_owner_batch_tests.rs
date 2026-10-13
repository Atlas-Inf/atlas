// SPDX-License-Identifier: AGPL-3.0-only

// Focused child module for mla_long_context_tests.rs: the semantic-index
// query projections of an owner-batched verify.
use super::*;
use crate::layers::qwen3_attention::prefill::GlmChunkOwner;

/// Two 8-row owners in one verify batch. The index queries and weights are
/// projected once over both owners' rows when one of them selects, and the
/// selector reads that owner's rows of the batch; when both owners are dense
/// nothing projects them and nothing reads where they would have been.
#[test]
fn owner_batch_projects_index_queries_only_for_a_selecting_owner() {
    fixture(|gpu, config, layer| {
        // 32 batch tokens: the scratch holds all 16 rows of every projection.
        let arena = BufferArena::new(config, 32, 32768, 16, 1, gpu).unwrap();
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let metadata = gpu.alloc(16 * 2049 * 4 + 1024).unwrap();
        let meta = AttnMetadataDev {
            positions: metadata,
            positions_h: metadata,
            positions_w: metadata,
            slot: metadata.offset(256),
            seq_len: metadata.offset(512),
            block_table: metadata.offset(768),
            max_blocks_per_seq: 2049,
            num_seqs: 2,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let ctx = ForwardContext {
            buffers: &arena,
            gpu,
            config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: Some(meta),
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            2050,
            gpu,
        )
        .unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();

        let indexer = layer.mla.as_ref().unwrap().glm_indexer.as_ref().unwrap();
        let (rows, latent_row, query_row, weight_row) = (16, 32 * 512 * 2, 32 * 128 * 2, 32 * 2);
        let index_query = arena.ssm_deinterleaved().offset(rows * latent_row);
        let weights = arena.ssm_gates();
        for (second_start, selects) in [(500, false), (2041, true)] {
            let owner = |row0, seq_len_start| GlmChunkOwner {
                row0,
                rows: 8,
                seq_len_start,
                meta,
            };
            let owners = [owner(0, 30), owner(8, second_start)];
            let before = gpu.1.lock().unwrap().len();
            layer
                .prefill_attention_glm_owners(
                    &owners,
                    arena.hidden_states(),
                    rows,
                    &mut cache,
                    &ctx,
                    0,
                )
                .unwrap();
            let calls = gpu.1.lock().unwrap()[before..].to_vec();
            let with = |p: DevicePtr| {
                let p = p.0.to_ne_bytes();
                calls
                    .iter()
                    .filter(|(_, a)| a.contains(&p.to_vec()))
                    .count()
            };
            // One key projection for both owners: the batch path is taken.
            assert_eq!(with(indexer.wk.weight), 1);
            let expected = usize::from(selects);
            assert_eq!(with(indexer.wq_b.weight), expected, "wq_b, {selects}");
            assert_eq!(with(indexer.weights_proj.weight), expected);
            // The projection writes all rows at the batch base; the selector
            // reads the second owner's rows of it.
            assert_eq!(with(index_query), expected);
            assert_eq!(with(index_query.offset(8 * query_row)), expected);
            assert_eq!(with(weights.offset(8 * weight_row)), expected);
        }
    });
}
