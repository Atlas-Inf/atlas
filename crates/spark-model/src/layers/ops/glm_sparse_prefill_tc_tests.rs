// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}
fn args(config: &ModelConfig) -> GlmSparsePrefillTc<'_> {
    GlmSparsePrefillTc {
        config,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr(16),
        k_cache: DevicePtr(32),
        v_cache: DevicePtr(48),
        indices: DevicePtr(64),
        output: DevicePtr(80),
        block_table: DevicePtr(96),
        rows: 1024,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    }
}

#[test]
fn actual_tc_dispatch_uses_one_query_token_per_cta_and_is_opt_in() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    assert!(!dispatch(&gpu, &a, 7, false, false, false).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), 0);
    a.rows = 1;
    assert!(!dispatch(&gpu, &a, 7, true, false, false).unwrap());
    assert_eq!(gpu.launches_snapshot().len(), 0);
    a.rows = 1024;
    assert!(dispatch(&gpu, &a, 7, true, false, false).unwrap());
    let launch = &gpu.launches_snapshot()[0];
    assert_eq!(launch.grid, [1, 1024, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
}

#[test]
fn enabled_unsupported_shapes_fail_before_launch() {
    let gpu = MockGpuBackend::new();
    let c = config();
    for variant in 0..7 {
        let mut a = args(&c);
        match variant {
            0 => a.heads = 64,
            1 => a.index_width = 2048,
            2 => a.rows = 0,
            3 => a.rows = 65536,
            4 => a.head_dim = 256,
            5 => a.scale = 0.125,
            _ => a.query = DevicePtr::NULL,
        }
        assert!(dispatch(&gpu, &a, 0, true, false, false).is_err());
        assert!(dispatch(&gpu, &a, 0, true, true, false).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
    for name in [
        "ATLAS_GLM_SPARSE_PREFILL_TC",
        "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE",
    ] {
        assert!(parse("glm5_next", name, Some("1")).unwrap());
        assert!(!parse("qwen3_next", name, Some("1")).unwrap());
        for value in ["", "true", "2"] {
            assert!(parse("glm5_next", name, Some(value)).is_err());
        }
    }
}

#[test]
fn kv_reuse_dispatch_requires_tc_and_identical_latent_writer() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    assert!(dispatch(&gpu, &a, 7, false, true, false).is_err());
    a.identical_kv_latent = false;
    assert!(dispatch(&gpu, &a, 7, true, true, false).is_err());
    assert!(gpu.launches_snapshot().is_empty());
    a.identical_kv_latent = true;
    a.rows = 1;
    assert!(!dispatch(&gpu, &a, 7, true, true, false).unwrap());
    a.rows = 2048;
    assert!(dispatch(&gpu, &a, 7, true, true, false).unwrap());
    let launch = &gpu.launches_snapshot()[0];
    assert_eq!(launch.grid, [1, 2048, 1]);
    assert_eq!(launch.block, [256, 1, 1]);
    assert_eq!(
        kernel_spec(true, KvCacheDtype::Bf16, false),
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            69376,
        )
    );
    assert_eq!(
        kernel_spec(false, KvCacheDtype::Bf16, false),
        (
            "glm_sparse_prefill_tc",
            "glm_sparse_mla_prefill_bf16_head32_tc",
            101120,
        )
    );
}

#[test]
fn fp8_g128_takes_single_rows_through_the_fp8_kv_pad_kernel_only() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    a.dtype = KvCacheDtype::Fp8G128;
    a.rows = 1;
    // Without the K=V kernel there is no fp8_g128 reader.
    assert!(dispatch(&gpu, &a, 7, true, false, false).is_err());
    assert!(gpu.launches_snapshot().is_empty());
    assert!(dispatch(&gpu, &a, 7, true, true, false).unwrap());
    assert_eq!(gpu.launches_snapshot()[0].grid, [1, 1, 1]);
    assert_eq!(
        kernel_spec(true, KvCacheDtype::Fp8G128, false),
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad",
            69376,
        )
    );
    a.dtype = KvCacheDtype::Fp8;
    a.rows = 8;
    assert!(dispatch(&gpu, &a, 7, true, true, false).is_err());
}

#[test]
fn pipe_swaps_only_the_unsplit_fp8_kernel_and_requires_tc() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let mut a = args(&c);
    a.dtype = KvCacheDtype::Fp8G128;
    a.rows = 4096;
    assert!(dispatch(&gpu, &a, 7, false, false, true).is_err());
    assert!(dispatch(&gpu, &a, 7, true, true, true).unwrap());
    assert_eq!(gpu.launches_snapshot()[0].grid, [1, 4096, 1]);
    assert_eq!(gpu.launches_snapshot()[0].block, [256, 1, 1]);
    assert_eq!(
        kernel_spec(true, KvCacheDtype::Fp8G128, true),
        (
            "glm_sparse_prefill_kv_reuse",
            "glm_sparse_mla_prefill_fp8g128_head32_tc_pipe",
            80128,
        )
    );
    // BF16 caches keep their kernels; the flag off keeps kv_pad.
    for kv_reuse in [false, true] {
        assert_eq!(
            kernel_spec(kv_reuse, KvCacheDtype::Bf16, true),
            kernel_spec(kv_reuse, KvCacheDtype::Bf16, false)
        );
    }
    assert_eq!(
        kernel_spec(true, KvCacheDtype::Fp8G128, false).1,
        "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad"
    );
    assert!(parse("glm5_next", PIPE, Some("1")).unwrap());
    assert!(!parse("glm5_next", PIPE, None).unwrap());
    assert!(parse("glm5_next", PIPE, Some("yes")).is_err());
}

#[test]
fn pipe_keeps_the_bf16_view_only_for_pieces_native_admits() {
    let never = || -> Result<bool> { panic!("native admission queried") };
    // Flag off (base) and short owners never ask native: same as `rows >= 2048`.
    for rows in [1, 2047, 2048, 4096] {
        assert_eq!(needs_view(rows, false, never).unwrap(), rows >= 2048);
        assert!(!needs_view(rows.min(2047), true, never).unwrap());
    }
    // Under the pipe the view survives only for native's own pieces.
    assert!(needs_view(4096, true, || Ok(true)).unwrap());
    assert!(!needs_view(4096, true, || Ok(false)).unwrap());
    assert!(needs_view(4096, true, || anyhow::bail!("x")).is_err());
    // Other models never read the flag: base behavior.
    assert!(glm_sparse_owner_needs_view("qwen3_next", 4096, never).unwrap());
}

#[test]
fn pipe_warm_launch_is_one_zero_row_cta() {
    let gpu = MockGpuBackend::new();
    warm(&gpu, &config(), KvCacheDtype::Fp8G128, true).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].grid, [1, 1, 1]);
    assert_eq!(launches[0].block, [256, 1, 1]);
    // Non-GLM models never read the flag's kernels.
    initialize_glm_sparse_prefill_pipe(&gpu, &ModelConfig::qwen3_next_80b_nvfp4()).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 1);
}

#[test]
fn the_pinned_split_count_does_not_move_with_the_verify_width() {
    // Unpinned, the count follows the launch's rows: one position gets
    // different softmax partials at each DFlash verify width.
    let launch = |rows| sparse_owner_splits(rows, 32, 2051, false);
    assert_eq!(
        (1..=8).map(launch).collect::<Vec<_>>(),
        [13, 13, 13, 11, 9, 8, 13, 6]
    );
    for rows in 1..=128 {
        assert_eq!(launch(rows), sparse_split_count(rows, 32, 2051));
    }
    // Pinned, every verify-sized owner takes the widest block's count, which
    // fills the 48 SMs in one wave there and keeps that block's bits.
    let pinned = |rows| sparse_owner_splits(rows, 32, 2051, true);
    for rows in 1..=8 {
        assert_eq!(pinned(rows), launch(8));
        // The scratch a pinned owner needs never exceeds the widest block's.
        assert!(
            sparse_split_scratch_bytes(pinned(rows), rows, 32, 512)
                <= sparse_split_scratch_bytes(launch(8), 8, 32, 512)
        );
    }
    // Wider owners (prefill pieces) keep the count they have today.
    for rows in 9..=128 {
        assert_eq!(pinned(rows), launch(rows));
    }
    assert!(parse("glm5_next", SPLIT_PIN, Some("1")).unwrap());
    assert!(!parse("glm5_next", SPLIT_PIN, None).unwrap());
    assert!(parse("glm5_next", SPLIT_PIN, Some("yes")).is_err());
}

#[test]
fn the_pin_sets_the_split_grid_of_verify_owners_and_nothing_else() {
    let c = config();
    let scratch = DevicePtr(1 << 20);
    let bytes = sparse_split_scratch_bytes(16, 8, 32, 512);
    let launch = |rows: u32, scratch: DevicePtr, flags: SplitFlags| {
        let gpu = MockGpuBackend::new();
        let mut a = args(&c);
        a.rows = rows;
        assert!(dispatch_split(&gpu, &a, scratch, bytes, 7, flags).unwrap());
        gpu.launches_snapshot()[0].grid
    };
    let off = SplitFlags {
        on: true,
        tc: true,
        kv_reuse: true,
        pipe: false,
        pin: false,
    };
    let pin = SplitFlags { pin: true, ..off };
    for rows in 1..=8 {
        // Off: the launch's own count, as before the flag existed.
        let own = sparse_split_count(rows, 32, 2051);
        assert_eq!(launch(rows, scratch, off), [1, rows, own]);
        // On: the widest verify block's count at every width.
        assert_eq!(launch(rows, scratch, pin), [1, rows, 6]);
    }
    // A launch that does not split stays unsplit under the pin: the split
    // switched off, no K=V kernel, or no scratch.
    let unsplit = [
        (scratch, SplitFlags { on: false, ..pin }),
        (
            scratch,
            SplitFlags {
                kv_reuse: false,
                ..pin
            },
        ),
        (DevicePtr::NULL, pin),
    ];
    for (scratch, flags) in unsplit {
        assert_eq!(launch(3, scratch, flags), [1, 3, 1]);
    }
}
