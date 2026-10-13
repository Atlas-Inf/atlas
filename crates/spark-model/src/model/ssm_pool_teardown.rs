// SPDX-License-Identifier: AGPL-3.0-only

//! Teardown for [`SsmStatePool`].
//!
//! Split from `ssm_pool.rs`, which is far past the 500-line cap.

use spark_runtime::gpu::GpuBackend;

use super::ssm_pool::SsmStatePool;

/// Release every allocation behind the state pools.
///
/// The pool vectors hold per-layer offset VIEWS into one block whenever the
/// contiguous path in `alloc_layer_pools` ran, which is the normal case. Freeing
/// them one by one (what this did until #122) succeeded for layer 0 only; every
/// other view came back `CUDA_ERROR_INVALID_VALUE`, went back onto the backend
/// ledger, and failed a second time in the sweep — the `ssm state pool:
/// cuMemFree_v2 failed: status 1` error on every model's shutdown. Same shape
/// as `SsmSnapshotPool`'s teardown: free the bases, drop the views.
impl atlas_core::scope::ModelResource<dyn GpuBackend> for SsmStatePool {
    fn label(&self) -> &'static str {
        "ssm state pool"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        // Views first: nothing may keep a pointer into memory that is gone.
        for views in [
            &mut self.h_state_pools,
            &mut self.conv_state_pools,
            &mut self.h_intermediate_pools,
            &mut self.conv_intermediate_pools,
            &mut self.h_checkpoint_pools,
            &mut self.conv_checkpoint_pools,
            &mut self.replay_input_rings,
            &mut self.gdn_commit_qkv_pools,
            &mut self.gdn_commit_gb_pools,
        ] {
            views.clear();
        }
        self.h_prefill_stage_pool = None;
        let mut first_error = None;
        for ptr in self.allocations.drain(..) {
            if let Err(e) = gpu.free(ptr)
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use atlas_core::config::ModelConfig;
    use atlas_core::scope::ModelResource;
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::SsmStatePool;
    use crate::ssm_reserve::SsmRollbackMode;

    /// The 80B hybrid layer pattern at doll-house blob widths (same shape as
    /// `ssm_batched_copy_tests::tiny_config`).
    fn config() -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.linear_num_key_heads = 2;
        c.linear_key_head_dim = 4;
        c.linear_num_value_heads = 2;
        c.linear_value_head_dim = 4;
        c.linear_conv_kernel_dim = 4;
        c
    }

    fn pool(gpu: &MockGpuBackend, h_f16: bool, deferred_commit: bool) -> SsmStatePool {
        SsmStatePool::new(
            &config(),
            4,
            true,
            4,
            3,
            h_f16,
            SsmRollbackMode::Snapshot,
            deferred_commit,
            gpu,
        )
        .unwrap()
    }

    /// Every allocation comes back, and no free targets a view: the real
    /// backend rejects a view free with status 1 and re-ledgers it.
    #[test]
    fn release_frees_each_allocation_once_and_no_views() {
        for (h_f16, deferred) in [(false, false), (true, false), (false, true), (true, true)] {
            let gpu = MockGpuBackend::new();
            let mut p = pool(&gpu, h_f16, deferred);
            assert!(p.num_ssm_layers > 1, "needs >1 layer to have views");
            assert!(gpu.alloc_count() > 0);
            p.release(&gpu).unwrap();
            assert_eq!(
                gpu.invalid_free_count(),
                0,
                "h_f16={h_f16} deferred={deferred}"
            );
            assert_eq!(gpu.alloc_count(), 0, "h_f16={h_f16} deferred={deferred}");
            // Idempotent: a second release frees nothing.
            p.release(&gpu).unwrap();
            assert_eq!(gpu.invalid_free_count(), 0);
        }
    }

    /// Guard on the mock itself: freeing a view IS counted, so the test above
    /// would have failed on the per-view release it replaces.
    #[test]
    fn mock_counts_a_view_free() {
        let gpu = MockGpuBackend::new();
        let p = pool(&gpu, false, false);
        gpu.free(p.h_state_pools[1]).unwrap();
        assert_eq!(gpu.invalid_free_count(), 1);
    }
}
