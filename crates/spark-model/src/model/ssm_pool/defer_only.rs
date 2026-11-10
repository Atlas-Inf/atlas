// SPDX-License-Identifier: AGPL-3.0-only

//! Verify pools for a model whose every GDN verify defers its commit.
//!
//! Under the exact verify with `ATLAS_QWEN38_VERIFY_DEFER=1` the verify of a
//! chain (`qwen35_gdn_verify_tree_defer`) writes no state at all: H0 and the
//! conv windows stay in the slot's live state, each row's raw conv inputs and
//! gate/beta go to the slot's deferred-commit staging (16 rows), and after the
//! verdict `qwen35_gdn_commit_rows` replays the accepted rows from H0. Nothing
//! reads a per-row H or conv snapshot, and a full reject needs no checkpoint
//! because the live state never moved. Those buffers are the bulk of the
//! verify pools: per slot (γ-1) H snapshots, γ conv snapshots and one
//! checkpoint of each, 1.2 GB a slot at γ=8 and 2.4 GB at γ=16 on the 27B.
//!
//! When every condition the deferred arm needs holds at boot, the pool keeps
//! only the staging. A verify that would still take a storing arm finds no
//! snapshot slots and fails loudly (the arms check the slot vectors before
//! writing), so a mismatch here can never become silent state corruption.
//! `ATLAS_VERIFY_POOL_STORE=1` (presence) allocates the full pools again.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::layers::ops::qwen38_decode_fuse as q38;

/// Whether `ATLAS_VERIFY_POOL_STORE` asks for the storing pools.
fn store_requested() -> bool {
    std::env::var_os("ATLAS_VERIFY_POOL_STORE").is_some()
}

/// The deferred exact GDN verify serves every verify this pool will see:
/// `rows` (the verify-width ceiling) fits its staging, the levers and kernels
/// are on, the geometry is the one its kernels are built for, and no exact
/// tree verify (whose branch parents keep an H snapshot) is requested.
pub(crate) fn applies(
    config: &ModelConfig,
    rows: usize,
    rollback: crate::ssm_reserve::SsmRollbackMode,
    h_f16_pool: bool,
    gpu: &dyn GpuBackend,
) -> bool {
    if store_requested() || rows == 0 {
        return false;
    }
    let (verify, commit) = if q38::defer_on() {
        q38::defer_kernels(gpu)
    } else {
        return false;
    };
    geometry_fits(config)
        && config.model_type == "qwen3_5"
        && crate::model::exact_verify::requested()
        && rows <= q38::DEFER_ROWS_MAX
        && rollback == crate::ssm_reserve::SsmRollbackMode::Snapshot
        && !h_f16_pool
        && !crate::layers::qwen3_ssm::ssm_h_fp16_enabled()
        && !crate::layers::qwen3_ssm::gdn_fused_norm_enabled()
        && crate::speculative::draft_tree::rows_lever() == 0
        && verify.0 != 0
        && commit.0 != 0
}

/// The GDN shape `q38_defer_arm` accepts.
fn geometry_fits(c: &ModelConfig) -> bool {
    c.linear_key_head_dim == 128
        && c.linear_value_head_dim == 128
        && c.linear_conv_kernel_dim == 4
        && c.linear_num_key_heads != 0
        && c.linear_num_value_heads == 3 * c.linear_num_key_heads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_27b_gdn_shape_fits_and_others_do_not() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.linear_key_head_dim = 128;
        c.linear_value_head_dim = 128;
        c.linear_conv_kernel_dim = 4;
        c.linear_num_key_heads = 16;
        c.linear_num_value_heads = 48;
        assert!(geometry_fits(&c));
        c.linear_num_value_heads = 32;
        assert!(!geometry_fits(&c));
    }

    #[test]
    fn off_without_the_levers() {
        // The test process sets neither ATLAS_EXACT_VERIFY nor the q38 levers.
        let c = ModelConfig::qwen3_next_80b_nvfp4();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        assert!(!applies(
            &c,
            8,
            crate::ssm_reserve::SsmRollbackMode::Snapshot,
            false,
            &gpu
        ));
    }
}
