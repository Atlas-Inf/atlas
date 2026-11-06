// SPDX-License-Identifier: AGPL-3.0-only

//! Fused chunked GDN prefill on BF16 tensor cores (`gdn_chunk_tc.cu`,
//! lever `ATLAS_GDN_CHUNK_TC=1`): one launch per GDN scan in place of the
//! FLA path's three, for the single-stream, uniform-batch and varlen scans.
//! The handle is looked up only with the lever set (and only the 27B dense
//! targets ship the module), so other boot audits are unchanged.

use spark_runtime::kernel_args::KernelLaunch;

use super::*;

/// Kernel module and entry point.
const MODULE: &str = "gdn_chunk_tc";
const FUNC: &str = "gdn_chunk_tc_prefill";
/// `gdn_chunk_tc::SMEM` in the kernel.
const SMEM: u32 = 99_328;
/// Value columns per CTA (`gdn_chunk_tc::DV`).
const DV: u32 = 64;

/// The handle when the lever is set and the target ships the kernel.
pub(super) fn load(gpu: &dyn GpuBackend) -> KernelHandle {
    let on = std::env::var("ATLAS_GDN_CHUNK_TC").as_deref() == Ok("1");
    super::super::try_kernel_gated(on, gpu, MODULE, FUNC)
}

/// One GDN scan's operands: the packed buffers, `batch` streams of
/// `seq_len` rows each (or `cu_seqlens` when varlen), and the state (a
/// contiguous base or a device table of per-stream bases).
pub(super) struct ChunkTcScan {
    pub h_state: DevicePtr,
    pub h_state_is_table: bool,
    pub batch: u32,
    pub seq_len: u32,
    pub cu_seqlens: Option<DevicePtr>,
}

impl Qwen3SsmLayer {
    /// Run `scan` on the fused kernel. `Ok(false)`: not eligible (lever off,
    /// kernel absent, exact replay, or not 128-dim heads); the caller keeps
    /// its own ladder.
    pub(super) fn try_gdn_chunk_tc(
        &self,
        gdn_bufs: &GdnPrefillBuffers,
        scan: ChunkTcScan,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let c = ctx.config;
        let (nk, kd) = (c.linear_num_key_heads, c.linear_key_head_dim);
        let (nv, vd) = (c.linear_num_value_heads, c.linear_value_head_dim);
        if self.gdn_chunk_tc_k.0 == 0
            || ctx.gdn_exact_replay
            || kd != 128
            || vd != 128
            || nk == 0
            || !nv.is_multiple_of(nk)
        {
            return Ok(false);
        }
        if ctx.stats.once("log:gdn_chunk_tc") {
            tracing::info!("GDN prefill: fused chunked tensor-core scan ACTIVE ({MODULE}::{FUNC})");
        }
        let key_dim = nk * kd;
        let conv_dim = (key_dim * 2 + nv * vd) as u32;
        let (bf16, fp32) = (2usize, 4usize);
        KernelLaunch::new(ctx.gpu, self.gdn_chunk_tc_k)
            .grid([vd as u32 / DV, nv as u32, scan.batch])
            .block([256, 1, 1])
            .shared_mem(SMEM)
            .arg_ptr(scan.h_state)
            .arg_ptr(gdn_bufs.qkv)
            .arg_ptr(gdn_bufs.qkv.offset(key_dim * bf16))
            .arg_ptr(gdn_bufs.qkv.offset(key_dim * 2 * bf16))
            .arg_ptr(gdn_bufs.gate_beta)
            .arg_ptr(gdn_bufs.gate_beta.offset(nv * fp32))
            .arg_ptr(gdn_bufs.output)
            .arg_u32(scan.batch)
            .arg_u32(scan.seq_len)
            .arg_u32(nk as u32)
            .arg_u32(nv as u32)
            .arg_u32(conv_dim)
            .arg_u32(conv_dim)
            .arg_u32((nv * 2) as u32)
            .arg_u32(scan.h_state_is_table as u32)
            .arg_ptr(scan.cu_seqlens.unwrap_or(DevicePtr::NULL))
            .arg_u32(scan.cu_seqlens.is_some() as u32)
            .launch(stream)?;
        Ok(true)
    }
}
