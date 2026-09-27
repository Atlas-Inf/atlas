// SPDX-License-Identifier: AGPL-3.0-only

//! In-place UNIFIED MoE transpose: the transposed [K/2,N] expert layout has
//! exactly the same byte size as the original [N,K/2] one, so instead of
//! allocating transposed slabs and freeing the originals per layer, the
//! transpose kernel writes into a per-call scratch and the result is copied
//! back into the original device allocations. No per-layer alloc/free churn.

use super::*;

/// Grow-only pair of device buffers used as the transpose destination.
/// Allocated at most once per needed size for the whole pass.
pub(super) struct TransposeScratch {
    packed: DevicePtr,
    packed_cap: usize,
    scale: DevicePtr,
    scale_cap: usize,
}

impl TransposeScratch {
    pub(super) fn new() -> Self {
        Self {
            packed: DevicePtr::NULL,
            packed_cap: 0,
            scale: DevicePtr::NULL,
            scale_cap: 0,
        }
    }

    fn ensure(&mut self, gpu: &dyn GpuBackend, packed: usize, scale: usize) -> Result<()> {
        if packed > self.packed_cap {
            if !self.packed.is_null() {
                gpu.free(self.packed)?;
            }
            self.packed = gpu.alloc(packed)?;
            self.packed_cap = packed;
        }
        if scale > self.scale_cap {
            if !self.scale.is_null() {
                gpu.free(self.scale)?;
            }
            self.scale = gpu.alloc(scale)?;
            self.scale_cap = scale;
        }
        Ok(())
    }

    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !self.packed.is_null() {
            gpu.free(self.packed)?;
            self.packed = DevicePtr::NULL;
            self.packed_cap = 0;
        }
        if !self.scale.is_null() {
            gpu.free(self.scale)?;
            self.scale = DevicePtr::NULL;
            self.scale_cap = 0;
        }
        Ok(())
    }
}

impl MoeLayer {
    /// Same two `moe_transpose_u8_batched` launches as `transpose_experts_gpu`,
    /// but the destination is `scratch`, and the transposed bytes are then
    /// copied back into each expert's original `weight`/`weight_scale`
    /// allocations. Returns QuantizedWeights carrying the ORIGINAL pointers
    /// (null stays null), so the caller's pointer tables address the
    /// transposed data in place.
    pub(super) fn transpose_experts_inplace(
        &self,
        gpu: &dyn GpuBackend,
        src: &[QuantizedWeight],
        n: usize,
        k: usize,
        group_size: usize,
        scratch: &mut TransposeScratch,
    ) -> Result<Vec<QuantizedWeight>> {
        let num_experts = src.len();
        let packed_each = n * (k / 2);
        let scale_each = n * (k / group_size);
        anyhow::ensure!(
            packed_each > 0 && scale_each > 0,
            "transpose_experts_inplace: zero-sized projection (n={n} k={k} gs={group_size})"
        );
        scratch.ensure(gpu, num_experts * packed_each, num_experts * scale_each)?;

        let mut dst = Vec::with_capacity(num_experts);
        for (e, w) in src.iter().enumerate() {
            if w.is_null() {
                dst.push(QuantizedWeight::null());
            } else {
                dst.push(QuantizedWeight {
                    weight: scratch.packed.offset(e * packed_each),
                    weight_scale: scratch.scale.offset(e * scale_each),
                    weight_scale_2: w.weight_scale_2,
                    input_scale: w.input_scale,
                    weight_scale_2_vec: w.weight_scale_2_vec,
                });
            }
        }

        let src_tbl = build_ptr_table_from_qw(src, gpu)?;
        let dst_tbl = build_ptr_table_from_qw(&dst, gpu)?;
        let stream = gpu.default_stream();
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.packed_ptrs,
            dst_tbl.packed_ptrs,
            n as u32,
            (k / 2) as u32,
            num_experts as u32,
            stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            gpu,
            self.moe_transpose_u8_batched_k,
            src_tbl.scale_ptrs,
            dst_tbl.scale_ptrs,
            n as u32,
            (k / group_size) as u32,
            num_experts as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        // The pointer tables were scratch for the launch only.
        gpu.free(src_tbl.packed_ptrs)?;
        gpu.free(src_tbl.scale_ptrs)?;
        gpu.free(src_tbl.scale2_vals)?;
        gpu.free(dst_tbl.packed_ptrs)?;
        gpu.free(dst_tbl.scale_ptrs)?;
        gpu.free(dst_tbl.scale2_vals)?;

        // Copy the transposed bytes back over the original allocations.
        for (e, w) in src.iter().enumerate() {
            if w.is_null() {
                continue;
            }
            gpu.copy_d2d(
                scratch.packed.offset(e * packed_each),
                w.weight,
                packed_each,
            )?;
            gpu.copy_d2d(
                scratch.scale.offset(e * scale_each),
                w.weight_scale,
                scale_each,
            )?;
        }
        gpu.synchronize(stream)?;

        Ok(src.to_vec())
    }
}
