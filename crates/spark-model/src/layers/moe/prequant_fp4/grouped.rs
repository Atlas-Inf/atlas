// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl MoeLayer {
    /// One prequant grouped projection `[rows, k] x [k, n]`: the K128W kernel
    /// over `wide` when it fits, else the K128 kernel (256 threads) when
    /// enabled and K128/N128-aligned, else `fallback` (128 threads) over the
    /// dense `(n/128, max_m_tiles, experts)` grid. All three agree bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn prequant_grouped(
        &self,
        fallback: KernelHandle,
        [a_packed, a_scale]: [DevicePtr; 2],
        weight: &ExpertPtrTable,
        output: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        [n, k]: [u32; 2],
        max_m_tiles: u32,
        wide: Option<MtileGrid>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(grid) = wide
            && n % 256 == 0
            && k % 128 == 0
        {
            return ops::moe_w4a4_grouped_gemm_prequant_k128w(
                ctx.gpu,
                grid.down,
                a_packed,
                a_scale,
                weight.packed_ptrs,
                weight.scale_ptrs,
                weight.scale2_vals,
                output,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n,
                k,
                grid.prefix,
                grid.schedule,
                stream,
            );
        }
        let (kernel, threads) =
            if self.moe_w4a4_prequant_t_k128.0 != 0 && n % 128 == 0 && k % 128 == 0 {
                (self.moe_w4a4_prequant_t_k128, 256)
            } else {
                (fallback, 128)
            };
        ops::moe_w4a4_grouped_gemm_prequant_n128(
            ctx.gpu,
            kernel,
            a_packed,
            a_scale,
            weight.packed_ptrs,
            weight.scale_ptrs,
            weight.scale2_vals,
            output,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n,
            k,
            max_m_tiles,
            threads,
            stream,
        )
    }
}
