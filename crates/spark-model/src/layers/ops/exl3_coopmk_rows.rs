// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl Exl3CoopMk {
    /// `m` rows in one call (upstream `CoopMK::run` at bsz = m): `x_bf16`
    /// [m, H], `indices` / `weights` [m, top_k] (u32 / f32, contiguous),
    /// `out_bf16` [m, H]. `m == 1` is exactly [`Self::run`]. For m >= 2 the
    /// rotation pre-kernel writes every slot's rotated input to `had_g/had_u`
    /// and builds the (expert, slot) run table, then kernel A/B run each run of
    /// up to 8 slots that share an expert as the rows of one MMA: the expert's
    /// trellis is read once per run instead of once per (row, slot). Per row the
    /// arithmetic is the bsz-1 arithmetic (same rotation, same k-split and
    /// reduction order; MMA rows do not mix). Launch-only, no host sync.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        x_bf16: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        shared_gate_w: DevicePtr,
        out_bf16: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if m == 1 {
            return self.run(
                gpu,
                x_bf16,
                indices,
                weights,
                shared_gate_w,
                out_bf16,
                stream,
            );
        }
        if m == 0 || m > self.rows_max {
            bail!("CoopMK run_rows: {m} rows outside 1..={}", self.rows_max);
        }
        let h = self.hidden;
        KernelLaunch::new(gpu, self.kernels.prep_rows)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x_bf16)
            .arg_ptr(self.xh)
            .arg_i32(h as i32)
            .arg_ptr(indices)
            .arg_ptr(weights)
            .arg_ptr(self.sel)
            .arg_ptr(self.rw)
            .arg_i32(self.top_k as i32)
            .arg_ptr(shared_gate_w)
            .arg_i32(self.shared_local as i32)
            .launch(stream)?;
        let mut q = self.params;
        q.bsz = m as i32;
        q.a_global = 1;
        let slots = m as u32 * (self.top_k + 1);
        let nproj = 2u32;
        // Narrow tile for both stages (the only instances built; upstream's
        // pick_wide would switch stage A to the wide tile at >= 32 slots).
        let (wa, wb) = wide_for(h, self.inter, slots);
        let grid_a = slots * nproj * (self.inter / if wa { 128 } else { COOPMK_COLS });
        let grid_b = slots * (h / if wb { 128 } else { COOPMK_COLS });
        let la = if wa {
            &self.launches_a_w
        } else {
            &self.launches_a
        };
        let lb = if wb {
            &self.launches_b_w
        } else {
            &self.launches_b
        };
        let rot_items = slots * (h / 128) * nproj;
        let rot_grid = div_ceil(rot_items, COOPMK_THREADS / 32);
        let bytes = q.as_bytes();
        KernelLaunch::new(gpu, self.kernels.rot)
            .grid([rot_grid, 1, 1])
            .block([COOPMK_THREADS, 1, 1])
            .arg_bytes(bytes)
            .launch(stream)?;
        for (k, _) in la {
            KernelLaunch::new(gpu, *k)
                .grid([grid_a, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_a)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        for (k, _) in lb {
            KernelLaunch::new(gpu, *k)
                .grid([grid_b, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_b)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        let total = m as u32 * h;
        KernelLaunch::new(gpu, self.kernels.f32_to_bf16)
            .grid([div_ceil(total, 256).min(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.out_f32)
            .arg_ptr(out_bf16)
            .arg_i32(total as i32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::MoeCoopParams;
    use std::mem::offset_of;

    #[test]
    fn layout_matches_c() {
        // Same numbers as the static_asserts in exl3_vendor/coopmk/moe_coop.cuh.
        assert_eq!(std::mem::size_of::<MoeCoopParams>(), 344);
        assert_eq!(offset_of!(MoeCoopParams, sel), 16);
        assert_eq!(offset_of!(MoeCoopParams, bsz), 32);
        assert_eq!(offset_of!(MoeCoopParams, max_expert), 64);
        assert_eq!(offset_of!(MoeCoopParams, g_trellis), 72);
        assert_eq!(offset_of!(MoeCoopParams, d_bias), 160);
        assert_eq!(offset_of!(MoeCoopParams, act), 168);
        assert_eq!(offset_of!(MoeCoopParams, gated), 176);
        assert_eq!(offset_of!(MoeCoopParams, had_g), 184);
        assert_eq!(offset_of!(MoeCoopParams, a_global), 200);
        assert_eq!(offset_of!(MoeCoopParams, gu_g), 208);
        assert_eq!(offset_of!(MoeCoopParams, gu_f32), 224);
        assert_eq!(offset_of!(MoeCoopParams, act_out), 232);
        assert_eq!(offset_of!(MoeCoopParams, ctr_a_len), 264);
        assert_eq!(offset_of!(MoeCoopParams, dbg), 280);
        assert_eq!(offset_of!(MoeCoopParams, runs), 288);
        assert_eq!(offset_of!(MoeCoopParams, sh_gate_n), 308);
        assert_eq!(offset_of!(MoeCoopParams, out), 312);
        assert_eq!(offset_of!(MoeCoopParams, out_stride), 320);
        assert_eq!(offset_of!(MoeCoopParams, sh_out), 328);
        assert_eq!(offset_of!(MoeCoopParams, sh_gate_w), 336);
    }
}
