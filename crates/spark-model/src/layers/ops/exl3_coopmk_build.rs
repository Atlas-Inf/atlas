// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl Exl3CoopMk {
    /// Build the tables for `routed` experts plus `shared` (appended as the
    /// last local expert). Every routed expert must be present (no expert
    /// parallel split on this path). `hidden` = H = Hi = Ho, `inter` = I.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &dyn GpuBackend,
        routed: &[(&Exl3Weight, &Exl3Weight, &Exl3Weight)],
        shared: (&Exl3Weight, &Exl3Weight, &Exl3Weight),
        hidden: usize,
        inter: usize,
        top_k: usize,
    ) -> Result<Self> {
        let kernels = Exl3CoopMkKernels::resolve(gpu)?;
        let n_local = routed.len() + 1;
        let slots = top_k + 1;
        if !hidden.is_multiple_of(128) || !inter.is_multiple_of(128) {
            bail!("CoopMK: hidden {hidden} / intermediate {inter} must be multiples of 128");
        }
        if slots > 256 || slots > COOPMK_THREADS as usize {
            bail!("CoopMK: {slots} slots exceed the kernel's 256-slot bound");
        }
        let all: Vec<(&Exl3Weight, &Exl3Weight, &Exl3Weight)> = routed
            .iter()
            .copied()
            .chain(std::iter::once(shared))
            .collect();
        let mut tabs: [Vec<u64>; 9] = Default::default();
        let (mut kg, mut ku, mut kd) = (Vec::new(), Vec::new(), Vec::new());
        let (mut kset_a, mut kset_b) = (0u32, 0u32);
        for (e, (g, u, d)) in all.iter().enumerate() {
            if g.is_null() || u.is_null() || d.is_null() {
                bail!("CoopMK: expert {e} has no packed weights");
            }
            let ok_gu =
                |w: &Exl3Weight| w.shape.in_features == hidden && w.shape.out_features == inter;
            if !ok_gu(g)
                || !ok_gu(u)
                || d.shape.in_features != inter
                || d.shape.out_features != hidden
            {
                bail!(
                    "CoopMK: expert {e} shapes gate {}x{} up {}x{} down {}x{} (in x out) do not match H {hidden} I {inter}",
                    g.shape.in_features,
                    g.shape.out_features,
                    u.shape.in_features,
                    u.shape.out_features,
                    d.shape.in_features,
                    d.shape.out_features
                );
            }
            let (bg, bu, bd) = (bits_ok(g)?, bits_ok(u)?, bits_ok(d)?);
            kset_a |= (1 << bg) | (1 << bu);
            kset_b |= 1 << bd;
            kg.push(bg as i32);
            ku.push(bu as i32);
            kd.push(bd as i32);
            for (t, p) in tabs.iter_mut().zip([
                g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh,
            ]) {
                t.push(p.0);
            }
        }

        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            let p = gpu.alloc(bytes.max(16))?;
            gpu.memset(p, 0, bytes.max(16))?;
            owned.push(p);
            Ok(p)
        };
        let mut tab_ptrs = [DevicePtr::NULL; 9];
        for (i, t) in tabs.iter().enumerate() {
            let p = alloc(t.len() * 8)?;
            gpu.copy_h2d(&u64_bytes(t), p)?;
            tab_ptrs[i] = p;
        }
        let ktab_host: Vec<u8> = kg
            .iter()
            .chain(ku.iter())
            .chain(kd.iter())
            .flat_map(|k| k.to_le_bytes())
            .collect();
        let k_tab = alloc(ktab_host.len())?;
        gpu.copy_h2d(&ktab_host, k_tab)?;

        let (h, i) = (hidden, inter);
        // Scratch for up to COOPMK_ROWS_MAX rows (bsz 1 uses the first row's
        // slice; the kernels index by slot / row, never by the maxima).
        let rows_max = if slots * COOPMK_ROWS_MAX <= 256 {
            COOPMK_ROWS_MAX
        } else {
            1
        };
        let slots_max = slots * rows_max;
        let xh = alloc(rows_max * h * 2)?;
        let sel = alloc(slots_max * 8)?;
        let rw = alloc(slots_max * 2)?;
        let had_g = alloc(slots_max * h * 2)?;
        let had_u = alloc(slots_max * h * 2)?;
        let gu_g = alloc(slots_max * i * 2)?;
        let gu_u = alloc(slots_max * i * 2)?;
        let act_out = alloc(slots_max * i * 2)?;
        let d_out = alloc(slots_max * h * 4)?;
        let ctr_a_len = slots_max * (i / 128);
        let ctr_b_len = rows_max * (h / 128);
        // exl3_moe_coop_ctr_len: counters + run table.
        let ctr_len = ctr_a_len + ctr_b_len + 2 + (slots_max + 1) + slots_max;
        let ctr = alloc(ctr_len * 4)?;
        let out_f32 = alloc(rows_max * h * 4)?;
        let rows_cap = rows_max;
        let parity_out = alloc(h * 2)?;

        let dbg = std::env::var("ATLAS_EXL3_COOPMK_DBG")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        let params = MoeCoopParams {
            x: xh.0,
            x_stride: h as i32,
            sel: sel.0,
            rw: rw.0,
            bsz: 1,
            topk: slots as i32,
            h: h as i32,
            hi: h as i32,
            i: i as i32,
            ho: h as i32,
            h_out: h as i32,
            min_expert: -1,
            max_expert: -1,
            g_trellis: tab_ptrs[0].0,
            g_suh: tab_ptrs[1].0,
            g_svh: tab_ptrs[2].0,
            u_trellis: tab_ptrs[3].0,
            u_suh: tab_ptrs[4].0,
            u_svh: tab_ptrs[5].0,
            d_trellis: tab_ptrs[6].0,
            d_suh: tab_ptrs[7].0,
            d_svh: tab_ptrs[8].0,
            act: ACT_SILU,
            act_limit: 0.0,
            gated: 1,
            had_g: had_g.0,
            had_u: had_u.0,
            a_global: 0,
            gu_g: gu_g.0,
            gu_u: gu_u.0,
            // Qwen4Exp (Flash Next) interm_dtype is fp16 in the fork.
            gu_f32: 0,
            act_out: act_out.0,
            d_out: d_out.0,
            ctr_a: ctr.0,
            ctr_b: ctr.0 + (ctr_a_len * 4) as u64,
            ctr_a_len: ctr_a_len as i32,
            ctr_b_len: ctr_b_len as i32,
            ksplit_a: 1,
            ksplit_b: 1,
            dbg,
            runs: ctr.0 + ((ctr_a_len + ctr_b_len) * 4) as u64,
            slots_max: slots_max as i32,
            rows_max: rows_max as i32,
            n_local: n_local as i32,
            sh_gate_n: 0,
            out: out_f32.0,
            out_stride: h as i32,
            sh_out: 0,
            sh_gate_w: 0,
            ..Default::default()
        };

        let plan = plan_from_env();
        let launches_a = stage_variants(&kernels, plan, kset_a, true, false);
        let launches_b = stage_variants(&kernels, plan, kset_b, false, false);
        let launches_a_w = stage_variants(&kernels, plan, kset_a, true, true);
        let launches_b_w = stage_variants(&kernels, plan, kset_b, false, true);
        // Upstream geometry at bsz 1, narrow tile (pick_wide on Blackwell:
        // kslices >= 256, or >= 128 with >= 32 slots — neither holds here).
        let grid_a = (slots as u32) * 2 * (i as u32 / COOPMK_COLS);
        let grid_b = (slots as u32) * (h as u32 / COOPMK_COLS);
        // smem_a_bytes_mk(Hi) = Hi*2 + WK*ROWS*COLS*4 + WK*STAGE_WORDS*4
        let red = 16 * 8 * 32 * 4;
        let stage = 16 * 128 * 4;
        let part = 16 * 128 * 4;
        let smem_a = (h * 2 + red + stage) as u32;
        let smem_b = (red + stage + part) as u32;
        Ok(Self {
            kernels,
            params,
            k_tab,
            xh,
            sel,
            rw,
            out_f32,
            parity_out,
            top_k: top_k as u32,
            hidden: h as u32,
            inter: i as u32,
            rows_max: rows_cap,
            shared_local: routed.len() as u32,
            launches_a,
            launches_b,
            launches_a_w,
            launches_b_w,
            smem_a,
            smem_b,
            grid_a,
            grid_b,
            kset_a,
            kset_b,
            owned,
        })
    }

    pub fn describe(&self) -> String {
        let names = |v: &[(KernelHandle, &'static str)]| {
            v.iter().map(|x| x.1).collect::<Vec<_>>().join("+")
        };
        let ks = |m: u32| {
            (1..=8)
                .filter(|k| m & (1 << k) != 0)
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "A[{}] K{{{}}} grid {} smem {} | B[{}] K{{{}}} grid {} smem {} | {} local experts, {} slots",
            names(&self.launches_a),
            ks(self.kset_a),
            self.grid_a,
            self.smem_a,
            names(&self.launches_b),
            ks(self.kset_b),
            self.grid_b,
            self.smem_b,
            self.params.n_local,
            self.params.topk
        )
    }

    /// One token: `x_bf16` [H] MoE input, `indices`/`weights` the router's
    /// top-k (u32 / f32 on device), `shared_gate_w` bf16 [H] (or NULL: weight
    /// 1). Writes bf16 [H] to `out_bf16`. Launch-only, no host sync.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        x_bf16: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        shared_gate_w: DevicePtr,
        out_bf16: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden;
        KernelLaunch::new(gpu, self.kernels.prep)
            .grid([div_ceil(h, 256).min(64), 1, 1])
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
        let bytes = self.params.as_bytes();
        let (wa, wb) = wide_for(h, self.inter, self.top_k + 1);
        let (la, ga) = if wa {
            (
                &self.launches_a_w,
                (self.top_k + 1) * 2 * (self.inter / 128),
            )
        } else {
            (&self.launches_a, self.grid_a)
        };
        let (lb, gb) = if wb {
            (&self.launches_b_w, (self.top_k + 1) * (h / 128))
        } else {
            (&self.launches_b, self.grid_b)
        };
        for (k, _) in la {
            KernelLaunch::new(gpu, *k)
                .grid([ga, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_a)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        for (k, _) in lb {
            KernelLaunch::new(gpu, *k)
                .grid([gb, 1, 1])
                .block([COOPMK_THREADS, 1, 1])
                .shared_mem(self.smem_b)
                .arg_bytes(bytes)
                .arg_ptr(self.k_tab)
                .launch(stream)?;
        }
        KernelLaunch::new(gpu, self.kernels.f32_to_bf16)
            .grid([div_ceil(h, 256).min(64), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.out_f32)
            .arg_ptr(out_bf16)
            .arg_i32(h as i32)
            .launch(stream)
    }
}
