// SPDX-License-Identifier: AGPL-3.0-only

use super::{GROUP, MarlinSidecar, SMEM, SMS, SORTED_CAP};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn pack_proj(
    gpu: &dyn GpuBackend,
    w: &QuantizedWeight,
    n: usize,
    k: usize,
    tmp_in: DevicePtr,
    tmp_out: DevicePtr,
    dest_w: DevicePtr,
    dest_s: DevicePtr,
    dest_gs: DevicePtr,
    repack: spark_runtime::gpu::KernelHandle,
) -> Result<()> {
    let mut host = vec![0u8; n * (k / 2)];
    gpu.copy_d2h(w.weight, &mut host)?;
    let mut sc = vec![0u8; n * (k / GROUP)];
    gpu.copy_d2h(w.weight_scale, &mut sc)?;
    ops::marlin_pack_nvfp4(
        gpu,
        repack,
        &host,
        &sc,
        w.weight_scale_2,
        n,
        k,
        tmp_in,
        tmp_out,
        dest_w,
        dest_s,
        dest_gs,
        SMS,
        SMEM,
    )
}

impl MarlinSidecar {
    pub fn try_build(
        gpu: &dyn GpuBackend,
        experts: &[crate::weight_map::NemotronExpertWeight],
        up_n: usize,
        up_k: usize,
        down_n: usize,
        down_k: usize,
    ) -> Result<Option<Self>> {
        if std::env::var_os("ATLAS_MOE_MARLIN").is_none() {
            return Ok(None);
        }
        let moe_up =
            crate::layers::try_kernel(gpu, "marlin_moe_nvfp4", "atlas_marlin_moe_nvfp4_m8");
        let moe_down =
            crate::layers::try_kernel(gpu, "marlin_moe_nvfp4", "atlas_marlin_moe_nvfp4_m8_k64n128");
        let lin_up = crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8");
        let lin_down =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8_k64n128");
        let cfg4_up =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4");
        let cfg4_down =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_cfg4_k64n128");
        let slot_up =
            crate::layers::try_kernel(gpu, "marlin_nvfp4_gemm", "atlas_marlin_nvfp4_m8_allslots");
        let slot_dn = crate::layers::try_kernel(
            gpu,
            "marlin_nvfp4_gemm",
            "atlas_marlin_nvfp4_m8_k64n128_allslots",
        );
        let pack = crate::layers::try_kernel(gpu, "marlin_pack_slots", "atlas_marlin_pack_slots");
        let scatter =
            crate::layers::try_kernel(gpu, "marlin_scatter_slots", "atlas_marlin_scatter_slots");
        let repack = crate::layers::try_kernel(gpu, "marlin_repack", "atlas_marlin_repack_w4");
        let align = crate::layers::try_kernel(gpu, "marlin_align", "atlas_marlin_align_block8");
        let repeat = crate::layers::try_kernel(gpu, "marlin_row_repeat", "atlas_row_repeat_bf16");
        let pack_rows =
            crate::layers::try_kernel(gpu, "marlin_pack_rows", "atlas_marlin_pack_rows");
        if lin_up.0 == 0
            || lin_down.0 == 0
            || moe_up.0 == 0
            || moe_down.0 == 0
            || repack.0 == 0
            || align.0 == 0
            || repeat.0 == 0
            || pack_rows.0 == 0
        {
            tracing::warn!("ATLAS_MOE_MARLIN set but kernels missing; leaving GEMV");
            return Ok(None);
        }
        let e = experts.len();
        let up_w_b = (up_k / 16) * (up_n * 16 / 8) * 4;
        let down_w_b = (down_k / 16) * (down_n * 16 / 8) * 4;
        let up_s_b = (up_k / GROUP) * up_n;
        let down_s_b = (down_k / GROUP) * down_n;
        let up_w = gpu.alloc(e * up_w_b)?;
        let down_w = gpu.alloc(e * down_w_b)?;
        let up_s = gpu.alloc(e * up_s_b)?;
        let down_s = gpu.alloc(e * down_s_b)?;
        let up_gs = gpu.alloc(e * 4)?;
        let down_gs = gpu.alloc(e * 4)?;
        let tmp_in = gpu.alloc(up_w_b.max(down_w_b))?;
        let tmp_out = gpu.alloc(up_w_b.max(down_w_b))?;
        for (i, ex) in experts.iter().enumerate() {
            pack_proj(
                gpu,
                &ex.up_proj,
                up_n,
                up_k,
                tmp_in,
                tmp_out,
                DevicePtr(up_w.0 + (i * up_w_b) as u64),
                DevicePtr(up_s.0 + (i * up_s_b) as u64),
                DevicePtr(up_gs.0 + (i * 4) as u64),
                repack,
            )?;
            pack_proj(
                gpu,
                &ex.down_proj,
                down_n,
                down_k,
                tmp_in,
                tmp_out,
                DevicePtr(down_w.0 + (i * down_w_b) as u64),
                DevicePtr(down_s.0 + (i * down_s_b) as u64),
                DevicePtr(down_gs.0 + (i * 4) as u64),
                repack,
            )?;
        }
        let _ = gpu.free(tmp_in);
        let _ = gpu.free(tmp_out);
        tracing::info!("Marlin sidecar packed {e} experts UP {up_n}x{up_k} DOWN {down_n}x{down_k}");
        Ok(Some(Self {
            up_w,
            up_s,
            up_gs,
            down_w,
            down_s,
            down_gs,
            locks: gpu.alloc(ops::MARLIN_SLOTS as usize * 256 * 4)?,
            c_tmp: gpu.alloc(ops::MARLIN_SLOTS as usize * 16 * down_n.max(up_n) * 4)?,
            sorted_ids: gpu.alloc(SORTED_CAP * 4)?,
            expert_ids: gpu.alloc(256 * 4)?,
            n_post: gpu.alloc(4)?,
            a_exp: gpu.alloc(16 * 6 * up_k * 2)?,
            moe_up_k: moe_up,
            moe_down_k: moe_down,
            lin_up_k: lin_up,
            lin_down_k: lin_down,
            cfg4_up_k: cfg4_up,
            cfg4_down_k: cfg4_down,
            pack_rows_k: pack_rows,
            lin_up_out: gpu.alloc(8 * up_n * 2)?,
            lin_dn_out: gpu.alloc(8 * down_n * 2)?,
            pack_k: pack,
            scatter_k: scatter,
            slot_up_k: slot_up,
            slot_dn_k: slot_dn,
            slot_eids: gpu.alloc(ops::MARLIN_SLOTS as usize * 4)?,
            slot_map: gpu.alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * 4)?,
            slot_a: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * up_k * 2)?,
            slot_up: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * up_n * 2)?,
            slot_dn: gpu
                .alloc(ops::MARLIN_SLOTS as usize * ops::MARLIN_M_TILE as usize * down_n * 2)?,
            slot_bars: gpu.alloc(ops::MARLIN_SLOTS as usize * 4)?,
            align_k: align,
            repeat_k: repeat,
            up_n: up_n as i32,
            up_k: up_k as i32,
            down_n: down_n as i32,
            down_k: down_k as i32,
            e: e as i32,
        }))
    }
}
