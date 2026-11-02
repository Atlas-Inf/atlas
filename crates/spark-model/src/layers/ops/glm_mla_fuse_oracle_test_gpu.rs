// SPDX-License-Identifier: AGPL-3.0-only

//! Unfused production-kernel oracle for the fused MLA front.

use super::*;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// The twelve launches, as the canonical verify runs them.
pub(super) fn unfused(
    gpu: &dyn GpuBackend,
    w: &Weights,
    run: &Run,
    meta: &Meta,
    input: DevicePtr,
    rows: u32,
    owner: ShardRank,
    s: u64,
) -> Result<()> {
    let tier = |m: u32| match m {
        1..=8 => "8",
        9..=16 => "16",
        _ => "32",
    };
    let family = if w.fp8_block { "fp8b" } else { "mxfp8" };
    let mxk = kernel(
        gpu,
        "mxfp8_gemv",
        &format!("{family}_gemv_tc{}", tier(rows)),
    )?;
    let dk = kernel(
        gpu,
        "dense_gemv_bf16_batchm",
        &format!("dense_gemv_bf16_tc{}", tier(rows)),
    )?;
    let norm = kernel(gpu, "glm_rms_norm_regs", "rms_norm_vanilla_regs")?;
    let idx = |name| kernel(gpu, "glm_indexer", name);
    let dw = |weight| DenseWeight { weight };
    let gemv = if w.fp8_block {
        native::gemv
    } else {
        ops::mxfp8_gemv
    };
    gemv(gpu, mxk, input, w.q_a.0, w.q_a.1, run.q, rows, QL, H, QL, s)?;
    ops::rms_norm(gpu, norm, run.q, &dw(w.q_norm), run.q, rows, QL, 1e-6, s)?;
    gemv(
        gpu, mxk, input, w.kv_a.0, w.kv_a.1, run.kv, rows, KVL, H, KVL, s,
    )?;
    ops::rms_norm(
        gpu,
        norm,
        run.kv,
        &dw(w.kv_norm),
        run.kv,
        rows,
        KVL,
        1e-6,
        s,
    )?;
    let v_entries = run.entries.offset(rows as usize * KVL as usize * 2);
    let assemble = kernel(gpu, "mla_absorbed", "mla_cache_assemble_batched")?;
    ops::mla_cache_assemble_batched(
        gpu,
        assemble,
        run.kv,
        DevicePtr::NULL,
        run.entries,
        v_entries,
        rows,
        KVL,
        0,
        KVL,
        s,
    )?;
    let slots = if owner.world > 1 {
        ops::glm_kv_shard_map_slots(gpu, meta.slots, run.local, rows, BS, owner, s)?;
        run.local
    } else {
        meta.slots
    };
    let write = kernel(gpu, "reshape_and_cache", "glm_latent_cache_write_fp8g128")?;
    ops::glm_latent_cache_write_fp8g128(
        gpu,
        write,
        run.entries,
        run.latent,
        slots,
        rows,
        BS,
        KVL,
        s,
    )?;
    ops::dense_gemv_bf16_tc(gpu, dk, input, &dw(w.wk), run.keys, rows, ID, H, ID, s)?;
    let ln = idx("glm_index_layernorm_bf16")?;
    ops::glm_index_layernorm(gpu, ln, run.keys, w.k_norm_w, w.k_norm_b, rows, ID, 1e-6, s)?;
    ops::dense_gemv_bf16_tc(gpu, dk, input, &dw(w.gate), run.gates, rows, ID, H, ID, s)?;
    let tail = idx("glm_index_tail_write_bf16")?;
    let (keys, gates, tm) = (run.keys, run.gates, DevicePtr::NULL);
    ops::glm_index_tail_write(
        gpu,
        tail,
        keys,
        gates,
        run.tail,
        tm,
        meta.slots,
        rows,
        BS,
        4,
        ID,
        TAIL_STRIDE,
        s,
    )?;
    let fin = idx("glm_index_kpool_finalize_bf16")?;
    ops::glm_index_kpool_finalize(
        gpu,
        fin,
        run.tail,
        tm,
        w.ape,
        run.pool,
        meta.slots,
        rows,
        BS,
        4,
        ID,
        TAIL_STRIDE,
        POOL_STRIDE,
        s,
    )?;
    if owner.world > 1 {
        let (ids, counts) = (run.ids, run.counts);
        ops::glm_kv_shard_localize(
            gpu, None, ids, counts, meta.table, rows, WIDTH, BS, owner, meta.start, s,
        )?;
    }
    Ok(())
}
