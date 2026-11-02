// SPDX-License-Identifier: AGPL-3.0-only
//! Optional large KDA projection: transient E4M3 operands, same-stream Lt.
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
#[path = "projection_fp8_plan.rs"]
mod bounds;
use bounds::Span;

#[allow(clippy::too_many_arguments)]
pub(super) fn try_project(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &super::Glm5Projection,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    decode: bool,
    capture: bool,
    scratch: DevicePtr,
    scratch_bytes: usize,
    arena_rows: usize,
    enabled: bool,
    cast: bool,
    stream: u64,
) -> Result<bool> {
    run(
        gpu,
        input,
        &weight.nvfp4,
        weight.prefill_nvfp4_t.is_some(),
        output,
        m,
        n,
        k,
        decode,
        capture,
        scratch,
        scratch_bytes,
        arena_rows,
        enabled,
        cast,
        stream,
        |a, w, o, m, n, k, s| {
            spark_runtime::cublaslt::fp8_gemm_act_weight_t_tensorwise(
                a,
                w,
                o,
                m,
                n,
                k,
                no_split_k(),
                s,
            )
        },
    )
}

/// Opt-in `ATLAS_GLM_KDA_PREFILL_LT_FP8_SPLITK1=1` (lossy path only): pin
/// the Lt FP8 projection to split-K 1. Changes the summation order of the
/// 2048..~4K-row chunks where the heuristic picks split-K 2, at ~half the time.
fn no_split_k() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_no_split_k(
            std::env::var("ATLAS_GLM_KDA_PREFILL_LT_FP8_SPLITK1")
                .ok()
                .as_deref(),
        )
    })
}

/// Off unless exactly `1`: the pin is its own opt-in, not implied by the
/// parent `ATLAS_GLM_KDA_PREFILL_LT_FP8` (whose dispatch `run` gates).
fn parse_no_split_k(value: Option<&str>) -> bool {
    value == Some("1")
}

#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &QuantizedWeight,
    transposed: bool,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    decode: bool,
    capture: bool,
    scratch: DevicePtr,
    scratch_bytes: usize,
    arena_rows: usize,
    enabled: bool,
    cast: bool,
    stream: u64,
    lt: impl FnOnce(u64, u64, u64, u32, u32, u32, u64) -> Result<()>,
) -> Result<bool> {
    if !bounds::selected(enabled, decode, capture, transposed, m, n, k) {
        return Ok(false);
    }
    if gpu.stream_is_capturing(stream) {
        return Ok(false);
    }
    ensure!(
        weight.weight_scale_2.is_finite()
            && weight.weight_scale_2 > 0.0
            && weight.weight_scale_2_vec.is_null(),
        "KDA Lt FP8 requires finite scalar NVFP4 weights"
    );
    let plan = bounds::plan(
        Span {
            ptr: scratch.0,
            bytes: scratch_bytes,
        },
        arena_rows,
        m,
        input.0,
        output.0,
        weight.weight.0,
        weight.weight_scale.0,
    )
    .map_err(anyhow::Error::msg)?;
    let cache = gpu.op_cache();
    let cast_k = cache.kernel(gpu, "w4a16", "bf16_to_fp8")?;
    let dequant = cache.kernel(gpu, "w4a16", "predequant_nvfp4_to_fp8")?;
    ensure!(
        cast_k.0 != 0 && dequant.0 != 0,
        "KDA Lt conversion kernel missing"
    );
    // expert_gate_out is dead during Q/K/V and after the recurrence before O.
    // The recurrence and FFN reuse it only later on this same caller stream.
    // The activation slot depends only on the scratch and `m`, so a caller
    // projecting the same input again may skip the cast (`cast == false`).
    if cast {
        ops::bf16_to_fp8(
            gpu,
            cast_k,
            input,
            DevicePtr(plan.activation),
            m * k,
            stream,
        )?;
    }
    ops::predequant_nvfp4_to_fp8(
        gpu,
        dequant,
        weight.weight,
        weight.weight_scale,
        weight.weight_scale_2,
        DevicePtr(plan.weight),
        n,
        k,
        stream,
    )?;
    lt(plan.activation, plan.weight, output.0, m, n, k, stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    fn weight() -> QuantizedWeight {
        let mut q = QuantizedWeight::null();
        q.weight = DevicePtr(0x4000_0000);
        q.weight_scale = DevicePtr(0x5000_0000);
        q.weight_scale_2 = 0.003;
        q
    }
    #[test]
    fn kda_lt_fp8_actual_dispatch_casts_then_lt_without_allocating() {
        let gpu = MockGpuBackend::new();
        for m in [2048, 4100] {
            let before = gpu.launch_count();
            assert!(
                run(
                    &gpu,
                    DevicePtr(0x2000_0000),
                    &weight(),
                    true,
                    DevicePtr(0x3000_0000),
                    m,
                    4096,
                    4096,
                    false,
                    false,
                    DevicePtr(0x1000_0000),
                    64 << 20,
                    4100,
                    true,
                    true,
                    77,
                    |a, w, o, got_m, n, k, s| {
                        assert_eq!(
                            (a, w, o, got_m, n, k, s),
                            (0x1100_0000, 0x1000_0000, 0x3000_0000, m, 4096, 4096, 77)
                        );
                        let calls = gpu.launches_snapshot();
                        assert_eq!(calls.len(), before + 2);
                        assert_eq!(calls[before].grid, [(m * 4096 / 2).div_ceil(256), 1, 1]);
                        assert_eq!(
                            calls[before + 1].grid,
                            [(4096 * 4096 / 2u32).div_ceil(256), 1, 1]
                        );
                        Ok(())
                    }
                )
                .unwrap()
            );
        }
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.sync_count(), 0);
    }
    #[test]
    fn kda_lt_fp8_short_decode_graph_and_flag_off_do_not_dispatch() {
        let gpu = MockGpuBackend::new();
        for (m, decode, capture, enabled) in [
            (3, false, false, true),
            (2048, true, false, true),
            (2048, false, true, true),
            (2048, false, false, false),
        ] {
            assert!(
                !run(
                    &gpu,
                    DevicePtr::NULL,
                    &weight(),
                    true,
                    DevicePtr::NULL,
                    m,
                    4096,
                    4096,
                    decode,
                    capture,
                    DevicePtr::NULL,
                    0,
                    0,
                    enabled,
                    true,
                    77,
                    |_, _, _, _, _, _, _| panic!("unexpected Lt")
                )
                .unwrap()
            );
        }
        assert_eq!(gpu.launch_count(), 0);
        assert_eq!(gpu.alloc_count(), 0);
    }
    #[test]
    fn kda_lt_fp8_split_k_pin_is_off_unless_set_to_one() {
        for value in [None, Some(""), Some("0"), Some("true"), Some("2")] {
            assert!(!parse_no_split_k(value), "{value:?}");
        }
        assert!(parse_no_split_k(Some("1")));
    }
    #[test]
    fn kda_lt_fp8_bounds_fail_before_casts_and_lt_error_propagates() {
        let gpu = MockGpuBackend::new();
        let call = |capacity, cast, lt: fn(u64, u64, u64, u32, u32, u32, u64) -> Result<()>| {
            run(
                &gpu,
                DevicePtr(0x2000_0000),
                &weight(),
                true,
                DevicePtr(0x3000_0000),
                2048,
                4096,
                4096,
                false,
                false,
                DevicePtr(0x1000_0000),
                capacity,
                4100,
                true,
                cast,
                77,
                lt,
            )
        };
        fn fail(_: u64, _: u64, _: u64, _: u32, _: u32, _: u32, _: u64) -> Result<()> {
            anyhow::bail!("Lt injected failure")
        }
        fn ok(_: u64, _: u64, _: u64, _: u32, _: u32, _: u32, _: u64) -> Result<()> {
            Ok(())
        }
        assert!(call(1, true, fail).is_err());
        assert_eq!(gpu.launch_count(), 0);
        assert!(call(64 << 20, true, fail).is_err());
        assert_eq!(gpu.launch_count(), 2);
        // A projection reusing the previous cast of the same input launches
        // only the weight conversion.
        assert!(call(64 << 20, false, ok).unwrap());
        assert_eq!(gpu.launch_count(), 3);
    }
}
