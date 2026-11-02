// SPDX-License-Identifier: AGPL-3.0-only

//! Launch-recording tests for the decode/verify exact-rows MXFP8 contract
//! (`ops::mxfp8_proj_exact_rows` and the shared expert's chunked fused
//! chain), split from `mod_tests.rs` for the 500-line cap.

use super::*;

/// `mxfp8_proj_exact_rows` serves any m as <=32-row GEMV-tier launches
/// (bit-identical rows at decode/verify); the plain `mxfp8_proj` still takes
/// the prefill dequant + GEMM arm at m > 32.
#[test]
fn mxfp8_exact_rows_chunks_and_never_dequants() {
    use crate::layers::moe::shared_fp8_cache_test_gpu::RecordingGpu;

    let gpu = RecordingGpu::new();
    let proj = ops::Mxfp8Proj {
        w: crate::weight_map::Mxfp8Weight {
            tiled: false,
            data: DevicePtr(0x10000),
            scales: DevicePtr(0x11000),
        },
        n: 640,
        k: 2560,
    };
    let kernels = ops::Mxfp8ProjKernels {
        gemv: KernelHandle(111),
        batchm: KernelHandle(222),
        dequant: KernelHandle(333),
        rows32: KernelHandle(444),
        tc: [KernelHandle(0); 3],
    };
    let scratch = ops::Mxfp8Scratch {
        ptr: DevicePtr(0x9000_0000),
        bytes: 1 << 24,
    };
    let launches_of = |f: &mut dyn FnMut() -> anyhow::Result<()>| {
        gpu.launches.lock().unwrap().clear();
        let r = f();
        let ls: Vec<u64> = gpu
            .launches
            .lock()
            .unwrap()
            .iter()
            .map(|l| l.kernel)
            .collect();
        (r, ls)
    };

    // m = 0 errors and launches nothing.
    let (r, ls) = launches_of(&mut || {
        ops::mxfp8_proj_exact_rows(&gpu, &kernels, &proj, DevicePtr(1), DevicePtr(2), 0, 19)
    });
    assert!(
        r.is_err() && ls.is_empty(),
        "m=0 must fail with zero launches"
    );

    // m > 32 through the exact fn: ceil(m / 32) launches, never dequant,
    // and every launch is a GEMV-tier kernel (gemv/batchm/rows32).
    for m in [33usize, 64, 65] {
        let (r, ls) = launches_of(&mut || {
            ops::mxfp8_proj_exact_rows(&gpu, &kernels, &proj, DevicePtr(1), DevicePtr(2), m, 19)
        });
        assert!(r.is_ok());
        let want = m.div_ceil(32);
        assert_eq!(
            ls.len(),
            want,
            "m={m}: expected {want} row-chunk launches, got {ls:?}"
        );
        assert!(
            ls.iter().all(|h| [111u64, 222, 444].contains(h)),
            "m={m}: non-GEMV launch in {ls:?}"
        );
    }

    // m > 32 through the plain fn still expands + GEMMs.
    let (r, ls) = launches_of(&mut || {
        ops::mxfp8_proj(
            &gpu,
            &kernels,
            &proj,
            DevicePtr(1),
            DevicePtr(2),
            33,
            scratch,
            19,
        )
    });
    // The mock cannot run the trailing GEMM; the dequant launch is the
    // signal that the >32-row arm was taken.
    let _ = r;
    assert!(
        ls.contains(&333),
        "m=33 via mxfp8_proj must launch dequant, got {ls:?}"
    );
}

/// With a fused `gate_up` arm, exact mode at m > 32 must run the serial chain
/// (gate_up projection -> fused_silu_mul -> down) once per <=32-row chunk —
/// never the unfused silu_mul arm, never dequant.
#[test]
fn mxfp8_shared_expert_exact_chunks_the_fused_chain() {
    use crate::layer::ForwardContext;
    use crate::layers::moe::shared_fp8_cache_test_gpu::{Arg, RecordingGpu};
    use atlas_core::config::ModelConfig;
    use spark_runtime::buffers::BufferArena;
    use spark_runtime::gpu::KernelHandle;

    let gpu = RecordingGpu::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 2560;
    config.num_experts = 512;
    config.num_experts_per_tok = 10;
    config.moe_intermediate_size = 640;
    config.shared_expert_intermediate_size = 640;
    config.vocab_size = 8;
    let mut layer = MoeLayer::new(MoeWeights::empty(512), 512, None, &gpu, &config).unwrap();
    let (h, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
    let proj = |n: usize, k: usize, ptr: u64| Mxfp8Proj {
        w: crate::weight_map::Mxfp8Weight {
            tiled: false,
            data: DevicePtr(ptr),
            scales: DevicePtr(ptr + 0x1000),
        },
        n: n as u32,
        k: k as u32,
    };
    let kernels = ops::Mxfp8ProjKernels {
        gemv: KernelHandle(111),
        batchm: KernelHandle(222),
        dequant: KernelHandle(333),
        rows32: KernelHandle(444),
        tc: [KernelHandle(0); 3],
    };
    let scratch = ops::Mxfp8Scratch {
        ptr: DevicePtr(0x9000_0000),
        bytes: 1 << 24,
    };
    let silu_fused = KernelHandle(777);
    let gu = crate::layers::moe::Mxfp8GateUp {
        proj: proj(inter * 2, h, 0x50000),
        gate_up_out: DevicePtr(0xA000_0000),
        silu_mul: silu_fused,
    };
    layer
        .set_mxfp8_shared_expert(
            proj(inter, h, 0x10000),
            proj(inter, h, 0x20000),
            proj(h, inter, 0x30000),
            Some(gu),
            kernels,
            scratch,
        )
        .unwrap();

    let buffers = BufferArena::new(&config, 64, 4096, 16, 1, &gpu).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
    let derived = ops::DerivedWeights::new();
    let levers = ops::ModelLevers::defaults();
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        ssm_batch: None,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    let (input, gate_out, up_out, down_out) = (
        DevicePtr(0x40000),
        DevicePtr(0x50000),
        DevicePtr(0x60000),
        DevicePtr(0x70000),
    );
    // Per chunk: gate_up rows via batchm/rows32, fused_silu_mul (777), down
    // rows via batchm/rows32. mxfp8 kernel args: (act, w, scales, out, m, n,
    // k); fused_silu_mul args: (gate_up, out, num_tokens, inter).
    for (m, want_chains) in [(1u32, 1usize), (64, 2), (40, 2)] {
        gpu.launches.lock().unwrap().clear();
        layer
            .run_separate_shared_expert(
                input,
                m,
                h as u32,
                inter as u32,
                gate_out,
                up_out,
                down_out,
                &ctx,
                true,
                19,
            )
            .unwrap();
        let ls = gpu.launches.lock().unwrap();
        let handles: Vec<u64> = ls.iter().map(|l| l.kernel).collect();
        assert_eq!(
            handles.iter().filter(|h| **h == 777).count(),
            want_chains,
            "m={m}: fused_silu_mul count"
        );
        assert!(
            !handles.contains(&333),
            "m={m}: dequant must not run in exact mode"
        );
        assert_eq!(
            handles
                .iter()
                .filter(|h| **h == layer.moe_act_mul.0)
                .count(),
            0,
            "m={m}: unfused silu_mul must not run"
        );
        let gemvs: Vec<_> = ls
            .iter()
            .filter(|l| [111u64, 222, 444].contains(&l.kernel))
            .collect();
        assert_eq!(gemvs.len(), 2 * want_chains, "m={m}: GEMV-tier count");
        let chunk_rows: Vec<usize> = (0..m as usize)
            .step_by(32)
            .map(|f| (m as usize - f).min(32))
            .collect();
        for (c, want_rows) in chunk_rows.iter().enumerate() {
            for (half, tag) in [(0usize, "gate_up"), (1, "down")] {
                let l = gemvs[c * 2 + half];
                // The 1-row chunk launches `mxfp8_proj_gemv`, whose args are
                // (act, w, scales, out, n, k) — no m slot.
                if *want_rows == 1 {
                    assert_eq!(l.kernel, 111, "m={m} chunk{c} {tag} kernel");
                    assert_eq!(l.args.len(), 6, "m={m} chunk{c} {tag} arg count");
                } else {
                    assert_eq!(
                        l.args[4],
                        Arg::Bytes((*want_rows as u32).to_le_bytes().to_vec()),
                        "m={m} chunk{c} {tag} rows arg"
                    );
                }
            }
            if c == 1 {
                let (gu_l, dn_l) = (gemvs[2], gemvs[3]);
                assert_eq!(gu_l.args[0], Arg::Ptr(input.offset(32 * h * 2)));
                assert_eq!(gu_l.args[3], Arg::Ptr(DevicePtr(0xA000_0000)));
                assert_eq!(dn_l.args[0], Arg::Ptr(up_out.offset(32 * inter * 2)));
                assert_eq!(dn_l.args[3], Arg::Ptr(down_out.offset(32 * h * 2)));
            }
        }
        let fuses: Vec<_> = ls.iter().filter(|l| l.kernel == 777).collect();
        for (c, fl) in fuses.iter().enumerate() {
            let want_rows = chunk_rows[c] as u32;
            assert_eq!(
                fl.args[2],
                Arg::Bytes(want_rows.to_le_bytes().to_vec()),
                "m={m} chunk{c} silu rows"
            );
        }
    }
}
