// SPDX-License-Identifier: AGPL-3.0-only
//! ATLAS_GLM_MOE_DECODE_M16 / _K128W / _STREAM dispatch through the actual
//! verify FFN entries; the numerics are gated on hardware by
//! scripts/moe-decode-bench.
use super::*;
use crate::layers::moe::decode_m16::{MAX_ROWS, STREAM_MAX_ROWS, m16_shape};

#[test]
fn m16_tiles_cover_their_row_slabs_of_k128_aligned_experts() {
    for max_rows in [MAX_ROWS, STREAM_MAX_ROWS] {
        for rows in 1..=max_rows {
            assert!(m16_shape(rows, max_rows, 4096, 1024));
            assert!(m16_shape(rows, max_rows, 4096, 2048));
        }
        // More rows than the slabs, or none: the M64 kernels keep the batch.
        assert!(!m16_shape(0, max_rows, 4096, 1024));
        assert!(!m16_shape(max_rows + 1, max_rows, 4096, 1024));
    }
    assert_eq!(STREAM_MAX_ROWS, 2 * MAX_ROWS);
    // Gate/up tiles are 128 columns wide, down tiles 256.
    assert!(!m16_shape(8, MAX_ROWS, 4096, 1000));
    assert!(!m16_shape(8, MAX_ROWS, 4096 + 128, 1024));
}

/// How a routed FFN batch reaches the MoE.
#[derive(Clone, Copy, PartialEq)]
enum Entry {
    /// One verify block through `forward_independent`.
    Independent,
    /// 3-row verify blocks of several owners through the C3 grouped
    /// `forward_prefill`.
    Owner,
    /// A plain `forward_prefill` batch: not verify decode.
    Prefill,
}

fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// One FFN of `rows` rows through `entry`, with every expert local at half
/// the width (`expert_tp`, the served topology) or this rank's half of the
/// experts at full width; checks the launches the flags swap and returns the
/// trace.
fn verify_ffn(
    gpu: &Gpu,
    rank: usize,
    vector: bool,
    expert_tp: bool,
    entry: Entry,
    rows: usize,
) -> Vec<Event> {
    let (_store, config, mut layer) = resident_tests::setup_with(gpu, rank, expert_tp);
    layer
        .transpose_for_prefill_unified_keep_shared(gpu, &config)
        .unwrap();
    layer.unified_layout = true;
    layer.nvfp4_vecscale = vector;
    layer.nvfp4_fused_silu_quant = true;
    let arena = BufferArena::new(&config, 8, 2048, 16, 8, gpu).unwrap();
    let metadata = gpu.alloc(512).unwrap();
    let meta = AttnMetadataDev {
        positions: metadata,
        positions_h: metadata,
        positions_w: metadata,
        slot: metadata.offset(32),
        seq_len: metadata.offset(64),
        block_table: metadata.offset(96),
        max_blocks_per_seq: 1,
        num_seqs: rows as u32,
        seq_slot: metadata,
        moe_row_adapter: DevicePtr::NULL,
    };
    let resources = ContextResources::new();
    let mut levers = ops::ModelLevers::defaults();
    levers.max_decode_seqs = 8;
    let comm = Comm {
        gpu,
        rank,
        reductions: Mutex::new(vec![]),
    };
    let mut ctx = resources.view(&arena, &config, gpu);
    ctx.levers = &levers;
    ctx.comm = Some(&comm);
    ctx.attn_metadata = Some(meta);
    let h = [gpu.alloc(8 * 2097152).unwrap()];
    let conv = [gpu.alloc(8 * 196608).unwrap()];
    let pool =
        crate::layer::ssm_batch::SsmPoolView::new(&h, &conv, 2097152, 2097152, 196608, 8).unwrap();
    let slots = [7, 0, 6, 1, 5, 2, 4, 3];
    if entry == Entry::Independent {
        ctx.ssm_batch = Some(
            crate::layer::ssm_batch::SsmBatchView::new(pool, metadata.offset(256), &slots[..rows])
                .unwrap(),
        );
    }
    let (up_out, scratch) = (arena.expert_up_out(), arena.moe_router_in_f32());
    gpu.clear();
    match entry {
        Entry::Independent => {
            layer
                .forward_independent(arena.norm_output(), rows, &ctx, 91)
                .unwrap();
        }
        Entry::Owner => crate::layers::moe::with_owner_rows(rows as u32, || {
            layer.forward_prefill(arena.norm_output(), rows, &ctx, 91)
        })
        .unwrap(),
        Entry::Prefill => layer
            .forward_prefill(arena.norm_output(), rows, &ctx, 91)
            .unwrap(),
    }
    let trace = gpu.trace();
    // The launches the flags swap, by kernel.
    let launches = |name: &str| -> Vec<([u32; 3], [u32; 3], Vec<Arg>)> {
        let Ok(handle) = gpu.kernel("moe_w4a16", name).map(|k| k.0) else {
            return vec![];
        };
        trace
            .iter()
            .filter_map(|e| match e {
                Event::Launch(h, grid, block, _, 91, args) if *h == handle => {
                    Some((*grid, *block, args.clone()))
                }
                _ => None,
            })
            .collect()
    };
    let position = |name: &str| {
        let handle = gpu.kernel("moe_w4a16", name).unwrap().0;
        trace
            .iter()
            .position(|e| matches!(e, Event::Launch(h, ..) if *h == handle))
    };
    let compact = if vector {
        "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up"
    } else {
        "moe_w4a4_grouped_gemm_prequant_t_k64_compact_gate_up"
    };
    // The worklist builder's launches, by their N tiles per row tile.
    let builder: Vec<Arg> = trace
        .iter()
        .filter_map(|e| match e {
            Event::Launch(h, _, _, _, 91, args) if *h == layer.moe_build_tile_worklist_k.0 => {
                Some(args[5].clone())
            }
            _ => None,
        })
        .collect();
    let n_tiles = |n: u32| vec![Arg::Bytes(n.to_ne_bytes().to_vec())];
    let silu = trace
        .iter()
        .filter(|e| matches!(e, Event::Launch(h, ..) if *h == layer.silu_mul_quant_nvfp4_k.0))
        .count();
    // Whether the target lacks a kernel whose name passes `kind`.
    let lacks = |kind: fn(&str) -> bool| gpu.missing.lock().unwrap().iter().any(|n| kind(n));
    // ATLAS_GLM_MOE_DECODE_STREAM (with the M16 flag) swaps in the stream
    // twins, two slabs above 16 rows, unless the target lacks one of them;
    // ATLAS_GLM_MOE_DOWN_ZSKIP only the down.
    let stream = env_on("ATLAS_GLM_MOE_DECODE_M16")
        && env_on("ATLAS_GLM_MOE_DECODE_STREAM")
        && !lacks(|n| !n.ends_with("_l2pf") && !n.ends_with("_l2pf_p"));
    let family = match (stream, rows as u32 <= MAX_ROWS) {
        (false, _) => "m16",
        (true, true) => "m16s",
        (true, false) => "m32s",
    };
    let skip = if env_on("ATLAS_GLM_MOE_DOWN_ZSKIP") {
        "_zskip"
    } else {
        ""
    };
    // ATLAS_GLM_MOE_DECODE_L2PF (with the stream twins) swaps in their
    // prefetching twins (=2: the gate/up only), unless the target lacks one.
    let l2pf = std::env::var("ATLAS_GLM_MOE_DECODE_L2PF").ok();
    let prefetch =
        stream && matches!(l2pf.as_deref(), Some("1" | "2")) && !lacks(|n| n.ends_with("_l2pf"));
    // ATLAS_GLM_MOE_DECODE_PERSIST (with L2PF=1 and the skip) swaps in the
    // persistent prefetching twins over one CTA per SM, unless the target
    // lacks one.
    let persist = prefetch
        && l2pf.as_deref() == Some("1")
        && env_on("ATLAS_GLM_MOE_DOWN_ZSKIP")
        && env_on("ATLAS_GLM_MOE_DECODE_PERSIST")
        && !lacks(|n| n.ends_with("_l2pf_p"));
    let suffix = |on: bool| match (on, persist) {
        (false, _) => "",
        (true, false) => "_l2pf",
        (true, true) => "_l2pf_p",
    };
    let gate_up_name = format!(
        "glm_moe_decode_{family}_gate_up_silu_k128w{}",
        suffix(prefetch)
    );
    let down_name = format!(
        "glm_moe_decode_{family}_k128w{skip}{}",
        suffix(prefetch && l2pf.as_deref() == Some("1"))
    );
    let [gate_up, down] = [gate_up_name.as_str(), down_name.as_str()].map(launches);
    // No other twin launches.
    for other in ["m16", "m16s", "m32s"] {
        for kernel in [
            "gate_up_silu_k128w",
            "k128w",
            "k128w_zskip",
            "gate_up_silu_k128w_l2pf",
            "k128w_l2pf",
            "k128w_zskip_l2pf",
            "gate_up_silu_k128w_l2pf_p",
            "k128w_zskip_l2pf_p",
        ] {
            let name = format!("glm_moe_decode_{other}_{kernel}");
            if name != gate_up_name && name != down_name {
                assert!(launches(&name).is_empty(), "{name}");
            }
        }
    }
    let k128w = [
        "moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w",
        "moe_w4a4_grouped_gemm_prequant_t_k128w_compact",
    ]
    .map(launches);
    let inter = config.routed_inter_local() as u32;
    let routed = rows as u32 * 8;
    // Gate/up tiles are 128 columns of the rank's width, down tiles 256 of H.
    let grids = |bound: u32| [[inter / 128, bound, 1], [16, bound, 1]];
    let decode = entry != Entry::Prefill;
    // The stream twins take every batch they fit, verify decode or not.
    let max_rows = if stream { STREAM_MAX_ROWS } else { MAX_ROWS };
    if (decode || stream) && env_on("ATLAS_GLM_MOE_DECODE_M16") && rows as u32 <= max_rows {
        // One tile per routed expert, at most one expert per routed row.
        assert_eq!((gate_up.len(), down.len()), (1, 1));
        // One worklist item per routed local expert, its count at the base.
        assert_eq!(builder, n_tiles(1));
        let grid = if persist {
            [[48, 1, 1]; 2]
        } else {
            grids(routed.min(288))
        };
        assert_eq!([gate_up[0].0, down[0].0], grid);
        assert_eq!([gate_up[0].1, down[0].1], [[256, 1, 1]; 2]);
        // The K128W argument lists: 17 fused gate/up, 12 down, the worklist
        // scratch where they take the prefix.
        assert_eq!((gate_up[0].2.len(), down[0].2.len()), (17, 12));
        assert_eq!(gate_up[0].2[11], Arg::Ptr(scratch));
        assert_eq!(down[0].2[11], Arg::Ptr(scratch));
        // Gate/up leaves the NVFP4 SiLU product where the down reads it.
        let scales = up_out.offset((routed * inter / 2) as usize);
        assert_eq!(gate_up[0].2[15..], [Arg::Ptr(up_out), Arg::Ptr(scales)]);
        assert_eq!(down[0].2[..2], [Arg::Ptr(up_out), Arg::Ptr(scales)]);
        assert!(position(&gate_up_name) < position(&down_name));
        // Nothing of the M64 paths runs.
        assert_eq!((silu, launches(compact).len()), (0, 0));
        assert_eq!((k128w[0].len(), k128w[1].len()), (0, 0));
        for dense in [
            "moe_w4a4_grouped_gemm_prequant_t_k128",
            "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale",
            "moe_w4a4_grouped_gemm_prequant_t_k64",
        ] {
            assert!(launches(dense).is_empty(), "{dense}");
        }
    } else {
        // No twin outside verify decode or without the flag.
        assert_eq!((gate_up.len(), down.len()), (0, 0));
        if !decode || env_on("ATLAS_GLM_MOE_DECODE_K128W") {
            // The prefill K128W pair over the M64 row-tile prefix.
            assert_eq!(builder, vec![]);
            assert_eq!((k128w[0].len(), k128w[1].len()), (1, 1));
            assert_eq!(
                [k128w[0][0].0, k128w[1][0].0],
                grids(routed.div_ceil(64) + 288)
            );
            assert_eq!(launches("moe_mtile_prefix").len(), 1);
            assert_eq!((silu, launches(compact).len()), (0, 0));
        } else {
            // The base launches.
            assert_eq!(builder, n_tiles(inter / 128));
            assert_eq!((silu, launches(compact).len()), (1, 1));
            assert_eq!((k128w[0].len(), k128w[1].len()), (0, 0));
            assert_eq!(launches("moe_w4a4_grouped_gemm_prequant_t_k128").len(), 1);
        }
    }
    if entry == Entry::Independent {
        assert_eq!(
            comm.reductions
                .lock()
                .unwrap()
                .iter()
                .map(|r| r.0)
                .collect::<Vec<_>>(),
            vec![rows * 8192]
        );
    }
    trace
}

#[test]
fn verify_ffn_launches_the_m16_twins_only_with_the_flag() {
    const SENTINEL: &str = "ATLAS_TEST_DECODE_M16";
    const MISSING: &str = "ATLAS_TEST_DECODE_M16_MISSING";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(
            module_path!(),
            "::verify_ffn_launches_the_m16_twins_only_with_the_flag"
        );
        let name = name.split_once("::").unwrap().1;
        // Flags M16, ZSKIP, K128W, STREAM, MISSING (a lacking kernel), L2PF,
        // PERSIST. Off; M16 with and without the skip, the skip alone, the
        // K128W control, both; the stream twins alone, with M16, the skip,
        // K128W, and on targets lacking a stream kernel; the prefetch twins
        // (1, 2: gate/up only) with and without the skip or stream twins and
        // lacking a prefetching down; the persistent twins with L2PF=1 and the
        // skip, without either or the prefetch twins, alone, and lacking a
        // persistent down. Ignored flags keep what runs without them.
        for mode in [
            "0000000", "1000000", "1100000", "0100000", "0010000", "1010000", "0001000", "1001000",
            "1101000", "1011000", "1001100", "1101200", "1001010", "1101010", "1101020", "1001020",
            "1000010", "0000010", "1101310", "1101011", "1001011", "1101021", "1101001", "0000001",
            "1101411",
        ] {
            let flag = |i: usize| &mode[i..=i];
            let mut cmd = ffn_child(name, SENTINEL, mode);
            cmd.env("ATLAS_GLM_MOE_DECODE_M16", flag(0))
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", flag(1))
                .env("ATLAS_GLM_MOE_DECODE_K128W", flag(2))
                .env("ATLAS_GLM_MOE_DECODE_STREAM", flag(3))
                .env(MISSING, flag(4))
                .env("ATLAS_GLM_MOE_DECODE_L2PF", flag(5))
                .env("ATLAS_GLM_MOE_DECODE_PERSIST", flag(6))
                .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
                .env("ATLAS_GLM_C3_GROUPED_MOE", "1")
                .env("ATLAS_MOE_PREQUANT_K128", "1")
                // Batches of 9 rows or more otherwise read the expert offsets
                // back for exact M64 tiles; the recording backend serves none.
                .env("ATLAS_MOE_PREFILL_EXACT_TILES", "0");
            assert_child_passed(mode, cmd);
        }
        return;
    }
    // One verify block of 2..=8 rows, 3-row blocks of four owners, and
    // prefill batches of 12, 24 and 40 rows, which only the stream twins take
    // (up to 32 rows).
    let batches = (2..=8).map(|rows| (Entry::Independent, rows)).chain([
        (Entry::Owner, 12),
        (Entry::Prefill, 12),
        (Entry::Prefill, 24),
        (Entry::Prefill, 40),
    ]);
    for (entry, rows) in batches {
        for (rank, vector, expert_tp) in [
            (0, false, false),
            (1, true, false),
            (0, true, true),
            (1, false, true),
        ] {
            let gpu = Gpu::new();
            let missing = match std::env::var(MISSING).as_deref() {
                Ok("1") => Some("glm_moe_decode_m32s_gate_up_silu_k128w"),
                Ok("2") => Some("glm_moe_decode_m16s_k128w_zskip"),
                Ok("3") => Some("glm_moe_decode_m32s_k128w_zskip_l2pf"),
                Ok("4") => Some("glm_moe_decode_m32s_k128w_zskip_l2pf_p"),
                _ => None,
            };
            gpu.missing
                .lock()
                .unwrap()
                .extend(missing.map(String::from));
            let trace = verify_ffn(&gpu, rank, vector, expert_tp, entry, rows);
            // Graph-safe either way: no allocation, free or host read.
            assert!(
                !trace
                    .iter()
                    .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..))),
                "{rows} rows"
            );
        }
    }
}

/// The test binary re-run as `name` alone with `sentinel` set to `mode` and
/// the routing every FFN test pins.
fn ffn_child(name: &str, sentinel: &str, mode: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", name, "--nocapture"])
        .env(sentinel, mode)
        .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
        .env("ATLAS_GLM_C3_GROUPED_MOE", "1")
        .env("ATLAS_MOE_PREQUANT_K128", "1")
        .env("ATLAS_EP_PROTOCOL", "v2");
    for off in [
        "ATLAS_NVFP4_PREQUANT_MOE",
        "ATLAS_NVFP4_FUSED_SILU_QUANT",
        "ATLAS_GLM_MOE_GATE_UP_M16",
        "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
        "ATLAS_GLM_C2_COMPACT_MOE",
        "ATLAS_GLM_C4_GROUPED_MOE",
        "ATLAS_GLM_K5_COMPACT_MOE",
        "ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP",
        "ATLAS_GLM_K5_HC_CUBLAS",
        "ATLAS_GLM_MOE_PREFILL_PERSIST",
    ] {
        cmd.env(off, "0");
    }
    cmd
}

fn assert_child_passed(mode: &str, mut cmd: std::process::Command) {
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "mode{mode}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

/// ATLAS_GLM_MOE_STREAM_NOSYNC: with the exact M64 tile count at its NVFP4
/// default, a batch the stream twins take launches exactly what it launches
/// without the count (`verify_ffn`'s checks) and reads nothing back; without
/// the switch it still reads the expert offsets (the recording backend fails
/// that read), and batches the twins do not take read them either way.
#[test]
fn stream_batches_read_no_expert_offsets_with_nosync() {
    const SENTINEL: &str = "ATLAS_TEST_STREAM_NOSYNC";
    let Ok(nosync) = std::env::var(SENTINEL) else {
        let name = concat!(
            module_path!(),
            "::stream_batches_read_no_expert_offsets_with_nosync"
        );
        let name = name.split_once("::").unwrap().1;
        for nosync in ["0", "1"] {
            let mut cmd = ffn_child(name, SENTINEL, nosync);
            cmd.env("ATLAS_GLM_MOE_DECODE_M16", "1")
                .env("ATLAS_GLM_MOE_DOWN_ZSKIP", "1")
                .env("ATLAS_GLM_MOE_DECODE_STREAM", "1")
                .env("ATLAS_GLM_MOE_STREAM_NOSYNC", nosync)
                .env_remove("ATLAS_MOE_PREFILL_EXACT_TILES");
            assert_child_passed(nosync, cmd);
        }
        return;
    };
    let reads = |entry, rows| {
        std::panic::catch_unwind(|| {
            let gpu = Gpu::new();
            let trace = verify_ffn(&gpu, 0, false, true, entry, rows);
            trace
                .iter()
                .filter(|e| matches!(e, Event::Read(..)))
                .count()
        })
    };
    // Batches of 9 to 32 rows: the owner-batched verify of two to four streams.
    for (entry, rows) in [
        (Entry::Owner, 12),
        (Entry::Prefill, 12),
        (Entry::Prefill, 24),
    ] {
        match nosync.as_str() {
            "1" => assert_eq!(reads(entry, rows).ok(), Some(0), "{rows} rows"),
            _ => assert!(reads(entry, rows).is_err(), "{rows} rows read nothing"),
        }
    }
    // Wider than the twins: the M64 grid needs the count.
    assert!(reads(Entry::Prefill, 40).is_err());
}

#[test]
fn decode_flags_require_0_or_1() {
    const SENTINEL: &str = "ATLAS_TEST_DECODE_M16_TOGGLE";
    let Some(flag) = std::env::var_os(SENTINEL) else {
        let name = concat!(module_path!(), "::decode_flags_require_0_or_1");
        for flag in [
            "ATLAS_GLM_MOE_DECODE_M16",
            "ATLAS_GLM_MOE_DOWN_ZSKIP",
            "ATLAS_GLM_MOE_DECODE_K128W",
            "ATLAS_GLM_MOE_DECODE_STREAM",
            "ATLAS_GLM_MOE_DECODE_L2PF",
            "ATLAS_GLM_MOE_DECODE_PERSIST",
            "ATLAS_GLM_MOE_STREAM_NOSYNC",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name.split_once("::").unwrap().1])
                .env(SENTINEL, flag)
                .env(flag, "true")
                .output()
                .unwrap();
            assert!(output.status.success(), "{flag}");
        }
        return;
    };
    let gpu = Gpu::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    let error = crate::layers::moe::decode_m16::DecodeM16::new(&gpu, &config)
        .err()
        .expect("a toggle other than 0 or 1 must fail the load");
    assert!(error.to_string().contains(flag.to_str().unwrap()));
}
