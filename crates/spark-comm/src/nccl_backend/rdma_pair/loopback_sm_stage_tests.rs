// SPDX-License-Identifier: AGPL-3.0-only

//! SM-staged one-shot exchanges through the NIC and their launch-chain timing.

use super::*;

fn sizes_from_env(key: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(key)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn launch_on(f: u64, stream: u64, blocks: u32, params: &mut [*mut c_void]) {
    let null = std::ptr::null_mut();
    ck!(cuLaunchKernel(
        f,
        blocks,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        params.as_mut_ptr(),
        null
    ));
}

/// Stress of what the NIC reads from the staging buffer (two ranks through
/// the NIC, GPU + RDMA device; the staging mode comes from
/// `ATLAS_RDMA_ONESHOT_SM_STAGE`): `ATLAS_ONESHOT_STRESS_OPS` (default
/// 200,000) in-place all-reduces per rank, sizes round-robin from
/// `ATLAS_ONESHOT_STRESS_SIZES` (default 24576,8192,65536,327680). Each op's
/// payload is filled and its result checked on the GPU (`oneshot_stress.cu`
/// in the cubin dir), every element different from the previous op's, with a
/// host sync only every 4096 ops; any stale staged byte is a wrong value.
/// `ATLAS_ONESHOT_STRESS_DRAG_MB` adds an L2-churning add of that many MiB
/// before every 16th op.
#[test]
#[ignore = "needs a GPU, an RDMA device and ATLAS_ONESHOT_CUBIN_DIR with oneshot_stress.cubin"]
fn nic_loopback_sm_stage_stress() {
    let Some((g, pairs)) = ranks() else {
        return;
    };
    let fill = load("oneshot_stress.cubin", "stress_fill");
    let check = load("oneshot_stress.cubin", "stress_check");
    let ops: u64 = std::env::var("ATLAS_ONESHOT_STRESS_OPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let drag_mb: usize = std::env::var("ATLAS_ONESHOT_STRESS_DRAG_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let sizes = sizes_from_env(
        "ATLAS_ONESHOT_STRESS_SIZES",
        &[24_576, 8192, 65_536, 327_680],
    );
    let streams = [g.stream, new_stream()];
    let max = *sizes.iter().max().unwrap();
    let bufs = [alloc(max), alloc(max)];
    let drag = alloc((drag_mb << 20).max(16));
    // Wrong elements, bad blocks, lowest bad op, ops checked.
    let bad = alloc(32);
    let init = [0u64, 0, u64::MAX, 0];
    ck!(cuMemcpyHtoDAsync_v2(
        bad,
        init.as_ptr().cast(),
        32,
        g.stream
    ));
    sync(streams);
    let t0 = Instant::now();
    for k in 1..=ops {
        let bytes = sizes[(k % sizes.len() as u64) as usize];
        let n = (bytes / 2) as u32;
        let blocks = n.div_ceil(256).min(96);
        for rank in 0..2 {
            let stream = streams[rank];
            if drag_mb != 0 && k % 16 == (rank as u64) * 8 {
                launch_add(&g, stream, drag, drag, (drag_mb << 20) / 2);
            }
            let (mut p, mut n, mut r, mut k) = (bufs[rank], n, rank as u32, k);
            let mut args: [*mut c_void; 4] = [
                (&raw mut p).cast(),
                (&raw mut n).cast(),
                (&raw mut r).cast(),
                (&raw mut k).cast(),
            ];
            launch_on(fill, stream, blocks, &mut args);
            let oneshot = pairs[rank].oneshot().unwrap();
            assert!(oneshot.enqueue(p, p, bytes, true, stream).unwrap());
            let (mut add, mut b) = (1u32, bad);
            let mut args: [*mut c_void; 6] = [
                (&raw mut p).cast(),
                (&raw mut n).cast(),
                (&raw mut r).cast(),
                (&raw mut k).cast(),
                (&raw mut add).cast(),
                (&raw mut b).cast(),
            ];
            launch_on(check, stream, blocks, &mut args);
        }
        if k % 4096 == 0 {
            sync(streams);
        }
    }
    sync(streams);
    let took = t0.elapsed();
    let mut got = [0u64; 4];
    ck!(cuMemcpyDtoHAsync_v2(
        got.as_mut_ptr().cast(),
        bad,
        32,
        g.stream
    ));
    sync(streams);
    let poison: Vec<_> = pairs
        .iter()
        .map(|p| p.oneshot().unwrap().poisoned())
        .collect();
    println!(
        "SM-stage stress (sm_stage={}, pdl={}, sizes {sizes:?}, drag {drag_mb} MiB): {} ops checked of {}, \
         {:.2} us/op; {} wrong elements in {} block checks, first bad op {}; poison {poison:?}",
        std::env::var("ATLAS_RDMA_ONESHOT_SM_STAGE").unwrap_or_default(),
        std::env::var("ATLAS_RDMA_ONESHOT_PDL").unwrap_or_default(),
        got[3],
        2 * ops,
        took.as_secs_f64() * 1e6 / ops as f64,
        got[0],
        got[1],
        if got[2] == u64::MAX {
            "none".to_string()
        } else {
            got[2].to_string()
        },
    );
    assert_eq!(got[3], 2 * ops);
    assert_eq!(got[0], 0);
    assert!(poison.iter().all(Option::is_none));
}

/// Launch `bf16_add_inplace` with programmatic dependent launch, as the
/// runtime launches the kernels on its PDL list.
fn launch_add_pdl(g: &Gpu, stream: u64, dst: u64, src: u64, n: usize) {
    let (mut d, mut s, mut c) = (dst, src, n as i32);
    let mut params: [*mut c_void; 3] = [
        (&raw mut d).cast(),
        (&raw mut s).cast(),
        (&raw mut c).cast(),
    ];
    let mut value = [0u8; 64];
    value[..4].copy_from_slice(&1i32.to_ne_bytes());
    let mut attr = CuLaunchAttribute {
        id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value,
    };
    let config = CuLaunchConfig {
        grid: [(n as u32).div_ceil(256), 1, 1],
        block: [256, 1, 1],
        shared_mem_bytes: 0,
        stream,
        attrs: &mut attr,
        num_attrs: 1,
    };
    let null = std::ptr::null_mut();
    ck!(cuLaunchKernelEx(&config, g.add, params.as_mut_ptr(), null));
}

/// Microbench (stand-in peer, no NIC; the peer's data is always there): the
/// all-reduce's added cost inside a PDL chain, producer -> all-reduce ->
/// consumer (`bf16_add_inplace` both, PDL-launched), per staging mode: copy
/// engine, SM staging, SM staging + PDL. Eager and as a replayed graph of 64
/// links; the chain without the all-reduce is the baseline. Each link starts
/// with a PDL-launched add over `ATLAS_ONESHOT_CHAIN_WORK_KB` (default 1024;
/// 0 = none) standing in for the layer's other kernels, so the previous send
/// has drained (as between decode all-reduces) and only the all-reduce's own
/// critical path is added. Sizes from `ATLAS_ONESHOT_CHAIN_SIZES` (default
/// 24576,65536,327680).
#[test]
#[ignore = "needs a GPU and ATLAS_ONESHOT_CUBIN_DIR"]
fn oneshot_chain_timing() {
    let Some(g) = gpu() else {
        return;
    };
    let sync = || ck!(cuStreamSynchronize(g.stream));
    let n = 2000;
    let time = |f: &dyn Fn()| {
        f();
        sync();
        let t0 = Instant::now();
        for _ in 0..n {
            f();
        }
        sync();
        t0.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    let work_kb: usize = std::env::var("ATLAS_ONESHOT_CHAIN_WORK_KB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let work = alloc((work_kb << 10).max(16));
    let layer = || {
        if work_kb != 0 {
            launch_add_pdl(&g, g.stream, work, work, (work_kb << 10) / 2);
        }
    };
    for bytes in sizes_from_env("ATLAS_ONESHOT_CHAIN_SIZES", &[24_576, 65_536, 327_680]) {
        let half = bytes / 2;
        let (x, buf, y) = (alloc(bytes), alloc(bytes), alloc(bytes));
        let base_link = || {
            layer();
            launch_add_pdl(&g, g.stream, buf, x, half);
            launch_add_pdl(&g, g.stream, y, buf, half);
        };
        let links = 64;
        let base_exec = capture(&g, || (0..links).for_each(|_| base_link()));
        let base = time(&base_link);
        let base_graph = time(&|| ck!(cuGraphLaunch(base_exec, g.stream))) / links as f64;
        let mut line = format!(
            "\nchain {bytes:>7} B, work {work_kb} KiB: baseline eager {base:.2} us/link, graph {base_graph:.2}; all-reduce adds"
        );
        for (name, sm_stage, pdl) in [
            ("copy engine", 0, false),
            ("SM", 1, false),
            ("SM+PDL", 1, true),
        ] {
            let cfg = Config {
                sm_stage,
                pdl,
                stage_fence: false,
                ..test_cfg(SPLIT_MIN)
            };
            let os = channel(&g, cfg, 1);
            let stop = Arc::new(AtomicBool::new(false));
            let peer = spawn_peer(&os, Peer::Early(bytes), stop.clone());
            let link = || {
                layer();
                launch_add_pdl(&g, g.stream, buf, x, half);
                assert!(os.enqueue(buf, buf, bytes, true, g.stream).unwrap());
                launch_add_pdl(&g, g.stream, y, buf, half);
            };
            let eager = time(&link);
            let exec = capture(&g, || (0..links).for_each(|_| link()));
            let graph = time(&|| ck!(cuGraphLaunch(exec, g.stream))) / links as f64;
            stop.store(true, Ordering::Release);
            peer.join().unwrap();
            assert!(os.poisoned().is_none());
            line += &format!(
                "; {name}: eager +{:.2} us, graph +{:.2} us",
                eager - base,
                graph - base_graph
            );
        }
        println!("{line}");
    }
}
