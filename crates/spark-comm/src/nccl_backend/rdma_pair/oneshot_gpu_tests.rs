// SPDX-License-Identifier: AGPL-3.0-only

//! Single-GPU loopback tests of the one-shot protocol and kernel (GPU only,
//! `#[ignore]`d). A host thread stands in for the proxy and both NICs: it
//! waits for `stage`, checks the staged bytes are this op's payload, writes
//! the peer's payload into receive slot `seq % 2` and the per-rail flags, then
//! clears `stage`. Landed results must equal `bf16_add_inplace` bit for bit
//! (add) or the peer's payload (copy), for eager calls and CUDA-graph replays
//! interleaved on one sequence.
//!
//!   ATLAS_ONESHOT_CUBIN_DIR=<dir holding rdma_oneshot.cubin, bf16_add.cubin> \
//!     cargo test -p spark-comm oneshot -- --ignored --test-threads=1 --nocapture
//!
//! `ATLAS_ONESHOT_FAULT=<fault>` makes `oneshot_trap_*` fire the kernel's
//! trap; it kills the CUDA context, so run it alone in its own process.

use super::*;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDevicePrimaryCtxRetain(ctx: *mut u64, dev: i32) -> i32;
    fn cuCtxSetCurrent(ctx: u64) -> i32;
    fn cuModuleLoad(module: *mut u64, path: *const std::ffi::c_char) -> i32;
    fn cuModuleGetFunction(f: *mut u64, module: u64, name: *const std::ffi::c_char) -> i32;
    fn cuStreamCreate(stream: *mut u64, flags: u32) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: u32) -> i32;
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut u64, p: *mut c_void, flags: u32) -> i32;
    fn cuStreamBeginCapture_v2(stream: u64, mode: u32) -> i32;
    fn cuStreamEndCapture(stream: u64, graph: *mut u64) -> i32;
    fn cuGraphInstantiateWithFlags(exec: *mut u64, graph: u64, flags: u64) -> i32;
    fn cuGraphLaunch(exec: u64, stream: u64) -> i32;
}

/// A driver call that must succeed.
macro_rules! ck {
    ($call:expr) => {
        cu(unsafe { $call }, stringify!($call)).unwrap()
    };
}

/// Capture what `enqueue` puts on the test stream into an instantiated graph.
fn capture(g: &Gpu, enqueue: impl Fn()) -> u64 {
    let (mut graph, mut exec) = (0u64, 0u64);
    ck!(cuStreamBeginCapture_v2(g.stream, 1));
    enqueue();
    ck!(cuStreamEndCapture(g.stream, &mut graph));
    ck!(cuGraphInstantiateWithFlags(&mut exec, graph, 0));
    exec
}

struct Gpu {
    ctx: u64,
    stream: u64,
    oneshot: u64,
    add: u64,
}

fn new_stream() -> u64 {
    let mut stream = 0u64;
    ck!(cuStreamCreate(&mut stream, 1));
    stream
}

/// The CUDA context and kernels, or `None` (test skipped) without a cubin dir.
fn gpu() -> Option<Gpu> {
    let dir = std::env::var("ATLAS_ONESHOT_CUBIN_DIR").ok()?;
    let load = |file: &str, name: &str| {
        let path = std::ffi::CString::new(format!("{dir}/{file}")).unwrap();
        let name = std::ffi::CString::new(name).unwrap();
        let (mut module, mut f) = (0u64, 0u64);
        ck!(cuModuleLoad(&mut module, path.as_ptr()));
        ck!(cuModuleGetFunction(&mut f, module, name.as_ptr()));
        f
    };
    let mut ctx = 0u64;
    ck!(cuInit(0));
    ck!(cuDevicePrimaryCtxRetain(&mut ctx, 0));
    ck!(cuCtxSetCurrent(ctx));
    Some(Gpu {
        ctx,
        stream: new_stream(),
        oneshot: load("rdma_oneshot.cubin", "rdma_oneshot_bf16"),
        add: load("bf16_add.cubin", "bf16_add_inplace"),
    })
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 31)).wrapping_mul(0x7fb5_d329_728e_a185);
    x = (x ^ (x >> 27)).wrapping_mul(0x81da_def4_bc2d_d44d);
    x ^ (x >> 33)
}

/// This rank's payload for op `seq`: every BF16 bit pattern (NaN, inf,
/// subnormal, -0 included) on every fifth op, pseudo-random otherwise.
fn local_payload(seq: u64, n: usize) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            if seq.is_multiple_of(5) {
                i as u16
            } else {
                mix(seq << 32 | i) as u16
            }
        })
        .collect()
}

fn peer_payload(seq: u64, n: usize) -> Vec<u16> {
    (0..n as u64).map(|i| mix(!seq << 32 | i) as u16).collect()
}

fn bytes_of(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding; the byte view covers the same allocation.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 2) }
}

fn upload(g: &Gpu, dst: u64, v: &[u16]) {
    ck!(cuMemcpyHtoDAsync_v2(
        dst,
        v.as_ptr().cast(),
        v.len() * 2,
        g.stream
    ));
}

fn download(g: &Gpu, src: u64, n: usize) -> Vec<u16> {
    let mut v = vec![0u16; n];
    ck!(cuMemcpyDtoHAsync_v2(
        v.as_mut_ptr().cast(),
        src,
        n * 2,
        g.stream
    ));
    ck!(cuStreamSynchronize(g.stream));
    v
}

fn alloc(bytes: usize) -> u64 {
    let mut p = 0u64;
    ck!(cuMemAlloc_v2(&mut p, bytes.max(16)));
    p
}

/// A one-shot channel over a fresh pinned region (no NIC).
fn channel(g: &Gpu, cfg: Config, rails: usize) -> OneShot {
    let (mut host, mut dev) = (std::ptr::null_mut(), 0u64);
    let bytes = region_bytes(cfg.max);
    ck!(cuMemHostAlloc(&mut host, bytes, 0x3));
    unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
    ck!(cuMemHostGetDevicePointer_v2(&mut dev, host, 0));
    let os = OneShot::new(cfg, rails, host as usize, dev).unwrap();
    os.set_kernel(g.oneshot);
    os
}

/// How the stand-in peer behaves.
#[derive(Clone, Copy)]
enum Peer {
    /// Check the staged payload, then deliver the peer payload for `seq`.
    Verify,
    /// Timing: deliver op `seq + 1` (all ops `bytes` long) as soon as op
    /// `seq` is staged, so each kernel finds its flags already set -- the
    /// late rank's critical path. Payload contents are not rewritten.
    Early(usize),
    /// Flag every op with the wrong size (the kernel must trap).
    Lie,
    /// Flag every op as two ops later (the kernel must trap).
    Ahead,
}

/// Run the stand-in proxy + peer until `stop`; returns staged-payload
/// mismatches (bytes of `S` that were not this op's payload).
fn spawn_peer(os: &OneShot, mode: Peer, stop: Arc<AtomicBool>) -> std::thread::JoinHandle<u64> {
    let (cfg, rails, host) = (os.cfg, os.rails, os.host);
    std::thread::spawn(move || {
        let ctrl = host + ctrl_off(cfg.max);
        // SAFETY: the region outlives the thread (joined before it is dropped).
        let stage = unsafe { &*((ctrl + STAGE) as *const AtomicU32) };
        let flag = |r: usize| unsafe { &*((ctrl + FLAGS + 64 * r) as *const AtomicU64) };
        let deliver = |seq: u64, bytes: usize, data: bool| {
            if data {
                let peer = peer_payload(seq, bytes / 2);
                let slot = (host + recv_off(cfg.max, (seq & 1) as usize)) as *mut u8;
                unsafe { std::ptr::copy_nonoverlapping(bytes_of(&peer).as_ptr(), slot, bytes) };
            }
            let word = match mode {
                Peer::Lie => flag_word(seq, bytes + 2),
                Peer::Ahead => flag_word(seq + 2, bytes),
                _ => flag_word(seq, bytes),
            };
            for r in 0..stripes(bytes, rails, cfg.stripe_min).len() {
                flag(r).store(word, Ordering::Release);
            }
        };
        let (mut served, mut bad) = (0u64, 0u64);
        if let Peer::Early(bytes) = mode {
            deliver(1, bytes, true);
            deliver(2, bytes, true);
        }
        while !stop.load(Ordering::Acquire) {
            let bytes = stage.load(Ordering::Acquire) as usize;
            if bytes == 0 {
                std::hint::spin_loop();
                continue;
            }
            let seq = served + 1;
            match mode {
                Peer::Verify | Peer::Lie | Peer::Ahead => {
                    let want = local_payload(seq, bytes / 2);
                    let staged = unsafe { std::slice::from_raw_parts(host as *const u8, bytes) };
                    bad += staged
                        .iter()
                        .zip(bytes_of(&want))
                        .filter(|(a, b)| a != b)
                        .count() as u64;
                    deliver(seq, bytes, true);
                }
                // Op `seq + 1` goes in slot (seq + 1) % 2, last read by kernel
                // `seq - 1`, which finished before this stage.
                Peer::Early(_) if seq > 1 => deliver(seq + 1, bytes, false),
                Peer::Early(_) => {}
            }
            served = seq;
            stage.store(0, Ordering::Release);
        }
        bad
    })
}

struct Op {
    bytes: usize,
    add: bool,
    /// Byte offset of `dst` from a 256-byte-aligned allocation (2 exercises
    /// the kernel's unaligned path).
    misalign: usize,
    src: u64,
    dst: u64,
    /// Reference buffers: local copy and peer payload.
    want: [u64; 2],
}

impl Op {
    fn new(bytes: usize, add: bool, misalign: usize) -> Self {
        let dst = alloc(bytes + 256) + misalign as u64;
        let src = if add { dst } else { alloc(bytes) };
        let want = [alloc(bytes), alloc(bytes)];
        Self {
            bytes,
            add,
            misalign,
            src,
            dst,
            want,
        }
    }

    fn run(&self, os: &OneShot, g: &Gpu) {
        assert!(
            os.enqueue(self.src, self.dst, self.bytes, self.add, g.stream)
                .unwrap()
        );
    }

    /// Load op `seq`'s inputs (stream-ordered before the op).
    fn prepare(&self, g: &Gpu, seq: u64) {
        let n = self.bytes / 2;
        upload(g, self.src, &local_payload(seq, n));
        if !self.add {
            upload(g, self.dst, &vec![0xffff; n]);
        }
    }

    /// Compare the landed result of op `seq` against the reference:
    /// `bf16_add_inplace(local, peer)` on the GPU (add) or the peer payload.
    fn check(&self, g: &Gpu, seq: u64) -> usize {
        let n = self.bytes / 2;
        let got = download(g, self.dst, n);
        let want = if self.add {
            let [r, p] = self.want;
            upload(g, r, &local_payload(seq, n));
            upload(g, p, &peer_payload(seq, n));
            launch_add(g, g.stream, r, p, n);
            download(g, r, n)
        } else {
            peer_payload(seq, n)
        };
        got.iter().zip(&want).filter(|(a, b)| a != b).count()
    }
}

fn launch_add(g: &Gpu, stream: u64, dst: u64, src: u64, n: usize) {
    let (mut d, mut s, mut c) = (dst, src, n as i32);
    let mut params: [*mut c_void; 3] = [
        (&raw mut d).cast(),
        (&raw mut s).cast(),
        (&raw mut c).cast(),
    ];
    let blocks = (n as u32).div_ceil(256);
    let null = std::ptr::null_mut();
    ck!(cuLaunchKernel(
        g.add,
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

fn test_cfg(stripe_min: usize) -> Config {
    Config {
        max: 1 << 20,
        stripe_min,
        timeout_ns: 10_000_000_000,
        stage_fence: std::env::var("ATLAS_RDMA_ONESHOT_STAGE_FENCE").as_deref() == Ok("1"),
    }
}

/// Eager ops of every shape, then a captured graph replayed between eager
/// ops, all on one sequence; every result bit-exact.
fn run_protocol(rails: usize, stripe_min: usize) {
    let Some(g) = gpu() else {
        eprintln!("ATLAS_ONESHOT_CUBIN_DIR unset: skipped");
        return;
    };
    let os = channel(&g, test_cfg(stripe_min), rails);
    let stop = Arc::new(AtomicBool::new(false));
    let peer = spawn_peer(&os, Peer::Verify, stop.clone());
    let ops = [
        Op::new(131_072, true, 0),
        Op::new(65_536, true, 0),
        Op::new(8192, true, 0),
        Op::new(327_680, true, 0),
        Op::new(1 << 20, true, 0),
        Op::new(2, true, 0),
        Op::new(18, true, 2),
        Op::new(65_538, true, 2),
        Op::new(128, false, 0),
        Op::new(54, false, 0),
        Op::new(65_536, false, 2),
    ];
    let (mut seq, mut wrong) = (0u64, 0usize);
    for op in &ops {
        seq += 1;
        op.prepare(&g, seq);
        op.run(&os, &g);
        let bad = op.check(&g, seq);
        if bad != 0 {
            eprintln!(
                "eager op {seq} ({} B add={} misalign={}): {bad} wrong",
                op.bytes, op.add, op.misalign
            );
        }
        wrong += bad;
    }
    // Capture every op once, then replay between eager calls.
    let exec = capture(&g, || ops.iter().for_each(|op| op.run(&os, &g)));
    for replay in 0..40 {
        let base = seq;
        for (k, op) in ops.iter().enumerate() {
            op.prepare(&g, base + 1 + k as u64);
        }
        ck!(cuGraphLaunch(exec, g.stream));
        seq += ops.len() as u64;
        for (k, op) in ops.iter().enumerate() {
            let bad = op.check(&g, base + 1 + k as u64);
            if bad != 0 {
                eprintln!("replay {replay} node {k}: {bad} wrong");
            }
            wrong += bad;
        }
        let op = &ops[replay % ops.len()];
        seq += 1;
        op.prepare(&g, seq);
        op.run(&os, &g);
        wrong += op.check(&g, seq);
    }
    stop.store(true, Ordering::Release);
    let staged_bad = peer.join().unwrap();
    println!(
        "one-shot loopback rails={rails}: {seq} ops, {wrong} wrong results, {staged_bad} stale staged bytes"
    );
    assert_eq!((wrong, staged_bad), (0, 0));
    assert!(os.poisoned().is_none());
    // Ineligible sizes enqueue nothing and leave the sequence alone.
    assert!(
        !os.enqueue(ops[0].src, ops[0].dst, 3, true, g.stream)
            .unwrap()
    );
    assert!(
        !os.enqueue(ops[0].src, ops[0].dst, (1 << 20) + 2, true, g.stream)
            .unwrap()
    );
}

#[test]
#[ignore = "needs a GPU and ATLAS_ONESHOT_CUBIN_DIR"]
fn oneshot_loopback_eager_and_graph_one_rail() {
    run_protocol(1, SPLIT_MIN);
}

#[test]
#[ignore = "needs a GPU and ATLAS_ONESHOT_CUBIN_DIR"]
fn oneshot_loopback_eager_and_graph_striped() {
    run_protocol(2, 4096);
}

/// Every fault must poison the channel and trap, never hang. By
/// `ATLAS_ONESHOT_FAULT`: `timeout` (no peer), `mismatch` (the peer flags
/// another size), `desync` (its flag is two ops ahead), `proxy` (the proxy
/// failed earlier), `proxy-stage` / `proxy-spin` (it fails while the stream
/// waits for `stage` / the kernel for the peer), `launch` (a launch fails
/// after its fenced stage).
#[test]
#[ignore = "kills the CUDA context: run alone with ATLAS_ONESHOT_FAULT"]
fn oneshot_trap_poisons_instead_of_hanging() {
    let Ok(fault) = std::env::var("ATLAS_ONESHOT_FAULT") else {
        return;
    };
    let Some(g) = gpu() else {
        return;
    };
    let (mode, want) = match fault.as_str() {
        "timeout" => (None, "op 1: the peer never arrived"),
        "mismatch" => (Some(Peer::Lie), "op 1: the peer sent a different size"),
        "desync" => (Some(Peer::Ahead), "op 1: the peer's flag is more than one"),
        "launch" => (None, "op 0: a kernel launch failed"),
        _ => (None, "op 1: the RDMA proxy failed"),
    };
    let cfg = Config {
        // Only `timeout` may take the time limit; the others must beat it.
        timeout_ns: 1_000_000 * if fault == "timeout" { 200 } else { 60_000 },
        stage_fence: fault == "launch" || test_cfg(SPLIT_MIN).stage_fence,
        ..test_cfg(SPLIT_MIN)
    };
    let os = channel(&g, cfg, 1);
    let stop = Arc::new(AtomicBool::new(false));
    let peer = mode.map(|m| spawn_peer(&os, m, stop.clone()));
    let buf = alloc(8192);
    upload(&g, buf, &local_payload(1, 4096));
    let proxy = Channel::new(os.cfg, os.host, 0);
    // SAFETY: the region outlives the test.
    let stage = unsafe { &*((os.host + ctrl_off(os.cfg.max) + STAGE) as *const AtomicU32) };
    match fault.as_str() {
        "proxy" => proxy.fail(),
        // An earlier send the proxy never finishes.
        "proxy-stage" => stage.store(8192, Ordering::Release),
        "launch" => {
            // A stream is no function, so the launch fails behind the stage
            // write; the proxy would serve and clear that orphan stage.
            os.set_kernel(g.stream);
            assert!(os.enqueue(buf, buf, 8192, true, g.stream).is_err());
            os.set_kernel(g.oneshot);
            ck!(cuStreamSynchronize(g.stream));
            stage.store(0, Ordering::Release);
        }
        _ => {}
    }
    let t0 = Instant::now();
    assert!(os.enqueue(buf, buf, 8192, true, g.stream).unwrap());
    if fault.starts_with("proxy-") {
        std::thread::sleep(Duration::from_millis(100));
        proxy.fail();
    }
    let status = unsafe { cuStreamSynchronize(g.stream) };
    let took = t0.elapsed();
    stop.store(true, Ordering::Release);
    if let Some(p) = peer {
        p.join().unwrap();
    }
    let why = os.poisoned();
    println!("fault {fault}: stream status {status} after {took:?}; poison: {why:?}");
    assert_ne!(status, 0, "the kernel must trap");
    assert!(why.expect("poison word set").contains(want));
    assert!(took.as_millis() < 5000);
}

#[path = "loopback_tests.rs"]
mod loopback;
