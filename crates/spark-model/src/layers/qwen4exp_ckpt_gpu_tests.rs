// SPDX-License-Identifier: AGPL-3.0-only

//! GPU checks of the qwen4_exp mid-chunk checkpoint captures
//! (`layers::qwen4exp_ckpt`) through the production launchers, at the TP2
//! rank's GDN shape (24 v-heads, 8 k-heads, 128 x 128): every capture leaves
//! the pass's outputs and final state byte-identical, and the state it
//! captures at row r equals the state a pass over only the first r rows ends
//! with. `#[ignore]`: run on a GB10 with `--ignored`.

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::layers::qwen4exp_ckpt as ckpt;

const NV: usize = 24;
const NK: usize = 8;
const D: usize = 128;
const CONV: usize = 2 * NK * D + NV * D;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(16)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn get(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

fn bf16(rng: &mut Lcg, n: usize, s: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| (((rng.f() * s).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32s(rng: &mut Lcg, n: usize, f: impl Fn(f32) -> f32) -> Vec<u8> {
    (0..n).flat_map(|_| f(rng.f()).to_le_bytes()).collect()
}

fn backend() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

/// GDN inputs for `rows` tokens: packed q|k|v rows and [gate | beta] rows.
struct Gdn {
    qkv: DevicePtr,
    gates: DevicePtr,
    h0: Vec<u8>,
}

fn gdn_inputs(g: &dyn GpuBackend, rows: usize) -> Gdn {
    let mut rng = Lcg(0xc0ffee);
    let qkv = up(g, &bf16(&mut rng, rows * CONV, 0.08));
    let gates: Vec<u8> = (0..rows)
        .flat_map(|_| {
            let mut row = f32s(&mut rng, NV, |x| 0.95 + 0.045 * x);
            row.extend(f32s(&mut rng, NV, |x| 0.5 + 0.4 * x));
            row
        })
        .collect();
    let h0 = f32s(&mut rng, NV * D * D, |x| 0.02 * x);
    Gdn {
        qkv,
        gates: up(g, &gates),
        h0,
    }
}

/// The FLA chunked prefill with the `_pipe` spine, through
/// `ops::gdn_prefill_fla`, over `rows` tokens from `h`.
fn fla(g: &dyn GpuBackend, x: &Gdn, h: DevicePtr, out: DevicePtr, rows: usize) {
    let k = |name: &str| g.kernel("gated_delta_rule_fla", name).unwrap();
    let nt = rows.div_ceil(64);
    let scratch = g
        .alloc(nt * NV * 64 * D * 2 * 3 + nt * NV * D * D * 2 + nt * NV * 64 * 4)
        .unwrap();
    let w = scratch;
    let u = w.offset(nt * NV * 64 * D * 2);
    let s = u.offset(nt * NV * 64 * D * 2);
    let uc = s.offset(nt * NV * D * D * 2);
    let gc = uc.offset(nt * NV * 64 * D * 2);
    let bf = 2usize;
    ops::gdn_prefill_fla(
        g,
        k("gated_delta_rule_recompute_wu"),
        k("gated_delta_rule_chunk_delta_h_ksplit"),
        spark_runtime::gpu::KernelHandle(0),
        k("gated_delta_rule_chunk_delta_h_pipe"),
        spark_runtime::gpu::KernelHandle(0),
        k("gated_delta_rule_chunk_fwd_o"),
        h,
        x.qkv,
        x.qkv.offset(NK * D * bf),
        x.qkv.offset(2 * NK * D * bf),
        x.gates,
        x.gates.offset(NV * 4),
        out,
        w,
        u,
        s,
        uc,
        gc,
        1,
        rows as u32,
        nt as u32,
        NK as u32,
        NV as u32,
        D as u32,
        D as u32,
        CONV as u32,
        CONV as u32,
        (2 * NV) as u32,
        false,
        DevicePtr::NULL,
        DevicePtr::NULL,
        false,
        false,
        g.default_stream(),
    )
    .unwrap();
    g.synchronize(g.default_stream()).unwrap();
    g.free(scratch).unwrap();
}

/// The FLA spine's `_cap` twin through `gdn_prefill_fla`: same outputs and
/// final state; the state captured at chunk 8 equals a 512-row pass's.
#[test]
#[ignore]
fn fla_spine_capture_is_exact() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let rows = 1000;
    let x = gdn_inputs(g, rows);
    let hb = NV * D * D * 4;
    let ob = rows * NV * D * 2;
    let (h1, h2, h3, cap) = (
        up(g, &x.h0),
        up(g, &x.h0),
        up(g, &x.h0),
        g.alloc(hb).unwrap(),
    );
    let (o1, o2, o3) = (
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
    );
    fla(g, &x, h1, o1, rows);
    ckpt::begin(512, DevicePtr::NULL);
    ckpt::arm_spine(8, cap);
    fla(g, &x, h2, o2, rows);
    let (done, _) = ckpt::end();
    assert_eq!(done, 1, "the `_pipe_cap` twin took the arm");
    fla(g, &x, h3, o3, 512);
    assert!(get(g, o1, ob) == get(g, o2, ob), "outputs differ");
    assert!(get(g, h1, hb) == get(g, h2, hb), "final states differ");
    assert!(
        get(g, cap, hb) == get(g, h3, hb),
        "captured != 512-row final state"
    );
}

/// The spine's `_capn` twin (dense / branch-point checkpoints): three rows
/// in one pass, each equal to a pass over only the rows below it, with the
/// outputs and final state unchanged.
#[test]
#[ignore]
fn fla_spine_capture_many_is_exact() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let rows = 1000;
    let x = gdn_inputs(g, rows);
    let hb = NV * D * D * 4;
    let ob = rows * NV * D * 2;
    let (h1, h2) = (up(g, &x.h0), up(g, &x.h0));
    let (o1, o2, o3) = (
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
    );
    let chunks = [3u32, 8, 12];
    let caps: Vec<DevicePtr> = chunks.iter().map(|_| g.alloc(hb).unwrap()).collect();
    fla(g, &x, h1, o1, rows);
    ckpt::begin_many(&[]);
    let arms: Vec<_> = chunks.iter().copied().zip(caps.iter().copied()).collect();
    ckpt::arm_spines(&arms);
    fla(g, &x, h2, o2, rows);
    assert_eq!(ckpt::end().0, 1, "the `_pipe_capn` twin took the list");
    assert!(get(g, o1, ob) == get(g, o2, ob), "outputs differ");
    assert!(get(g, h1, hb) == get(g, h2, hb), "final states differ");
    for (&c, &cap) in chunks.iter().zip(&caps) {
        let short = up(g, &x.h0);
        fla(g, &x, short, o3, c as usize * ckpt::CHUNK);
        assert!(
            get(g, cap, hb) == get(g, short, hb),
            "captured at chunk {c} != a {}-row pass",
            c as usize * ckpt::CHUNK
        );
    }
}

/// The token-sequential warm-replay recurrence split at a row: same outputs
/// and final state; the state at the split equals a pass over the first rows.
#[test]
#[ignore]
fn regresident_split_is_exact() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kernel = g
        .kernel(
            "gated_delta_rule_regresident",
            "gated_delta_rule_prefill_regresident",
        )
        .unwrap();
    let rows = 1000usize;
    let x = gdn_inputs(g, rows);
    let hb = NV * D * D * 4;
    let ob = rows * NV * D * 2;
    let bf = 2usize;
    let run = |h: DevicePtr, out: DevicePtr, start: usize, len: usize| {
        let gate = x.gates.offset(start * 2 * NV * 4);
        ops::gdn_prefill_regresident(
            g,
            kernel,
            h,
            x.qkv.offset(start * CONV * bf),
            x.qkv.offset(start * CONV * bf + NK * D * bf),
            x.qkv.offset(start * CONV * bf + 2 * NK * D * bf),
            gate,
            gate.offset(NV * 4),
            out.offset(start * NV * D * bf),
            1,
            len as u32,
            NK as u32,
            NV as u32,
            D as u32,
            D as u32,
            CONV as u32,
            CONV as u32,
            (2 * NV) as u32,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
    };
    let (h1, h2, h3) = (up(g, &x.h0), up(g, &x.h0), up(g, &x.h0));
    let (o1, o2, o3) = (
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
    );
    let split = 333;
    run(h1, o1, 0, rows);
    run(h2, o2, 0, split);
    let at_split = get(g, h2, hb);
    run(h2, o2, split, rows - split);
    run(h3, o3, 0, split);
    assert!(get(g, o1, ob) == get(g, o2, ob), "outputs differ");
    assert!(get(g, h1, hb) == get(g, h2, hb), "final states differ");
    assert!(
        at_split == get(g, h3, hb),
        "state at the split != a short pass's"
    );
}

/// The PLE conv split at a row (`PleLayer::conv_span`'s two launches):
/// same outputs and carry; the carry at the split equals a short pass's.
#[test]
#[ignore]
fn ple_conv_split_is_exact() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let kernel = g.kernel("ple", "ple_conv").unwrap();
    let (c, k, dil, rows) = (10240usize, 4usize, 3usize, 300usize);
    let st = (k - 1) * dil;
    let mut rng = Lcg(77);
    let xs = up(g, &f32s(&mut rng, rows * c, |v| v));
    let gated = up(g, &f32s(&mut rng, rows * c, |v| 0.5 * v));
    let w = up(g, &bf16(&mut rng, c * k, 0.3));
    let s0 = f32s(&mut rng, st * c, |v| 0.1 * v);
    let run = |state: DevicePtr, out: DevicePtr, r0: usize, n: usize| {
        let off = r0 * c * 4;
        ops::ple_conv(
            g,
            kernel,
            xs.offset(off),
            gated.offset(off),
            w,
            state,
            out.offset(off),
            n as u32,
            c as u32,
            k as u32,
            dil as u32,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
    };
    let (sb, ob) = (st * c * 4, rows * c * 4);
    let (a, b, short) = (up(g, &s0), up(g, &s0), up(g, &s0));
    let (o1, o2, o3) = (
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
        g.alloc(ob).unwrap(),
    );
    let split = 97;
    run(a, o1, 0, rows);
    run(b, o2, 0, split);
    let at_split = get(g, b, sb);
    run(b, o2, split, rows - split);
    run(short, o3, 0, split);
    assert!(get(g, o1, ob) == get(g, o2, ob), "outputs differ");
    assert!(get(g, a, sb) == get(g, b, sb), "final carries differ");
    assert!(
        at_split == get(g, short, sb),
        "carry at the split != a short pass's"
    );
}
