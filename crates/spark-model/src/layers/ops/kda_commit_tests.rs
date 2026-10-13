// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-identity gate for the layer-batched `kda_commit_records`: one launch
//! over every layer of a uniformly strided pool must leave exactly the state
//! of one launch per layer, and both must equal the host fold (the kernel's
//! `kda_fold` is `fma(delta, key, h * decay)` in IEEE round-to-nearest, which
//! `f32::mul_add` reproduces bit for bit). Owners commit different record
//! ranges, including none, a leaf-split tail and the full verify; the spare
//! slot between them must stay untouched. Also times both forms at the
//! production shape (34 KDA layers, 32 heads per rank).
//!
//! GPU test: `#[ignore]` per repo convention. Run with
//! ```text
//! cargo test -p spark-model --release --lib kda_commit -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const HEADS: usize = 32;
const STATE: usize = HEADS * 128 * 128;
const ROW: usize = HEADS * ops::KDA_RECORD_FLOATS;
/// Records rows per slot (`num_intermediates`).
const NI: usize = 8;

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn up(g: &dyn GpuBackend, d: &[f32]) -> DevicePtr {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len()).unwrap();
    g.copy_h2d(&b, p).unwrap();
    p
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u32> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b).unwrap();
    crate::layers::glm_gpu_test_backend::finite_f32("KDA committed state", &b).unwrap();
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Pools laid out as `SsmStatePool` does: per layer `slots` states / `slots *
/// NI` record rows, layers contiguous.
struct Pools {
    layers: usize,
    slots: usize,
    state: Vec<f32>,
    records: Vec<f32>,
}

impl Pools {
    fn new(layers: usize, slots: usize, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let state = (0..layers * slots * STATE)
            .map(|_| rng.r(-1.0, 1.0))
            .collect();
        let mut records = vec![0.0f32; layers * slots * NI * ROW];
        for row in records.chunks_exact_mut(ops::KDA_RECORD_FLOATS) {
            for (i, x) in row.iter_mut().enumerate() {
                *x = match i / 128 {
                    0 => rng.r(0.9, 1.0),
                    1 => rng.r(-0.1, 0.1),
                    _ => rng.r(-0.5, 0.5),
                };
            }
        }
        Self {
            layers,
            slots,
            state,
            records,
        }
    }
    fn state_at(&self, layer: usize, slot: usize) -> usize {
        (layer * self.slots + slot) * STATE
    }
    fn records_at(&self, layer: usize, slot: usize, row: usize) -> usize {
        ((layer * self.slots + slot) * NI + row) * ROW
    }
    /// Host fold of `rows` into `slot`'s state in every layer.
    fn fold(&self, state: &mut [f32], slot: usize, rows: &std::ops::Range<usize>) {
        for l in 0..self.layers {
            for t in rows.clone() {
                let rec = &self.records[self.records_at(l, slot, t)..][..ROW];
                let h = &mut state[self.state_at(l, slot)..][..STATE];
                for head in 0..HEADS {
                    let r = &rec[head * ops::KDA_RECORD_FLOATS..];
                    for k in 0..128 {
                        for v in 0..128 {
                            let x = &mut h[(head * 128 + k) * 128 + v];
                            *x = r[256 + v].mul_add(r[128 + k], *x * r[k]);
                        }
                    }
                }
            }
        }
    }
}

struct Dev<'a> {
    g: &'a dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    /// 0: the grid kernel; else the stride kernel over this many CTAs.
    ctas: u32,
    stream: u64,
    state: DevicePtr,
    records: DevicePtr,
    p: &'a Pools,
}

impl Dev<'_> {
    /// Commit `rows` of `slot`: one launch for all layers, or one per layer.
    fn commit(&self, slot: usize, rows: &std::ops::Range<usize>, batched: bool) {
        let p = self.p;
        let launch = |l: usize, layers: usize| {
            ops::kda_commit_records(
                self.g,
                self.kernel,
                self.state.offset(p.state_at(l, slot) * 4),
                p.slots * STATE,
                self.records.offset(p.records_at(l, slot, rows.start) * 4),
                p.slots * NI * ROW,
                ROW,
                rows.len() as u32,
                HEADS as u32,
                layers as u32,
                self.ctas,
                self.stream,
            )
            .unwrap()
        };
        if batched {
            launch(0, p.layers);
        } else {
            (0..p.layers).for_each(|l| launch(l, 1));
        }
    }
}

fn setup(g: &dyn GpuBackend, p: &Pools) -> (DevicePtr, DevicePtr) {
    (up(g, &p.state), up(g, &p.records))
}

fn backend() -> &'static spark_runtime::cuda_backend::AtlasCudaBackend {
    crate::layers::glm_gpu_test_backend::backend().expect("GLM GPU test backend")
}

#[test]
#[ignore]
fn kda_commit_layer_batched_byte_identical() {
    let gpu = backend();
    let g: &dyn GpuBackend = gpu;
    // Slot 2 is a spare no owner commits; owner ranges cover none, one row,
    // a leaf-split tail and the full verify.
    let owners: [(usize, std::ops::Range<usize>); 4] =
        [(0, 0..0), (1, 0..1), (3, 2..5), (4, 0..NI)];
    let p = Pools::new(3, 5, 0x6b64_6163);
    let mut want = p.state.clone();
    for (slot, rows) in &owners {
        p.fold(&mut want, *slot, rows);
    }
    let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
    // The stride kernel over 1, 7, 48 and more CTAs than blocks: every
    // block-to-CTA phase.
    for (ctas, batched) in [
        (0, true),
        (0, false),
        (1, true),
        (7, true),
        (48, true),
        (9999, true),
    ] {
        let (state, records) = setup(g, &p);
        let dev = Dev {
            g,
            kernel: ops::kda_commit_kernel(g, ctas).unwrap(),
            ctas,
            stream: g.default_stream(),
            state,
            records,
            p: &p,
        };
        for (slot, rows) in &owners {
            dev.commit(*slot, rows, batched);
        }
        g.synchronize(g.default_stream()).unwrap();
        let got = down(g, state, p.state.len());
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            bad, 0,
            "ctas={ctas} batched={batched}: {bad} state words differ"
        );
        g.free(state).unwrap();
        g.free(records).unwrap();
    }
}

#[test]
#[ignore]
fn kda_commit_layer_batched_timing() {
    let gpu = backend();
    let g: &dyn GpuBackend = gpu;
    let p = Pools::new(34, 5, 7);
    let (state, records) = setup(g, &p);
    let dev = Dev {
        g,
        kernel: ops::kda_commit_kernel(g, 0).unwrap(),
        ctas: 0,
        stream: g.default_stream(),
        state,
        records,
        p: &p,
    };
    const ITERS: usize = 50;
    for owners in [1usize, 4] {
        for rows in [1usize, 4, NI] {
            for batched in [false, true] {
                let step = || {
                    for slot in 0..owners {
                        dev.commit(slot, &(0..rows), batched);
                    }
                };
                step();
                g.synchronize(g.default_stream()).unwrap();
                let t = std::time::Instant::now();
                for _ in 0..ITERS {
                    step();
                }
                g.synchronize(g.default_stream()).unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                eprintln!(
                    "kda_commit owners={owners} rows={rows} {}: {us:.1} us/step",
                    if batched {
                        "one launch per owner"
                    } else {
                        "one launch per layer"
                    }
                );
            }
        }
    }
    g.free(state).unwrap();
    g.free(records).unwrap();
}

/// The serving shape of a records commit (34 layers, 32 heads, three
/// accepted rows) on a second stream, as `commit_accepted_prefix` issues it,
/// and an 8 KiB device copy plus host sync on the default stream right
/// after, as the step's drain does: how long the copy's sync waits behind
/// the commit, and how long the commit takes, for the grid kernel and the
/// stride kernel over several CTA counts.
#[test]
#[ignore]
fn kda_commit_beside_a_copy_timing() {
    let gpu = backend();
    let g: &dyn GpuBackend = gpu;
    let p = Pools::new(34, 1, 11);
    let (state, records) = setup(g, &p);
    let side = g.create_stream().unwrap();
    let main = g.default_stream();
    let (src, dst) = (g.alloc(8192).unwrap(), g.alloc(8192).unwrap());
    const ITERS: usize = 20;
    for ctas in [0u32, 24, 48, 96, 192] {
        let dev = Dev {
            g,
            kernel: ops::kda_commit_kernel(g, ctas).unwrap(),
            ctas,
            stream: side,
            state,
            records,
            p: &p,
        };
        let (mut copy_us, mut commit_us) = (Vec::new(), Vec::new());
        for i in 0..=ITERS {
            g.synchronize_device().unwrap();
            let t = std::time::Instant::now();
            dev.commit(0, &(0..3), true);
            g.copy_d2d_async(src, dst, 8192, main).unwrap();
            g.synchronize(main).unwrap();
            let copied = t.elapsed();
            g.synchronize(side).unwrap();
            if i > 0 {
                copy_us.push(copied.as_secs_f64() * 1e6);
                commit_us.push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        eprintln!(
            "kda_commit ctas={ctas}: copy+sync {:.1} us, commit {:.1} us (median of {ITERS})",
            median(&mut copy_us),
            median(&mut commit_us)
        );
    }
    g.free(state).unwrap();
    g.free(records).unwrap();
}
