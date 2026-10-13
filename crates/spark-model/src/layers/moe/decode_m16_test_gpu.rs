// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): the persistent prefetching twins
//! (`ATLAS_GLM_MOE_DECODE_PERSIST`) against the grid twins they replace, bit
//! for bit, through the ops serving launches. Every batch of 1 to 32 rows
//! under three routings (uniform, a hot set that leaves experts from 1 to all
//! of the rows, and few experts with many rows each), both slab families
//! (`m16s` past 16 rows per expert is the downs' NaN backstop), the downs
//! also over signed-zero inputs, and the persistent twins over 48, 7 and 1
//! CTAs (other item-to-CTA phases): the fused gate/up's NVFP4 bytes and the
//! down's BF16 must equal the grid kernels' everywhere, poison included.
//! `persistent_rows_do_not_depend_on_the_batch` holds each row of a batch to
//! the bits it gets alone.
//!
//!   cargo test --release -p spark-model --lib persistent_twins_match_the_grid_twins -- --ignored --nocapture
//!   cargo test --release -p spark-model --lib persistent_rows_do_not_depend_on_the_batch -- --ignored --nocapture

use super::*;
use anyhow::{Context, ensure};

const H: u32 = 4096;
const I: u32 = 1024;
const E: u32 = 64;
const TOPK: usize = 8;
const POISON: u8 = 0x5a;

#[path = "decode_m16_chain_test_gpu.rs"]
mod chain;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn bytes(&mut self, n: usize, scale: bool) -> Vec<u8> {
        (0..n)
            .map(|_| match scale {
                // E4M3 block scales of moderate magnitude, as the bench's.
                true => 0x28 + self.below(24) as u8,
                false => self.next() as u8,
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn words<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain integer/float slices, read as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// One projection's `[k/2, n]` packed and `[k/16, n]` scale tables of every
/// expert, carved from one slab each, and the `[packed, scales, scale2]`
/// pointer tables.
fn tables(gpu: &dyn GpuBackend, rng: &mut Rng, n: u32, k: u32, s2: f32) -> Result<[DevicePtr; 3]> {
    let (wb, sb) = ((k / 2 * n) as usize, (k / 16 * n) as usize);
    let w = upload(gpu, &rng.bytes(wb * E as usize, false))?;
    let s = upload(gpu, &rng.bytes(sb * E as usize, true))?;
    let ptrs = |base: DevicePtr, per: usize| -> Vec<u64> {
        (0..E as usize).map(|e| base.offset(e * per).0).collect()
    };
    let scale2: Vec<f32> = (0..E)
        .map(|e| s2 * (1.0 + (e % 7) as f32 * 0.125))
        .collect();
    Ok([
        upload(gpu, words(&ptrs(w, wb)))?,
        upload(gpu, words(&ptrs(s, sb)))?,
        upload(gpu, words(&scale2))?,
    ])
}

/// A routing of `rows` rows, top-8 distinct experts each, sorted by expert as
/// `moe_sort_by_expert` leaves it: offsets, sorted token ids, and each (row,
/// k)'s sorted position.
struct Route {
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    pos: Vec<[usize; TOPK]>,
}

fn route(ids: &[[u32; TOPK]]) -> Route {
    let mut offsets = vec![0i32; E as usize + 1];
    for row in ids {
        for &e in row {
            offsets[e as usize + 1] += 1;
        }
    }
    for e in 0..E as usize {
        offsets[e + 1] += offsets[e];
    }
    let mut fill: Vec<i32> = offsets[..E as usize].to_vec();
    let mut sorted = vec![0i32; ids.len() * TOPK];
    let mut pos = vec![[0usize; TOPK]; ids.len()];
    for (t, row) in ids.iter().enumerate() {
        for (k, &e) in row.iter().enumerate() {
            let p = fill[e as usize] as usize;
            fill[e as usize] += 1;
            sorted[p] = t as i32;
            pos[t][k] = p;
        }
    }
    Route {
        offsets,
        sorted,
        pos,
    }
}

/// Top-8 picks of `rows` rows: uniform (0), four from a hot set of six then
/// uniform (1: experts with 1 to `rows` rows), or from ten experts (2).
fn picks(rng: &mut Rng, rows: usize, kind: u32) -> Vec<[u32; TOPK]> {
    (0..rows)
        .map(|_| {
            let mut row = [u32::MAX; TOPK];
            for k in 0..TOPK {
                loop {
                    let e = match kind {
                        1 if k < 4 => rng.below(6) as u32,
                        2 => 20 + rng.below(10) as u32,
                        _ => rng.below(E as usize) as u32,
                    };
                    if !row[..k].contains(&e) {
                        row[k] = e;
                        break;
                    }
                }
            }
            row
        })
        .collect()
}

struct Rig<'a> {
    gpu: &'a dyn GpuBackend,
    gate: [DevicePtr; 3],
    up: [DevicePtr; 3],
    down: [DevicePtr; 3],
    builder: KernelHandle,
    /// Activations of up to 32 rows: packed, scales.
    a: [DevicePtr; 2],
    /// Gate/up output (the down's input): packed, scales; down output.
    q: [DevicePtr; 2],
    c: DevicePtr,
    offsets: DevicePtr,
    sorted: DevicePtr,
    scratch: DevicePtr,
}

const MAX_ROUTED: usize = 32 * TOPK;

impl<'a> Rig<'a> {
    fn new(gpu: &'a dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            gpu,
            gate: tables(gpu, rng, I, H, 1.0 / 256.0)?,
            up: tables(gpu, rng, I, H, 1.0 / 128.0)?,
            down: tables(gpu, rng, H, I, 1.0 / 64.0)?,
            builder: gpu.kernel("moe", "moe_build_tile_worklist")?,
            a: [
                upload(gpu, &rng.bytes(32 * H as usize / 2, false))?,
                upload(gpu, &rng.bytes(32 * H as usize / 16, true))?,
            ],
            q: [
                gpu.alloc(MAX_ROUTED * I as usize / 2)?,
                gpu.alloc(MAX_ROUTED * I as usize / 16)?,
            ],
            c: gpu.alloc(MAX_ROUTED * H as usize * 2)?,
            offsets: gpu.alloc((E as usize + 1) * 4)?,
            sorted: gpu.alloc(MAX_ROUTED * 4)?,
            scratch: gpu.alloc(16 + MAX_ROUTED * 8)?,
        })
    }

    /// Uploads `r` (rows from row `first` of the activations) and builds the
    /// decode worklist as `decode_m16_grid` does.
    fn set_route(&self, r: &Route, first: usize) -> Result<()> {
        let sorted: Vec<i32> = r.sorted.iter().map(|&t| t + first as i32).collect();
        self.gpu.copy_h2d(words(&r.offsets), self.offsets)?;
        self.gpu.copy_h2d(words(&sorted), self.sorted)?;
        ops::moe_build_tile_worklist(
            self.gpu,
            self.builder,
            self.offsets,
            self.gate[0],
            self.scratch.offset(16),
            self.scratch,
            E,
            1,
            64,
            0,
        )
    }

    fn schedule(&self, rows: u32, ctas: u32) -> ops::K128wSchedule {
        match ctas {
            0 => ops::K128wSchedule::Grid {
                bound: (rows * TOPK as u32).min(E),
            },
            ctas => ops::K128wSchedule::Stride { ctas },
        }
    }

    /// The fused gate/up over the routed rows into poisoned outputs:
    /// returns its packed and scale bytes.
    fn gate_up(&self, kernel: KernelHandle, rows: u32, ctas: u32) -> Result<Vec<u8>> {
        let routed = rows as usize * TOPK;
        let (qb, sb) = (routed * I as usize / 2, routed * I as usize / 16);
        self.gpu.memset(self.q[0], POISON, qb)?;
        self.gpu.memset(self.q[1], POISON, sb)?;
        ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
            self.gpu,
            ops::K128wKernel {
                grid: kernel,
                persist: KernelHandle(0),
            },
            self.a[0],
            self.a[1],
            self.gate,
            self.up,
            self.q[0],
            self.q[1],
            self.offsets,
            self.sorted,
            E,
            I,
            H,
            self.scratch,
            self.schedule(rows, ctas),
            0,
        )?;
        self.gpu.synchronize(0)?;
        let mut out = vec![0u8; qb + sb];
        self.gpu.copy_d2h(self.q[0], &mut out[..qb])?;
        self.gpu.copy_d2h(self.q[1], &mut out[qb..])?;
        Ok(out)
    }

    /// The down over `input` (the gate/up bytes) into a poisoned output:
    /// returns its BF16 bytes.
    fn down(
        &self,
        kernel: KernelHandle,
        rows: u32,
        ctas: u32,
        input: &[u8],
        routing: (&Route, usize),
    ) -> Result<Vec<u8>> {
        let routed = rows as usize * TOPK;
        let qb = routed * I as usize / 2;
        self.gpu.copy_h2d(&input[..qb], self.q[0])?;
        self.gpu.copy_h2d(&input[qb..], self.q[1])?;
        self.gpu.memset(self.c, POISON, routed * H as usize * 2)?;
        ops::moe_w4a4_grouped_gemm_prequant_k128w(
            self.gpu,
            ops::K128wKernel {
                grid: kernel,
                persist: KernelHandle(0),
            },
            self.q[0],
            self.q[1],
            self.down[0],
            self.down[1],
            self.down[2],
            self.c,
            self.offsets,
            DevicePtr::NULL,
            E,
            H,
            I,
            self.scratch,
            self.schedule(rows, ctas),
            0,
        )?;
        self.gpu.synchronize(0)?;
        let mut out = vec![0u8; routed * H as usize * 2];
        self.gpu.copy_d2h(self.c, &mut out)?;
        check_down_output(&out, &routing.0.offsets, routing.1)?;
        Ok(out)
    }
}

/// The grid and persistent kernels of a slab family: `[gate_up, down]` each.
fn family(gpu: &dyn GpuBackend, family: &str) -> Result<[[KernelHandle; 2]; 2]> {
    let k = |name: String| gpu.kernel("moe_w4a16", &name).with_context(|| name.clone());
    let pair = |suffix: &str| -> Result<[KernelHandle; 2]> {
        Ok([
            k(format!(
                "glm_moe_decode_{family}_gate_up_silu_k128w{suffix}"
            ))?,
            k(format!("glm_moe_decode_{family}_k128w_zskip{suffix}"))?,
        ])
    };
    Ok([pair("_l2pf")?, pair("_l2pf_p")?])
}

/// The down input of `q` with every third sorted row all -0 codes and every
/// third a mix of +0/-0 codes (tiles skip none, some and all of their rows).
fn signed_zeros(q: &[u8], routed: usize) -> Vec<u8> {
    const MIX: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
    let mut z = q.to_vec();
    let row_bytes = I as usize / 2;
    for row in 0..routed {
        if row % 3 == 2 {
            continue;
        }
        for kp in 0..row_bytes {
            z[row * row_bytes + kp] = if row % 3 == 0 {
                0x88
            } else {
                MIX[(kp + row) & 3]
            };
        }
    }
    z
}

fn backend() -> Result<&'static spark_runtime::cuda_backend::AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn persistent_twins_match_the_grid_twins() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    let mut rng = Rng(0x5eed_cafe);
    let rig = Rig::new(gpu, &mut rng)?;
    let families = [family(gpu, "m16s")?, family(gpu, "m32s")?];
    let (mut checked, mut failed) = (0usize, vec![]);
    for rows in 1..=32u32 {
        for kind in 0..3 {
            let r = route(&picks(&mut rng, rows as usize, kind));
            let most = (0..E as usize)
                .map(|e| r.offsets[e + 1] - r.offsets[e])
                .max()
                .unwrap();
            rig.set_route(&r, 0)?;
            for (slab, [grid, persist]) in families.iter().enumerate() {
                let want_q = rig.gate_up(grid[0], rows, 0)?;
                let routed = rows as usize * TOPK;
                let zeros = signed_zeros(&want_q, routed);
                let want_c = [&want_q, &zeros]
                    .map(|input| rig.down(grid[1], rows, 0, input, (&r, (slab + 1) * 16)));
                for ctas in [48, 7, 1] {
                    let case = format!(
                        "rows {rows} routing {kind} most {most} slabs {} ctas {ctas}",
                        slab + 1
                    );
                    if rig.gate_up(persist[0], rows, ctas)? != want_q {
                        failed.push(format!("{case}: gate/up"));
                    }
                    for (input, want) in [&want_q, &zeros].into_iter().zip(&want_c) {
                        if &rig.down(persist[1], rows, ctas, input, (&r, (slab + 1) * 16))?
                            != want.as_ref().unwrap()
                        {
                            failed.push(format!("{case}: down"));
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("{checked} cases, {} differ", failed.len());
    ensure!(failed.is_empty(), "persistent twins differ: {failed:#?}");
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn persistent_rows_do_not_depend_on_the_batch() -> Result<()> {
    let gpu = backend()?;
    let gpu: &dyn GpuBackend = gpu;
    let mut rng = Rng(0x0dd_ba11);
    let rig = Rig::new(gpu, &mut rng)?;
    let families = [family(gpu, "m16s")?, family(gpu, "m32s")?];
    let mut failed = vec![];
    for rows in [2u32, 5, 8, 12, 16, 24, 32] {
        for kind in 0..3 {
            let ids = picks(&mut rng, rows as usize, kind);
            let batch = route(&ids);
            let [_, persist] = families[usize::from(rows > 16)];
            rig.set_route(&batch, 0)?;
            let q = rig.gate_up(persist[0], rows, 48)?;
            let c = rig.down(
                persist[1],
                rows,
                48,
                &q,
                (&batch, if rows > 16 { 32 } else { 16 }),
            )?;
            let (qb, routed) = (I as usize / 2, rows as usize * TOPK);
            let sb = I as usize / 16;
            for (t, row_ids) in ids.iter().enumerate() {
                // Row t alone: its own activations, the same experts.
                let alone = route(std::slice::from_ref(row_ids));
                rig.set_route(&alone, t)?;
                let q1 = rig.gate_up(persist[0], 1, 48)?;
                let c1 = rig.down(
                    persist[1],
                    1,
                    48,
                    &q1,
                    (&alone, if rows > 16 { 32 } else { 16 }),
                )?;
                for k in 0..TOPK {
                    let (p, p1) = (batch.pos[t][k], alone.pos[0][k]);
                    let same = q[p * qb..(p + 1) * qb] == q1[p1 * qb..(p1 + 1) * qb]
                        && q[routed * qb + p * sb..routed * qb + (p + 1) * sb]
                            == q1[TOPK * qb + p1 * sb..TOPK * qb + (p1 + 1) * sb]
                        && c[p * H as usize * 2..(p + 1) * H as usize * 2]
                            == c1[p1 * H as usize * 2..(p1 + 1) * H as usize * 2];
                    if !same {
                        failed.push(format!(
                            "rows {rows} routing {kind} row {t} expert {}",
                            row_ids[k]
                        ));
                    }
                }
            }
        }
    }
    ensure!(failed.is_empty(), "rows depend on their batch: {failed:#?}");
    Ok(())
}

/// Unsupported expert widths deliberately produce NaNs in the down kernel.
/// Keep that backstop under test, without allowing a NaN in any row the
/// chosen slab supports (every row production is allowed to consume).
fn check_down_output(out: &[u8], offsets: &[i32], slab_rows: usize) -> Result<()> {
    let row_bytes = H as usize * 2;
    ensure!(
        out.len() == *offsets.last().unwrap() as usize * row_bytes,
        "down output shape"
    );
    for (expert, span) in offsets.windows(2).enumerate() {
        let range = span[0] as usize * row_bytes..span[1] as usize * row_bytes;
        let bytes = &out[range];
        if (span[1] - span[0]) as usize <= slab_rows {
            crate::layers::glm_gpu_test_backend::finite_bf16("MoE supported down rows", bytes)
                .with_context(|| format!("expert {expert}, slabs {slab_rows}"))?;
        } else {
            ensure!(
                bytes.chunks_exact(2).all(|word| {
                    f32::from_bits((u16::from_le_bytes([word[0], word[1]]) as u32) << 16).is_nan()
                }),
                "expert {expert} exceeds {slab_rows} rows without the NaN capacity backstop"
            );
        }
    }
    Ok(())
}

#[test]
fn down_finiteness_allows_only_the_capacity_backstop() {
    let offsets = [0, 17, 18];
    let mut out = vec![0u8; 18 * H as usize * 2];
    assert!(check_down_output(&out, &offsets, 16).is_err());
    for word in out[..17 * H as usize * 2].chunks_exact_mut(2) {
        word.copy_from_slice(&0x7fc0u16.to_le_bytes());
    }
    assert!(check_down_output(&out, &offsets, 16).is_ok());
    assert!(check_down_output(&out, &offsets, 32).is_err());
    for bad in [0x7fc0u16, 0x7f80, 0xff80] {
        out[17 * H as usize * 2..][..2].copy_from_slice(&bad.to_le_bytes());
        assert!(check_down_output(&out, &offsets, 16).is_err());
    }
}
