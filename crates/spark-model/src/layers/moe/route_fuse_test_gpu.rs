// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): `ATLAS_GLM_MOE_ROUTE_FUSE`'s two launches against the five
//! they replace (the canonical `dense_gemv_bf16_tc{8,16,32}` router,
//! `moe_topk_sigmoid_batched`, `moe_sort_by_expert_scan`,
//! `moe_build_tile_worklist_scan` and `quantize_bf16_to_nvfp4`) at GLM's
//! shape, for every batch of 1..=32 rows, with near-tied router logits and
//! remote experts in the pointer table: the same logit, route, offset,
//! worklist and NVFP4 words, the sort's rows as a valid permutation of the
//! same expert groups. `fused_routes_do_not_depend_on_the_batch` holds a
//! probe row to the same words at every width and slot.
//!
//!   cargo test --release -p spark-model --lib route_fuse -- --ignored --nocapture

use super::*;
use anyhow::ensure;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const H: u32 = 4096;
const E: u32 = 288;
const TOPK: u32 = 8;
const SCALING: f32 = 2.5;
const POISON: u8 = 0xa5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: plain integer/float slices, read as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
        .to_vec()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn words(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn raw(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// The router weight, correction bias and local-expert pointer table (every
/// fifth expert remote) of one GLM MoE layer.
struct Layer {
    gate: DevicePtr,
    bias: DevicePtr,
    ptrs: DevicePtr,
}

impl Layer {
    fn new(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<Self> {
        // A few experts share their neighbour's weight row, so their logits
        // tie exactly and the top-k's lower-index rule decides.
        let mut w: Vec<u16> = (0..E * H).map(|_| bf16(rng.unit() * 0.03)).collect();
        for e in (8..E as usize).step_by(37) {
            let (a, b) = w.split_at_mut(e * H as usize);
            b[..H as usize].copy_from_slice(&a[(e - 1) * H as usize..]);
        }
        let mut bias: Vec<f32> = (0..E).map(|_| rng.unit() * 0.01).collect();
        for e in (8..E as usize).step_by(37) {
            bias[e] = bias[e - 1];
        }
        let ptrs: Vec<u64> = (0..E as u64)
            .map(|e| if e % 5 == 3 { 0 } else { 0x1000 + e * 64 })
            .collect();
        Ok(Self {
            gate: upload(gpu, &bytes_of(&w))?,
            bias: upload(gpu, &bytes_of(&bias))?,
            ptrs: upload(gpu, &bytes_of(&ptrs))?,
        })
    }
}

/// Every output of one routing of `rows` rows, read back.
#[derive(Clone, PartialEq, Debug)]
struct Out {
    logits: Vec<u8>,
    ids: Vec<u32>,
    weights: Vec<u32>,
    offsets: Vec<u32>,
    sorted_experts: Vec<u32>,
    tiles: Vec<u32>,
    packed: Vec<u8>,
    /// (token, expert) of each sorted row, in sorted order within a group
    /// normalized: the multiset per group.
    groups: Vec<Vec<(u32, u32)>>,
}

struct Bufs {
    args: RouteArgs,
    overlay: bool,
}

impl Bufs {
    fn new(gpu: &dyn GpuBackend, layer: &Layer, input: DevicePtr, rows: u32) -> Result<Self> {
        Self::with_layout(gpu, layer, input, rows, false)
    }

    fn with_layout(
        gpu: &dyn GpuBackend,
        layer: &Layer,
        input: DevicePtr,
        rows: u32,
        overlay: bool,
    ) -> Result<Self> {
        let te = (rows * TOPK) as usize;
        let poison = |n: usize| upload(gpu, &vec![POISON; n]);
        let sort_bytes = (3 * te + E as usize + 1) * 4;
        let logits = poison(((rows * E) as usize * 2).max(sort_bytes))?;
        let sort = |offset, bytes| {
            if overlay {
                Ok(logits.offset(offset))
            } else {
                poison(bytes)
            }
        };
        Ok(Self {
            overlay,
            args: RouteArgs {
                input,
                gate: layer.gate,
                bias: layer.bias,
                logits,
                indices: poison(te * 4)?,
                weights: poison(te * 4)?,
                sorted_token_ids: sort(0, te * 4)?,
                sorted_expert_ids: sort(te * 4, te * 4)?,
                expert_offsets: sort(te * 8, (E as usize + 1) * 4)?,
                token_to_perm: sort(te * 8 + (E as usize + 1) * 4, te * 4)?,
                local_ptrs: layer.ptrs,
                total_tiles: poison(16 + te * 8)?,
                a_packed: poison((rows * H) as usize * 9 / 16)?,
                rows,
                experts: E,
                hidden: H,
                top_k: TOPK,
                normalize: true,
                scaling: SCALING,
            },
        })
    }

    /// The five launches the fused pair replaces, as serving issues them.
    fn chain(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let a = &self.args;
        let tier = match a.rows {
            1..=8 => "dense_gemv_bf16_tc8",
            9..=16 => "dense_gemv_bf16_tc16",
            _ => "dense_gemv_bf16_tc32",
        };
        let router = gpu.kernel("dense_gemv_bf16_batchm", tier)?;
        let gate = DenseWeight { weight: a.gate };
        ops::dense_gemv_bf16_tc(
            gpu, router, a.input, &gate, a.logits, a.rows, E, H, E, stream,
        )?;
        let topk = gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched")?;
        ops::moe_topk_sigmoid_batched(
            gpu, topk, a.logits, a.bias, a.indices, a.weights, E, TOPK, true, SCALING, a.rows,
            stream,
        )?;
        let sort = gpu.kernel("moe", "moe_sort_by_expert_scan")?;
        ops::moe_sort_by_expert(
            gpu,
            sort,
            a.indices,
            a.sorted_token_ids,
            a.sorted_expert_ids,
            a.expert_offsets,
            a.token_to_perm,
            a.rows * TOPK,
            E,
            TOPK,
            stream,
        )?;
        let worklist = gpu.kernel("moe", "moe_build_tile_worklist_scan")?;
        ops::moe_build_tile_worklist(
            gpu,
            worklist,
            a.expert_offsets,
            a.local_ptrs,
            a.total_tiles.offset(16),
            a.total_tiles,
            E,
            1,
            64,
            stream,
        )?;
        let quant = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let scales = a.a_packed.offset((a.rows * H / 2) as usize);
        ops::quantize_bf16_to_nvfp4(gpu, quant, a.input, a.a_packed, scales, a.rows, H, stream)
    }

    fn read(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<Out> {
        gpu.synchronize(stream)?;
        let a = &self.args;
        let te = (a.rows * TOPK) as usize;
        let ids = words(gpu, a.indices, te)?;
        let offsets = words(gpu, a.expert_offsets, E as usize + 1)?;
        let sorted_tokens = words(gpu, a.sorted_token_ids, te)?;
        let sorted_experts = words(gpu, a.sorted_expert_ids, te)?;
        let perm = words(gpu, a.token_to_perm, te)?;
        // The sort's contract: token_to_perm and sorted_token_ids describe
        // one permutation into the expert groups `offsets` bound.
        let mut seen = vec![false; te];
        for (i, &p) in perm.iter().enumerate() {
            let p = p as usize;
            ensure!(
                p < te && !seen[p],
                "rows={}: token_to_perm not a permutation",
                a.rows
            );
            seen[p] = true;
            let e = ids[i] as usize;
            ensure!(
                sorted_tokens[p] as usize == i / TOPK as usize
                    && sorted_experts[p] == ids[i]
                    && (offsets[e] as usize..offsets[e + 1] as usize).contains(&p),
                "rows={}: sorted row {p} does not hold route {i}",
                a.rows
            );
        }
        let groups = (0..E as usize)
            .map(|e| {
                let mut g: Vec<(u32, u32)> = (offsets[e]..offsets[e + 1])
                    .map(|p| (sorted_tokens[p as usize], sorted_experts[p as usize]))
                    .collect();
                g.sort_unstable();
                g
            })
            .collect();
        let total = words(gpu, a.total_tiles, 1)?[0] as usize;
        ensure!(
            total <= te,
            "rows={}: {total} tiles for {te} routes",
            a.rows
        );
        let mut tiles = vec![total as u32];
        tiles.extend(words(gpu, a.total_tiles.offset(16), total * 2)?);
        let logits = if self.overlay {
            vec![]
        } else {
            raw(gpu, a.logits, (a.rows * E) as usize * 2)?
        };
        crate::layers::glm_gpu_test_backend::finite_bf16("routing logits", &logits)?;
        let weight_bytes = raw(gpu, a.weights, te * 4)?;
        crate::layers::glm_gpu_test_backend::finite_f32("routing weights", &weight_bytes)?;
        Ok(Out {
            logits,
            ids,
            weights: words(gpu, a.weights, te)?,
            offsets,
            sorted_experts,
            tiles,
            packed: raw(gpu, a.a_packed, (a.rows * H) as usize * 9 / 16)?,
            groups,
        })
    }
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

/// `rows` rows of activations; row `probe_at` (if any) is `probe`.
fn input(
    gpu: &dyn GpuBackend,
    rng: &mut Rng,
    rows: u32,
    probe: Option<(&[u16], u32)>,
) -> Result<DevicePtr> {
    let mut x: Vec<u16> = (0..rows * H).map(|_| bf16(rng.unit() * 2.0)).collect();
    if let Some((p, at)) = probe {
        x[(at * H) as usize..((at + 1) * H) as usize].copy_from_slice(p);
    }
    upload(gpu, &bytes_of(&x))
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_routing_matches_the_five_launches() -> Result<()> {
    let gpu = backend()?;
    let fuse = RouteFuse::load(gpu);
    ensure!(fuse.loaded(), "glm_moe_route kernels not in this build");
    let stream = gpu.default_stream();
    let mut rng = Rng(0x5eed_f00d);
    let layer = Layer::new(gpu, &mut rng)?;
    for rows in 1..=32u32 {
        for trial in 0..3 {
            let x = input(gpu, &mut rng, rows, None)?;
            let (chain, fused) = (
                Bufs::new(gpu, &layer, x, rows)?,
                Bufs::new(gpu, &layer, x, rows)?,
            );
            chain.chain(gpu, stream)?;
            fuse.launch(gpu, &fused.args, stream)?;
            let (want, got) = (chain.read(gpu, stream)?, fused.read(gpu, stream)?);
            ensure!(want.tiles[0] > 0, "rows={rows}: no local tiles");
            ensure!(
                got == want,
                "rows={rows} trial={trial}: fused routing differs"
            );
        }
    }
    println!("fused routing == five launches, rows 1..=32 x 3 inputs");
    Ok(())
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_routes_do_not_depend_on_the_batch() -> Result<()> {
    let gpu = backend()?;
    let fuse = RouteFuse::load(gpu);
    ensure!(fuse.loaded(), "glm_moe_route kernels not in this build");
    let stream = gpu.default_stream();
    let mut rng = Rng(0xbadc_0ffe);
    let layer = Layer::new(gpu, &mut rng)?;
    let probe: Vec<u16> = (0..H).map(|_| bf16(rng.unit() * 2.0)).collect();
    let row = |out: &Out, at: u32| {
        let (k, e) = ((at * TOPK) as usize, (at * E) as usize * 2);
        let q = (at * H / 2) as usize;
        let rows = out.ids.len() / TOPK as usize;
        let s = rows * (H / 2) as usize + (at * H / 16) as usize;
        (
            out.logits[e..e + E as usize * 2].to_vec(),
            out.ids[k..k + TOPK as usize].to_vec(),
            out.weights[k..k + TOPK as usize].to_vec(),
            out.packed[q..q + (H / 2) as usize].to_vec(),
            out.packed[s..s + (H / 16) as usize].to_vec(),
        )
    };
    let alone = {
        let b = Bufs::new(gpu, &layer, input(gpu, &mut rng, 1, Some((&probe, 0)))?, 1)?;
        fuse.launch(gpu, &b.args, stream)?;
        row(&b.read(gpu, stream)?, 0)
    };
    for rows in 1..=32u32 {
        for at in [0, rows / 2, rows - 1] {
            let b = Bufs::new(
                gpu,
                &layer,
                input(gpu, &mut rng, rows, Some((&probe, at)))?,
                rows,
            )?;
            fuse.launch(gpu, &b.args, stream)?;
            ensure!(
                row(&b.read(gpu, stream)?, at) == alone,
                "probe at {at} of {rows} rows routes differently"
            );
        }
    }
    println!("probe row: same logits, routes and NVFP4 bytes at every width 1..=32 and slot");
    Ok(())
}

/// The serving arena reuses router logits for every sorted-output array.
/// Compare its live outputs to the disjoint five-launch oracle, retaining
/// the original disjoint tests' logit comparisons as a separate invariant.
#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn fused_routing_supports_the_production_scratch_overlay() -> Result<()> {
    let gpu = backend()?;
    let fuse = RouteFuse::load(gpu);
    ensure!(fuse.loaded(), "glm_moe_route kernels not in this build");
    let mut rng = Rng(0xa11a_5327);
    let layer = Layer::new(gpu, &mut rng)?;
    for rows in 1..=32 {
        let x = input(gpu, &mut rng, rows, None)?;
        let oracle = Bufs::new(gpu, &layer, x, rows)?;
        oracle.chain(gpu, gpu.default_stream())?;
        let mut want = oracle.read(gpu, gpu.default_stream())?;
        want.logits.clear(); // dead after sorting in the production arena
        for fused in [false, true] {
            let overlaid = Bufs::with_layout(gpu, &layer, x, rows, true)?;
            if fused {
                fuse.launch(gpu, &overlaid.args, gpu.default_stream())?;
            } else {
                overlaid.chain(gpu, gpu.default_stream())?;
            }
            let got = overlaid.read(gpu, gpu.default_stream())?;
            ensure!(
                got == want,
                "rows={rows} fused={fused}: scratch overlay differs"
            );
        }
    }
    Ok(())
}
