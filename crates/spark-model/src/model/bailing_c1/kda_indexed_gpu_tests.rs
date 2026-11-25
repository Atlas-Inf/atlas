// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity harness for the Ling pool-indexed KDA decode pair
//! (`ling_kda_conv_indexed` + `ling_kda_recurrent_indexed`): the kernels
//! were ported for this family but never executed on a device before this
//! slice, so this pins the pair against the per-sequence kernels
//! (`causal_conv1d_update` + `kda_recurrent_bf16`) over the SAME
//! pseudo-random pool states, rows, and slot permutations. Both paths run
//! the identical conv/recurrence bodies, so the comparison is bit-exact —
//! any mismatch is a latent pool-addressing bug, exactly the class this
//! slice is on the hook to surface and fix. The padding lanes (-1 slots)
//! must leave the pool untouched.
//!
//! `#[ignore]`d — needs a GB10, a `ling-3.2-air` kernel build:
//! ```text
//! cargo test -p spark-model --release --lib \
//!   bailing_c1::kda_indexed_gpu_tests -- --ignored --nocapture
//! ```

use anyhow::{Context, Result, ensure};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels::LingKernels;
use super::spec::{KDA_HEAD_DIM, KDA_HEADS};
use crate::layer::ssm_batch::{SsmBatchView, SsmPoolView};
use crate::layers::ops;
use crate::layers::ops::{KdaBuffer, KdaIndexedConv, KdaIndexedRecurrent, KdaIndexedShape};
use crate::weight_map::DenseWeight;

const T: usize = KDA_HEADS * KDA_HEAD_DIM;
const CHANNELS: usize = 3 * T;
const CONV_TAPS: usize = 4;
/// Pool slots exercised: 8 (the lane's row ceiling; unique live ids).
const SLOTS: usize = 8;
const H_BYTES: usize = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4;
const CONV_BYTES: usize = CHANNELS * CONV_TAPS * 4;
const LOWER_BOUND: f32 = -5.0;

/// Deterministic LCG — stable across runs, no external RNG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
    fn f32(&mut self) -> f32 {
        (self.next() % 4096) as f32 / 2048.0 - 1.0
    }
    /// Random bf16 bits in the normal working range: magnitude ~2^-9..2^0
    /// (exponent field 118..126), random sign and mantissa — dot products
    /// and decay gates land in the ranges sigmoid/expf keep informative
    /// instead of denormal-flushing to zero everywhere.
    fn bf16(&mut self) -> u16 {
        let sign = (self.next() & 1) as u16;
        let exp = (118 + self.next() % 9) as u16;
        let man = (self.next() & 0x7f) as u16;
        (sign << 15) | (exp << 7) | man
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// bf16 row-major payload of `rows x width` elements.
fn bf16_rows(rng: &mut Lcg, rows: usize, width: usize) -> Vec<u8> {
    (0..rows * width)
        .flat_map(|_| rng.bf16().to_le_bytes().to_vec())
        .collect()
}

fn ling_gpu() -> Result<&'static AtlasCudaBackend> {
    static GPU: std::sync::OnceLock<Result<AtlasCudaBackend, String>> = std::sync::OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        configure_pdl("ling-3.2-air");
        let set = atlas_kernels::ptx_for_exact_target("ling-3.2-air", "nvfp4")
            .ok_or_else(|| "ling-3.2-air/nvfp4 absent from this build".to_owned())?;
        AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("CUDA backend: {e:#}"))
    });
    let gpu = gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    gpu.bind_to_thread()?;
    Ok(gpu)
}

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build"]
fn ling_kda_indexed_pair_parity_gpu() -> Result<()> {
    let gpu = ling_gpu()?;
    let stream = gpu.default_stream();
    let kernels = LingKernels::resolve(gpu).context("ling kernel resolution")?;
    let conv_update_k = gpu.kernel("causal_conv1d", "causal_conv1d_update")?;
    ensure!(conv_update_k.0 != 0, "causal_conv1d_update missing");

    // Pool init: [SLOTS][h] FP32 + [SLOTS][conv] FP32 — one contiguous blob
    // each, the `SsmStatePool` per-layer layout (per-slot stride the full
    // blob width).
    let mut rng = Lcg(0x5eed_0001);
    let h_init: Vec<f32> = (0..SLOTS * H_BYTES / 4).map(|_| rng.f32()).collect();
    let conv_init: Vec<f32> = (0..SLOTS * CONV_BYTES / 4).map(|_| rng.f32()).collect();
    // Conv weights [CHANNELS, CONV_TAPS] BF16 + a_log [32] / dt_bias [T] f32.
    let conv_w = bf16_rows(&mut rng, CHANNELS, CONV_TAPS);
    let a_log: Vec<f32> = (0..KDA_HEADS).map(|_| rng.f32()).collect();
    let dt_bias: Vec<f32> = (0..T).map(|_| rng.f32()).collect();

    let conv_w_d = upload(gpu, &conv_w)?;
    let a_log_d = upload(gpu, &f32_bytes(&a_log))?;
    let dt_bias_d = upload(gpu, &f32_bytes(&dt_bias))?;
    let conv_weight = DenseWeight { weight: conv_w_d };

    for (rows, perm) in [
        (2usize, vec![0usize, 2usize]),
        (3, vec![3, 1, 0]),
        (4, vec![1, 2, 3, 0]),
        (5, vec![7, 5, 4, 2, 0]),
        (8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        // Slot ids: the rows' live slots; every unused lane is a pad (-1).
        // The slot-id stream: the rows' live slots, every unused lane a
        // pad (-1) — the layout `ssm_indexed_decode::slot_upload` writes.
        let mut full_ids: Vec<i32> = perm.iter().map(|&s| s as i32).collect();
        full_ids.extend(std::iter::repeat(-1).take(8 - rows));
        let host_live: Vec<i32> = perm.iter().map(|&s| s as i32).collect();
        // Row inputs: token-major q|k|v [rows, CHANNELS] + gate + beta bf16.
        let qkv_in = bf16_rows(&mut rng, rows, CHANNELS);
        let gate = bf16_rows(&mut rng, rows, T);
        let beta = bf16_rows(&mut rng, rows, KDA_HEADS);
        // Both paths' conv runs write their SiLU rows in place, so each
        // path reads from its own seeded copy of the raw projections.
        let qkv_ref_d = upload(gpu, &qkv_in)?;
        let qkv_idx_d = upload(gpu, &qkv_in)?;
        let gate_d = upload(gpu, &gate)?;
        let beta_d = upload(gpu, &beta)?;

        // Two identical copies of the pool: per-sequence reference vs the
        // indexed pair (drift cannot cross the paths).
        let h_ref = upload(gpu, &f32_bytes(&h_init))?;
        let conv_ref = upload(gpu, &f32_bytes(&conv_init))?;
        let h_pool = upload(gpu, &f32_bytes(&h_init))?;
        let conv_pool = upload(gpu, &f32_bytes(&conv_init))?;

        // Reference: the per-sequence kernels, row r on its perm slot.
        // The conv writes its SiLU bf16 row back in place (the decode
        // path's own contract) and the recurrence advances the slot's H.
        let ref_out = gpu.alloc(rows * T * 2)?;
        for r in 0..rows {
            let slot = perm[r];
            ops::conv1d_update(
                gpu,
                conv_update_k,
                conv_ref.offset(slot * CONV_BYTES),
                qkv_ref_d.offset(r * CHANNELS * 2),
                &conv_weight,
                qkv_ref_d.offset(r * CHANNELS * 2),
                CHANNELS as u32,
                CONV_TAPS as u32,
                1,
                stream,
            )?;
            ops::kda_recurrent(
                gpu,
                kernels.kda_recurrent_k,
                qkv_ref_d.offset(r * CHANNELS * 2),
                gate_d.offset(r * T * 2),
                beta_d.offset(r * KDA_HEADS * 2),
                a_log_d,
                dt_bias_d,
                h_ref.offset(slot * H_BYTES),
                ref_out.offset(r * T * 2),
                1,
                KDA_HEADS as u32,
                KDA_HEAD_DIM as u32,
                LOWER_BOUND,
                stream,
            )?;
        }

        // The indexed pair over the pool: the SsmBatchView a real step
        // builds (validating the same slot stream the kernels read).
        // Pool-view bases: bound locals — the view borrows the slices.
        let h_bases = [h_pool];
        let conv_bases = [conv_pool];
        let pool_view =
            SsmPoolView::new(&h_bases, &conv_bases, H_BYTES, H_BYTES, CONV_BYTES, SLOTS)
                .context("harness pool view")?;
        // The view validates the LIVE rows (unique, in-capacity); the
        // kernels index the same device stream, whose padding lanes hold
        // the -1 pads the production upload writes.
        let slots_all_d = upload(gpu, &i32_bytes(&full_ids))?;
        let view =
            SsmBatchView::new(pool_view, slots_all_d, &host_live).context("harness batch view")?;
        let layer = view.layer(0)?;
        // The lane submits with the LIVE rows only; the kernels read the
        // first `rows` ids of the stream — exactly the live permutation.
        let shape = KdaIndexedShape::select(rows, KDA_HEADS, KDA_HEAD_DIM, CONV_TAPS)
            .context("harness geometry selects the indexed shape")?;
        let out_idx = gpu.alloc(rows * T * 2)?;
        let qkv_bytes = rows * CHANNELS * 2;
        let conv = KdaIndexedConv {
            input: KdaBuffer {
                ptr: qkv_idx_d,
                bytes: qkv_bytes,
            },
            weight: KdaBuffer {
                ptr: conv_weight.weight,
                bytes: CHANNELS * CONV_TAPS * 2,
            },
            bias: None,
            output: KdaBuffer {
                ptr: qkv_idx_d,
                bytes: qkv_bytes,
            },
            input_stride: CHANNELS,
            output_stride: CHANNELS,
        };
        let recurrent = KdaIndexedRecurrent {
            qkv: KdaBuffer {
                ptr: qkv_idx_d,
                bytes: qkv_bytes,
            },
            gate: KdaBuffer {
                ptr: gate_d,
                bytes: rows * T * 2,
            },
            beta: KdaBuffer {
                ptr: beta_d,
                bytes: rows * KDA_HEADS * 2,
            },
            a_log: KdaBuffer {
                ptr: a_log_d,
                bytes: KDA_HEADS * 4,
            },
            dt_bias: KdaBuffer {
                ptr: dt_bias_d,
                bytes: T * 4,
            },
            output: KdaBuffer {
                ptr: out_idx,
                bytes: rows * T * 2,
            },
            lower_bound: LOWER_BOUND,
        };
        ops::validate_kda_indexed_pair(
            shape,
            layer,
            kernels.kda_conv_indexed_k,
            kernels.kda_recurrent_indexed_k,
            conv,
            recurrent,
        )?;
        ops::kda_conv_indexed(gpu, kernels.kda_conv_indexed_k, shape, layer, conv, stream)?;
        ops::kda_recurrent_indexed(
            gpu,
            kernels.kda_recurrent_indexed_k,
            shape,
            layer,
            recurrent,
            stream,
        )?;
        gpu.synchronize(stream)?;

        // Bit-exact outputs per row...
        let mut out_ref_b = vec![0u8; rows * T * 2];
        let mut out_idx_b = vec![0u8; rows * T * 2];
        gpu.copy_d2h(ref_out, &mut out_ref_b)?;
        gpu.copy_d2h(out_idx, &mut out_idx_b)?;
        for (i, (a, b)) in out_ref_b.iter().zip(&out_idx_b).enumerate() {
            ensure!(
                a == b,
                "indexed KDA output bf16 byte mismatch rows={rows} at \
                 {i}: ref {a:#06x} vs indexed {b:#06x}"
            );
        }
        // ...and bit-exact state columns, untouched slots included.
        for slot in 0..SLOTS {
            let mut h = vec![0u8; H_BYTES];
            let mut hc = vec![0u8; H_BYTES];
            let mut cv = vec![0u8; CONV_BYTES];
            let mut cvc = vec![0u8; CONV_BYTES];
            gpu.copy_d2h(h_pool.offset(slot * H_BYTES), &mut h)?;
            gpu.copy_d2h(h_ref.offset(slot * H_BYTES), &mut hc)?;
            gpu.copy_d2h(conv_pool.offset(slot * CONV_BYTES), &mut cv)?;
            gpu.copy_d2h(conv_ref.offset(slot * CONV_BYTES), &mut cvc)?;
            let (hv, hcv) = (bytes_to_f32(&h), bytes_to_f32(&hc));
            for (i, (a, b)) in hv.iter().zip(&hcv).enumerate() {
                ensure!(
                    a.to_bits() == b.to_bits(),
                    "indexed KDA h_state mismatch rows={rows} slot={slot} \
                     elem {i}: {a:e} vs {b:e}"
                );
            }
            for (i, (a, b)) in cv.iter().zip(&cvc).enumerate() {
                ensure!(
                    a == b,
                    "indexed KDA conv_state mismatch rows={rows} slot={slot} elem {i}"
                );
            }
        }
        for p in [
            h_pool,
            conv_pool,
            h_ref,
            conv_ref,
            qkv_ref_d,
            qkv_idx_d,
            gate_d,
            beta_d,
            slots_all_d,
            out_idx,
            ref_out,
        ] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
