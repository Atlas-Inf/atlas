// SPDX-License-Identifier: AGPL-3.0-only

//! Packed-EXL3 MoE forward. Two arms, both on the packed trellis (never NVFP4,
//! never a resident BF16 copy of the experts):
//!
//! * **CoopMK** (`ATLAS_EXL3_COOPMK=1`, off by default): the vendored
//!   vcruz305/exllamav3 runtime-K cooperative kernels (`coopmk_a_kernel` /
//!   `coopmk_b_kernel`) decode every selected expert straight from its
//!   trellis, bitrate from a device table, shared expert as one extra slot.
//!   Routing stays on device: no host sync, so CUDA-graph capture is allowed.
//!   Multi-token calls (prefill) loop rows through the same bsz-1 launch
//!   unless `ATLAS_EXL3_COOPMK_PREFILL=0`.
//! * **Interim** (default, and the fallback when a layer did not bind
//!   CoopMK): for each selected expert, reconstruct the trellis, apply the
//!   Hadamard scales and run the fp16 hgemm. Host-routed (top-k ids read
//!   back), so CUDA-graph capture is refused.
//!
//! Debug / measurement knobs:
//! * `ATLAS_EXL3_COOPMK_TOGGLE_FILE=<path>`: a poller thread re-reads the file
//!   every 200 ms; content `1` selects CoopMK, anything else the interim arm
//!   (A/B/A inside one serve; tables are built at load when this is set).
//! * `ATLAS_EXL3_COOPMK_PARITY=N`: the first N single-row calls run BOTH arms
//!   on the same input and log max |diff| and cosine of the MoE output
//!   (host readback; debug only).

use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::{Exl3CoopMk, exl3_linear_bf16, exl3_reconstruct};
use crate::weight_map::exl3::Exl3Weight;
use crate::weight_map::exl3::moe_pack::{Exl3MoePack, with_exl3_moe};

use super::*;

static LOGGED: AtomicBool = AtomicBool::new(false);
static LOGGED_COOPMK: AtomicBool = AtomicBool::new(false);
static COOPMK_ON: AtomicBool = AtomicBool::new(false);
static COOPMK_INIT: Once = Once::new();
/// Rows dispatched per arm (proof of which kernels ran in each A/B phase).
static CALLS_COOPMK: AtomicU64 = AtomicU64::new(0);
static CALLS_INTERIM: AtomicU64 = AtomicU64::new(0);
static PARITY_DONE: AtomicU64 = AtomicU64::new(0);

fn env_is(name: &str, v: &str) -> bool {
    std::env::var(name).map(|s| s.trim() == v).unwrap_or(false)
}

/// Whether the CoopMK arm is selected right now (env default, overridden by
/// the toggle file when one is configured).
fn coopmk_selected() -> bool {
    COOPMK_INIT.call_once(|| {
        let off = std::env::var("ATLAS_EXL3_COOPMK")
            .map(|s| s.trim() == "0")
            .unwrap_or(false);
        COOPMK_ON.store(!off, Ordering::Relaxed);
        if let Ok(path) = std::env::var("ATLAS_EXL3_COOPMK_TOGGLE_FILE") {
            if let Ok(s) = std::fs::read_to_string(&path) {
                COOPMK_ON.store(s.trim() == "1", Ordering::Relaxed);
            }
            tracing::info!(
                "EXL3 CoopMK toggle file {path}: start {} (poll 200 ms)",
                if COOPMK_ON.load(Ordering::Relaxed) {
                    "coopmk"
                } else {
                    "interim"
                }
            );
            let _ = std::thread::Builder::new()
                .name("exl3-coopmk-toggle".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        let Ok(s) = std::fs::read_to_string(&path) else {
                            continue;
                        };
                        let v = s.trim() == "1";
                        if COOPMK_ON.swap(v, Ordering::Relaxed) != v {
                            tracing::info!(
                                "EXL3 CoopMK toggle -> {} (rows so far: coopmk {}, interim {})",
                                if v { "coopmk" } else { "interim" },
                                CALLS_COOPMK.load(Ordering::Relaxed),
                                CALLS_INTERIM.load(Ordering::Relaxed)
                            );
                        }
                    }
                });
        }
    });
    COOPMK_ON.load(Ordering::Relaxed)
}

fn coopmk_prefill_allowed() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !env_is("ATLAS_EXL3_COOPMK_PREFILL", "0"))
}

fn parity_budget() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_EXL3_COOPMK_PARITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn bf16_to_f64(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect()
}

impl MoeLayer {
    /// `true` when this layer's experts were registered as packed EXL3.
    pub(super) fn exl3_packed(&self) -> bool {
        crate::weight_map::exl3::moe_pack::exl3_moe_registered(
            self.weights.shared_expert_gate.weight.0,
        )
    }

    /// Run `num_tokens` rows on the packed path. Returns whether it ran.
    pub(super) fn try_forward_exl3(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !self.exl3_packed() {
            return Ok(false);
        }
        self.forward_exl3_rows(input, num_tokens, ctx, stream)?;
        Ok(true)
    }

    fn forward_exl3_rows(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let out = ctx.buffers.moe_output();
        let gate_key = self.weights.shared_expert_gate.weight.0;
        let use_coopmk = coopmk_selected()
            && (num_tokens == 1 || coopmk_prefill_allowed())
            && with_exl3_moe(gate_key, |p| p.coopmk.is_some()).unwrap_or(false);
        let parity_left = parity_budget().saturating_sub(PARITY_DONE.load(Ordering::Relaxed));
        let parity = parity_left > 0 && num_tokens == 1 && !ctx.graph_capture;
        if ctx.graph_capture && (!use_coopmk || parity) {
            bail!(
                "EXL3 packed MoE interim path reads top-k on the host and cannot run inside a CUDA graph"
            );
        }
        if use_coopmk {
            if LOGGED_COOPMK
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                tracing::info!(
                    "EXL3 MoE CoopMK: routed + shared experts decode from packed trellis via coopmk_a/b (no reconstruct, no host sync)"
                );
            }
        } else if LOGGED
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            tracing::info!(
                "EXL3 MoE interim: selected experts run reconstruct + Hadamard + hgemm (not NVFP4, not CoopMK)"
            );
        }
        for t in 0..num_tokens {
            let row_in = input.offset(t * h * 2);
            let row_out = out.offset(t * h * 2);
            self.forward_exl3_one(row_in, row_out, use_coopmk, parity, ctx, stream)?;
        }
        Ok(())
    }

    fn forward_exl3_one(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        use_coopmk: bool,
        parity: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        let router_in = self.router_input(input, 1, h, ctx, stream)?;
        let gate_logits = ctx.buffers.gate_logits();
        // Router stays BF16 on the packed path: the checkpoint gate is BF16
        // and a 4-bit router cannot separate 512 tightly clustered logits.
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv,
            router_in,
            &self.weights.gate,
            gate_logits,
            self.router_logits_n,
            h,
            stream,
        )?;
        if self.tid2eid_dev.is_some() || self.correction_bias_dev.is_some() {
            bail!("EXL3 packed MoE path supports softmax top-k only (no hash or bias routing)");
        }
        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk,
            gate_logits,
            indices_dev,
            weights_dev,
            ctx.config.num_experts as u32,
            top_k,
            ctx.config.norm_topk_prob,
            stream,
        )?;

        let gate_key = self.weights.shared_expert_gate.weight.0;
        let shared_gate_w = self.weights.shared_expert_gate.weight;
        if use_coopmk && !parity {
            CALLS_COOPMK.fetch_add(1, Ordering::Relaxed);
            return with_exl3_moe(gate_key, |pack| {
                let c = pack.coopmk.as_ref().expect("coopmk checked by caller");
                c.run(
                    ctx.gpu,
                    input,
                    indices_dev,
                    weights_dev,
                    shared_gate_w,
                    output,
                    stream,
                )
            })
            .ok_or_else(|| anyhow::anyhow!("EXL3 MoE pack missing for gate {gate_key:#x}"))?;
        }

        // Interim arm: host readback of the top-k ids.
        CALLS_INTERIM.fetch_add(1, Ordering::Relaxed);
        let k = top_k as usize;
        let mut idx_buf = vec![0u8; k * 4];
        ctx.gpu
            .copy_d2h_on_stream(indices_dev, &mut idx_buf, stream)?;
        let indices: Vec<u32> = (0..k)
            .map(|i| u32::from_le_bytes(idx_buf[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();

        with_exl3_moe(gate_key, |pack| -> Result<()> {
            self.run_selected(input, output, pack, &indices, weights_dev, ctx, stream)?;
            if let Some(c) = pack.coopmk.as_ref().filter(|_| parity) {
                self.coopmk_parity(
                    pack,
                    c,
                    input,
                    output,
                    indices_dev,
                    weights_dev,
                    use_coopmk,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        })
        .ok_or_else(|| anyhow::anyhow!("EXL3 MoE pack missing for gate {gate_key:#x}"))??;
        Ok(())
    }

    /// Debug: CoopMK on the same input and routing as the interim result in
    /// `output`; logs max |diff| and cosine. When the CoopMK arm is selected
    /// its result replaces `output`, so the trajectory follows the chosen arm.
    #[allow(clippy::too_many_arguments)]
    fn coopmk_parity(
        &self,
        pack: &Exl3MoePack,
        c: &Exl3CoopMk,
        input: DevicePtr,
        output: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        use_coopmk: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = PARITY_DONE.fetch_add(1, Ordering::Relaxed);
        let h = ctx.config.hidden_size;
        c.run(
            ctx.gpu,
            input,
            indices_dev,
            weights_dev,
            self.weights.shared_expert_gate.weight,
            c.parity_out,
            stream,
        )?;
        ctx.gpu.synchronize(stream)?;
        let mut a = vec![0u8; h * 2];
        let mut b = vec![0u8; h * 2];
        ctx.gpu.copy_d2h_on_stream(output, &mut a, stream)?;
        ctx.gpu.copy_d2h_on_stream(c.parity_out, &mut b, stream)?;
        let (a, b) = (bf16_to_f64(&a), bf16_to_f64(&b));
        let (mut dot, mut na, mut nb, mut maxd, mut maxa, mut sq) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
            maxd = maxd.max((x - y).abs());
            maxa = maxa.max(x.abs());
            sq += (x - y) * (x - y);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-300);
        let finite = b.iter().all(|v| v.is_finite());
        tracing::info!(
            "EXL3 CoopMK parity #{n} {}: max|d| {:.6e} (max|interim| {:.4e}, rel {:.3e}) rms_d {:.4e} cos {:.8} finite {}",
            pack.name,
            maxd,
            maxa,
            maxd / maxa.max(1e-30),
            (sq / h as f64).sqrt(),
            cos,
            finite
        );
        if use_coopmk {
            ctx.gpu
                .copy_d2d_async(c.parity_out, output, h * 2, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_selected(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        pack: &Exl3MoePack,
        indices: &[u32],
        weights_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let down = ctx.buffers.expert_down_out();
        for (slot, &id) in indices.iter().enumerate() {
            let expert = pack.experts.get(id as usize).ok_or_else(|| {
                anyhow::anyhow!(
                    "EXL3 MoE expert id {id} out of range ({})",
                    pack.experts.len()
                )
            })?;
            if expert.is_null() {
                bail!("EXL3 MoE selected expert {id} has no packed weights");
            }
            let slot_out = down.offset(slot * h * 2);
            self.exl3_swiglu(
                input,
                &expert.gate,
                &expert.up,
                &expert.down,
                slot_out,
                pack,
                ctx,
                stream,
            )?;
        }
        if pack.shared.is_null() {
            ctx.gpu
                .memset_async(pack.scratch.shared_out, 0, h * 2, stream)?;
        } else {
            self.exl3_swiglu(
                input,
                &pack.shared.gate,
                &pack.shared.up,
                &pack.shared.down,
                pack.scratch.shared_out,
                pack,
                ctx,
                stream,
            )?;
        }
        ops::moe_weighted_sum_blend(
            ctx.gpu,
            self.moe_weighted_sum_blend,
            output,
            down,
            weights_dev,
            pack.scratch.shared_out,
            input,
            self.weights.shared_expert_gate.weight,
            h as u32,
            indices.len() as u32,
            h as u32,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn exl3_swiglu(
        &self,
        input: DevicePtr,
        gate: &Exl3Weight,
        up: &Exl3Weight,
        down_w: &Exl3Weight,
        out: DevicePtr,
        pack: &Exl3MoePack,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        self.exl3_proj(input, gate, gate_out, pack, ctx, stream)?;
        self.exl3_proj(input, up, up_out, pack, ctx, stream)?;
        let n = gate.shape.out_features as u32;
        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            gate_out,
            up_out,
            pack.scratch.silu,
            n,
            stream,
        )?;
        self.exl3_proj(pack.scratch.silu, down_w, out, pack, ctx, stream)
    }

    fn exl3_proj(
        &self,
        x: DevicePtr,
        w: &Exl3Weight,
        out: DevicePtr,
        pack: &Exl3MoePack,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // `w_inner` is one shared tile: fill it for THIS projection first.
        exl3_reconstruct(ctx.gpu, &pack.kernels, w, pack.scratch.w_inner, stream)?;
        exl3_linear_bf16(
            ctx.gpu,
            &pack.kernels,
            x,
            1,
            w,
            pack.scratch.w_inner,
            pack.scratch.xh,
            pack.scratch.y,
            out,
            stream,
        )
    }
}
