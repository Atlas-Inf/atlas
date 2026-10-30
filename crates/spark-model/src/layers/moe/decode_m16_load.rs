// SPDX-License-Identifier: AGPL-3.0-only

//! [`DecodeM16`]'s kernel selection from the decode flags, split from
//! `decode_m16.rs` for the file-size cap.

use super::*;

impl DecodeM16 {
    pub(in crate::layers::moe) fn new(
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let toggle = |name: &str| match std::env::var(name).as_deref() {
            Err(_) | Ok("0") => Ok(false),
            Ok("1") => Ok(true),
            Ok(value) => anyhow::bail!("{name} requires 0 or 1, got {value:?}"),
        };
        let requested = toggle("ATLAS_GLM_MOE_DECODE_M16")?;
        let zskip = toggle("ATLAS_GLM_MOE_DOWN_ZSKIP")?;
        let stream = toggle("ATLAS_GLM_MOE_DECODE_STREAM")?;
        // Some(whether the downs prefetch too).
        let l2pf = match std::env::var("ATLAS_GLM_MOE_DECODE_L2PF").as_deref() {
            Err(_) | Ok("0") => None,
            Ok("1") => Some(true),
            Ok("2") => Some(false),
            Ok(value) => {
                anyhow::bail!("ATLAS_GLM_MOE_DECODE_L2PF requires 0, 1 or 2, got {value:?}")
            }
        };
        let glm = config.model_type == "glm5_next";
        let on = requested && glm;
        let kernel = |on, name: &str| crate::layers::try_kernel_gated(on, gpu, "moe_w4a16", name);
        // A family's fused gate/up and (zero-skipping) down.
        let pair = |on, family: &str, suffix: &str| {
            let skip = if zskip { "_zskip" } else { "" };
            [
                kernel(
                    on,
                    &format!("glm_moe_decode_{family}_gate_up_silu_k128w{suffix}"),
                ),
                kernel(on, &format!("glm_moe_decode_{family}_k128w{skip}{suffix}")),
            ]
        };
        // A set of twins of both slabs, all or none: a target lacking any of
        // the stream twins runs the M16 pair, as without the flag, and one
        // lacking any of their prefetching twins the stream twins.
        let twins = |on, suffix| {
            let twins = [pair(on, "m16s", suffix), pair(on, "m32s", suffix)];
            twins.iter().flatten().all(|k| k.0 != 0).then_some(twins)
        };
        let streams = twins(on && stream, "");
        let prefetch = twins(streams.is_some() && l2pf.is_some(), "_l2pf");
        let persist = toggle("ATLAS_GLM_MOE_DECODE_PERSIST")?;
        let persistent = twins(
            persist && prefetch.is_some() && l2pf == Some(true) && zskip,
            "_l2pf_p",
        );
        let [[mut gate_up_silu, mut down], mut wide] =
            streams.unwrap_or_else(|| [pair(on, "m16", ""), [KernelHandle(0); 2]]);
        if let Some([[gate_up16, down16], [gate_up32, down32]]) = persistent.or(prefetch) {
            [gate_up_silu, wide[0]] = [gate_up16, gate_up32];
            if l2pf == Some(true) {
                [down, wide[1]] = [down16, down32];
            }
        }
        // Both or neither, as every pair.
        let wide4 = Some(pair(persistent.is_some(), "m64s", "_l2pf_p"))
            .filter(|p| p.iter().all(|k| k.0 != 0))
            .unwrap_or([KernelHandle(0); 2]);
        let ctas = match persistent {
            Some(_) => gpu.sm_count()?,
            None => 0,
        };
        let mut this = Self {
            gate_up_silu,
            down,
            wide,
            wide4,
            zskip,
            k128w: toggle("ATLAS_GLM_MOE_DECODE_K128W")? && glm,
            nosync: toggle("ATLAS_GLM_MOE_STREAM_NOSYNC")? && glm,
            ctas,
        };
        if !this.loaded() {
            // Both or neither: half a pair never launches.
            this.gate_up_silu = KernelHandle(0);
        }
        if this.nosync && gpu.op_cache().once("moe:stream_nosync") {
            if this.stream() {
                tracing::info!(
                    "ATLAS_GLM_MOE_STREAM_NOSYNC: no expert-offset read for the stream twins' batches"
                );
            } else {
                tracing::warn!(
                    "ATLAS_GLM_MOE_STREAM_NOSYNC=1 ignored: it needs the stream twins (ATLAS_GLM_MOE_DECODE_M16=1 ATLAS_GLM_MOE_DECODE_STREAM=1)"
                );
            }
        }
        if (requested || zskip || stream || l2pf.is_some() || persist)
            && gpu.op_cache().once("moe:decode_m16")
        {
            if this.loaded() {
                tracing::info!(
                    "ATLAS_GLM_MOE_DECODE_M16: M16 routed gate/up+SiLU and down for verify decode (zero-row skip: {zskip})"
                );
                if stream && this.stream() {
                    tracing::info!(
                        "ATLAS_GLM_MOE_DECODE_STREAM: stream-loaded M16 twins for every routed FFN of up to {STREAM_MAX_ROWS} rows"
                    );
                } else if stream {
                    tracing::warn!(
                        "ATLAS_GLM_MOE_DECODE_STREAM=1 ignored: target lacks the stream kernels"
                    );
                }
                if let Some(downs) = l2pf {
                    if prefetch.is_some() {
                        tracing::info!(
                            "ATLAS_GLM_MOE_DECODE_L2PF: stream twins with the L2 row prefetch (gate/up, downs: {downs})"
                        );
                    } else {
                        tracing::warn!(
                            "ATLAS_GLM_MOE_DECODE_L2PF ignored: needs the stream twins and the target's _l2pf twins"
                        );
                    }
                }
                if persistent.is_some() {
                    tracing::info!(
                        "ATLAS_GLM_MOE_DECODE_PERSIST: persistent prefetching twins over {ctas} CTAs (four slabs to {} rows: {})",
                        this.max_rows(),
                        wide4[0].0 != 0
                    );
                } else if persist {
                    tracing::warn!(
                        "ATLAS_GLM_MOE_DECODE_PERSIST=1 ignored: needs ATLAS_GLM_MOE_DECODE_L2PF=1, ATLAS_GLM_MOE_DOWN_ZSKIP=1 and the target's _l2pf_p twins"
                    );
                }
            } else if on {
                tracing::warn!("ATLAS_GLM_MOE_DECODE_M16=1 ignored: target lacks the M16 kernels");
            } else if !requested {
                tracing::warn!(
                    "ATLAS_GLM_MOE_DOWN_ZSKIP / _DECODE_STREAM / _DECODE_L2PF / _DECODE_PERSIST ignored: they need ATLAS_GLM_MOE_DECODE_M16=1"
                );
            }
        }
        Ok(this)
    }
}
