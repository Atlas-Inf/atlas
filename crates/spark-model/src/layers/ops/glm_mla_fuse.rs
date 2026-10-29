// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_MLA_FUSE=1` (default off): the front of a GLM sparse-MLA layer,
//! from the attention norm's output to q_b, in two launches for a canonical
//! verify block of 1..=32 rows.
//!
//! The canonical path runs twelve dependent launches there: q_a and kv_a
//! (MXFP8 touch twins) each with its RMS norm, the latent cache assembly, the
//! shard's slot map and the fp8_g128 latent write, the indexer's key and pool
//! gate projections (eight CTAs each) around the key LayerNorm, the raw tail
//! write, the pool finalize and, for a sharded causal owner, the ID
//! localization (2026-10-12 RC3 nsys, C1 prose: about 88 us of a layer's
//! critical path, of which the twelve launches' own work is a third). With the
//! switch, `glm_mla_down_fuse_tc{8,16,32}` runs the four projections side by
//! side (their weights asked into L2 during the PDL wait) and
//! `glm_mla_mid_fuse` the norms, the latent write, the index cache update and
//! the localization (kernels/gb10/glm-5.3-flash/nvfp4/glm_mla_fuse.cu). Every
//! body is the per-row code of the launch it replaces, so every byte left for
//! q_b, the attention and later layers is the same
//! (`glm_mla_fuse_test_gpu.rs`).
//!
//! Both ranks must run the same value (`model::startup_parity`): the outputs
//! agree either way, but a pair is measured as one configuration.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::ShardRank;

/// The kernel module (`glm_mla_fuse.cu`).
const MODULE: &str = "glm_mla_fuse";
/// Widest block the fused front takes: the canonical tiers' (`MAX_ROWS`).
pub const MLA_FUSE_MAX_ROWS: u32 = 32;

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("ATLAS_GLM_MLA_FUSE must be 0 or 1, got {other:?}"),
    }
}

/// `ATLAS_GLM_MLA_FUSE` as set (read once).
pub fn glm_mla_fuse_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        parse(std::env::var("ATLAS_GLM_MLA_FUSE").ok().as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// The fused front's kernels: `down` per activation tier (up to 8, 16 and
/// 32 rows), and `mid`.
#[derive(Clone, Copy, Debug)]
pub struct MlaFuse {
    down: [KernelHandle; 3],
    mid: KernelHandle,
}

impl MlaFuse {
    /// The kernels of the target when the switch is on and every one of them
    /// is shipped (resolved once per process), else `None`.
    pub fn get(gpu: &dyn GpuBackend) -> Result<Option<Self>> {
        if !glm_mla_fuse_requested()? {
            return Ok(None);
        }
        static KERNELS: std::sync::OnceLock<Option<MlaFuse>> = std::sync::OnceLock::new();
        Ok(*KERNELS.get_or_init(|| {
            let this = Self::load(gpu);
            match this {
                Some(_) => tracing::info!("ATLAS_GLM_MLA_FUSE: GLM MLA front in two launches"),
                None => tracing::warn!("ATLAS_GLM_MLA_FUSE=1 ignored: target lacks {MODULE}"),
            }
            this
        }))
    }

    /// Every kernel of the target, or `None`.
    pub fn load(gpu: &dyn GpuBackend) -> Option<Self> {
        let kernel = |name| super::super::try_kernel(gpu, MODULE, name);
        let this = Self {
            down: [
                "glm_mla_down_fuse_tc8",
                "glm_mla_down_fuse_tc16",
                "glm_mla_down_fuse_tc32",
            ]
            .map(kernel),
            mid: kernel("glm_mla_mid_fuse"),
        };
        (this.mid.0 != 0 && this.down.iter().all(|k| k.0 != 0)).then_some(this)
    }
}

/// Resolve the fused front's kernels at boot (before the kernel audit seals)
/// when the switch is on for a GLM model, so its first verify never looks
/// them up.
pub fn initialize_glm_mla_fuse(
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<()> {
    if config.model_type == "glm5_next" {
        MlaFuse::get(gpu)?;
    }
    Ok(())
}

/// One MXFP8 projection: values `[n, k]` and E8M0 scales `[n, k / 32]`.
#[derive(Clone, Copy, Debug)]
pub struct MxProj {
    pub data: DevicePtr,
    pub scales: DevicePtr,
    pub out: DevicePtr,
    pub n: u32,
}

/// The four projections of `rows` rows of `input` (`[rows, k]` BF16): the
/// indexer's key and gate (BF16 `[ni, k]` weights into `[rows, ni]`), q_a and
/// kv_a. `touch`: the bytes and CTAs of the q_a / kv_a touch twins'
/// geometry ([`super::GemvTouch`]), or `None` for no touch.
#[derive(Clone, Copy, Debug)]
pub struct DownArgs {
    pub input: DevicePtr,
    pub rows: u32,
    pub k: u32,
    pub wk: DevicePtr,
    pub gate: DevicePtr,
    pub keys: DevicePtr,
    pub gates: DevicePtr,
    pub ni: u32,
    pub q_a: MxProj,
    pub kv_a: MxProj,
    pub touch: Option<(u64, u32)>,
}

/// Bytes of each weight asked into L2, in the kernel's order (q_a values and
/// scales, kv_a values and scales, index key, index gate): from the front,
/// within `budget` bytes in all.
fn touch_bytes(a: &DownArgs, budget: u64) -> [u32; 6] {
    let k = u64::from(a.k);
    let sizes = [
        u64::from(a.q_a.n) * k,
        u64::from(a.q_a.n) * k / 32,
        u64::from(a.kv_a.n) * k,
        u64::from(a.kv_a.n) * k / 32,
        u64::from(a.ni) * k * 2,
        u64::from(a.ni) * k * 2,
    ];
    let mut left = budget;
    sizes.map(|size| {
        let take = size.min(left).min(u64::from(u32::MAX));
        left -= take;
        take as u32
    })
}

/// The norms, the latent write, the index cache update and (optionally) the
/// causal localization of `rows` rows.
#[derive(Clone, Copy, Debug)]
pub struct MidArgs {
    pub rows: u32,
    pub q_latent: DevicePtr,
    pub q_norm: DevicePtr,
    pub q_dim: u32,
    pub kv_latent: DevicePtr,
    pub kv_norm: DevicePtr,
    pub kv_dim: u32,
    pub eps: f32,
    /// The layer's fp8_g128 latent pool (a shard's local pool), `slots` the
    /// rows' global cache slots (i64), `owner` the shard's rule (world 1:
    /// unsharded).
    pub latent_cache: DevicePtr,
    pub slots: DevicePtr,
    pub block_size: u32,
    pub owner: ShardRank,
    pub index: IndexCache,
    /// Causal localization of the merge form: `(table, ids, counts, width,
    /// causal_start)`.
    pub localize: Option<(DevicePtr, DevicePtr, DevicePtr, u32, u32)>,
}

/// The semantic-index cache update's operands (`glm_index_prefill_cache_update`).
#[derive(Clone, Copy, Debug)]
pub struct IndexCache {
    pub keys: DevicePtr,
    pub gates: DevicePtr,
    pub norm_weight: DevicePtr,
    pub norm_bias: DevicePtr,
    pub eps: f32,
    pub dim: u32,
    pub tail: DevicePtr,
    pub tail_map: DevicePtr,
    pub tail_stride: u64,
    pub ape: DevicePtr,
    pub pool: DevicePtr,
    pub pool_stride: u64,
    pub tokens_per_pool: u32,
}

impl MlaFuse {
    pub fn launch_down(&self, gpu: &dyn GpuBackend, a: &DownArgs, stream: u64) -> Result<()> {
        ensure!(
            (1..=MLA_FUSE_MAX_ROWS).contains(&a.rows)
                && a.k.is_multiple_of(64)
                && a.ni.is_multiple_of(16)
                && a.q_a.n.is_multiple_of(16)
                && a.kv_a.n.is_multiple_of(16),
            "fused GLM MLA projections: unsupported shape (rows {}, k {}, n {}/{}/{})",
            a.rows,
            a.k,
            a.ni,
            a.q_a.n,
            a.kv_a.n
        );
        let tier = match a.rows {
            1..=8 => 0,
            9..=16 => 1,
            _ => 2,
        };
        let (budget, ctas) = a.touch.unwrap_or((0, 1));
        let touch = touch_bytes(a, budget);
        let grid = 2 * div_ceil(a.ni, 16) + div_ceil(a.q_a.n, 16) + div_ceil(a.kv_a.n, 16);
        let mut launch = KernelLaunch::new(gpu, self.down[tier])
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.input)
            .arg_ptr(a.wk)
            .arg_ptr(a.gate)
            .arg_ptr(a.q_a.data)
            .arg_ptr(a.q_a.scales)
            .arg_ptr(a.kv_a.data)
            .arg_ptr(a.kv_a.scales)
            .arg_ptr(a.keys)
            .arg_ptr(a.gates)
            .arg_ptr(a.q_a.out)
            .arg_ptr(a.kv_a.out)
            .arg_u32(a.rows)
            .arg_u32(a.k)
            .arg_u32(a.ni)
            .arg_u32(a.q_a.n)
            .arg_u32(a.kv_a.n);
        for bytes in touch {
            launch = launch.arg_u32(bytes);
        }
        launch.arg_u32(ctas).launch(stream)
    }

    pub fn launch_mid(&self, gpu: &dyn GpuBackend, a: &MidArgs, stream: u64) -> Result<()> {
        let i = &a.index;
        ensure!(
            (1..=MLA_FUSE_MAX_ROWS).contains(&a.rows)
                && a.q_dim <= 2048
                && a.kv_dim == 512
                && i.dim == 128
                && a.owner.world >= 1
                && a.owner.rank < a.owner.world,
            "fused GLM MLA front: unsupported shape (rows {}, q {}, kv {}, index {})",
            a.rows,
            a.q_dim,
            a.kv_dim,
            i.dim
        );
        let (table, ids, counts, width, causal_start) =
            a.localize
                .unwrap_or((DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL, 0, 0));
        ensure!(
            a.localize.is_none() || (1..=1023 * a.block_size).contains(&width),
            "fused GLM MLA localization of {width} IDs over {}-token blocks",
            a.block_size
        );
        let loc_rows = if a.localize.is_some() { a.rows } else { 0 };
        KernelLaunch::new(gpu, self.mid)
            .grid([1 + 2 * a.rows + loc_rows, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(a.q_latent)
            .arg_ptr(a.q_norm)
            .arg_ptr(a.kv_latent)
            .arg_ptr(a.kv_norm)
            .arg_u32(a.q_dim)
            .arg_u32(a.kv_dim)
            .arg_f32(a.eps)
            .arg_ptr(a.latent_cache)
            .arg_ptr(a.slots)
            .arg_u32(a.block_size)
            .arg_u32(a.owner.rank)
            .arg_u32(a.owner.world)
            .arg_ptr(i.keys)
            .arg_ptr(i.gates)
            .arg_ptr(i.norm_weight)
            .arg_ptr(i.norm_bias)
            .arg_f32(i.eps)
            .arg_u32(i.dim)
            .arg_ptr(i.tail)
            .arg_ptr(i.tail_map)
            .arg_u64(i.tail_stride)
            .arg_ptr(i.ape)
            .arg_ptr(i.pool)
            .arg_u64(i.pool_stride)
            .arg_u32(i.tokens_per_pool)
            .arg_ptr(table)
            .arg_ptr(ids)
            .arg_ptr(counts)
            .arg_u32(width)
            .arg_u32(causal_start)
            .arg_u32(a.rows)
            .arg_u32(loc_rows)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_explicit() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        for value in ["", "true", "2", " 1"] {
            assert!(parse(Some(value)).is_err());
        }
    }

    #[test]
    fn touch_fills_the_budget_from_the_front() {
        let p = |n| MxProj {
            data: DevicePtr::NULL,
            scales: DevicePtr::NULL,
            out: DevicePtr::NULL,
            n,
        };
        let a = DownArgs {
            input: DevicePtr::NULL,
            rows: 3,
            k: 4096,
            wk: DevicePtr::NULL,
            gate: DevicePtr::NULL,
            keys: DevicePtr::NULL,
            gates: DevicePtr::NULL,
            ni: 128,
            q_a: p(1536),
            kv_a: p(512),
            touch: None,
        };
        // GLM-5.3: every weight fits the 12 MiB default.
        let all = [
            1536 * 4096,
            1536 * 128,
            512 * 4096,
            512 * 128,
            128 * 8192,
            128 * 8192,
        ];
        assert_eq!(touch_bytes(&a, 12 << 20), all);
        // A smaller budget stops partway, then touches nothing.
        let budget = 1536 * 4096 + 1000;
        assert_eq!(touch_bytes(&a, budget), [1536 * 4096, 1000, 0, 0, 0, 0]);
        assert_eq!(touch_bytes(&a, 0), [0; 6]);
    }
}

#[cfg(test)]
#[path = "glm_mla_fuse_test_gpu.rs"]
mod gpu_tests;
