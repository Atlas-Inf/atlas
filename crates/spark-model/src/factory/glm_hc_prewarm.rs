// SPDX-License-Identifier: AGPL-3.0-only
//! Optional HC TF32 initialization using dead startup arena storage, before KV sizing.
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const FLAG: &str = "ATLAS_GLM_HC_TF32_PREWARM";
/// Warmed shapes after the full arena: a 4K chunk and a common final chunk.
const ROWS: [u32; 3] = [4100, 4096, 3515];

fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("{FLAG} must be 0 or 1"),
    }
}

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Copy)]
struct Storage {
    activation: DevicePtr,
    activation_capacity: usize,
    scratch: DevicePtr,
    scratch_capacity: usize,
}

struct Plan {
    storage: Storage,
    activation_bytes: usize,
    scratch_bytes: usize,
    output: DevicePtr,
    max_rows: u32,
    n: u32,
    k: u32,
}

impl Plan {
    /// The full-arena shape first, so its library residency lands before
    /// the KV free snapshot, then the remaining `ROWS`.
    fn rows(&self) -> Vec<u32> {
        let mut rows = vec![self.max_rows];
        rows.extend(ROWS.into_iter().filter(|&m| m < self.max_rows));
        rows
    }
}

fn bytes(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm byte count overflow"))
}

fn span_end(ptr: DevicePtr, capacity: usize) -> Result<u64> {
    ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(256),
        "HC TF32 prewarm requires non-null 256-byte-aligned arena pointers"
    );
    ptr.0
        .checked_add(u64::try_from(capacity)?)
        .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm pointer span overflow"))
}

impl Plan {
    fn new(
        config: &ModelConfig,
        max_rows: usize,
        hc_flag: Option<&str>,
        storage: Storage,
    ) -> Result<Self> {
        ensure!(
            config.model_type == "glm5_next"
                && config.hidden_size == 4096
                && config.hc_mult == 4
                && max_rows >= ROWS[0] as usize
                && config.max_batch_tokens == max_rows,
            "HC TF32 prewarm requires GLM H4096/HC4 and a >=4100-row arena"
        );
        ensure!(
            matches!(hc_flag, Some("1" | "true" | "yes")),
            "HC TF32 prewarm requires ATLAS_HC_CUBLAS_PREFILL enabled"
        );
        let n = config
            .hc_mult
            .checked_add(2)
            .and_then(|count| count.checked_mul(config.hc_mult))
            .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm mix dimension overflow"))?;
        let k = config
            .hidden_size
            .checked_mul(config.hc_mult)
            .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm input dimension overflow"))?;
        let activation_bytes = bytes(max_rows, k)?;
        let weight_bytes = bytes(n, k)?;
        let scratch_bytes = weight_bytes
            .checked_add(bytes(max_rows, n)?)
            .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm scratch span overflow"))?;
        ensure!(
            storage.activation_capacity >= activation_bytes
                && storage.scratch_capacity >= scratch_bytes,
            "HC TF32 prewarm arena spans are too small"
        );
        let activation_end = span_end(storage.activation, storage.activation_capacity)?;
        let scratch_end = span_end(storage.scratch, storage.scratch_capacity)?;
        ensure!(
            activation_end <= storage.scratch.0 || scratch_end <= storage.activation.0,
            "HC TF32 prewarm arena allocations overlap"
        );
        let output = DevicePtr(
            storage
                .scratch
                .0
                .checked_add(u64::try_from(weight_bytes)?)
                .ok_or_else(|| anyhow::anyhow!("HC TF32 prewarm output pointer overflow"))?,
        );
        ensure!(
            output.0.is_multiple_of(256),
            "HC TF32 prewarm output alignment"
        );
        Ok(Self {
            storage,
            activation_bytes,
            scratch_bytes,
            output,
            max_rows: u32::try_from(max_rows)?,
            n: u32::try_from(n)?,
            k: u32::try_from(k)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Projection {
    activation: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
}

/// The launch boundary is injected so the CPU test exercises the real sequence
/// and arguments without pretending that the mock backend executes cuBLAS.
fn execute(
    gpu: &dyn GpuBackend,
    plan: &Plan,
    stream: u64,
    mut project: impl FnMut(Projection) -> Result<()>,
) -> Result<()> {
    let result = (|| {
        gpu.memset_async(plan.storage.activation, 0, plan.activation_bytes, stream)?;
        gpu.memset_async(plan.storage.scratch, 0, plan.scratch_bytes, stream)?;
        for m in plan.rows() {
            project(Projection {
                activation: plan.storage.activation,
                weight: plan.storage.scratch,
                output: plan.output,
                m,
                n: plan.n,
                k: plan.k,
                stream,
            })?;
        }
        Ok(())
    })();
    // Always drain queued work, including a partial warmup, before returning.
    // On success this also places library residency before the KV free snapshot.
    let drained = gpu
        .synchronize(stream)
        .context("HC TF32 prewarm stream drain");
    result.and(drained)
}

pub(super) fn initialize(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    buffers: &BufferArena,
    max_rows: usize,
) -> Result<()> {
    if !parse(env_value(FLAG)?.as_deref())? {
        return Ok(());
    }
    ensure!(cfg!(feature = "cuda"), "{FLAG}=1 requires CUDA");
    let hc_flag = env_value("ATLAS_HC_CUBLAS_PREFILL")?;
    let sizes = buffers.sizes();
    let plan = Plan::new(
        config,
        max_rows,
        hc_flag.as_deref(),
        Storage {
            activation: buffers.hc_streams(),
            activation_capacity: sizes.hc_streams,
            scratch: buffers.gate_logits_f32(),
            scratch_capacity: sizes.gate_logits_f32,
        },
    )?;
    let stream = gpu.default_stream();
    let free_before = gpu.free_memory()?;
    let started = std::time::Instant::now();
    execute(gpu, &plan, stream, |call| {
        spark_runtime::cublaslt::tf32_gemm_act_weight_t(
            call.activation.0,
            call.weight.0,
            call.output.0,
            call.m,
            call.n,
            call.k,
            call.stream,
        )
    })
    .context("requested HC TF32 startup prewarm failed")?;
    let free_after = gpu.free_memory()?;
    tracing::info!(
        rank = config.ep_rank,
        rows = ?plan.rows(),
        n = plan.n,
        k = plan.k,
        elapsed_ms = started.elapsed().as_secs_f64() * 1000.,
        free_before,
        free_after,
        "HC TF32 prewarm completed before KV accounting"
    );
    Ok(())
}

#[cfg(test)]
#[path = "glm_hc_prewarm_tests.rs"]
mod tests;
