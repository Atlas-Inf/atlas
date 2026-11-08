// SPDX-License-Identifier: AGPL-3.0-only

//! DSpine code-bridge weights (`tencent/Qwen3.8-27B-DSpine-b16`, Apache-2.0).
//!
//! DSpine (arXiv 2609.36173, github.com/Tencent/DSpine) is a DFlash block
//! drafter whose every layer ends in a *code bridge*: each block row reads a
//! unit code from its own hidden, and every row past the anchor receives a
//! gated, low-rank message from its predecessor's code. The tensor names and
//! shapes below are the checkpoint's (`selected_prefix_bridge.*`); this file
//! only loads and validates them — the forward is `layers/dflash_head/dspine`.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{DflashConfig, DspineSpec};
use crate::weight_map::DenseWeight;

/// The bridge of an `L`-layer DSpine drafter with code size `C`, rank `R`,
/// hidden `H` and vocabulary `V`.
pub struct DspineBridgeWeights {
    pub spec: DspineSpec,
    /// `[V, C]` anchor codes (scaled by `sqrt(C)` at use).
    pub codebook: DenseWeight,
    /// Per layer `[C, H]`: the code read of layer `l` (`early_read_weights[l]`
    /// below the last layer, `read_weight` for it).
    pub read: Vec<DevicePtr>,
    /// Per layer `[H, R]`: the feature write of layer `l`.
    pub write: Vec<DevicePtr>,
    /// `[3R, C]` = `[gate; predecessor_gate; prev]`, packed once at load.
    pub packed: DevicePtr,
    /// `[R]` gate bias.
    pub gate_bias: DenseWeight,
}

fn tensor<'a>(
    store: &'a WeightStore,
    name: &str,
    shape: &[usize],
) -> Result<&'a spark_runtime::weights::WeightTensor> {
    let t = store.get(name).with_context(|| format!("DSpine: missing {name}"))?;
    anyhow::ensure!(
        t.dtype == WeightDtype::BF16 && t.shape == shape,
        "DSpine: {name} is {:?} {:?}, expected BF16 {shape:?}",
        t.dtype,
        t.shape
    );
    Ok(t)
}

/// The I64 scalar `name` (a checkpoint format version) must read 1.
fn version_one(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<()> {
    let t = store.get(name).with_context(|| format!("DSpine: missing {name}"))?;
    anyhow::ensure!(
        t.dtype == WeightDtype::Int64 && t.num_elements() == 1,
        "DSpine: {name} is not an I64 scalar"
    );
    let mut b = [0u8; 8];
    gpu.copy_d2h(t.ptr, &mut b)?;
    anyhow::ensure!(
        i64::from_le_bytes(b) == 1,
        "DSpine: unsupported {name} = {}",
        i64::from_le_bytes(b)
    );
    Ok(())
}

/// Load the bridge when `config` declares one (`Ok(None)` otherwise).
pub fn load_dspine_bridge(
    store: &WeightStore,
    config: &DflashConfig,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<Option<DspineBridgeWeights>> {
    let Some(spec) = config.dspine()? else {
        return Ok(None);
    };
    let (l, h, v) = (
        config.num_hidden_layers,
        config.hidden_size,
        config.vocab_size,
    );
    let (c, r) = (spec.code_size, spec.rank);
    anyhow::ensure!(l >= 1, "DSpine: drafter has no layers");
    let b = |n: &str| format!("{prefix}selected_prefix_bridge.{n}");
    for name in [
        b("independent_write_version"),
        b("terminal_disabled_version"),
        format!("{prefix}draft_block_causal_version"),
        format!("{prefix}last_write_refine_version"),
    ] {
        version_one(store, gpu, &name)?;
    }
    // The terminal projection is disabled in this format: it must be zero.
    let out = tensor(store, &b("out.weight"), &[c, r])?;
    let mut bytes = vec![0u8; c * r * 2];
    gpu.copy_d2h(out.ptr, &mut bytes)?;
    anyhow::ensure!(
        bytes.chunks_exact(2).all(|w| (u16::from_le_bytes([w[0], w[1]]) & 0x7fff) == 0),
        "DSpine: terminal-disabled checkpoint carries a nonzero out.weight"
    );

    let codebook = tensor(store, &b("codebook"), &[v, c])?.ptr;
    let last = tensor(store, &b("read_weight"), &[c, h])?.ptr;
    let early = if l > 1 {
        tensor(store, &b("early_read_weights"), &[l - 1, c, h])?.ptr
    } else {
        DevicePtr::NULL
    };
    let write = tensor(store, &b("feature_write_weights"), &[l, h, r])?.ptr;
    let read = (0..l)
        .map(|i| {
            if i + 1 == l {
                last
            } else {
                early.offset(i * c * h * 2)
            }
        })
        .collect();
    let write = (0..l).map(|i| write.offset(i * h * r * 2)).collect();

    let rc = r * c * 2;
    let packed = gpu.alloc(3 * rc)?;
    let stream = gpu.default_stream();
    for (i, name) in ["gate.weight", "predecessor_gate.weight", "prev.weight"]
        .iter()
        .enumerate()
    {
        let src = tensor(store, &b(name), &[r, c])?.ptr;
        gpu.copy_d2d_async(src, packed.offset(i * rc), rc, stream)?;
    }
    gpu.synchronize(stream)?;
    let gate_bias = tensor(store, &b("gate.bias"), &[r])?.ptr;
    tracing::info!(
        "DSpine code bridge loaded: {l} layers, code {c}, rank {r}, eps {:e}, top-{} serial table",
        spec.eps(),
        spec.head_topk
    );
    Ok(Some(DspineBridgeWeights {
        spec,
        codebook: DenseWeight { weight: codebook },
        read,
        write,
        packed,
        gate_bias: DenseWeight { weight: gate_bias },
    }))
}
