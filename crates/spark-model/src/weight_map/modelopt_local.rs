// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time expansion of NVIDIA's ModelOpt "FP4-Local" storage formats into
//! the layouts Atlas already serves (nvidia/Qwen3.8-Flash-Next-FP4-Local).
//!
//! That pack differs from the NVFP4 pack Atlas runs only in HOW it stores
//! three things, never in what they compute:
//!
//! * routed-expert NVFP4 block scales are CSF-compressed
//!   (`weight_scale.nvfp4_csf_{fixed,exceptions}`) — decoded losslessly into
//!   the plain E4M3 `weight_scale` every NVFP4 kernel reads;
//! * GDN / attention / QSA-indexer / shared-expert projections are MXFP8.
//!   Those with a native MXFP8 consumer ([`native_consumers`]: the GDN and
//!   attention projections) are uploaded byte for byte, `weight` E4M3 + `weight_scale`
//!   E8M0, and read by `mxfp8_proj.cu`; the rest are expanded exactly to the
//!   BF16 `weight` the NVFP4 pack ships;
//! * vision MLP/merger projections are W4A16 NVFP4 — dequantized to the BF16
//!   the vision tower reads;
//! * the hyper-connection `hc_norm`s and the three PLE norms are F32 where
//!   other releases ship BF16 — left exactly as shipped; the kernels that
//!   read them are `const float*`, and the other packs' BF16 norms are
//!   widened to F32 at layer-load (`dense_keep_f32`).
//!
//! The loader DEFERS every source tensor (`FastSafetensorsLoader::
//! host_decode_modules`, `is_csf_stream`), so the compressed originals are
//! read from disk here and never occupy device memory; only the expanded
//! tensor is uploaded. A loader that uploaded them anyway (the non-fast path)
//! still works: resident sources are read back and freed.
//!
//! The PLE n-gram table's NVFP4 rows are a different consumer (gathered by
//! row off NVMe) and are handled by the PLE loader, not here.

use std::collections::HashSet;

use super::modelopt_plan::{ModeloptQuantPlan, QuantAlgo, native_consumers};
use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use atlas_core::modelopt_codec::{decode_csf_scales, mxfp8_to_bf16, nvfp4_rows_to_bf16};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

const CSF_FIXED: &str = ".nvfp4_csf_fixed";
const CSF_EXCEPTIONS: &str = ".nvfp4_csf_exceptions";

/// Modules whose weight/scale pair the loader must defer: every `MXFP8`
/// entry of the plan (native upload and host expansion both read the
/// deferred bytes), plus — on FP4-Local — every module with no registered
/// native consumer (its packed bytes are never meant to be resident). The
/// pre-plan rule for non-FP4-Local packs is the `MXFP8`-only arm.
pub fn host_decode_modules(config: &ModelConfig) -> HashSet<String> {
    let Some(qc) = config.quantization_config.as_ref() else {
        return HashSet::new();
    };
    if let Some(plan) = ModeloptQuantPlan::from_config(config)
        && plan.is_fp4_local()
    {
        let natives = native_consumers(&config.model_type);
        return plan
            .modules()
            .filter(|m| {
                let f = plan.format_of(m).expect("plan module");
                matches!(f.algo, QuantAlgo::Mxfp8) || !natives.covers(m, &f.algo)
            })
            .cloned()
            .collect();
    }
    qc.quantized_layers
        .iter()
        .filter(|(_path, spec)| spec.quant_algo.eq_ignore_ascii_case("MXFP8"))
        .map(|(path, _)| path.clone())
        .collect()
}

/// What [`materialize`] did, for the load log and for tests.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Materialized {
    pub csf_scales: usize,
    pub mxfp8_native: usize,
    pub mxfp8_to_bf16: usize,
    pub nvfp4_to_bf16: usize,
    pub f32_norms_native: usize,
    pub uploaded_bytes: u64,
}

/// A norm weight the FP4-Local pack keeps in F32 where every other release of
/// the model ships BF16: the 97 hyper-connection `hc_norm`s and the three PLE
/// norms (model card: "100 FP32 tensors"; 103 with the MTP block's). The
/// kernels reading them are declared `const float*`, so the resident tensor
/// is served as shipped — nothing to do here but count it for the log.
fn is_f32_norm(name: &str, t: &WeightTensor) -> bool {
    // `hc_norm`, and the PLE's `norm_conv` / `norm_key` / `norm_query`: the
    // last module segment names a norm.
    let names_norm = name
        .strip_suffix(".weight")
        .and_then(|m| m.rsplit('.').next())
        .is_some_and(|seg| seg.contains("norm"));
    t.dtype == WeightDtype::FP32 && t.shape.len() == 1 && names_norm
}

/// Raw bytes and geometry of a source tensor, from disk or from the device.
struct Source {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: WeightDtype,
}

/// Take `name` out of the store, preferring its deferred on-disk copy. A
/// resident copy is read back and its buffer freed (unless already reclaimed).
fn take(store: &mut WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<Option<Source>> {
    if let Some(d) = store.undefer(name) {
        let bytes = d
            .read_bytes()
            .with_context(|| format!("reading deferred {name}"))?;
        return Ok(Some(Source {
            bytes,
            shape: d.shape,
            dtype: d.dtype,
        }));
    }
    let reclaimed = store.was_reclaimed(name);
    let Some(t) = store.remove(name) else {
        return Ok(None);
    };
    ensure!(
        !reclaimed,
        "{name} was reclaimed before it could be expanded"
    );
    let mut bytes = vec![0u8; t.byte_size()];
    gpu.copy_d2h(t.ptr, &mut bytes)?;
    gpu.free(t.ptr)?;
    Ok(Some(Source {
        bytes,
        shape: t.shape,
        dtype: t.dtype,
    }))
}

/// Read a resident F32 scalar (`weight_scale_2`), removing it from the store.
fn take_scalar(store: &mut WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<f32> {
    let s = take(store, gpu, name)?.with_context(|| format!("{name} is missing"))?;
    ensure!(
        s.dtype == WeightDtype::FP32 && s.bytes.len() == 4,
        "{name}: expected one F32, got {:?} {:?}",
        s.dtype,
        s.shape
    );
    Ok(f32::from_le_bytes([
        s.bytes[0], s.bytes[1], s.bytes[2], s.bytes[3],
    ]))
}

fn upload(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
    name: String,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: WeightDtype,
) -> Result<u64> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    store.insert(name, WeightTensor { ptr, shape, dtype })?;
    Ok(bytes.len() as u64)
}

/// Names of every CSF fixed stream the store knows, deferred or resident.
fn csf_bases(store: &WeightStore) -> Vec<String> {
    let mut bases: Vec<String> = store
        .deferred_sorted()
        .into_iter()
        .map(|(n, _)| n.as_str())
        .chain(store.names())
        .filter_map(|n| n.strip_suffix(CSF_FIXED).map(str::to_string))
        .collect();
    bases.sort();
    bases.dedup();
    bases
}

fn decode_one_csf(store: &mut WeightStore, gpu: &dyn GpuBackend, base: &str) -> Result<u64> {
    let fixed = take(store, gpu, &format!("{base}{CSF_FIXED}"))?.context("fixed stream")?;
    let exc = take(store, gpu, &format!("{base}{CSF_EXCEPTIONS}"))?
        .with_context(|| format!("{base}: CSF fixed stream without its exceptions partner"))?;
    ensure!(
        fixed.dtype == WeightDtype::UInt8 && fixed.shape.len() == 2,
        "{base}: CSF fixed stream is {:?} {:?}, expected 2-D U8",
        fixed.dtype,
        fixed.shape
    );
    let (fr, fc) = (fixed.shape[0], fixed.shape[1]);
    let (rows, cols) = atlas_core::modelopt_codec::csf_scale_shape(fr, fc)?;
    let scales = decode_csf_scales(&fixed.bytes, fr, fc, &exc.bytes)
        .with_context(|| format!("decoding CSF scales of {base}"))?;
    upload(
        store,
        gpu,
        base.to_string(),
        &scales,
        vec![rows, cols],
        WeightDtype::FP8E4M3,
    )
}

/// Upload one MXFP8 module unchanged: `weight` E4M3 `[n, k]` and
/// `weight_scale` E8M0 `[n, k / 32]`. `None` when the checkpoint lacks it.
fn upload_native_mxfp8(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
    module: &str,
) -> Result<Option<u64>> {
    let Some(w) = take(store, gpu, &format!("{module}.weight"))? else {
        return Ok(None);
    };
    let s = take(store, gpu, &format!("{module}.weight_scale"))?
        .with_context(|| format!("{module}.weight has no weight_scale"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3
            && w.shape.len() == 2
            && w.shape[1] % 32 == 0
            && s.dtype == WeightDtype::UInt8
            && s.shape == [w.shape[0], w.shape[1] / 32],
        "{module}: expected MXFP8 (E4M3 [n, k] + U8 [n, k/32]), got {:?} {:?} + {:?} {:?}",
        w.dtype,
        w.shape,
        s.dtype,
        s.shape
    );
    let mut n = upload(
        store,
        gpu,
        format!("{module}.weight"),
        &w.bytes,
        w.shape,
        w.dtype,
    )?;
    n += upload(
        store,
        gpu,
        format!("{module}.weight_scale"),
        &s.bytes,
        s.shape,
        s.dtype,
    )?;
    Ok(Some(n))
}

/// Expand one host-decoded module into a BF16 `.weight`. Returns
/// `(was_mxfp8, bytes)`, or `None` when the checkpoint does not carry it.
fn expand_module(
    store: &mut WeightStore,
    gpu: &dyn GpuBackend,
    module: &str,
) -> Result<Option<(bool, u64)>> {
    let Some(w) = take(store, gpu, &format!("{module}.weight"))? else {
        return Ok(None);
    };
    let s = take(store, gpu, &format!("{module}.weight_scale"))?
        .with_context(|| format!("{module}.weight has no weight_scale"))?;
    ensure!(
        w.shape.len() == 2,
        "{module}.weight is {:?}, expected 2-D",
        w.shape
    );
    let out = w.shape[0];
    let (bytes, inp, mx) = match w.dtype {
        WeightDtype::FP8E4M3 => {
            let inp = w.shape[1];
            (mxfp8_to_bf16(&w.bytes, &s.bytes, out, inp)?, inp, true)
        }
        WeightDtype::UInt8 => {
            let inp = w.shape[1] * 2;
            ensure!(
                s.shape.len() == 2 && s.shape[1] > 0,
                "{module}: bad NVFP4 scale shape"
            );
            let group = inp / s.shape[1];
            let global = take_scalar(store, gpu, &format!("{module}.weight_scale_2"))?;
            // The activation scale is a W4A4 input; a BF16 projection has none.
            if let Some(t) = store.remove(&format!("{module}.input_scale")) {
                gpu.free(t.ptr)?;
            }
            let mut dst = vec![0u8; out * inp * 2];
            nvfp4_rows_to_bf16(&w.bytes, &s.bytes, global, inp, group, &mut dst)?;
            (dst, inp, false)
        }
        other => anyhow::bail!("{module}.weight is {other:?}; expected MXFP8 E4M3 or NVFP4 U8"),
    };
    let n = upload(
        store,
        gpu,
        format!("{module}.weight"),
        &bytes,
        vec![out, inp],
        WeightDtype::BF16,
    )?;
    Ok(Some((mx, n)))
}

/// Expand every FP4-Local storage format present in `store`. A no-op (and
/// cheap) for any checkpoint without CSF streams or host-decoded modules.
pub fn materialize(
    store: &mut WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Materialized> {
    let mut done = Materialized::default();
    let plan = ModeloptQuantPlan::from_config(config);
    let natives = native_consumers(&config.model_type);
    for base in csf_bases(store) {
        // The scale-encoding column decides the codec; `"csf"` is the only
        // one Atlas decodes today. A module the plan doesn't list keeps the
        // stream's name-driven decode — don't guess at an unknown layout.
        if let Some(enc) = base
            .strip_suffix(".weight_scale")
            .and_then(|m| plan.as_ref().and_then(|p| p.format_of(m)))
            .and_then(|f| f.scale_encoding.as_deref())
        {
            ensure!(
                enc == "csf",
                "{base}: unsupported weight_scale_encoding {enc:?}; cannot decode"
            );
        }
        done.uploaded_bytes += decode_one_csf(store, gpu, &base)?;
        done.csf_scales += 1;
    }
    let mut modules: Vec<String> = host_decode_modules(config).into_iter().collect();
    modules.sort();
    // A/B only: expand the native MXFP8 set too (both paths are exact).
    let force_expand = std::env::var("ATLAS_FP4LOCAL_EXPAND_MXFP8").as_deref() == Ok("1");
    for module in modules {
        let fmt = plan.as_ref().and_then(|p| p.format_of(&module));
        let algo = fmt
            .map(|f| &f.algo)
            .cloned()
            .unwrap_or(QuantAlgo::Other(String::new()));
        if !force_expand && natives.covers(&module, &algo) {
            if algo == QuantAlgo::Mxfp8
                && let Some(n) = upload_native_mxfp8(store, gpu, &module)?
            {
                done.mxfp8_native += 1;
                done.uploaded_bytes += n;
            }
            // An NVFP4/algo-native module stays deferred for its own loader.
            continue;
        }
        let Some((mx, n)) = expand_module(store, gpu, &module)? else {
            continue;
        };
        if mx {
            done.mxfp8_to_bf16 += 1;
        } else {
            done.nvfp4_to_bf16 += 1;
        }
        done.uploaded_bytes += n;
    }
    if done != Materialized::default() {
        // Only an FP4-Local checkpoint reaches here: another release's F32
        // tensors are never counted.
        done.f32_norms_native = store
            .names()
            .filter(|n| store.get(n).is_ok_and(|t| is_f32_norm(n, t)))
            .count();
        tracing::info!(
            "FP4-Local: decoded {} CSF scale tensors, kept {} MXFP8 projections native, \
             expanded {} MXFP8 and {} NVFP4 projections to BF16, kept {} F32 norms \
             native ({:.2} GiB uploaded; compressed originals never resident)",
            done.csf_scales,
            done.mxfp8_native,
            done.mxfp8_to_bf16,
            done.nvfp4_to_bf16,
            done.f32_norms_native,
            done.uploaded_bytes as f64 / (1u64 << 30) as f64
        );
    }
    Ok(done)
}

#[cfg(test)]
#[path = "modelopt_local_tests.rs"]
mod tests;
