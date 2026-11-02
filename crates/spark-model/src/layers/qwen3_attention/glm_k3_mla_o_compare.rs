// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in actual-input differential diagnostic; never enable for performance.
use super::*;
const COMPARE_FLAG: &str = "ATLAS_GLM_K3_MLA_O_COMPARE";
fn parse_compare(value: Option<&str>, batchm: bool) -> Result<bool> {
    let compare = match value {
        None | Some("0") => false,
        Some("1") => true,
        _ => bail!("{COMPARE_FLAG} must be 0 or 1"),
    };
    ensure!(!compare || batchm, "{COMPARE_FLAG} requires {FLAG}=1");
    Ok(compare)
}
pub(in crate::layers::qwen3_attention) fn enabled(model: &str) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    let value = match std::env::var(COMPARE_FLAG) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => bail!("{COMPARE_FLAG} must be 0 or 1"),
    };
    parse_compare(value.as_deref(), super::enabled(model)?)
}
fn compare_bytes(reference: &[u8], candidate: &[u8], rows: usize) -> Result<()> {
    let bytes = output_bytes(rows);
    ensure!(
        reference.len() == bytes && candidate.len() == bytes,
        "{COMPARE_FLAG}: expected full {rows}x4096 BF16 outputs"
    );
    let mut mismatches = 0;
    let mut first = None;
    let mut max_abs = 0.0_f32;
    for (index, (a, b)) in reference
        .chunks_exact(2)
        .zip(candidate.chunks_exact(2))
        .enumerate()
    {
        let a = u16::from_le_bytes([a[0], a[1]]);
        let b = u16::from_le_bytes([b[0], b[1]]);
        let af = f32::from_bits(u32::from(a) << 16);
        let bf = f32::from_bits(u32::from(b) << 16);
        ensure!(
            af.is_finite() && bf.is_finite(),
            "{COMPARE_FLAG}: nonfinite first_index={index} row={} column={} scalar_bits={a:04x} batchm_bits={b:04x}",
            index / 4096,
            index % 4096
        );
        if a != b {
            mismatches += 1;
            first.get_or_insert((index, a, b));
            let delta = (af - bf).abs();
            max_abs = max_abs.max(if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            });
        }
    }
    if let Some((index, a, b)) = first {
        bail!(
            "{COMPARE_FLAG}: mismatches={mismatches} first_index={index} row={} column={} scalar_bits={a:04x} batchm_bits={b:04x} max_abs={max_abs}",
            index / 4096,
            index % 4096
        );
    }
    Ok(())
}
impl StagePlan {
    pub(in crate::layers::qwen3_attention) fn with_compare(
        mut self,
        candidate: DevicePtr,
        capacity: usize,
        live: &[(DevicePtr, usize)],
    ) -> Result<Self> {
        let bytes = output_bytes(self.rows);
        ensure!(
            capacity >= bytes && candidate.0.is_multiple_of(2),
            "{COMPARE_FLAG}: need aligned {bytes}-byte comparison output"
        );
        let out = span(candidate, bytes)?;
        ensure!(
            disjoint(out, span(self.input, self.rows * ROW)?)
                && disjoint(out, span(self.output, bytes)?),
            "{COMPARE_FLAG}: comparison output aliases staged/reference rows"
        );
        for &(ptr, bytes) in live {
            ensure!(
                disjoint(out, span(ptr, bytes)?),
                "{COMPARE_FLAG}: comparison output aliases live input/weight"
            );
        }
        self.comparison = Some(candidate);
        Ok(self)
    }
    /// Run immediately after this row's V extraction, before the next causal row.
    pub(in crate::layers::qwen3_attention) fn capture_scalar(
        self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        weight: &DenseWeight,
        row: usize,
        stream: u64,
    ) -> Result<()> {
        if self.comparison.is_none() {
            return Ok(());
        }
        ensure!(kernel.0 != 0, "{COMPARE_FLAG}: scalar kernel unavailable");
        ops::dense_gemv(
            gpu,
            kernel,
            self.row(row)?,
            weight,
            self.output.offset(row * OUTPUT_ROW),
            4096,
            8192,
            stream,
        )
    }
    /// Candidate lives in dead attention scratch; baseline remains the real output.
    pub(in crate::layers::qwen3_attention) fn finish_compare(
        self,
        gpu: &dyn GpuBackend,
        stream: u64,
        layer: usize,
        rank: usize,
    ) -> Result<()> {
        let Some(candidate) = self.comparison else {
            return Ok(());
        };
        gpu.synchronize(stream)?;
        // Explicitly diagnostic host allocations; no GPU allocation or state replay.
        let bytes = output_bytes(self.rows);
        let mut reference = vec![0_u8; bytes];
        let mut actual = vec![0_u8; bytes];
        gpu.copy_d2h(self.output, &mut reference)?;
        gpu.copy_d2h(candidate, &mut actual)?;
        compare_bytes(&reference, &actual, self.rows).map_err(|error| {
            anyhow::anyhow!("{COMPARE_FLAG}: rank={rank} attention_layer={layer}: {error}")
        })?;
        tracing::info!(
            rank,
            attention_layer = layer,
            rows = self.rows,
            elements = bytes / 2,
            "GLM K3 MLA O compare passed: immediate scalar vs deferred batchm, bit-identical"
        );
        Ok(())
    }
}
#[cfg(test)]
#[path = "glm_k3_mla_o_compare_tests.rs"]
mod tests;
