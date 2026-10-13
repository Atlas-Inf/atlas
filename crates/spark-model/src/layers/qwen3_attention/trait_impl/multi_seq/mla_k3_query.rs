// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in query projection staging for the GLM-5 long-verify MLA path: the
//! repaired K3 verify or a DFlash block of up to `MAX_ROWS` rows.
//!
//! The rows are independent only through the Qa/RMS/Qb/index-Q
//! projections.  Cache mutation, semantic-index maintenance, selection,
//! attention, and value extraction remain in the caller's causal row loop.
//! This module owns the checked scratch partition and the diagnostic scalar
//! comparison; it does not add a persistent allocation or replay model state.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::glm_k3_mla_o::{MAX_ROWS, rows_supported};
use crate::layers::qwen3_attention::{MlaWeights, Qwen3AttentionLayer};

const FLAG: &str = "ATLAS_GLM_K3_MLA_QUERY_BATCHM";
const COMPARE_FLAG: &str = "ATLAS_GLM_K3_MLA_QUERY_COMPARE";

const HIDDEN: usize = 4096;
const Q_LORA: usize = 1536;
const Q_DIM: usize = 8192;
const INDEX_DIM: usize = 4096;
const BF16: usize = 2;

const LATENT_ROW: usize = Q_LORA * BF16;
const Q_ROW: usize = Q_DIM * BF16;
const INDEX_ROW: usize = INDEX_DIM * BF16;
const INPUT_ROW: usize = HIDDEN * BF16;

// Partition sizes for `rows` staged rows (2..=MAX_ROWS, so no overflow). Q
// rows are contiguous and the index-Q rows follow the last Q row.
const fn input_bytes(rows: usize) -> usize {
    rows * INPUT_ROW
}
const fn latent_bytes(rows: usize) -> usize {
    rows * LATENT_ROW
}
const fn index_offset(rows: usize) -> usize {
    rows * Q_ROW
}
const fn query_bytes(rows: usize) -> usize {
    index_offset(rows) + rows * INDEX_ROW
}

// ssm_qkvz is dead before the causal row loop. Diagnostic references borrow
// the whole arena for the scalar snapshots, then the row loop reuses its first
// 512 bytes for the indexer's key and gate scratch and its remaining bytes for
// retained O rows.
const fn diagnostic_bytes(rows: usize) -> usize {
    query_bytes(rows)
}

fn parse_flag(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("{FLAG} must be 0 or 1"),
    }
}

fn parse_compare(value: Option<&str>, batchm: bool) -> Result<bool> {
    let compare = match value {
        None | Some("0") => false,
        Some("1") => true,
        _ => bail!("{COMPARE_FLAG} must be 0 or 1"),
    };
    ensure!(!compare || batchm, "{COMPARE_FLAG} requires {FLAG}=1");
    Ok(compare)
}

pub(super) fn enabled(model: &str) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match std::env::var(FLAG) {
        Ok(value) => parse_flag(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(_) => bail!("{FLAG} must be 0 or 1"),
    }
}

fn compare_enabled(model: &str, batchm: bool) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    let value = match std::env::var(COMPARE_FLAG) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => bail!("{COMPARE_FLAG} must be 0 or 1"),
    };
    parse_compare(value.as_deref(), batchm)
}

fn span(ptr: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(ptr.0 != 0, "{FLAG}: null query staging pointer");
    Ok((
        ptr.0,
        ptr.0
            .checked_add(bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("{FLAG}: query staging address overflow"))?,
    ))
}

fn disjoint(a: (u64, u64), b: (u64, u64)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}

fn aligned_nonnull(ptr: DevicePtr, alignment: u64, name: &str) -> Result<()> {
    ensure!(
        ptr.0 != 0 && ptr.0.is_multiple_of(alignment),
        "{FLAG}: {name} must be non-null and {alignment}-byte aligned"
    );
    Ok(())
}

/// Return the bytes remaining in the known arena that contains `ptr`.
/// Multi-sequence callers normally pass arena bases, but using the containing
/// range also makes alias checks safe for a row-offset or an FP32 scratch
/// pointer supplied by a future caller.
fn arena_remaining_capacity(
    ptr: DevicePtr,
    arenas: &[(DevicePtr, usize)],
    name: &str,
) -> Result<usize> {
    ensure!(ptr.0 != 0, "{FLAG}: {name} pointer is null");
    for &(base, capacity) in arenas {
        if base.0 == 0 || capacity == 0 {
            continue;
        }
        let end = base
            .0
            .checked_add(capacity as u64)
            .ok_or_else(|| anyhow::anyhow!("{FLAG}: {name} arena address overflow"))?;
        if (base.0..end).contains(&ptr.0) {
            return Ok(capacity - (ptr.0 - base.0) as usize);
        }
    }
    bail!("{FLAG}: {name} pointer is outside known arenas")
}

/// A retained row of the stateless query stage.  The pointers stay valid until
/// the causal loop has consumed that row; no row may be recomputed after cache
/// mutation because the query scratch is intentionally shared with the indexer.
#[derive(Clone, Copy)]
pub(super) struct QueryRow {
    pub(super) q_latent: DevicePtr,
    pub(super) q_full: DevicePtr,
    pub(super) index_query: DevicePtr,
}

#[derive(Clone, Copy)]
pub(super) struct QueryPlan {
    rows: usize,
    latent: DevicePtr,
    q_full: DevicePtr,
    index_query: DevicePtr,
    diagnostic: Option<DevicePtr>,
}

impl QueryPlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        rows: usize,
        input: DevicePtr,
        input_capacity: usize,
        latent: DevicePtr,
        latent_capacity: usize,
        q_scratch: DevicePtr,
        q_capacity: usize,
        diagnostic: Option<(DevicePtr, usize)>,
        live: &[(DevicePtr, usize)],
    ) -> Result<Self> {
        ensure!(
            rows_supported(rows),
            "{FLAG}: {rows} query rows outside 2..={MAX_ROWS}"
        );
        let (input_bytes, latent_bytes, query_bytes) =
            (input_bytes(rows), latent_bytes(rows), query_bytes(rows));
        ensure!(
            input_capacity >= input_bytes,
            "{FLAG}: normalized input capacity is too small"
        );
        ensure!(
            latent_capacity >= latent_bytes,
            "{FLAG}: Q latent capacity requires {latent_bytes} bytes"
        );
        ensure!(
            q_capacity >= query_bytes,
            "{FLAG}: Q/index staging capacity requires {query_bytes} bytes"
        );
        aligned_nonnull(input, 16, "normalized input")?;
        aligned_nonnull(latent, 16, "Q latent")?;
        aligned_nonnull(q_scratch, 16, "Q/index staging")?;

        let input_span = span(input, input_bytes)?;
        let latent_span = span(latent, latent_bytes)?;
        let query_span = span(q_scratch, query_bytes)?;
        ensure!(
            disjoint(input_span, latent_span) && disjoint(input_span, query_span),
            "{FLAG}: query outputs alias normalized input"
        );
        ensure!(
            disjoint(latent_span, query_span),
            "{FLAG}: Q latent aliases Q/index staging"
        );
        for &(ptr, bytes) in live {
            let other = span(ptr, bytes)?;
            ensure!(
                disjoint(input_span, other)
                    && disjoint(latent_span, other)
                    && disjoint(query_span, other),
                "{FLAG}: query plan aliases live range"
            );
        }

        let diagnostic_owner_index = diagnostic.and_then(|target| {
            live.iter()
                .enumerate()
                .find_map(|(index, entry)| (*entry == target).then_some(index))
        });
        let diagnostic_ptr = if let Some((ptr, capacity)) = diagnostic {
            let diagnostic_bytes = diagnostic_bytes(rows);
            ensure!(
                capacity >= diagnostic_bytes,
                "{COMPARE_FLAG}: diagnostic scratch requires {diagnostic_bytes} bytes"
            );
            aligned_nonnull(ptr, 16, "diagnostic scratch")?;
            let scratch = span(ptr, diagnostic_bytes)?;
            ensure!(
                disjoint(scratch, input_span)
                    && disjoint(scratch, latent_span)
                    && disjoint(scratch, query_span),
                "{COMPARE_FLAG}: diagnostic scratch aliases query state"
            );
            for (index, &(ptr, bytes)) in live.iter().enumerate() {
                // Skip only the designated qkvz owner entry. A duplicate
                // exact tuple is still checked and therefore rejected.
                if diagnostic_owner_index == Some(index) {
                    continue;
                }
                ensure!(
                    disjoint(scratch, span(ptr, bytes)?),
                    "{COMPARE_FLAG}: diagnostic scratch aliases live range"
                );
            }
            Some(ptr)
        } else {
            None
        };

        Ok(Self {
            rows,
            latent,
            q_full: q_scratch,
            index_query: q_scratch.offset(index_offset(rows)),
            diagnostic: diagnostic_ptr,
        })
    }

    pub(super) fn row(self, row: usize) -> Result<QueryRow> {
        ensure!(
            row < self.rows,
            "{FLAG}: query row {row} outside {}",
            self.rows
        );
        Ok(QueryRow {
            q_latent: self.latent.offset(row * LATENT_ROW),
            q_full: self.q_full.offset(row * Q_ROW),
            index_query: self.index_query.offset(row * INDEX_ROW),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        self,
        layer: &Qwen3AttentionLayer,
        c: &MultiSeqCtx<'_>,
        mla: &MlaWeights,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let stream = c.stream;
        let batch_kernel = layer.dense_gemv_batchm_k;
        ensure!(batch_kernel.0 != 0, "{FLAG}: batchm kernel unavailable");

        ops::dense_gemv_batchm(
            gpu,
            batch_kernel,
            c.normed,
            &mla.wq_a,
            self.latent,
            self.rows as u32,
            Q_LORA as u32,
            HIDDEN as u32,
            Q_LORA as u32,
            stream,
        )?;

        if let Some(reference) = self.diagnostic {
            // Compare the raw Qa result before the in-place RMS pass can
            // overwrite it.  This runs before any KV/index state write.
            for row in 0..self.rows {
                ops::dense_gemv(
                    gpu,
                    layer.dense_gemv_k,
                    c.normed.offset(row * INPUT_ROW),
                    &mla.wq_a,
                    reference.offset(row * LATENT_ROW),
                    Q_LORA as u32,
                    HIDDEN as u32,
                    stream,
                )?;
            }
            self.compare_rows(
                gpu,
                stream,
                "raw Qa",
                self.latent,
                reference,
                LATENT_ROW,
                layer.attn_layer_idx,
                c.fwd.config.tp_rank,
            )?;
        }

        // Keep the production RMS arithmetic and alias exactly as the scalar
        // lane.  A single packed launch would be mathematically equivalent,
        // but this first integration does not widen the normalization change.
        for row in 0..self.rows {
            let candidate = self.latent.offset(row * LATENT_ROW);
            ops::rms_norm(
                gpu,
                layer.rms_norm_w_k,
                candidate,
                &mla.q_a_norm,
                candidate,
                1,
                Q_LORA as u32,
                c.eps,
                stream,
            )?;
        }

        if let Some(reference) = self.diagnostic {
            for row in 0..self.rows {
                let row_ref = reference.offset(row * LATENT_ROW);
                ops::rms_norm(
                    gpu,
                    layer.rms_norm_w_k,
                    row_ref,
                    &mla.q_a_norm,
                    row_ref,
                    1,
                    Q_LORA as u32,
                    c.eps,
                    stream,
                )?;
            }
            self.compare_rows(
                gpu,
                stream,
                "normalized latent",
                self.latent,
                reference,
                LATENT_ROW,
                layer.attn_layer_idx,
                c.fwd.config.tp_rank,
            )?;
        }

        ops::dense_gemv_batchm(
            gpu,
            batch_kernel,
            self.latent,
            &mla.wq_b,
            self.q_full,
            self.rows as u32,
            Q_DIM as u32,
            Q_LORA as u32,
            Q_DIM as u32,
            stream,
        )?;

        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("{FLAG}: validated GLM indexer missing");
        ops::dense_gemv_batchm(
            gpu,
            batch_kernel,
            self.latent,
            &indexer.wq_b,
            self.index_query,
            self.rows as u32,
            INDEX_DIM as u32,
            Q_LORA as u32,
            INDEX_DIM as u32,
            stream,
        )?;

        if let Some(reference) = self.diagnostic {
            let reference_q = reference;
            let reference_index = reference.offset(index_offset(self.rows));
            for row in 0..self.rows {
                ops::dense_gemv(
                    gpu,
                    layer.dense_gemv_k,
                    self.latent.offset(row * LATENT_ROW),
                    &mla.wq_b,
                    reference_q.offset(row * Q_ROW),
                    Q_DIM as u32,
                    Q_LORA as u32,
                    stream,
                )?;
                ops::dense_gemv(
                    gpu,
                    layer.dense_gemv_k,
                    self.latent.offset(row * LATENT_ROW),
                    &indexer.wq_b,
                    reference_index.offset(row * INDEX_ROW),
                    INDEX_DIM as u32,
                    Q_LORA as u32,
                    stream,
                )?;
            }
            self.compare_rows(
                gpu,
                stream,
                "Qb",
                self.q_full,
                reference_q,
                Q_ROW,
                layer.attn_layer_idx,
                c.fwd.config.tp_rank,
            )?;
            self.compare_rows(
                gpu,
                stream,
                "index-Q",
                self.index_query,
                reference_index,
                INDEX_ROW,
                layer.attn_layer_idx,
                c.fwd.config.tp_rank,
            )?;
        }
        Ok(())
    }
}

#[path = "mla_k3_query_compare.rs"]
mod compare;

#[path = "mla_k3_query_stage.rs"]
mod stage;

#[cfg(test)]
#[path = "mla_k3_query_tests.rs"]
mod tests;
