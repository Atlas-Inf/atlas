// SPDX-License-Identifier: AGPL-3.0-only
//! SparkGLM 0.2.0's KDA prefill (`ATLAS_KDA_FLASH_LIBRARY`): its FlashKDA
//! bridge, as `Enntity/atlas` c87f2790 drove the dlopened library, over
//! `spark_runtime::glm_kda::sparkglm`'s in-tree build. The same scratch
//! (packed in `ssm_qkvz`, workspace in `expert_gate_out`, aux in
//! `expert_up_out`), the same pieces, the same calls.
use anyhow::{Context, Result, ensure};
use spark_runtime::glm_kda::{self, sparkglm as bridge};
use spark_runtime::gpu::DevicePtr;

use super::{PIECE_ROWS, pieces};
use crate::layer::ForwardContext;

pub(in crate::layers::glm5_kda) struct SparkGlm {
    /// Local KDA heads: 32 at TP2, 16 at TP4.
    heads: usize,
    /// `ATLAS_GLM_KDA_FLASH_PLANES=1`: q / k / v arrive as planes.
    pub(super) planes: bool,
}

/// `[qkv or q / k / v planes, gate, beta, a, bias, state, output]`.
pub(super) type Operands = [DevicePtr; 7];

fn checked_ranges(ranges: &[(DevicePtr, usize)]) -> Result<()> {
    for (i, &(ptr, bytes)) in ranges.iter().enumerate() {
        ensure!(
            !ptr.is_null() && bytes > 0,
            "FlashKDA null or empty operand"
        );
        let end = ptr
            .0
            .checked_add(bytes as u64)
            .context("FlashKDA address overflow")?;
        for &(other, n) in &ranges[..i] {
            let other_end = other
                .0
                .checked_add(n as u64)
                .context("FlashKDA address overflow")?;
            ensure!(
                end <= other.0 || other_end <= ptr.0,
                "FlashKDA overlapping operands"
            );
        }
    }
    Ok(())
}

impl SparkGlm {
    pub(super) fn load(heads: usize, dim: usize, lower: f32, planes: bool) -> Result<Self> {
        ensure!(
            bridge::HEADS.contains(&heads) && dim == 128 && lower == -5.0,
            "FlashKDA screen is qualified only for H16/H32, D128, lower_bound=-5 (got H{heads}/D{dim})"
        );
        ensure!(glm_kda::available(), glm_kda::Error::Unavailable);
        tracing::info!(
            planes,
            "SparkGLM FlashKDA prefill (FP32 recurrence, in-tree) enabled for >=2048 rows \
             (2048..=4100 per call); decode and verification unchanged"
        );
        Ok(Self { heads, planes })
    }

    /// Interleaved `[tokens, 3, heads, 128]` q / k / v at `ops[0]`.
    pub(super) fn forward(
        &self,
        ops: Operands,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let plane_row = self.heads * 128 * 2;
        let (b, sizes) = (ctx.buffers, ctx.buffers.sizes());
        for (row, rows) in pieces(tokens) {
            self.run_piece(
                self.piece(ops, row, 3 * plane_row),
                rows,
                [
                    (b.ssm_qkvz(), sizes.ssm_qkvz),
                    (b.expert_gate_out(), sizes.expert_gate_out),
                    (b.expert_up_out(), sizes.expert_up_out),
                ],
                stream,
            )?;
        }
        Ok(())
    }

    /// q / k / v planes (`3 x [tokens, heads * 128]`) at `ops[0]`.
    pub(super) fn forward_planes(
        &self,
        ops: Operands,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let plane_row = self.heads * 128 * 2;
        let plane = tokens * plane_row;
        let (b, sizes) = (ctx.buffers, ctx.buffers.sizes());
        for (row, rows) in pieces(tokens) {
            let [q, k, v] = [0, 1, 2].map(|i| ops[0].offset(i * plane + row * plane_row));
            self.run_piece_planes(
                [q, k, v],
                self.piece(ops, row, 0),
                rows,
                [
                    (b.expert_gate_out(), sizes.expert_gate_out),
                    (b.expert_up_out(), sizes.expert_up_out),
                ],
                stream,
            )?;
        }
        Ok(())
    }

    /// `ops` advanced to row `row` (`qkv_row` bytes a row for `ops[0]`).
    fn piece(&self, ops: Operands, row: usize, qkv_row: usize) -> Operands {
        let [qkv, gate, beta, a, bias, state, output] = ops;
        let plane_row = self.heads * 128 * 2;
        [
            qkv.offset(row * qkv_row),
            gate.offset(row * plane_row),
            beta.offset(row * self.heads * 2),
            a,
            bias,
            state,
            output.offset(row * plane_row),
        ]
    }

    /// The operands common to both entries, checked: capacities first, then
    /// every live range (`inputs` are the q / k / v ranges) disjoint.
    fn operands(
        &self,
        [_, gate, beta, a, bias, state, output]: Operands,
        inputs: &[(DevicePtr, usize)],
        tokens: usize,
        [(workspace, workspace_cap), (aux, aux_cap)]: [(DevicePtr, usize); 2],
    ) -> Result<bridge::Operands> {
        ensure!(PIECE_ROWS.contains(&tokens), "Unqualified FlashKDA piece");
        let heads = self.heads;
        let plane = bridge::plane_bytes(tokens, heads);
        let aux_required = bridge::aux_bytes(tokens, heads);
        ensure!(
            aux_cap >= aux_required,
            "FlashKDA scratch capacity exceeded"
        );
        // Ask the compiled library, keeping its sizing formula authoritative.
        let workspace_required = bridge::workspace_bytes(tokens, heads)?;
        ensure!(
            workspace_cap >= workspace_required,
            "FlashKDA scratch capacity exceeded"
        );
        let mut ranges = inputs.to_vec();
        ranges.extend([
            (gate, plane),
            (beta, tokens * heads * 2),
            (a, heads * 4),
            (bias, heads * 128 * 4),
            (state, bridge::state_bytes(heads)),
            (output, plane),
            (workspace, workspace_required),
            (aux, aux_required),
        ]);
        checked_ranges(&ranges)?;
        Ok(bridge::Operands {
            gate: gate.0,
            beta: beta.0,
            a_log: a.0,
            dt_bias: bias.0,
            state: state.0,
            output: output.0,
            workspace: (workspace.0, workspace_cap),
            aux: (aux.0, aux_cap),
        })
    }

    /// One library call over `tokens` rows, with the packed, workspace and
    /// aux scratch given as `(ptr, capacity)`.
    fn run_piece(
        &self,
        ops: Operands,
        tokens: usize,
        [(packed, packed_cap), workspace, aux]: [(DevicePtr, usize); 3],
        stream: u64,
    ) -> Result<()> {
        let plane = bridge::plane_bytes(tokens, self.heads);
        ensure!(
            packed_cap >= 3 * plane,
            "FlashKDA scratch capacity exceeded"
        );
        let inputs = [(ops[0], 3 * plane), (packed, 3 * plane)];
        let o = self.operands(ops, &inputs, tokens, [workspace, aux])?;
        // SAFETY: qualified geometry, capacities and disjoint live byte ranges
        // were checked above; the forward's stream owns every pointer. The
        // bridge enqueues all work on it and restores the canonical state.
        unsafe {
            bridge::prefill(
                ops[0].0,
                (packed.0, packed_cap),
                &o,
                tokens,
                self.heads,
                stream,
            )
        }
        .context("FlashKDA launch failed; state may be partially updated")
    }

    /// One planar library call over `tokens` rows of `[q, k, v]`.
    fn run_piece_planes(
        &self,
        [q, k, v]: [DevicePtr; 3],
        ops: Operands,
        tokens: usize,
        scratch: [(DevicePtr, usize); 2],
        stream: u64,
    ) -> Result<()> {
        let plane = bridge::plane_bytes(tokens, self.heads);
        let inputs = [(q, plane), (k, plane), (v, plane)];
        let o = self.operands(ops, &inputs, tokens, scratch)?;
        // SAFETY: as `run_piece`.
        unsafe { bridge::prefill_planes([q.0, k.0, v.0], &o, tokens, self.heads, stream) }
            .context("FlashKDA launch failed; state may be partially updated")
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "flash_prefill_sparkglm_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlaps_and_address_wrap_are_rejected() {
        assert!(checked_ranges(&[(DevicePtr(128), 128), (DevicePtr(256), 16)]).is_ok());
        assert!(checked_ranges(&[(DevicePtr(128), 129), (DevicePtr(256), 16)]).is_err());
        assert!(checked_ranges(&[(DevicePtr(u64::MAX), 2)]).is_err());
        assert!(checked_ranges(&[(DevicePtr::NULL, 4)]).is_err());
    }

    #[test]
    fn short_scratch_and_unqualified_pieces_fail_before_any_call() {
        let s = SparkGlm {
            heads: 16,
            planes: false,
        };
        let ops: Operands = std::array::from_fn(|i| DevicePtr((i as u64 + 1) << 32));
        let big = 1usize << 30;
        let aux = bridge::aux_bytes(4096, 16);
        let scratch = |packed, aux| {
            [
                (DevicePtr(9 << 32), packed),
                (DevicePtr(10 << 32), big),
                (DevicePtr(11 << 32), aux),
            ]
        };
        let plane = bridge::plane_bytes(4096, 16);
        for (rows, packed, aux_cap) in [
            (4096, 3 * plane - 1, aux),
            (4096, 3 * plane, aux - 1),
            (2047, 3 * plane, aux),
            (4101, 3 * plane, aux),
        ] {
            let error = s
                .run_piece(ops, rows, scratch(packed, aux_cap), 0)
                .expect_err("rejected before the bridge");
            let message = error.to_string();
            assert!(
                message.contains("capacity") || message.contains("Unqualified"),
                "{rows}: {message}"
            );
        }
    }
}
