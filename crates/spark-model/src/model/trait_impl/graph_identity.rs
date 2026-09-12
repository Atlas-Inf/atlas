// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::graph_runtime::{
    GraphCost, GraphIdentity, GraphPayload, GraphRuntimeError, GraphSegment,
};

use super::super::types::TransformerModel;

impl TransformerModel {
    pub(super) fn decode_graph_identity(
        &self,
        request_count: usize,
        padded_request_count: usize,
        slots: Vec<u32>,
    ) -> Result<GraphIdentity, GraphRuntimeError> {
        self.graph_runtime.identity(
            GraphSegment::DecodeBody,
            GraphPayload::Decode {
                request_count: request_count as u32,
                padded_request_count: padded_request_count as u32,
                slots,
            },
        )
    }

    pub(super) fn verify_graph_identity(
        &self,
        slots: Vec<u32>,
        depths: Vec<u32>,
        row_count: usize,
        layout: Vec<u32>,
    ) -> Result<GraphIdentity, GraphRuntimeError> {
        self.graph_runtime.identity(
            GraphSegment::VerifyBody,
            GraphPayload::Verify {
                request_count: slots.len() as u32,
                row_count: row_count as u32,
                slots,
                depths,
                layout,
            },
        )
    }

    pub(super) fn fused_graph_identity(
        &self,
        slot: usize,
        rows: usize,
    ) -> Result<GraphIdentity, GraphRuntimeError> {
        self.graph_runtime.identity(
            GraphSegment::FusedBody,
            GraphPayload::Fused {
                request_count: 1,
                row_count: rows as u32,
                draft_depth: rows.saturating_sub(1) as u32,
                slots: vec![slot as u32],
            },
        )
    }

    pub(super) fn model_graph_cost(&self) -> GraphCost {
        GraphCost {
            estimated_bytes: (self.layers.len() as u64).saturating_mul(1024 * 1024),
            node_count: (self.layers.len() as u64).saturating_mul(64),
            child_count: 0,
            staging_bytes: 0,
        }
    }
}
