// SPDX-License-Identifier: AGPL-3.0-only
//! One DeepSeek-V4.1 block on one token. A Hyper-Connection site is two
//! launches and the experts are three (`hc::site`, `moe::experts`); each is
//! held to the bytes of the plain kernels it stands for, which replay the
//! reference a step at a time.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::hc::{self, HcSite, SiteOut, SublayerParts};
use super::parts::Held;
use super::profile;
pub use super::shape::{BlockKernels, Scratch, Shape};
pub use super::state::{CompressedKeys, Handoff, SeqState};
use super::weights::BlockWeights;
use crate::layers::ops::rms_norm;
use crate::weight_map::DenseWeight;

/// The pieces every block of a model shares.
#[derive(Clone)]
pub struct Shared {
    pub shape: Shape,
    /// The parts of the wide projections this rank computes.
    pub held: Held,
    pub kernels: BlockKernels,
    pub scratch: Arc<Scratch>,
    pub handoff: Arc<Mutex<Handoff>>,
}

/// A layer's part in the two-level top-k: the candidate source keeps the
/// best blocks of compressed positions, and the index sources after it
/// choose only inside them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Candidates {
    None,
    Source,
    Reader,
}

pub struct Block {
    pub index: usize,
    /// This layer's compression ratio: 0 for a window-only layer.
    pub ratio: usize,
    pub candidates: Candidates,
    /// `[rope / 2]` FP32 radians per position: the base ladder on a
    /// window-only layer, the compressed one elsewhere.
    pub freqs: DevicePtr,
    pub weights: BlockWeights,
    pub shared: Shared,
}

impl Block {
    pub(super) fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        width: usize,
        stream: u64,
    ) -> Result<()> {
        let weight = DenseWeight { weight };
        rms_norm(
            gpu,
            self.shared.kernels.rms_norm,
            x,
            &weight,
            out,
            1,
            width as u32,
            self.shared.shape.norm_eps,
            stream,
        )
    }

    /// One Hyper-Connection site of a token (`hc::site`): the sublayer
    /// `before` into the stream, then the input of the next, collapsed with
    /// `pre_in`, in `scratch.normed` and `scratch.quantized`.
    #[allow(clippy::too_many_arguments)]
    fn site(
        &self,
        gpu: &dyn GpuBackend,
        weights: &HcSite,
        before: Option<SublayerParts>,
        streams: DevicePtr,
        pre_in: DevicePtr,
        pre_out: DevicePtr,
        norm_weight: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        let out = SiteOut {
            collapsed: sc.x,
            normed: sc.normed,
            quantized: sc.quantized,
            partials: sc.hc_partials,
        };
        profile::stage(gpu, stream, "site", || {
            hc::site(
                gpu,
                &k.hc,
                &s.hc,
                weights,
                before,
                streams,
                pre_in,
                pre_out,
                sc.post,
                sc.comb,
                norm_weight,
                out,
                stream,
            )
        })
    }

    /// The attention's output, as the parts [`Block::attention`] leaves.
    pub(super) fn attention_parts(&self) -> SublayerParts {
        SublayerParts {
            parts: self.shared.scratch.attn_parts,
            slots: 1,
        }
    }

    /// The attention sublayer of one token, up to its output in
    /// [`Block::attention_parts`]. Its site collapses with `pre`, the
    /// coefficients of the site before, and leaves its own in scratch.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        streams: DevicePtr,
        pre: DevicePtr,
        state: &mut SeqState,
        position: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (sc, w) = (&self.shared.scratch, &self.weights);
        self.site(
            gpu,
            &w.hc_attn,
            None,
            streams,
            pre,
            sc.site_pre,
            w.attn.norm,
            stream,
        )?;
        self.attention(gpu, row, state, position, comm, stream)
    }

    /// The rest of the block: the attention's output into the stream, the
    /// FFN, collapsed with the attention site's pre, and its output into the
    /// stream. `pre` is left holding the FFN site's. `comm` reaches the
    /// other ranks when the routed experts are split over them.
    fn finish(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights,
        );
        self.site(
            gpu,
            &w.hc_ffn,
            Some(self.attention_parts()),
            streams,
            sc.site_pre,
            pre,
            w.moe.norm,
            stream,
        )?;
        self.moe(gpu, comm, stream)?;
        profile::stage(gpu, stream, "expand", || {
            hc::expand_parts(
                gpu,
                k.hc.expand_parts,
                &s.hc,
                self.moe_parts(),
                streams,
                sc.post,
                sc.comb,
                stream,
            )
        })
    }

    /// Run the block on one token of row `row` of a pass. `streams` is the
    /// token's `[copies, hidden]` BF16 stream, updated in place. `pre` is
    /// `[copies]` FP32: the coefficients this block's attention is collapsed
    /// with on entry, the ones the next block's is on return.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        streams: DevicePtr,
        pre: DevicePtr,
        state: &mut SeqState,
        position: usize,
        comm: Option<&dyn CommBackend>,
        stream: u64,
    ) -> Result<()> {
        self.attend(gpu, row, streams, pre, state, position, comm, stream)?;
        self.finish(gpu, streams, pre, comm, stream)
    }
}

#[cfg(test)]
impl Block {
    /// The attention's output, rounded from its parts, in
    /// `scratch.sublayer_out`.
    pub(super) fn attention_out(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<DevicePtr> {
        let (s, k, sc) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
        );
        super::parts::round(
            gpu,
            k.parts.round,
            sc.attn_parts,
            sc.sublayer_out,
            1,
            s.hidden,
            stream,
        )?;
        Ok(sc.sublayer_out)
    }

    /// Stand `output` (`[hidden]` BF16, as bytes) in for the attention's
    /// parts: the first part holds it and the others nothing.
    pub(super) fn set_attention_out(&self, gpu: &dyn GpuBackend, output: &[u8]) -> Result<()> {
        let mut parts = vec![0u8; super::parts::PARTS * self.shared.shape.hidden * 4];
        for (value, bits) in parts.chunks_exact_mut(4).zip(output.chunks_exact(2)) {
            let bits = u16::from_le_bytes([bits[0], bits[1]]);
            value.copy_from_slice(&super::cache_recipe::bf16_to_f32(bits).to_le_bytes());
        }
        gpu.synchronize(gpu.default_stream())?;
        gpu.copy_h2d(&parts, self.shared.scratch.attn_parts)
    }
}

#[cfg(test)]
#[path = "block_test_gpu.rs"]
mod test_gpu;
