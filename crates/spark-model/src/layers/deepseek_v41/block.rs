// SPDX-License-Identifier: AGPL-3.0-only
//! One DeepSeek-V4.1 block on one token, in the plain form: every step a
//! launch of a kernel that replays the reference on its own. It is slow by
//! design. It defines what the block computes, and the faster paths are held
//! to its bytes.
//!
//! Not built yet: the indexer (so a query may see at most `index_topk`
//! compressed groups, all of which it attends to), a compressor at ratio 1,
//! Engram, and more than one rank.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::hc::{self, HcSite};
use super::ops;
pub use super::shape::{BlockKernels, Scratch, Shape};
pub use super::state::{CompressedKeys, Handoff, SeqState};
use super::weights::{BlockWeights, Fp8Proj};
use crate::layers::ops::rms_norm;
use crate::weight_map::DenseWeight;

/// The pieces every block of a model shares.
#[derive(Clone)]
pub struct Shared {
    pub shape: Shape,
    pub kernels: BlockKernels,
    pub scratch: Arc<Scratch>,
    pub handoff: Arc<Mutex<Handoff>>,
}

pub struct Block {
    pub index: usize,
    /// This layer's compression ratio: 0 for a window-only layer.
    pub ratio: usize,
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

    /// `out` = `w` times the FP8 form of `x`, which is left in `quantized`.
    pub(super) fn fp8(
        &self,
        gpu: &dyn GpuBackend,
        w: &Fp8Proj,
        x: DevicePtr,
        quantized: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let k = &self.shared.kernels;
        ops::fp8_linear(
            gpu, &k.cache, &k.tiers, &w.w, x, quantized, out, 1, w.n, w.k, stream,
        )
    }

    /// `position` as the device word a rotation reads, in `word`.
    pub(super) fn at(
        &self,
        gpu: &dyn GpuBackend,
        word: DevicePtr,
        position: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        // The kernels of the previous token may still read this word.
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&(position as u32).to_le_bytes(), word)?;
        Ok(word)
    }

    fn site(
        &self,
        gpu: &dyn GpuBackend,
        weights: &HcSite,
        streams: DevicePtr,
        pre: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (k, sc) = (&self.shared.kernels, &self.shared.scratch);
        hc::mix(
            gpu,
            k.hc.mix,
            &self.shared.shape.hc,
            weights,
            streams,
            pre,
            sc.post,
            sc.comb,
            1,
            stream,
        )
    }

    /// The attention sublayer of one token, up to its output in
    /// `scratch.sublayer_out`; the site's post and comb are left in scratch.
    fn attend(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        streams: DevicePtr,
        pre: DevicePtr,
        state: &mut SeqState,
        position: usize,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights,
        );
        // Its own site's post and comb, the previous site's pre.
        self.site(gpu, &w.hc_attn, streams, sc.site_pre, stream)?;
        hc::collapse(gpu, k.hc.collapse, &s.hc, streams, pre, sc.x, 1, stream)?;
        self.norm(gpu, sc.x, w.attn.norm, sc.normed, s.hidden, stream)?;
        self.attention(gpu, row, state, position, stream)
    }

    /// The rest of the block: the attention output into the stream, then the
    /// FFN, collapsed with the attention site's pre.
    fn finish(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        pre: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (s, k, sc, w) = (
            &self.shared.shape,
            &self.shared.kernels,
            &self.shared.scratch,
            &self.weights,
        );
        let expand = |streams| {
            hc::expand(
                gpu,
                k.hc.expand,
                &s.hc,
                sc.sublayer_out,
                streams,
                sc.post,
                sc.comb,
                streams,
                1,
                stream,
            )
        };
        expand(streams)?;
        self.site(gpu, &w.hc_ffn, streams, sc.next_pre, stream)?;
        hc::collapse(
            gpu,
            k.hc.collapse,
            &s.hc,
            streams,
            sc.site_pre,
            sc.x,
            1,
            stream,
        )?;
        self.norm(gpu, sc.x, w.moe.norm, sc.normed, s.hidden, stream)?;
        self.moe(gpu, stream)?;
        expand(streams)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2d(sc.next_pre, pre, s.hc.copies * 4)
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
        stream: u64,
    ) -> Result<()> {
        self.attend(gpu, row, streams, pre, state, position, stream)?;
        self.finish(gpu, streams, pre, stream)
    }
}

#[cfg(test)]
#[path = "block_test_gpu.rs"]
mod test_gpu;
