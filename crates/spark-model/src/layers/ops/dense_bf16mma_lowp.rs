// SPDX-License-Identifier: AGPL-3.0-only

//! Low-precision activations for the dense prefill GEMMs
//! (`ATLAS_DENSE_PREFILL_LOWP`): the projection groups, the activation-quant
//! scratch, and the choice and launch of the per-token quantizer.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{DenseBf16mma, QUANT_MAX_K};

/// Projections whose activations `ATLAS_DENSE_PREFILL_LOWP` quantizes per
/// token (on top of `ATLAS_DENSE_PREFILL_BF16MMA=1`). NVFP4 weights take
/// NVFP4 tokens (W4A4, `kind::mxf4nvf4`: `gu,dn,qkv,o`) or E4M3 tokens
/// (W4A8, the weights re-encoded as block-scaled E4M3: `gu8,dn8,qkv8,o8`);
/// the FP8 GDN projections take E4M3 tokens (W8A8: `qkvz,gout`). The value
/// is `1` for [`LowpMix::DEFAULT`] or a comma list of those names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowpMix(pub(super) u16);

impl LowpMix {
    pub const GATE_UP: u16 = 1;
    pub const DOWN: u16 = 2;
    pub const ATTN_QKV: u16 = 4;
    pub const ATTN_O: u16 = 8;
    pub const GDN_QKVZ: u16 = 16;
    pub const GDN_OUT: u16 = 32;
    /// An NVFP4 group's W4A8 bit: its W4A4 bit shifted by this.
    pub(super) const A8: u32 = 6;
    const NAMES: [(&str, u16); 10] = [
        ("gu", Self::GATE_UP),
        ("dn", Self::DOWN),
        ("qkv", Self::ATTN_QKV),
        ("o", Self::ATTN_O),
        ("qkvz", Self::GDN_QKVZ),
        ("gout", Self::GDN_OUT),
        ("gu8", Self::GATE_UP << Self::A8),
        ("dn8", Self::DOWN << Self::A8),
        ("qkv8", Self::ATTN_QKV << Self::A8),
        ("o8", Self::ATTN_O << Self::A8),
    ];
    /// `ATLAS_DENSE_PREFILL_LOWP=1`: W4A8 MLP and attention, W8A8 GDN. On the
    /// d3 oracle this stays at the BF16-MMA path's distance from d3's BF16
    /// runtime; NVFP4 tokens on the MLP (`gu,dn`, faster) land at or past the
    /// old default kernels' distance, depending on the rest of the pass.
    pub const DEFAULT: Self = Self(
        (Self::GATE_UP | Self::DOWN | Self::ATTN_QKV | Self::ATTN_O) << Self::A8
            | Self::GDN_QKVZ
            | Self::GDN_OUT,
    );

    pub fn parse(v: &str) -> Self {
        match v.trim() {
            "1" => Self::DEFAULT,
            "all" => Self(0x3f),
            list => Self(
                list.split(',')
                    .filter_map(|n| Self::NAMES.iter().find(|(k, _)| *k == n.trim()))
                    .fold(0, |m, (_, b)| m | b),
            ),
        }
    }

    pub fn from_env() -> Self {
        std::env::var("ATLAS_DENSE_PREFILL_LOWP").map_or(Self(0), |v| Self::parse(&v))
    }

    pub fn has(self, group: u16) -> bool {
        self.0 & group != 0
    }

    /// Whether NVFP4 group `group` takes E4M3 tokens (W4A8).
    pub fn has_a8(self, group: u16) -> bool {
        self.has(group << Self::A8)
    }
}

/// A projection's group in [`LowpMix`] and the arena's activation-quant
/// scratch (`ffn_act_a`: codes; `ffn_act_scale`: NVFP4 block scales, then
/// the FP32 per-token scales).
#[derive(Clone, Copy, Debug)]
pub struct LowpArgs {
    pub group: u16,
    pub codes: DevicePtr,
    pub scales: DevicePtr,
}

impl DenseBf16mma {
    /// Whether `group` takes low-precision activations through `handle`
    /// for a `k`-wide input.
    pub(super) fn lowp_on(
        &self,
        group: u16,
        lowp: &LowpArgs,
        k: u32,
        [handle, quant]: [KernelHandle; 2],
    ) -> bool {
        self.lowp.has(group)
            && k <= QUANT_MAX_K
            && handle.0 != 0
            && quant.0 != 0
            && !lowp.codes.is_null()
            && !lowp.scales.is_null()
    }

    /// Quantizes `input[m, k]` per token into the scratch: NVFP4 codes +
    /// block scales, or E4M3 codes; returns the per-token scale pointer
    /// (after the block scales, 16-byte aligned).
    pub(super) fn quantize(
        &self,
        gpu: &dyn GpuBackend,
        nvfp4: bool,
        input: DevicePtr,
        lowp: &LowpArgs,
        [m, k]: [u32; 2],
        stream: u64,
    ) -> Result<DevicePtr> {
        let (kernel, block_scales) = match nvfp4 {
            true => (self.quant_nvfp4, m as usize * (k as usize / 16)),
            false => (self.quant_fp8, 0),
        };
        let token_scales = lowp.scales.offset(block_scales.next_multiple_of(16));
        let launch = KernelLaunch::new(gpu, kernel)
            .grid([m, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(lowp.codes);
        let launch = match nvfp4 {
            true => launch.arg_ptr(lowp.scales),
            false => launch,
        };
        launch.arg_ptr(token_scales).arg_u32(k).launch(stream)?;
        Ok(token_scales)
    }

    /// The low-precision twin serving this launch, if any: `(NVFP4 tokens,
    /// kernel, shared memory)`. `a8` is the E4M3-token kernel and its group bit.
    pub(super) fn lowp_kernel(
        &self,
        lowp: &LowpArgs,
        k: u32,
        a4: (KernelHandle, u32),
        (a8_group, a8, a8_smem): (u16, KernelHandle, u32),
    ) -> Option<(bool, KernelHandle, u32)> {
        if self.lowp_on(a8_group, lowp, k, [a8, self.quant_fp8]) {
            Some((false, a8, a8_smem))
        } else if self.lowp_on(lowp.group, lowp, k, [a4.0, self.quant_nvfp4]) {
            Some((true, a4.0, a4.1))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LowpMix;

    #[test]
    fn lowp_mix_parses_groups_and_formats() {
        assert_eq!(LowpMix::parse("0"), LowpMix(0));
        assert_eq!(LowpMix::parse(""), LowpMix(0));
        assert_eq!(
            LowpMix::parse("1"),
            LowpMix::parse("gu8,dn8,qkv8,o8,qkvz,gout")
        );
        let m = LowpMix::parse(" gu8, dn8 ,gout,bogus");
        assert!(m.has_a8(LowpMix::GATE_UP) && m.has_a8(LowpMix::DOWN) && m.has(LowpMix::GDN_OUT));
        assert!(
            !m.has(LowpMix::GATE_UP) && !m.has(LowpMix::GDN_QKVZ) && !m.has_a8(LowpMix::ATTN_O)
        );
        assert_eq!(
            LowpMix::parse("all"),
            LowpMix::parse("gu,dn,qkv,o,qkvz,gout")
        );
    }
}
