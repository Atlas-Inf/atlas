// SPDX-License-Identifier: AGPL-3.0-only

//! Low-precision activations for the dense prefill GEMMs
//! (`ATLAS_DENSE_PREFILL_LOWP`): the projection groups, the activation-quant
//! scratch, and the choice and launch of the per-token quantizer.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{DenseBf16mma, QUANT_MAX_K, SMEM_W4A8};

/// The MX8 down: the W4A8 stages plus two groups of 128 tokens' 16 scale
/// bytes (`db_smem_bytes` with `DB_A_MX8`).
const SMEM_W4A8_MX: u32 = SMEM_W4A8 + 2 * 128 * 16;

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
    /// `dnmx` (with `gu8`): the W4A8 gate+up writes the down projection's
    /// tokens as E4M3 with a UE8M0 scale per 32-K MMA block, and the W4A8
    /// down reads them (no per-token quantizer pass). Changes the numerics.
    pub const DOWN_MX: u16 = 1 << 12;
    /// An NVFP4 group's W4A8 bit: its W4A4 bit shifted by this (bits 6..12
    /// are the shifted groups, so [`Self::DOWN_MX`] sits above them).
    pub(super) const A8: u32 = 6;
    /// The NVFP4 groups, which have both a W4A4 and a W4A8 form.
    const NVFP4_GROUPS: [(&str, u16); 4] = [
        ("gu", Self::GATE_UP),
        ("dn", Self::DOWN),
        ("qkv", Self::ATTN_QKV),
        ("o", Self::ATTN_O),
    ];
    const NAMES: [(&str, u16); 11] = [
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
        ("dnmx", Self::DOWN_MX),
    ];
    /// `ATLAS_DENSE_PREFILL_LOWP=1`: W4A8 MLP and attention, W8A8 GDN.
    /// Against a BF16 reference this stays at the BF16-MMA path's distance;
    /// NVFP4 tokens on the MLP (`gu,dn`, faster) land at or past the old
    /// default kernels' distance, depending on the rest of the pass.
    pub const DEFAULT: Self = Self(
        (Self::GATE_UP | Self::DOWN | Self::ATTN_QKV | Self::ATTN_O) << Self::A8
            | Self::GDN_QKVZ
            | Self::GDN_OUT,
    );

    pub fn parse(v: &str) -> Self {
        Self::parse_checked(v).0
    }

    /// [`Self::parse`] plus a note for every name it drops or that has no
    /// effect: unknown names, a group named in both token formats (W4A8
    /// wins), and `dnmx` without `gu8`.
    pub fn parse_checked(v: &str) -> (Self, Vec<String>) {
        let mut notes = Vec::new();
        let mix = match v.trim() {
            "1" => Self::DEFAULT,
            "all" => Self(0x3f),
            "" | "0" => Self(0),
            list => Self(list.split(',').map(str::trim).fold(0, |m, n| {
                match Self::NAMES.iter().find(|(k, _)| *k == n) {
                    Some((_, b)) => m | b,
                    None => {
                        notes.push(format!("unknown name {n:?} ignored"));
                        m
                    }
                }
            })),
        };
        for (name, g) in Self::NVFP4_GROUPS {
            if mix.has(g) && mix.has_a8(g) {
                notes.push(format!(
                    "{name} and {name}8 both set: {name}8 (W4A8) is used"
                ));
            }
        }
        if mix.has(Self::DOWN_MX) && !mix.has_a8(Self::GATE_UP) {
            notes.push("dnmx has no effect without gu8".into());
        }
        (mix, notes)
    }

    /// The lever, read where the layers are built (once per layer); its
    /// notes are logged once per process.
    pub fn from_env() -> Self {
        let Ok(v) = std::env::var(LOWP_VAR) else {
            return Self(0);
        };
        let (mix, notes) = Self::parse_checked(&v);
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            for n in &notes {
                tracing::warn!("{LOWP_VAR}={v:?}: {n}");
            }
        });
        mix
    }

    /// The lever set while `ATLAS_DENSE_PREFILL_BF16MMA` is off: it does
    /// nothing then, so say so (once per process).
    pub(super) fn warn_without_bf16mma() {
        if std::env::var(LOWP_VAR).is_ok_and(|v| Self::parse(&v) != Self(0)) {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!("{LOWP_VAR} has no effect without ATLAS_DENSE_PREFILL_BF16MMA=1");
            });
        }
    }

    pub fn has(self, group: u16) -> bool {
        self.0 & group != 0
    }

    /// Whether NVFP4 group `group` takes E4M3 tokens (W4A8).
    pub fn has_a8(self, group: u16) -> bool {
        self.has(group << Self::A8)
    }
}

const LOWP_VAR: &str = "ATLAS_DENSE_PREFILL_LOWP";

/// A projection's group in [`LowpMix`] and the arena's activation-quant
/// scratch (`ffn_act_a`: codes; `ffn_act_scale`: NVFP4 block scales, then
/// the FP32 per-token scales).
#[derive(Clone, Copy, Debug)]
pub struct LowpArgs {
    pub group: u16,
    pub codes: DevicePtr,
    pub scales: DevicePtr,
}

/// The MX handoff's K granularity: its scales stage eight 64-K stages at a time.
const MX_K: u32 = 512;
/// The warp-specialized W4A8 kernels (`dw_body`): 8 MMA + 8 producer warps,
/// persistent; up to 144 token rows a tile (288 rows as two tiles, so the
/// 20 feature tiles of an N = 5120 projection fit one wave of 48 SMs); four
/// token stages, two re-encoded operand stages (`DpA`), three NVFP4 stages.
const WS_THREADS: u32 = 512;
pub(super) const WS_TILE_ROWS: u32 = 144;
const SMEM_W4A8_WS: u32 =
    4 * WS_TILE_ROWS * 64 + 2 * (8 * 4 * 32 * 16 + 8 * 8 * 4) + 3 * 2 * (4096 + 512);
/// A tile's cost beyond its 8-row token tiles in `chunks`: the per-stage
/// re-encode, about 16 tiles' worth of MMAs (1056 rows: 9 chunks, not 12).
pub(super) const WS_FIXED: u32 = 16;
/// Rows from which they replace the 8-warp kernels they reproduce bit for
/// bit. Preliminary microbenchmark at the 27B shapes, us 8-warp / 16-warp
/// (round 4) / warp-specialized: gate+up 96 rows 524 / 523 / 525, 288 918 /
/// 844 / 729, 1056 2973 / 2735 / 2654, 4128 11114 / 10673 / 9957; down 288
/// 569 / 618 / 368, 1056 1534 / 1658 / 1262; q 288 301 / 300 / 235; o 288
/// 209 / 220 / 135. At 96 rows the engine measured them 1 to 6 % slower
/// (gate+up 554 / 547 us, down+o 283 / 278), below up to 15 %. The MX
/// handoff (dnmx) keeps the 8-warp kernels.
const WS_ROWS: u32 = 129;
/// The persistent W4A4 kernels (`dz_body`): 8 MMA warps + a loader
/// warpgroup; db_body's token layout and 6 stages from Z_LONG_ROWS rows,
/// below that the per-lane layout and 4 stages (`wsl`). No re-encode, so a
/// tile costs about CTA_FIXED beyond its token tiles. Preliminary
/// microbenchmark, us 8-warp / persistent: gate+up 96 rows 537 / 509, 288
/// 660 / 693, 1056 1839 / 1663, 4128 7368 / 6387; down 96 246 / 235, 288
/// 344 / 300, 1056 864 / 786; q 288 228 / 222, 2112 1272 / 1001.
const Z_THREADS: u32 = 384;
const SMEM_W4A4_WS: u32 = 6 * (WS_TILE_ROWS * 36 + 2 * (4096 + 512));
const SMEM_W4A4_WSL: u32 = 4 * (WS_TILE_ROWS * 64 + 2 * (4096 + 512));
const Z_ROWS: u32 = 96;
const Z_LONG_ROWS: u32 = 384;

/// A bit-identical persistent twin of a low-precision kernel.
pub(super) struct Twin {
    pub kernel: KernelHandle,
    pub threads: u32,
    pub smem: u32,
    /// `chunks`' per-CTA cost.
    pub fixed: u32,
}

impl DenseBf16mma {
    /// The persistent twin of a W4A8 / W4A4 kernel for `m` rows, where it is
    /// faster.
    pub(super) fn lowp_twin(&self, kernel: KernelHandle, m: u32) -> Option<Twin> {
        let [w8_gate_up, w8_nvfp4] = self.w4a8ws;
        let [z_gate_up, z_nvfp4, zl_gate_up, zl_nvfp4] = self.w4a4ws;
        let long = m >= Z_LONG_ROWS;
        let (twin, threads, smem, fixed) = match kernel.0 {
            k if k == self.w4a8_gate_up.0 && m >= WS_ROWS => {
                (w8_gate_up, WS_THREADS, SMEM_W4A8_WS, WS_FIXED)
            }
            k if k == self.w4a8.0 && m >= WS_ROWS => (w8_nvfp4, WS_THREADS, SMEM_W4A8_WS, WS_FIXED),
            k if k == self.w4a4_gate_up.0 && m >= Z_ROWS => match long {
                true => (z_gate_up, Z_THREADS, SMEM_W4A4_WS, super::CTA_FIXED),
                false => (zl_gate_up, Z_THREADS, SMEM_W4A4_WSL, super::CTA_FIXED),
            },
            k if k == self.w4a4.0 && m >= Z_ROWS => match long {
                true => (z_nvfp4, Z_THREADS, SMEM_W4A4_WS, super::CTA_FIXED),
                false => (zl_nvfp4, Z_THREADS, SMEM_W4A4_WSL, super::CTA_FIXED),
            },
            _ => return None,
        };
        (twin.0 != 0).then_some(Twin {
            kernel: twin,
            threads,
            smem,
            fixed,
        })
    }

    /// Whether the gate+up -> down handoff runs as MX8 tokens (`dnmx`) for a
    /// `k_down`-wide intermediate: both kernels shipped, W4A8 gate+up on.
    pub(super) fn mx_on(&self, lowp: &LowpArgs, k_down: u32) -> bool {
        self.lowp.has(LowpMix::DOWN_MX)
            && self.lowp.has_a8(LowpMix::GATE_UP)
            && self.w4a8_gate_up_mx.0 != 0
            && self.w4a8_mx.0 != 0
            && self.quant_fp8.0 != 0
            && k_down.is_multiple_of(MX_K)
            && !lowp.codes.is_null()
            && !lowp.scales.is_null()
    }

    /// The W4A8 down projection on MX8 tokens `input` (codes `[m, k]`, then
    /// the `[m, k/32]` scales), as written by `dense_w4a8_gate_up_silu_mx`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mx_down(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        w: &crate::weight_map::QuantizedWeight,
        out: DevicePtr,
        [m, n, k]: [u32; 3],
        stream: u64,
    ) -> Result<()> {
        let c = super::chunks(
            m,
            n / super::FEATURES,
            (self.slots / super::CTAS_PER_SM).max(1),
            super::LOWP_ROWS,
            super::CTA_FIXED,
        );
        KernelLaunch::new(gpu, self.w4a8_mx)
            .grid([(n / super::FEATURES) * c, 1, 1])
            .block([super::THREADS, 1, 1])
            .shared_mem(SMEM_W4A8_MX)
            .arg_ptr(input)
            .arg_ptr(input.offset(m as usize * k as usize))
            .arg_ptr(w.weight)
            .arg_ptr(w.weight_scale)
            .arg_f32(w.weight_scale_2)
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(n)
            .arg_u32(c)
            .launch(stream)
    }
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
        let (m, notes) = LowpMix::parse_checked(" gu8, dn8 ,gout,bogus");
        assert!(m.has_a8(LowpMix::GATE_UP) && m.has_a8(LowpMix::DOWN) && m.has(LowpMix::GDN_OUT));
        assert!(
            !m.has(LowpMix::GATE_UP) && !m.has(LowpMix::GDN_QKVZ) && !m.has_a8(LowpMix::ATTN_O)
        );
        // The unknown name is dropped, and said so.
        assert_eq!(notes, vec!["unknown name \"bogus\" ignored".to_string()]);
        assert!(LowpMix::parse("gu8,dnmx").has(LowpMix::DOWN_MX));
        assert!(!LowpMix::parse("1").has(LowpMix::DOWN_MX));
        assert_eq!(
            LowpMix::parse("all"),
            LowpMix::parse("gu,dn,qkv,o,qkvz,gout")
        );
        for v in ["1", "all", "0", "", "gu8,dnmx", "qkv,o8"] {
            assert!(LowpMix::parse_checked(v).1.is_empty(), "{v}");
        }
    }

    #[test]
    fn lowp_mix_notes_conflicts_and_dead_names() {
        let (m, notes) = LowpMix::parse_checked("gu,gu8");
        assert!(m.has_a8(LowpMix::GATE_UP));
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("gu8 (W4A8) is used"));
        let (_, notes) = LowpMix::parse_checked("dnmx,gu");
        assert_eq!(notes, vec!["dnmx has no effect without gu8".to_string()]);
    }

    #[test]
    fn lowp_mix_bits_are_distinct() {
        // DOWN_MX used to share bit 10 with GDN_QKVZ's shifted (W4A8) bit.
        let shifted = [LowpMix::GDN_QKVZ, LowpMix::GDN_OUT].map(|g| g << LowpMix::A8);
        assert!(shifted.iter().all(|&b| b & LowpMix::DOWN_MX == 0));
        assert!(!LowpMix(LowpMix::DOWN_MX).has_a8(LowpMix::GDN_QKVZ));
        let mut seen = 0u16;
        for (_, b) in LowpMix::NAMES {
            assert_eq!(seen & b, 0, "bit {b:#x} reused");
            seen |= b;
        }
    }
}
