// SPDX-License-Identifier: AGPL-3.0-only

//! What the score pack store keeps of a pack ([`Part`]s: GDN h state and K/V
//! as E4M3 rows with a scale a row, the conv window as is) and the kernels that
//! move them into a stored copy and back (`score_store.rs`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::super::types::TransformerModel;
use super::score_pack::ssm_ptrs_of;
use crate::traits::SequenceState;

/// One piece of a pack: rows of `len` elements (`dtype` 0 = FP32, 1 = BF16)
/// kept as E4M3 rows, or bytes kept as they are.
pub(super) enum Part {
    Fp8 {
        at: DevicePtr,
        rows: usize,
        len: usize,
        dtype: u32,
    },
    Raw {
        at: DevicePtr,
        bytes: usize,
    },
}

impl Part {
    /// Its store bytes (16-byte aligned pieces): codes, then scales.
    pub(super) fn bytes(&self) -> usize {
        match *self {
            Part::Fp8 { rows, len, .. } => {
                (rows * len).next_multiple_of(16) + (rows * 4).next_multiple_of(16)
            }
            Part::Raw { bytes, .. } => bytes.next_multiple_of(16),
        }
    }
}

impl TransformerModel {
    /// The pieces of the pack sequence `seq` the store keeps, in store order.
    pub(super) fn store_parts(&self, seq: &SequenceState, kv: &PagedKvCache) -> Result<Vec<Part>> {
        ensure!(
            self.ssm_pool.h_stored_bytes == self.config.ssm_h_state_bytes(),
            "score store: the SSM pool must hold FP32 state"
        );
        let kd = self.config.linear_key_head_dim;
        let h_rows = self.config.ssm_h_state_bytes() / 4 / kd;
        let mut parts = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                let (h, conv) = ssm_ptrs_of(seq, i)?;
                parts.push(Part::Fp8 {
                    at: h,
                    rows: h_rows,
                    len: kd,
                    dtype: 0,
                });
                parts.push(Part::Raw {
                    at: conv,
                    bytes: self.ssm_pool.conv_bytes,
                });
            }
        }
        let d = self.config.head_dim;
        for l in 0..kv.num_layers() {
            let (k, v) = (
                kv.k_block_stride_bytes_for_layer(l),
                kv.v_block_stride_bytes_for_layer(l),
            );
            if k == 0 && v == 0 {
                continue;
            }
            ensure!(
                kv.dtype_for_layer(l) == KvCacheDtype::Bf16 && k % (2 * d) == 0 && v % (2 * d) == 0,
                "score store: needs a BF16 KV cache"
            );
            for &b in &seq.block_table {
                parts.push(Part::Fp8 {
                    at: kv.k_cache_ptr(l, b),
                    rows: k / 2 / d,
                    len: d,
                    dtype: 1,
                });
                parts.push(Part::Fp8 {
                    at: kv.v_cache_ptr(l, b),
                    rows: v / 2 / d,
                    len: d,
                    dtype: 1,
                });
            }
        }
        Ok(parts)
    }

    /// Move `parts` into the store buffer `buf` (`into`) or back out of it.
    pub(super) fn store_move(
        &self,
        parts: &[Part],
        buf: DevicePtr,
        into: bool,
        stream: u64,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();
        let name = if into { "rows_to_fp8" } else { "rows_from_fp8" };
        // Asked first: looked up here (after the boot audit sealed), a kernel this
        // build lacks would abort the server instead of refusing the store.
        ensure!(
            gpu.ships_kernel("residual_add", name),
            "score store: {name} is not in this build"
        );
        let kernel = crate::layers::try_kernel(gpu, "residual_add", name);
        ensure!(kernel.0 != 0, "score store: {name} is not in this build");
        let mut off = 0;
        for part in parts {
            match *part {
                Part::Fp8 {
                    at,
                    rows,
                    len,
                    dtype,
                } => {
                    let (codes, scales) = (
                        buf.offset(off),
                        buf.offset(off + (rows * len).next_multiple_of(16)),
                    );
                    let launch = KernelLaunch::new(gpu, kernel)
                        .grid([rows.div_ceil(8) as u32, 1, 1])
                        .block([256, 1, 1]);
                    let launch = match into {
                        true => launch.arg_ptr(at).arg_ptr(codes).arg_ptr(scales),
                        false => launch.arg_ptr(codes).arg_ptr(scales).arg_ptr(at),
                    };
                    launch
                        .arg_u32(rows as u32)
                        .arg_u32(len as u32)
                        .arg_u32(dtype)
                        .launch(stream)?;
                }
                Part::Raw { at, bytes } => match into {
                    true => gpu.copy_d2d_async(at, buf.offset(off), bytes, stream)?,
                    false => gpu.copy_d2d_async(buf.offset(off), at, bytes, stream)?,
                },
            }
            off += part.bytes();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Part;
    use spark_runtime::gpu::DevicePtr;

    #[test]
    fn parts_take_aligned_codes_and_scales() {
        let fp8 = Part::Fp8 {
            at: DevicePtr::NULL,
            rows: 6144,
            len: 128,
            dtype: 0,
        };
        assert_eq!(fp8.bytes(), 6144 * 128 + 6144 * 4);
        let odd = Part::Fp8 {
            at: DevicePtr::NULL,
            rows: 3,
            len: 5,
            dtype: 1,
        };
        assert_eq!(odd.bytes(), 16 + 16);
        assert_eq!(
            Part::Raw {
                at: DevicePtr::NULL,
                bytes: 17
            }
            .bytes(),
            32
        );
    }
}
