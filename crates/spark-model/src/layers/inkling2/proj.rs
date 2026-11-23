// SPDX-License-Identifier: AGPL-3.0-only

//! A visit's input projections: one list, which decode and verify batches of up
//! to [`BATCHM_MAX_ROWS`] rows run as one `inkling2_proj_gemv` launch when every
//! projection is BF16 (the checkpoint's weights, Strict), each output row
//! computed as the one-projection GEMV computes it. With NVFP4 projections
//! among them (the Spark overlay) one decode row runs as one
//! `inkling2_proj_gemv_mixed` launch, each output computed as `w4a16_gemv` or
//! `dense_gemv_bf16` computes it; more rows take one launch a projection.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::glue::BATCHM_MAX_ROWS;
use super::layer::Inkling2Layer;
use super::weights::{KvConv, Linear};

/// `n` outputs of `w` over the input rows at `x`, into `y`.
#[derive(Clone, Copy)]
pub(super) struct Proj {
    x: DevicePtr,
    w: Linear,
    y: DevicePtr,
    n: usize,
}

/// Segments a visit has at most: q, k, v, relative bias, gate, sinks and the
/// two dynamic convs' taps.
const MAX_SEGS: usize = 8;
/// The mixed kernel's table entry (`MixSeg`): input, weight, block scales and
/// output pointers, outputs, first block, the weight's global scale, padding.
const MIX_SEG_BYTES: usize = 48;
/// One visit's table for one row count: `MAX_SEGS` of the kernel's 32-byte
/// `ProjSeg` (input, weight and output pointers, outputs, first block), or
/// for one row with NVFP4 projections `MAX_SEGS` of `MixSeg`.
pub(crate) const TABLE_BYTES: usize = MAX_SEGS * MIX_SEG_BYTES;

/// One projection of a mixed table: `scale` is the NVFP4 block scales, or
/// null for a BF16 weight.
#[derive(Clone, Copy)]
struct MixEntry {
    a: u64,
    w: u64,
    scale: u64,
    c: u64,
    n: u32,
    scale2: f32,
}

/// The mixed kernel's table for `entries` and the blocks its launch needs.
fn mixed_table(entries: &[MixEntry]) -> (Vec<u8>, u32) {
    let mut table = Vec::with_capacity(entries.len() * MIX_SEG_BYTES);
    let mut block0 = 0u32;
    for e in entries {
        for ptr in [e.a, e.w, e.scale, e.c] {
            table.extend(ptr.to_le_bytes());
        }
        table.extend(e.n.to_le_bytes());
        table.extend(block0.to_le_bytes());
        table.extend(e.scale2.to_le_bytes());
        table.extend(0u32.to_le_bytes());
        block0 += e.n.div_ceil(OUTPUTS_PER_BLOCK as u32);
    }
    (table, block0)
}
/// Outputs a GEMV block computes (the kernels' `N_PER_BLOCK`).
const OUTPUTS_PER_BLOCK: usize = 4;

impl Inkling2Layer {
    /// Whether the q, k and v projections read their own normed inputs (a full
    /// layer, or a loop pass whose message gates made three) rather than one.
    pub(super) fn inputs_split(&self) -> bool {
        self.visit.full || self.visit.pass > 0
    }

    /// The visit's input projections over `rows` rows, in launch order.
    fn projections(&self, rows: usize) -> Vec<Proj> {
        let (s, w, d) = (&self.shared, &self.w, &self.dims);
        let normed = |i: usize| s.normed3.offset(i * rows * d.hidden * 2);
        let (aq, ak, av) = if self.inputs_split() {
            (normed(0), normed(1), normed(2))
        } else {
            (normed(0), normed(0), normed(0))
        };
        let mut out = vec![
            Proj {
                x: aq,
                w: w.wq,
                y: s.q,
                n: d.q_dim,
            },
            Proj {
                x: ak,
                w: w.wk,
                y: s.k_lin,
                n: d.kv_dim,
            },
            Proj {
                x: av,
                w: w.wv,
                y: s.v_lin,
                n: d.kv_dim,
            },
            Proj {
                x: aq,
                w: w.wr,
                y: s.r,
                n: d.rel_dim,
            },
            Proj {
                x: aq,
                w: w.gate,
                y: s.gate,
                n: d.q_dim,
            },
            Proj {
                x: aq,
                w: Linear::Bf16(w.sink),
                y: s.sink,
                n: self.p.heads,
            },
        ];
        for (conv, x, y) in [(w.k_conv, ak, s.dyn_k), (w.v_conv, av, s.dyn_v)] {
            if let KvConv::Dynamic { proj, .. } = conv {
                out.push(Proj {
                    x,
                    w: Linear::Bf16(proj),
                    y,
                    n: d.dyn_dim,
                });
            }
        }
        out
    }

    fn proj_table(&self, rows: usize) -> DevicePtr {
        let slot = self.visit_idx * BATCHM_MAX_ROWS + rows - 1;
        self.shared.proj_tables.offset(slot * TABLE_BYTES)
    }

    /// Writes the visit's fused-projection tables (one a row count) once its
    /// shared buffers are attached; a visit with a quantized projection keeps
    /// one launch a projection.
    pub(crate) fn upload_proj_tables(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.fused_proj = false;
        self.fused_mixed = false;
        if !self
            .projections(1)
            .iter()
            .all(|p| matches!(p.w, Linear::Bf16(_)))
        {
            return self.upload_mixed_table(gpu);
        }
        for rows in 1..=BATCHM_MAX_ROWS {
            let mut table = Vec::with_capacity(TABLE_BYTES);
            let mut block0 = 0u32;
            for p in self.projections(rows) {
                let Linear::Bf16(w) = p.w else { unreachable!() };
                for ptr in [p.x, w, p.y] {
                    table.extend(ptr.0.to_le_bytes());
                }
                table.extend((p.n as u32).to_le_bytes());
                table.extend(block0.to_le_bytes());
                block0 += p.n.div_ceil(OUTPUTS_PER_BLOCK) as u32;
            }
            gpu.copy_h2d(&table, self.proj_table(rows))?;
        }
        self.fused_proj = true;
        Ok(())
    }

    /// One decode row's table for a visit with NVFP4 projections, where the
    /// target ships the mixed kernel and every NVFP4 weight has one global
    /// scale (the kernel reads no per-row scales).
    fn upload_mixed_table(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.gk.proj_gemv_mixed.0 == 0 {
            return Ok(());
        }
        let mut entries = Vec::new();
        for p in self.projections(1) {
            let (w, scale, scale2) = match p.w {
                Linear::Bf16(w) => (w.0, 0, 0.0),
                Linear::Nvfp4 { q, .. } if q.weight_scale_2_vec.is_null() => {
                    (q.weight.0, q.weight_scale.0, q.weight_scale_2)
                }
                Linear::Nvfp4 { .. } => return Ok(()),
            };
            entries.push(MixEntry {
                a: p.x.0,
                w,
                scale,
                c: p.y.0,
                n: p.n as u32,
                scale2,
            });
        }
        gpu.copy_h2d(&mixed_table(&entries).0, self.proj_table(1))?;
        self.fused_mixed = true;
        Ok(())
    }

    /// The visit's input projections over `rows` rows.
    pub(super) fn project(&self, gpu: &dyn GpuBackend, rows: usize, stream: u64) -> Result<()> {
        let projs = self.projections(rows);
        let k = self.dims.hidden;
        if self.fused_mixed && rows == 1 {
            let blocks: usize = projs.iter().map(|p| p.n.div_ceil(OUTPUTS_PER_BLOCK)).sum();
            return KernelLaunch::new(gpu, self.gk.proj_gemv_mixed)
                .grid([blocks as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.proj_table(1))
                .arg_u32(projs.len() as u32)
                .arg_u32(k as u32)
                .launch(stream);
        }
        if self.fused_proj && rows <= BATCHM_MAX_ROWS {
            let blocks: usize = projs.iter().map(|p| p.n.div_ceil(OUTPUTS_PER_BLOCK)).sum();
            return KernelLaunch::new(gpu, self.gk.proj_gemv)
                .grid([blocks as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.proj_table(rows))
                .arg_u32(projs.len() as u32)
                .arg_u32(rows as u32)
                .arg_u32(k as u32)
                .launch(stream);
        }
        for p in projs {
            self.gk.proj(gpu, rows, p.x, p.w, p.y, p.n, k, stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{MIX_SEG_BYTES, MixEntry, mixed_table};

    #[test]
    fn a_mixed_table_lays_out_the_kernels_entries() {
        let entries = [
            // NVFP4, 3,072 outputs: 768 blocks
            MixEntry {
                a: 0x10,
                w: 0x20,
                scale: 0x30,
                c: 0x40,
                n: 3072,
                scale2: 0.5,
            },
            // BF16 (no block scales), 24 outputs: 6 blocks
            MixEntry {
                a: 0x10,
                w: 0x50,
                scale: 0,
                c: 0x60,
                n: 24,
                scale2: 0.0,
            },
            // 6 outputs: 2 blocks, the second partly filled
            MixEntry {
                a: 0x10,
                w: 0x70,
                scale: 0x80,
                c: 0x90,
                n: 6,
                scale2: 2.0,
            },
        ];
        let (table, blocks) = mixed_table(&entries);
        assert_eq!(table.len(), 3 * MIX_SEG_BYTES);
        assert_eq!(blocks, 768 + 6 + 2);
        let u64_at = |o: usize| u64::from_le_bytes(table[o..o + 8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(table[o..o + 4].try_into().unwrap());
        // entry 0: a, w, scale, c, n, block0, scale2
        assert_eq!(
            [u64_at(0), u64_at(8), u64_at(16), u64_at(24)],
            [0x10, 0x20, 0x30, 0x40]
        );
        assert_eq!([u32_at(32), u32_at(36)], [3072, 0]);
        assert_eq!(f32::from_le_bytes(table[40..44].try_into().unwrap()), 0.5);
        // entry 1 starts where entry 0's blocks end, and is BF16
        let e1 = MIX_SEG_BYTES;
        assert_eq!(u64_at(e1 + 16), 0);
        assert_eq!([u32_at(e1 + 32), u32_at(e1 + 36)], [24, 768]);
        assert_eq!(u32_at(2 * MIX_SEG_BYTES + 36), 774);
    }
}
