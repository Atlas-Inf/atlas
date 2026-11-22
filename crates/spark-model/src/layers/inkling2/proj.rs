// SPDX-License-Identifier: AGPL-3.0-only

//! A visit's input projections: one list, which decode and verify batches of up
//! to [`BATCHM_MAX_ROWS`] rows run as one `inkling2_proj_gemv` launch when every
//! projection is BF16 (the checkpoint's weights, Strict), each output row
//! computed as the one-projection GEMV computes it; otherwise one launch each.

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
/// One visit's table for one row count: `MAX_SEGS` of the kernel's 32-byte
/// `ProjSeg` (input, weight and output pointers, outputs, first block).
pub(crate) const TABLE_BYTES: usize = MAX_SEGS * 32;
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
        if !self
            .projections(1)
            .iter()
            .all(|p| matches!(p.w, Linear::Bf16(_)))
        {
            return Ok(());
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

    /// The visit's input projections over `rows` rows.
    pub(super) fn project(&self, gpu: &dyn GpuBackend, rows: usize, stream: u64) -> Result<()> {
        let projs = self.projections(rows);
        let k = self.dims.hidden;
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
