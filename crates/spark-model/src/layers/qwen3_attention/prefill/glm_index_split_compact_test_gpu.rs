// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): `glm_index_topk_expand` rows packed compact
//! (`ATLAS_GLM_INDEX_SPLIT_COMPACT`) and expanded again are the same words,
//! for selections past the top-k budget (with tied scores) and within it.
//!
//!   cargo test --release -p spark-model --lib index_split_compact -- --ignored --nocapture

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

use crate::layers::ops;

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn index_split_compact_rows_expand_to_the_same_words() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (topk, pool) = (2048u32, 4u32);
    let width = topk + pool - 1;
    let topk_k = g.kernel("glm_indexer", "glm_index_topk_expand")?;
    let compact = ops::GlmSelCompact {
        pack: g.kernel("glm_indexer", "glm_index_sel_pack")?,
        unpack: g.kernel("glm_indexer", "glm_index_sel_unpack")?,
        topk,
        pool,
        width,
    };
    let mut s = 0x5e1ec7u64;
    // (history, rows, distinct scores): long rows with ties, and short rows
    // whose whole history fits the budget.
    for (start, rows, levels) in [
        (30000u32, 512usize, 64u64),
        (30000, 256, 1 << 20),
        (900, 300, 7),
    ] {
        let stride = (start as usize + rows).div_ceil(pool as usize);
        let logits: Vec<u8> = (0..rows * stride)
            .flat_map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % levels) as f32 * 0.01 - 0.3).to_le_bytes()
            })
            .collect();
        let lp = g.alloc(logits.len())?;
        g.copy_h2d(&logits, lp)?;
        let bytes = rows * width as usize * 4;
        let (sel, back) = (g.alloc(bytes)?, g.alloc(bytes)?);
        let packed = g.alloc(rows * compact.row_words() * 4)?;
        g.memset_async(back, 0x5a, bytes, stream)?;
        ops::glm_index_topk_expand(
            g,
            topk_k,
            lp,
            sel,
            rows as u32,
            start,
            stride as u32,
            topk,
            pool,
            width,
            stream,
        )?;
        compact.pack(g, sel, packed, rows, stream)?;
        compact.unpack(g, packed, back, rows, stream)?;
        g.synchronize(stream)?;
        let (mut want, mut got) = (vec![0u8; bytes], vec![0u8; bytes]);
        g.copy_d2h(sel, &mut want)?;
        g.copy_d2h(back, &mut got)?;
        ensure!(
            want == got,
            "history {start}, {rows} rows, {levels} score levels: expanded compact rows differ"
        );
        for p in [lp, sel, back, packed] {
            g.free(p)?;
        }
    }
    println!(
        "index split compact rows: {} of {width} words a row, expand to the same words",
        compact.row_words()
    );
    Ok(())
}
