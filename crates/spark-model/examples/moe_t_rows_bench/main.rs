// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-identity + speed for the multi-row dedup `_rows` kernels vs K
//! single-row `moe_expert_gate_up_shared_t`/`silu_down_shared_t` launches.
//!
//! Cold pattern: 48 layers x 40 separately-allocated expert tables (the
//! FP4-Local expert count), a fresh table set each iteration, and routing
//! built to hit an exact routed-expert union of 19 / 26 / 33 at K=4 rows
//! (the 10-19 measured p10 / mean / p90 of the 4-row verify union). K=2/3
//! run the union targets that fit in K*TOP_K slots; K=1 is the degenerate
//! arm (production wiring only uses rows >= 2). NVFP4 shared expert on.
//!
//! Run on sparky:
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example moe_t_rows_bench

mod cases;
mod tables;

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;

use crate::cases::{bench_case, mem_kb};
use crate::tables::{LAYER_SET, Layer, Lcg, TOP_K};

fn main() -> Result<()> {
    let g = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let mut rng = Lcg(0xC0FFEE);
    let layers: Vec<Layer> = (0..LAYER_SET)
        .map(|_| Layer::new(&g, &mut rng))
        .collect::<Result<_>>()?;
    println!("MemAvailable at start: {:?} kB", mem_kb());
    if let Ok(sel) = std::env::var("MOE_ROWS_ONLY") {
        let un = std::env::var("MOE_UNION")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(26);
        let sh = std::env::var("MOE_SHARED")
            .map(|v| v == "1")
            .unwrap_or(true);
        for rows in sel.split(',').filter_map(|x| x.parse::<usize>().ok()) {
            bench_case(&g, &layers, rows, un, sh, "v0", &format!("_r{rows}"), 32)?;
        }
        return Ok(());
    }
    let variants: [(&str, u32); 1] = [("v3_b256", 256)];
    let suffix = |vname: &str, rows: usize| -> String {
        format!("_{}_r{rows}_{}", &vname[..2], &vname[3..])
    };
    for (vname, vbs) in variants {
        let rows = 1;
        bench_case(
            &g,
            &layers,
            rows,
            TOP_K,
            true,
            vname,
            &suffix(vname, rows),
            vbs,
        )?; // degenerate
        for rows in [2usize, 3, 4] {
            for union in [19usize, 26, 33] {
                if union <= rows * TOP_K {
                    bench_case(
                        &g,
                        &layers,
                        rows,
                        union,
                        true,
                        vname,
                        &suffix(vname, rows),
                        vbs,
                    )?;
                }
            }
        }
    }
    println!("MemAvailable at end: {:?} kB", mem_kb());
    Ok(())
}
