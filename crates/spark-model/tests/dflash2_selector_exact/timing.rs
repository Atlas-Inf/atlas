// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Microseconds per launch of the rank split's selector halves: one rank's
/// top-k pass over half the vocabulary, and the walk.
#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn split_selector_timing() -> Result<()> {
    let (gpu, stream) = setup()?;
    let (topk, walk, walk_conf) = split_kernels(&gpu)?;
    let inp = inputs(&gpu, &CASES[0])?;
    let cut = VOCAB / 2 / 16 * 16;
    let scratch_bytes = ops::dflash2_selector_scratch_bytes(GAMMA);
    let d_scratch = gpu.alloc(scratch_bytes)?;
    gpu.memset(d_scratch, 0, scratch_bytes)?;
    let d_lists = gpu.alloc(ops::dflash2_selector_lists_bytes(GAMMA))?;
    let d_out = gpu.alloc(2 * GAMMA * 4)?;
    let time = |name: &str, launch: &dyn Fn() -> Result<()>| -> Result<()> {
        for _ in 0..20 {
            launch()?;
        }
        gpu.synchronize(stream)?;
        const N: usize = 400;
        let t = std::time::Instant::now();
        for _ in 0..N {
            launch()?;
        }
        gpu.synchronize(stream)?;
        let us = t.elapsed().as_secs_f64() * 1e6 / N as f64;
        println!("TIMING {name}: {us:.1} us/launch");
        Ok(())
    };
    time("split top-k (half vocabulary)", &|| {
        ops::dflash2_selector_topk(
            &gpu,
            topk,
            inp.d_logits,
            VOCAB as u32,
            cut as u32,
            0,
            GAMMA as u32,
            d_lists,
            d_scratch,
            stream,
        )
    })?;
    for (name, kernel) in [("split walk prod", walk), ("split walk conf", walk_conf)] {
        time(name, &|| {
            ops::dflash2_selector_walk(
                &gpu,
                kernel,
                [d_lists, d_lists],
                inp.d_proj,
                inp.d_pred,
                inp.d_succ,
                d_out,
                inp.d_anchor,
                inp.d_ban,
                inp.end_ids,
                GAMMA as u32,
                VOCAB as u32,
                RANK as u32,
                TOP_K as u32,
                stream,
            )
        })?;
    }
    Ok(())
}

/// Microseconds per fused selector launch at the GLM geometry.
#[test]
#[ignore = "requires CUDA GPU + non-stub kernel build"]
fn selector_timing() -> Result<()> {
    let (gpu, stream) = setup()?;
    let (prod, conf) = kernels(&gpu)?;
    let inp = inputs(&gpu, &CASES[0])?;
    let scratch_bytes = ops::dflash2_selector_scratch_bytes(GAMMA);
    let d_scratch = gpu.alloc(scratch_bytes)?;
    gpu.memset(d_scratch, 0, scratch_bytes)?;
    let d_out = gpu.alloc(2 * GAMMA * 4)?;
    for (name, kernel) in [("prod", prod), ("conf", conf)] {
        let launch = || {
            ops::dflash2_candidate_selector(
                &gpu,
                kernel,
                inp.d_logits,
                inp.d_proj,
                inp.d_pred,
                inp.d_succ,
                d_out,
                inp.d_anchor,
                inp.d_ban,
                inp.end_ids,
                GAMMA as u32,
                VOCAB as u32,
                RANK as u32,
                TOP_K as u32,
                d_scratch,
                stream,
            )
        };
        for _ in 0..20 {
            launch()?;
        }
        gpu.synchronize(stream)?;
        const N: usize = 400;
        let t = std::time::Instant::now();
        for _ in 0..N {
            launch()?;
        }
        gpu.synchronize(stream)?;
        let us = t.elapsed().as_secs_f64() * 1e6 / N as f64;
        println!("TIMING fused selector {name}: {us:.1} us/launch");
    }
    Ok(())
}
