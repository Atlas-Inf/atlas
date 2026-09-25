// SPDX-License-Identifier: AGPL-3.0-only

//! Stages of `QsaIndexer::prefill_select` (host top-k arm, attention arm) and
//! the packed rank key, split out of `qsa_select.rs` for the 500-LoC cap.
//! Bodies are moved verbatim; `qsa_select.rs` keeps the control flow.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::QsaIndexer;
use crate::layers::ops;

/// Pack `(score DESCENDING, index ASCENDING)` into one `u64` so the top-k
/// selection is a plain integer partition.
///
/// The float goes through the standard monotone `f32 -> u32` map (flip the
/// sign bit for positives, invert every bit for negatives), which preserves
/// IEEE ordering; inverting that gives DESCENDING score, and the index in the
/// low 32 bits breaks ties by ascending index. That is exactly the comparator
/// this replaces -- `partial_cmp(b, a).then(a.cmp(&b))` -- so the selected set
/// AND its order are unchanged, which matters: `qsa_prefill_attn` walks the
/// list warp-striped and its online softmax accumulates in list order.
///
/// It is also strictly better defined. The old comparator collapsed a NaN
/// comparison to `Equal`, which is not a total order and makes `sort_by`'s
/// output unspecified; here every distinct element has a distinct key. Scores
/// are `relu`'d sums (or `-1e30` for out-of-range blocks), so NaN should not
/// arise -- but "should not" is not a sort precondition.
#[inline]
fn rank_key(score: f32, idx: u32) -> u64 {
    // -0.0 and +0.0 have DIFFERENT bit patterns but compare Equal in IEEE, so
    // the bit map alone would order them while `partial_cmp` would fall
    // through to the index. Canonicalise first. (`acc` here is a sum of
    // `fmaxf(dot, 0.0f)` scaled by a positive, so -0.0 should be unreachable --
    // but a differential check found this as the ONLY disagreement with the
    // comparator over 500 random values, and "should be unreachable" is a bad
    // reason to leave a selection subtly wrong.)
    // NaN ranks LAST, as in `qsa_decode_select::rank_cmp` (main's total order,
    // shared with the device arm). Raw bits would put a positive NaN FIRST.
    let score = if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    };
    let b = if score == 0.0 { 0 } else { score.to_bits() };
    let mono = if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    };
    ((!mono) as u64) << 32 | idx as u64
}

impl QsaIndexer {
    /// Host arm of the prefill top-k: D2H the slab's block scores, select
    /// per row on all cores, H2D the lists. Moved verbatim out of
    /// `prefill_select` for the 500-LoC cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_host_select(
        &self,
        gpu: &dyn GpuBackend,
        scores: DevicePtr,
        lists: DevicePtr,
        rows: usize,
        stride: usize,
        first_pos: usize,
        ratio: usize,
        topk: usize,
        stream: u64,
    ) -> Result<()> {
        // Host top-k per row (the D2H drains the stream first). Torch
        // tie-break: larger score first, lower index on ties.
        //
        // THIS LOOP IS DEAD GPU TIME, and no kernel-time profile can see
        // it because no kernel is running. nsys on a 2769-token prefill
        // measured the gap `qsa_score_rows -> qsa_prefill_attn` at
        // **6.17 ms x 24 launches = 148 ms**, 6.1% of the window. It is not
        // the transfer (488 KB); it is the sorting, and it scales like
        // `layers x slabs x rows x complete log complete`. At 32k that is
        // 12 layers x 15 slabs x 2048 rows x ~2000-element sorts -- of the
        // order of 1e10 comparisons on ONE core.
        //
        // Two changes, neither of which moves a single output bit:
        //
        //  * Only the first `topk` of the ordering is ever read, so
        //    `select_nth_unstable` (O(n)) partitions and then only the
        //    prefix is sorted. The order is TOTAL -- ties on score are
        //    broken by index, so no two distinct elements compare Equal --
        //    which means the partition point is unique and the prefix is
        //    exactly what the full sort produced, in the same order.
        //  * The comparison is a plain integer compare on a packed key
        //    (`rank_key`), not a closure that indexes back into the score
        //    matrix twice per comparison. That second indirection was the
        //    cost: nsys at 30k measured this gap at **49.6 ms**, still the
        //    largest single item in a 43 s prefill, on ~20k comparisons per
        //    row x 2048 rows.
        //  * The rows are independent. `std::thread::scope` (already used
        //    in this crate's mistral loader; no new dependency) fans them
        //    over the cores. Each thread writes a disjoint slice of
        //    `host_lists`, so the output is byte-identical regardless of
        //    how the rows are split or in what order the threads finish.
        //
        // ORDER MATTERS, so this must stay an exact reproduction rather
        // than any top-k that returns the same SET: `qsa_prefill_attn`
        // walks the list warp-striped (`t = warp; t < n_tok; t += 8`) and
        // its online softmax accumulates in that order. Permuting the list
        // reassociates the sum.
        // Receive straight into an f32 buffer. Landing in a `Vec<u8>` and
        // then running `chunks_exact(4).map(from_le_bytes).collect()`
        // walked every score a second time and allocated the matrix twice.
        // That matrix is `rows x stride`: at 30k context it is 2048 x 7500
        // = 15.4M floats PER SLAB, and a prefill runs ~14 slabs x 12
        // layers of them, so the conversion pass alone was seconds.
        //
        // The reinterpretation is sound in the direction used: `u8` has no
        // alignment requirement and the f32 allocation is already
        // 4-aligned, so the D2H writes exactly the same bytes to exactly
        // the same place and there is nothing left to convert. Both sides
        // are little-endian, which the original `from_le_bytes` also
        // assumed; the assertion below turns that into a build error
        // rather than silent garbage if this is ever cross-compiled.
        const _: () = assert!(
            cfg!(target_endian = "little"),
            "QSA score D2H reinterprets device f32 bytes in host order"
        );
        let mut sc = vec![0f32; rows * stride];
        {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    sc.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(sc.as_slice()),
                )
            };
            gpu.copy_d2h_on_stream(scores, bytes, stream)?;
        }
        let mut host_lists = vec![0u8; rows * topk * 4];
        // One row is ~2000 comparisons of work; below a few dozen rows the
        // spawn cost dominates, and a prefill issues thousands of these.
        let threads = if rows >= 64 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(rows / 32)
                .max(1)
        } else {
            1
        };
        let rows_per = rows.div_ceil(threads);
        let sc_ref = &sc;
        std::thread::scope(|scope| {
            for (ti, out) in host_lists.chunks_mut(rows_per * topk * 4).enumerate() {
                let r0 = ti * rows_per;
                scope.spawn(move || {
                    let mut keys: Vec<u64> = Vec::with_capacity(stride);
                    for (rl, orow) in out.chunks_mut(topk * 4).enumerate() {
                        let r = r0 + rl;
                        let complete = (first_pos + r + 1) / ratio;
                        let row_sc = &sc_ref[r * stride..r * stride + complete];
                        keys.clear();
                        keys.extend(
                            row_sc
                                .iter()
                                .enumerate()
                                .map(|(i, &s)| rank_key(s, i as u32)),
                        );
                        if complete > topk {
                            keys.select_nth_unstable(topk - 1);
                        }
                        keys[..topk].sort_unstable();
                        for (i, k) in keys[..topk].iter().enumerate() {
                            orow[i * 4..i * 4 + 4]
                                .copy_from_slice(&((*k as u32) as i32).to_le_bytes());
                        }
                    }
                });
            }
        });
        gpu.copy_h2d_async(&host_lists, lists, stream)?;
        Ok(())
    }

    /// Attention stage of `prefill_select` for one slab: every arm reads the
    /// same per-row lists. Moved verbatim for the 500-LoC cap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attend_slab(
        &self,
        gpu: &dyn GpuBackend,
        q_roped: DevicePtr,
        attn_ctx: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        lists: DevicePtr,
        first_row: usize,
        q_row: usize,
        rows: usize,
        first_pos: usize,
        topk: usize,
        block_size: u32,
        nq: u32,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<()> {
        // Every q head of a row attends over the SAME selected set --
        // the list is indexed by row, not by head -- so one block can
        // serve a whole group of heads and read each K/V row once instead
        // of once per head. nsys on an 11k prefill: the per-head kernel
        // was 3.70 s, 32.6% of a 12.3 s window, moving ~51 GB of L2
        // traffic per launch. Same accumulation order, so same bits; the
        // one-head kernel stays for geometries the group cannot divide.
        // `ATLAS_QSA_ATTN_L8` selects the 8-lanes-per-head reduction. It
        // is NOT bit-identical (the dot-product tree changes), so it is
        // opt-in and gated on `scripts/ppl.py`; see TTFT_GAP.md 22.
        // `ATLAS_QSA_ATTN_TC` selects the tensor-core attention. Same
        // selected set and same per-row semantics; NOT bit-identical (a
        // different summation tree), so it is opt-in and gated on
        // `scripts/ppl.py`. See TTFT_GAP.md 27.
        // `ATLAS_QSA_ATTN_TC2`: the same tensor-core attention with BOTH
        // kv heads in one CTA, which halves the M padding.
        // `ATLAS_QSA_ATTN_TC3`: tc2's kv-head packing with tc1's BC=32 tile.
        let tc3 = matches!(
            std::env::var("ATLAS_QSA_ATTN_TC3").as_deref(),
            Ok("1") | Ok("true")
        ) && self.k_prefill_attn_tc3_k.0 != 0
            && ops::qsa_prefill_attn_tc3_ok(nq, self.nkv_attn, self.hd_attn);
        if tc3 {
            ops::qsa_prefill_attn_tc3(
                gpu,
                self.k_prefill_attn_tc3_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        }
        let tc2 =
            !tc3 && matches!(
                std::env::var("ATLAS_QSA_ATTN_TC2").as_deref(),
                Ok("1") | Ok("true")
            ) && self.k_prefill_attn_tc2_k.0 != 0
                && ops::qsa_prefill_attn_tc2_ok(nq, self.nkv_attn, self.hd_attn);
        if tc2 {
            ops::qsa_prefill_attn_tc2(
                gpu,
                self.k_prefill_attn_tc2_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        }
        let tc =
            !tc2 && matches!(
                std::env::var("ATLAS_QSA_ATTN_TC").as_deref(),
                Ok("1") | Ok("true")
            ) && self.k_prefill_attn_tc_k.0 != 0
                && ops::qsa_prefill_attn_tc_ok(nq, self.nkv_attn, self.hd_attn);
        if tc {
            ops::qsa_prefill_attn_tc(
                gpu,
                self.k_prefill_attn_tc_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        }
        let l8 =
            !tc && matches!(
                std::env::var("ATLAS_QSA_ATTN_L8").as_deref(),
                Ok("1") | Ok("true")
            ) && ops::qsa_prefill_attn_l8_ok(nq, self.nkv_attn, self.hd_attn);
        if tc3 || tc2 || tc {
            // already dispatched above
        } else if l8 {
            ops::qsa_prefill_attn_l8(
                gpu,
                self.k_prefill_attn_l8_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        } else if ops::qsa_prefill_attn_grouped_ok(nq, self.nkv_attn, self.hd_attn) {
            ops::qsa_prefill_attn_g(
                gpu,
                self.k_prefill_attn_g_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        } else {
            ops::qsa_prefill_attn(
                gpu,
                self.k_prefill_attn_k,
                q_roped.offset(first_row * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_row * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod rank_key_tests {
    use super::rank_key;

    /// `rank_key` ascending must reproduce `qsa_decode_select::rank_cmp`
    /// exactly -- the selected set AND its order, because `qsa_prefill_attn`
    /// accumulates its online softmax in list order.
    #[test]
    fn matches_the_comparator_it_replaces() {
        let mut vals: Vec<f32> = vec![
            -1e30,
            -5.0,
            -1.0,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            0.5,
            1.0,
            3.25,
            1e30,
        ];
        // Deterministic spread, including repeats so ties are exercised.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..500 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            vals.push(((s >> 40) as f32 / 1024.0) - 8.0);
        }
        // NaN (both signs) and the infinities: the reference is main's total
        // order, which ranks NaN LAST, so the packed key must too.
        vals.extend([f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0]);
        let n = vals.len();

        let mut want: Vec<u32> = (0..n as u32).collect();
        want.sort_by(|&a, &b| super::super::qsa_decode_select::rank_cmp(&vals, a, b));

        let mut keys: Vec<u64> = (0..n).map(|i| rank_key(vals[i], i as u32)).collect();
        keys.sort_unstable();
        let got: Vec<u32> = keys.iter().map(|k| *k as u32).collect();

        assert_eq!(want, got, "packed key disagrees with the f32 comparator");
    }
}
