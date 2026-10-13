# GLM-5.3-Flash (SparkGLM) prior art and third-party material

What the GLM-5.3-Flash work on this fork took from other projects, and
where. `CITATIONS.md` at the repository root belongs to the separate
TurboQuant+ branch and is not about this work.

Three kinds of debt, kept apart:

- **Code or data copied or adapted**: kept only under a license compatible
  with AGPL-3.0-only, with the origin, copyright and license notice in the
  file and below.
- **Ideas or designs taken** from a named project: no license obligation,
  but credited here and in the module that implements them.
- **Related work** we did not derive from: listed so readers can compare.

Commit messages cannot be amended, so these credits live in the code and
in this file.

## Code and data from other projects

| Material in this tree | Origin | License |
|---|---|---|
| `EDGES` and `PRIOR` in `crates/spark-server/src/scheduler/dflash_conf_width.rs` | knapcio, [`overlay/glm_bav_table_seg.json`](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/d80f4fd/overlay/glm_bav_table_seg.json) @ d80f4fd: `EDGES` is its `edges[7..]` exactly; `PRIOR` is a rounded reading of its `g[0][7..]` (table source "mtpshadow capture 1633") | MIT, Copyright (c) 2026 knapcio (notice below) |
| `scripts/dev/glm_sparse_native/reference/traits/model/model_type.h`; the restated declarations in `native-bridge.cpp` and `native-init.cpp` | NVIDIA's sparse-MLA SM120 prefill sources in [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (`include/flashinfer/attention/sparse_mla_sm120/`, `csrc/sparse_mla_sm120_prefill.cu`) | BSD-3-Clause, Copyright 2026 NVIDIA CORPORATION & AFFILIATES; notice in the header and `NATIVE-BRIDGE-NOTICE.txt` |
| FlashKDA, loaded at run time by `crates/spark-model/src/layers/glm5_kda/flash_prefill.rs` and `research/flash-kda-prefill/` (not in this tree) | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) @ 1ce47ea3 | MIT (see `research/flash-kda-prefill/README.md`) |

### knapcio MIT notice (for `EDGES` and `PRIOR`)

```text
MIT License

Copyright (c) 2026 knapcio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ideas and designs taken

No code from these projects is in this tree.

| Feature (switch) | Where | Prior art | Ours |
|---|---|---|---|
| DFlash verify width from drafter confidence (`ATLAS_DFLASH_CONF_WIDTH`) | `scheduler/dflash_conf_width.rs`, `dflash_head/draft_conf.rs`, `DF2_SEL_CONF` in `kernels/gb10/common/dflash2_candidate_selector.cu` | knapcio's draft-shape truncation `GLM_DRAFT_TRUNC` ([`overlay/glm_draft_trunc.py`](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/982e258/overlay/glm_draft_trunc.py) @ 982e258): log max softmax per draft, a position x confidence-bin acceptance table, expected tokens from survival products, one uniform width maximizing expected tokens less a per-row price | Online decayed calibration pooled over depths, fixed `TAU` price, periodic full-width probe |
| One-shot RDMA exchange (`ATLAS_RDMA_ONESHOT`, `ATLAS_RDMA_PAIR_CHAIN`) | `kernels/gb10/common/rdma_oneshot.cu`, `spark-comm/.../rdma_pair/oneshot.rs`, `atlas-rdma/src/rdma_shim.c` (`rs_post_write_flag`) | b12x RoCEnante ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) `b12x/comm/roce/`, `docs/rocenante.md`; RoCEnante by Jason Cook in [#295](https://github.com/local-inference-lab/b12x/pull/295), b12x maintained by Luke Alonso; Apache-2.0): parity slots, a proxy that WRITEs data then a seq flag on the same QP, poison on timeout. The graph-replay wedge hazard is [b12x#313](https://github.com/local-inference-lab/b12x/issues/313), which reached us through [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) `docs/SFXNZ-AUDIT.md`. The kernel's block structure (device-resident seq, block 0 polls host flags and republishes a device go word, last block out stores seq) follows mmastrac's arx one-shot all-reduce ([mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) `experimental/arx/arx_vllm.cu`, PR #4; that directory has no license, so idea only) | Size-checked flags (`seq << 24 \| bytes`), desync detection, host-side poison, copy-engine staging with the `stage` word handshake. The base pair channel (`rdma_pair`, 2026-09-26) was written before any note of ours mentions RoCEnante |
| PDL weight touch (`ATLAS_GLM_DECODE_GEMV_BATCH`) | `kernels/gb10/glm-5.3-flash/nvfp4/atlas_pdl_touch.cuh`, `layers/ops/gemv_touch.rs` | Same technique as TensorFold's L2 weight touch ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patches 0040 and 0440, the latter before `griddepcontrol.wait`; Apache-2.0) and knapcio's `GLM_L2_PREFETCH`; compare mmastrac's arx L2 prefetch during all-reduce waits. Our notes do not record which prompted it | Discarded byte loads from the kernel's own CTAs; measured against our own nsys wait gaps |
| Routed expert L2 row prefetch (`ATLAS_GLM_MOE_DECODE_L2PF`) | `kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_decode_stream.cuh` (`pqs_prefetch`, the `*_l2pf` twins), `layers/moe/decode_m16.rs` | jayleaton's L2 prefetch of the weights the next kernels read ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patches/0460, Apache-2.0), carried in Mia's TensorFold recipe as patch `0046-glm-l2-prefetch` (Apache-2.0); TensorFold `0047`'s weight loads issued a step ahead (jayleaton patches/0580) | Inside the routed gate/up and down kernels after the PDL wait, not a side stream (theirs has no expert sites): each CTA asks for its share of every K stage's whole table rows a few stages ahead, so an expert's stages leave DRAM as contiguous runs instead of 128- and 256-byte strips |
| Decode seam on the BF16 mix weight, touched before the PDL wait (`ATLAS_GLM_HC_SEAM_ILP`) | `glm_hc_decode_{post_,}partial_ilp_bf16` in `kernels/gb10/glm-5.3-flash/nvfp4/glm_hc_prefill_vec.cu`, `layers/ops/glm_decode_fuse.rs` | TensorFold's `hc_partial` reads the hyper-connection `fn` as BF16 (Mia's TensorFold recipe patch `0019-glm-decode-kernels2`, Apache-2.0), and its L2 prefetch (patch `0046-glm-l2-prefetch`, after jayleaton's patches/0460, Apache-2.0) takes the next hyper-connection weights at the all-gather | Same FP32 products and reduction trees as our kernels (the BF16 tensor is the checkpoint's own, which the FP32 copy widens exactly), so outputs keep their bits; the touch is our PDL weight touch from the kernel's own CTAs; the butterfly-equivalent reduce-scatter and the Sinkhorn beside the collapse are ours |
| Prefill indexer row split (`ATLAS_GLM_INDEX_SPLIT`) | `qwen3_attention/prefill/glm_index_split*.rs`, `scripts/dev/glm_index_split_bench.cu` | RiNGSiDE's `GLM53_INDEXER_ROW_SPLIT` (othexmr, [GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE) `src/tp4/glm53_indexer_rowsplit.py`; Apache-2.0), itself after rhys101's SG18 prefill TP split and knapcio's DeepSeek-V4.1 TP4 adaptation | Zigzag quarters for equal causal work, the swap over the RDMA pair, the row-subset proof |
| Drafter split across ranks (`ATLAS_GLM_DRAFT_TP`, batched proposes with `ATLAS_GLM_DRAFT_TP_BATCH`, context appends with `ATLAS_GLM_DRAFT_TP_CTX`) | `dflash_head/rank_split.rs`, `rank_split_batch.rs`, `rank_split_ctx.rs` | MiaAI-Lab's `DFLASH_DRAFT_TP=2` ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) `.env.example`, `start.sh`; vLLM `draft_tensor_parallel_size`, which also shards the drafter's `fc` and K/V projections) | Output-row share on 16-row CTA boundaries, swapped over the pair, bit-identical drafts; the batched B×gamma split and the context rows announced with the propose are ours |
| KDA fold-record rollback (`--ssm-rollback-mode records`) | `model/ssm_pool/kda_records.rs`, `kda_recurrent_bf16_verify_rec_owners` / `kda_commit_records` in `kernels/gb10/glm-5.3-flash/nvfp4/kda.cu` | vLLM RecoverSSM ([vllm#51855](https://github.com/vllm-project/vllm/pull/51855), ZJY0516 with benchislett), from ReplaySSM ([vllm#48018](https://github.com/vllm-project/vllm/pull/48018), Johnny-Liou, Dao AI Lab, NVIDIA), as RiNGSiDE ships it for GLM-5.3 (`--use-replayssm`); Apache-2.0 | CUDA kernels on our snapshot-verify template, our record layout (decays, normalized keys, corrections) and one fold rounding order shared by verify and commit |
| SSM snapshot chain eviction | `spark-runtime/src/radix_tree/snapshot_chain.rs` | Reederey87's prefix-cache overlays (Artem Matskevych, [glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) `overlay/patch_apc_tail_boundary.py`, `cache_tail_evict.py`, `patch_cache_hot_protect.py`; Apache-2.0); branch points from Marconi (MLSys'25, arXiv:2411.19379) | The chain / superseded / one-protected-slot-per-session policy |
| Speculative decoding with strict structured output (`ATLAS_GLM_STRICT_SPEC`) | `scheduler/strict_spec.rs`, `model/glm_verify_masks.rs`, `argmax_bf16_value_ban_allow` in `kernels/gb10/glm-5.3-flash/nvfp4/argmax_bf16.cu`, `scripts/dev/glm_strict_split_argmax_bitcheck.cu` | vLLM [#14702](https://github.com/vllm-project/vllm/pull/14702) ("Enable Speculative Decoding with Structured Outputs": walk the matcher through the drafts, one bitmask per verify row plus the bonus row, roll back, trim at the first rejected draft), [#44297](https://github.com/vllm-project/vllm/pull/44297) ("Constrain bitmask and trim grammar advance at the reasoning boundary": rows before `</think>` unmasked, rows after it from the grammar's start state, the matcher never fed reasoning), tracked by RFC [#48197](https://github.com/vllm-project/vllm/issues/48197); Apache-2.0 | The TP2 vocab-split masked argmax: the masks broadcast to both ranks and each rank's slice applied before its partial argmax and the existing pair merge; the dead first draft that keeps a verify at width 2 |

## Related work, not derived from

- Token-sharded MLA latents (`docs/glm-kv-shard.md`): vLLM decode context
  parallelism (`decode_context_parallel_size`, Apache-2.0) also shards KV
  by token. The design came from our own 512K x 4 pool target.
- NVMe prefix cache (`docs/glm-nvme-prefix-cache.md`): MiaAI-Lab
  [PR #232](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/232)
  (`overlay/kvoffload/nvme_direct.py`) offloads vLLM prefix blocks to NVMe,
  one file per chunk. Ours spills radix evictions into a reserved O_DIRECT
  record file.
