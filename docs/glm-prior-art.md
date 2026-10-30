# GLM-5.3-Flash (SparkGLM) prior art and third-party material

What the GLM-5.3-Flash work on this fork took from other projects, and
where. `CITATIONS.md` at the repository root belongs to the separate
TurboQuant+ branch and is not about this work.

Three kinds of debt, kept apart:

- **Code or data copied or adapted**: kept only under a license compatible
  with AGPL-3.0-only, with the origin, copyright and license notice in the
  file and below.
- **Ideas or designs taken** from a named project: credited here and in the
  module that implements them, separately from copied source material.
- **Related work** we did not derive from: listed so readers can compare.

Credits are retained in the code and this file independently of commit-message
normalization in the local review history.

The October 12 [native RC4 source note](porting/GLM_5_3_FLASH_RC4.md) and
[import history](porting/glm53-validation/rc4-import-map.json) identify Jason
McCartney / Enntity's subsequent engine contributions. These complement the
original contributor and license records below.

## Code and data from other projects

| Material in this tree | Origin | License |
|---|---|---|
| `EDGES` and `PRIOR` in `crates/spark-server/src/scheduler/dflash_conf_width.rs` | knapcio, [`overlay/glm_bav_table_seg.json`](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/blob/d80f4fd/overlay/glm_bav_table_seg.json) @ d80f4fd: `EDGES` is its `edges[7..]` exactly; `PRIOR` is a rounded reading of its `g[0][7..]` (table source "mtpshadow capture 1633") | MIT, Copyright (c) 2026 knapcio (notice below) |
| `third_party/nvidia_sparse_mla_sm120/` (17 device headers) | NVIDIA's sparse-MLA SM120 device source in [FlashInfer](https://github.com/flashinfer-ai/flashinfer) @ `8eccd0c1352165302840c0e19066bc42d36dbd7a`; no FlashInfer host dispatcher/package | BSD-3-Clause, Copyright 2026 NVIDIA CORPORATION & AFFILIATES; per-file notices, LICENSE and provenance.json retained |
| `crates/spark-runtime/cuda/glm_kda/vendor/` (three device headers) | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) @ `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | MIT; LICENSE, NOTICE and provenance.json retained. Shared-memory state conversion adapted to Atlas's canonical key-major layout |
| `crates/spark-runtime/cuda/glm_kda_fp32/` (`ATLAS_KDA_FLASH_LIBRARY`, off by default) | SparkGLM 0.2.0's KDA prefill: [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) @ `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` with Jason McCartney / Enntity's state-slot, TMA-swizzle and FP32-recurrence patches, his C bridge and the `Enntity/atlas` c87f2790 runtime bridge, all verbatim and compiled in-tree with SparkGLM's flags | MIT upstream AND AGPL-3.0-only patches and bridges; LICENSE, NOTICE and provenance.json (every hash) retained |
| Atlas KDA launcher and sparse prep/split/merge integration | Atlas-owned parameter, workspace, state and launch code; prior GLM integration and main/tail composition by Jason McCartney / Enntity and Reiner Schmidt / Mango-kid. KDA launch layouts follow the credited MIT device source | AGPL-3.0-only Atlas integration; mixed source identified as AGPL-3.0-only AND MIT. Original imported history/notices retained |

These kernels now compile into Atlas and use its own state/workspace/dispatch
contracts. The final image does not build/load an external prefill bridge or
require a SparkGLM recipe, FlashKDA Python API or FlashInfer host API. CUDA and
the engine CUTLASS pin remain standard build dependencies. Numerical GPU and
full-model qualification are separate from source integration; see
[`docs/porting/GLM_5_3_FLASH.md`](porting/GLM_5_3_FLASH.md).

### Historical comparison scaffolding (removed after qualification)

The native GPU tests no longer require the external bridge builders. Local
archive commit `3c991613462c79730e2936a7b0f97add40f74370` retains
`scripts/build_glm_prefill/`, `research/flash-kda-prefill/` and historical files
under `scripts/dev/glm_sparse_native/`. That commit and its rewritten counterpart
are not ancestors of the proposed PR branches: cloning this stack does not
provide those local archive objects.

The original SparkGLM [FlashKDA bridge](https://github.com/Enntity/sparkglm/blob/84310dc3bbdfb99526e1b17342859241b447606d/install/flash_kda/atlas_flash_kda_bridge.cu),
[state-slot patch](https://github.com/Enntity/sparkglm/blob/84310dc3bbdfb99526e1b17342859241b447606d/install/flash_kda/flash_kda_sm121_slots.patch)
and [native build recipe](https://github.com/Enntity/sparkglm/blob/84310dc3bbdfb99526e1b17342859241b447606d/install/build-native.sh)
remain accessible at their pinned upstream revision. These recover the credited
upstream inputs, not all later local comparison scaffolding. The inventory
below records the original paths and hashes; native builds need none of these
removed reference implementations.

| Retained record | Origin and purpose |
|---|---|
| [Sparse source map and NVIDIA notice](../third_party/nvidia_sparse_mla_sm120/history/README.md) | NVIDIA sparse-MLA sources and Jason McCartney / Enntity's earlier experiment; original maps and notices preserved verbatim |
| [FlashKDA bridge/patch inventory](../third_party/glm_prefill/provenance.json) | Jason McCartney / Enntity, [SparkGLM](https://github.com/Enntity/sparkglm) at `84310dc3bbdfb99526e1b17342859241b447606d`, with MoonshotAI MIT source; original paths, hashes and license texts retained |

The [native GPU tests and fixture converter](../scripts/dev/glm_sparse_native/README.md)
remain available. Neither those tests nor deployment require rebuilding an
external prefill library.

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
| DFlash2 selector edge weight (`ATLAS_DFLASH2_EDGE=0.6\|0.8`) | `DF2_SEL_EDGE` in `kernels/gb10/common/dflash2_candidate_selector.cu`, `kernels/gb10/glm-5.3-flash/nvfp4/glm_dflash2_selector_conf_e06.cu` / `_e08.cu`, `dflash_head/draft_conf.rs` | MiaAI-Lab's TensorFold recipe scores DFlash2 candidates as unary + 0.6 × edge (TensorFold `glm5_next/cuda/dflash2.py`, `EDGE`; Apache-2.0), where the DFlash2 blog, vLLM and SGLang use 1 | Compile-time twins of our confidence selector, chosen at startup; experimental, draft-only |
| Copy (prompt-lookup) drafts beside DFlash2 (`ATLAS_DFLASH_COPY_DRAFTS`) | `scheduler/copy_drafts.rs`, the copy bins in `scheduler/dflash_conf_width.rs` | MiaAI-Lab's TensorFold recipe ([MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) `patches/0007-glm-copy-drafts.patch`, `patches/0032-glm-code-copy-drafts.patch`, patches of [TensorFold](https://github.com/ashhart/TensorFold) by ashhart; Apache-2.0): an 8-token match including the pending token, the latest earlier occurrence with a full continuation, the copy in place of the drafter's block, the longer match for occurrences inside the reply and the narrow copy after a miss; the copy bins' priors are per-draft rates derived from its measured shares of copied drafts kept (74% of the prompt's, 23% of the reply's, five drafts a round). Prompt lookup itself is older (vLLM and llama.cpp prompt lookup, as `src/ngram.rs` credits) | The incremental per-request index, the continuation over the copy itself for short repeats, copies priced by their own calibrated bins (prompt and reply) in the confidence width, and a merge that never widens DFlash2's block |
| One-shot RDMA exchange (`ATLAS_RDMA_ONESHOT`, `ATLAS_RDMA_PAIR_CHAIN`) | `kernels/gb10/common/rdma_oneshot.cu`, `spark-comm/.../rdma_pair/oneshot.rs`, `atlas-rdma/src/rdma_shim.c` (`rs_post_write_flag`) | b12x RoCEnante ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) `b12x/comm/roce/`, `docs/rocenante.md`; RoCEnante by Jason Cook in [#295](https://github.com/local-inference-lab/b12x/pull/295), b12x maintained by Luke Alonso; Apache-2.0): parity slots, a proxy that WRITEs data then a seq flag on the same QP, poison on timeout. The graph-replay wedge hazard is [b12x#313](https://github.com/local-inference-lab/b12x/issues/313), which reached us through [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) `docs/SFXNZ-AUDIT.md`. The kernel's block structure (device-resident seq, block 0 polls host flags and republishes a device go word, last block out stores seq) follows mmastrac's arx one-shot all-reduce ([mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) `experimental/arx/arx_vllm.cu`, PR #4; that directory has no license, so idea only) | Size-checked flags (`seq << 24 \| bytes`), desync detection, host-side poison, copy-engine staging with the `stage` word handshake. The base pair channel (`rdma_pair`, 2026-09-26) was written before any note of ours mentions RoCEnante |
| PDL weight touch (`ATLAS_GLM_DECODE_GEMV_BATCH`) | `kernels/gb10/glm-5.3-flash/nvfp4/atlas_pdl_touch.cuh`, `layers/ops/gemv_touch.rs` | Same technique as TensorFold's L2 weight touch ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patches 0040 and 0440, the latter before `griddepcontrol.wait`; Apache-2.0) and knapcio's `GLM_L2_PREFETCH`; compare mmastrac's arx L2 prefetch during all-reduce waits. Our notes do not record which prompted it | Discarded byte loads from the kernel's own CTAs; measured against our own nsys wait gaps |
| Routed expert L2 row prefetch (`ATLAS_GLM_MOE_DECODE_L2PF`) | `kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_decode_stream.cuh` (`pqs_prefetch`, the `*_l2pf` twins), `layers/moe/decode_m16.rs` | jayleaton's L2 prefetch of the weights the next kernels read ([jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) patches/0460, Apache-2.0), carried in Mia's TensorFold recipe as patch `0046-glm-l2-prefetch` (Apache-2.0); TensorFold `0047`'s weight loads issued a step ahead (jayleaton patches/0580) | Inside the routed gate/up and down kernels after the PDL wait, not a side stream (theirs has no expert sites): each CTA asks for its share of every K stage's whole table rows a few stages ahead, so an expert's stages leave DRAM as contiguous runs instead of 128- and 256-byte strips |
| Decode seam on the BF16 mix weight, touched before the PDL wait (`ATLAS_GLM_HC_SEAM_ILP`) | `glm_hc_decode_{post_,}partial_ilp_bf16` in `kernels/gb10/glm-5.3-flash/nvfp4/glm_hc_prefill_vec.cu`, `layers/ops/glm_decode_fuse.rs` | TensorFold's `hc_partial` reads the hyper-connection `fn` as BF16 (Mia's TensorFold recipe patch `0019-glm-decode-kernels2`, Apache-2.0), and its L2 prefetch (patch `0046-glm-l2-prefetch`, after jayleaton's patches/0460, Apache-2.0) takes the next hyper-connection weights at the all-gather | Same FP32 products and reduction trees as our kernels (the BF16 tensor is the checkpoint's own, which the FP32 copy widens exactly), so outputs keep their bits; the touch is our PDL weight touch from the kernel's own CTAs; the butterfly-equivalent reduce-scatter and the Sinkhorn beside the collapse are ours |
| Prefill indexer row split (`ATLAS_GLM_INDEX_SPLIT`) | `qwen3_attention/prefill/glm_index_split*.rs`, `scripts/dev/glm_index_split_bench.cu` | RiNGSiDE's `GLM53_INDEXER_ROW_SPLIT` (othexmr, [GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE) `src/tp4/glm53_indexer_rowsplit.py`; Apache-2.0), itself after rhys101's SG18 prefill TP split and knapcio's DeepSeek-V4.1 TP4 adaptation | Zigzag quarters for equal causal work, the swap over the RDMA pair, the row-subset proof |
| Drafter split across ranks (`ATLAS_GLM_DRAFT_TP`, batched proposes with `ATLAS_GLM_DRAFT_TP_BATCH`, context appends with `ATLAS_GLM_DRAFT_TP_CTX`) | `dflash_head/rank_split.rs`, `rank_split_batch.rs`, `rank_split_ctx.rs` | MiaAI-Lab's `DFLASH_DRAFT_TP=2` ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) `.env.example`, `start.sh`; vLLM `draft_tensor_parallel_size`, which also shards the drafter's `fc` and K/V projections) | Output-row share on 16-row CTA boundaries, swapped over the pair, bit-identical drafts; the batched B×gamma split and the context rows announced with the propose are ours |
| KDA fold-record rollback (`--ssm-rollback-mode records`) | `model/ssm_pool/kda_records.rs`, `kda_recurrent_bf16_verify_rec_owners` / `kda_commit_records` in `kernels/gb10/glm-5.3-flash/nvfp4/kda.cu` | vLLM RecoverSSM ([vllm#51855](https://github.com/vllm-project/vllm/pull/51855), ZJY0516 with benchislett), from ReplaySSM ([vllm#48018](https://github.com/vllm-project/vllm/pull/48018), Johnny-Liou, Dao AI Lab, NVIDIA), as RiNGSiDE ships it for GLM-5.3 (`--use-replayssm`); Apache-2.0 | CUDA kernels on our snapshot-verify template, our record layout (decays, normalized keys, corrections) and one fold rounding order shared by verify and commit |
| SSM snapshot chain eviction | `spark-runtime/src/radix_tree/snapshot_chain.rs` | Reederey87's prefix-cache overlays (Artem Matskevych, [glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) `overlay/patch_apc_tail_boundary.py`, `cache_tail_evict.py`, `patch_cache_hot_protect.py`; Apache-2.0); branch points from Marconi (MLSys'25, arXiv:2411.19379) | The chain / superseded / one-protected-slot-per-session policy |
| Speculative decoding with strict structured output (`ATLAS_GLM_STRICT_SPEC`) | `scheduler/strict_spec.rs`, `model/glm_verify_masks.rs`, `argmax_bf16_value_ban_allow` in `kernels/gb10/glm-5.3-flash/nvfp4/argmax_bf16.cu`, `scripts/dev/glm_strict_split_argmax_bitcheck.cu` | vLLM [#14702](https://github.com/vllm-project/vllm/pull/14702) ("Enable Speculative Decoding with Structured Outputs": walk the matcher through the drafts, one bitmask per verify row plus the bonus row, roll back, trim at the first rejected draft), [#44297](https://github.com/vllm-project/vllm/pull/44297) ("Constrain bitmask and trim grammar advance at the reasoning boundary": rows before `</think>` unmasked, rows after it from the grammar's start state, the matcher never fed reasoning), tracked by RFC [#48197](https://github.com/vllm-project/vllm/issues/48197); Apache-2.0 | The TP2 vocab-split masked argmax: the masks broadcast to both ranks and each rank's slice applied before its partial argmax and the existing pair merge; the dead first draft that keeps a verify at width 2 |
| GB10 display carveout as KV memory (`spark display-carveout`; originally SparkGLM-only, now an optional native Atlas path) | `spark-server/src/display_carveout.rs`, `display_carveout/rm.rs`, `spark-runtime/src/gpu/carveout.rs`, `cuda_backend/display_carveout.rs`, `kv_cache/placement.rs`, `spark-model/src/factory/build/kv_carveout.rs` | kindling-spark-os's `dispram` (kindlingai: mmastrac, coffee-the-dev, adapt-ai-systems; [kindlingai/kindling-spark-os](https://github.com/kindlingai/kindling-spark-os) `dispram/`; AGPL-3.0 daemon, GPL-3.0 client): the 2 GiB `DISPLAY_FRM` carveout that GB10's driver never allocates from, wrapped in an `NV01_MEMORY_LIST_SYSTEM` object, exported as an fd and imported with `cuMemImportFromShareableHandle`, glued behind ordinary memory for vLLM's KV tail. They credit an earlier NVIDIA developer forum post by emihuang | Re-derived from open-gpu-kernel-modules 580.173.02/580.178.04 (why the heap is unused, why exports outlive the exporter); no daemon: a launcher exports, takes a host flock, drops CAP_SYS_ADMIN and execs the server; whole per-layer KV pools placed so system-memory KV bytes never grow through the ranks' block agreement |

## Related work, not derived from

- Token-sharded MLA latents (`docs/glm-kv-shard.md`): vLLM decode context
  parallelism (`decode_context_parallel_size`, Apache-2.0) also shards KV
  by token. The design came from our own 512K x 4 pool target.
- NVMe prefix cache (`docs/glm-nvme-prefix-cache.md`): MiaAI-Lab
  [PR #232](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/232)
  (`overlay/kvoffload/nvme_direct.py`) offloads vLLM prefix blocks to NVMe,
  one file per chunk. Ours spills radix evictions into a reserved O_DIRECT
  record file.
