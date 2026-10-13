# `glm5_next` — GLM-5.3-Flash on GB10: what is in Atlas, and what is not

Status as of 2026-09-28: **port in progress; not yet in-tree.** Tracking issue:
#59. Target checkpoint:
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
This file lands with the first GLM PR and is updated in place as each PR in the
series merges. Every number here is either measured on the tree it names or is
marked as pending.

## The headline constraint: two Sparks minimum

GLM-5.3-Flash has 320B total parameters, of which 18B are active per token. At NVFP4, the weights take about 180 GB. A single GB10 has about 119 GB of unified memory. The minimum topology is therefore EP2 across two Sparks, as with the DeepSeek-V4-Flash EP2 recipe.

## Architecture

| Piece | Shape | Atlas building block |
|---|---|---|
| Layers | 45: 34 linear-attention (KDA) + 11 full-attention (MLA) | hybrid SSM/attention runtime (ADR-0003) |
| Linear attention | Kimi Delta Attention, 64 heads x 128, short conv 4, FP32 recurrent state | SSM state pool; new KDA kernels |
| Full attention | MLA, `kv_lora_rank` 512, sparse top-2048 selection from a semantic indexer above 2048 tokens; exact dense MLA below | MLA runtime shared with DeepSeek-V4-Flash |
| Residual | mHC hyper-connections, `hc_mult` 4, 20 Sinkhorn iterations | DeepSeek-V4-Flash mHC kernels |
| MoE | 288 routed experts, top-8, sigmoid `noaux_tc` routing, one shared expert; the first `first_k_dense_replace` layers are dense | NVFP4 grouped MoE |
| Vocab | 154,880 | |
| MTP | extra decoder layers appended to the checkpoint | follow-up job |
| Vision | vision tower present in the checkpoint | follow-up; image input refused at pre-flight until then |

Tool calls use the existing `poolside_v1` parser.

## Series plan

| PR | Scope | Status |
|---|---|---|
| 1a | parser, NVIDIA ModelOpt loader, KDA, MLA up to 2048 tokens, EP2, kernel tree `kernels/gb10/glm-5.3-flash/` | in preparation |
| 1b | semantic indexer (long context), `BENCH.toml`, `bfcl-subset` (ST-995) | pending |
| 2 | prefill levers, one A/B each | pending |
| later | decode concurrency, fp8 latent KV, MTP, DFlash2 on the upstream DFlash2 lane, vision | follow-up jobs |

## Provenance and credits

**Authors**
- **Reiner Schmidt ([@Mango-kid](https://github.com/Mango-kid))** wrote the original Atlas port: [`Mango-kid/atlas` `feat/glm53-dual-spark`](https://github.com/Mango-kid/atlas/tree/feat/glm53-dual-spark) @ `90b3584a`, AGPL-3.0-only. It includes the `glm5_next` parser, weight loader, KDA layers and kernels, the GLM indexer, and native MTP.
- **Jason McCartney ([@data-angel](https://github.com/data-angel), Enntity)** extended it in [`Enntity/atlas` `sparkglm/installable-20260924`](https://github.com/Enntity/atlas/tree/sparkglm/installable-20260924), AGPL-3.0-only. The work covers the NVIDIA ModelOpt dense-FFN loader, sparse-MLA decode and prefill kernels, MoE prefill, fp8 latent KV, and DFlash2 integration. Research notes: [`Enntity/sparkglm` `research/atlas`](https://github.com/Enntity/sparkglm/tree/main/research/atlas).
- **AI assistance.** Both branches were written largely by AI coding agents under their authors' direction: Claude and Codex for the Enntity work, and agent-written work in Reiner's branch. The PRs in this series are ported by Claude. Commits that carry Reiner's code credit him with `Co-authored-by`.

**Reference implementations and papers** — ideas and numerics only; no source copied:
- vLLM (Apache-2.0): GLM-5.3 model code, used as the design and parity reference for KDA token order and mHC storage precision. [revision: TBD]
- HuggingFace model card and `config.json` for `nvidia/GLM-5.3-Flash-NVFP4`: tensor layout and quantization metadata.
- Kimi Linear / Kimi Delta Attention (Moonshot AI) [arXiv id: TBD].
- DeepSeek-V2, Multi-head Latent Attention [arXiv id: TBD].
- DeepSeek-V3.2, DeepSeek Sparse Attention (lightning indexer) [arXiv id: TBD].
- Hyper-Connections and Manifold-Constrained Hyper-Connections (mHC) [arXiv ids: TBD].

**Chat template.** Supplied by the checkpoint, under the GLM-5.3 License from Z.ai. [Decision pending on #59: override or not.]

**Deliberately not carried** from the source branches:
- Out-of-tree `.so` bridges: FlashKDA (MIT) and an NVIDIA sparse-MLA object whose source revision is unresolved.
- The paired-process C2 serving guard, development scripts, and experiment logs.
