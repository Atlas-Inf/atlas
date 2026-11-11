# FP4-Local reference fixtures

The following fixture data came through Jason McCartney / Enntity's Atlas
FP4-Local integration. They are checkpoint excerpts and reference results,
not Atlas-authored model weights. The repository's source-code license does
not replace their upstream terms.

| Files | Original source |
| --- | --- |
| `nvfp4_csf32/l3_*`, `nvfp4_csf32/l44_*`, GLM FP4-Local config and header extracts | [NVIDIA GLM-5.3-Flash-FP4-Local at `76348ddb`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5) |
| `nvfp4_csf/`, `mxfp8/`, Qwen FP4-Local config extract | [NVIDIA Qwen3.8-Flash-Next-FP4-Local at `b9ce18a6`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300) |
| `glm5_nvfp4_headers.json` | [NVIDIA GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |

The FP4-Local checkpoints name the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
in their model cards. Refer to the pinned cards for the complete checkpoint
and base-model terms.

Each `golden.json` records the source revision, tensor names, geometry and
reference output hashes. `nvfp4_csf32/syn_160x133.*` is synthetic, generated
by the documented formula. The generator scripts describe their independent
reference implementations: vLLM and FlashInfer compressed-scale decoders
(Apache-2.0), and PyTorch MXFP8 expansion. The CSF generator scripts embed adapted Apache-2.0 reference functions; their
original license and notices are retained in
[`third_party/csf_references`](../third_party/csf_references/). The surrounding
Atlas generator code remains AGPL-3.0-only. These source-code licenses do not
relicense the checkpoint excerpts listed above.

Exact Enntity source commits and adaptations are recorded in
[`docs/porting/glm53-fp4local-sources.json`](../docs/porting/glm53-fp4local-sources.json).
