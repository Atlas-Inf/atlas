<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3-Flash FP4-Local checkpoint support

Atlas loads NVIDIA's `nvidia/GLM-5.3-Flash-FP4-Local` checkpoint through its
native GLM implementation. The existing `nvidia/GLM-5.3-Flash-NVFP4` checkpoint
remains supported. Both use the `glm-5.3-flash/nvfp4` kernel target.

This update adapts Jason McCartney / Enntity's FP4-Local work from
[Enntity/atlas at `b3b016ab`](https://github.com/Enntity/atlas/tree/b3b016ab7e91b73bea220b7280c9e9aa1a2cfb13).
The [source ledger](glm53-fp4local-sources.json) identifies the individual
commits and integration changes. NVIDIA supplied the quantized checkpoint;
the compressed-scale format references and independent test oracles are
identified in the decoder source and fixture manifests.

## Loading and numerical boundaries

- Compressed `csf_32x2_v1` expert scales decode losslessly into the E4M3 scales
  used by existing NVFP4 kernels. Expert tensor parallelism decodes deferred
  components when loading each rank's slice.
- MXFP8 projections expand exactly to BF16 within the checked exponent range.
  FP8 block projections expand with BF16 rounding. The ordinary GLM path then
  uses its existing projection formats, including runtime requantization.
- The split output head takes its MXFP8 rows directly from the checkpoint.
  Quantized vision projections expand through the native vision loader.
- Quantized source buffers are reclaimed after their last reader. Atlas main's
  packed allocations and checked ownership tracking are retained.

This is native Atlas execution, without an external prefill bridge or model
conversion service. It does not implement SparkGLM's optional
`ATLAS_GLM_FP4L_NATIVE` format-selection experiments.

## Deployment

Use the [two-Spark recipe](../../docker/gb10/glm-5.3-flash/nvfp4/README.md)
with the complete FP4-Local checkpoint as `MODEL_DIR`, including its nested
`tensors/` directory. The revision used for validation is
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`.

The target KV cache remains FP8 `fp8_g128`; FP4-Local names the weight
checkpoint, not a four-bit KV cache. Start with the recipe's 32K session and
aggregate limits. Larger pools require measurements on the actual hosts.
The checkpoint's 1,048,576-position metadata limit is not a measured capacity.
DFlash2 remains a separate, explicitly selected checkpoint.

## Validation

CPU coverage includes compressed-scale reference hashes and refusal cases,
cross-shard loading, expert TP slices, packed allocations, MXFP8 reference
values, source retirement and allocation-failure cleanup.

Ignored GPU tests compare real checkpoint projections, TP2 KDA weights,
shared experts, output-head slices and vision linears with host references.
They select the exact GLM kernel target and check finite outputs. Run them on
an otherwise idle Spark with `ATLAS_GLM_FP4L_CKPT` set to the checkpoint path.
Fresh serving, equivalence and performance results are pending for this update;
older validation records do not qualify these changes.

## Checkpoint and fixture terms

Atlas source retains its source licenses. The FP4-Local checkpoint is governed
by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
as stated in its [pinned model card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/README.md).
Small checkpoint-derived reference fixtures retain their upstream terms; see
[fixture provenance](../../test_data/FP4_LOCAL_NOTICE.md). Model weights and
DFlash2 weights are acquired separately and are not bundled in the runtime.
