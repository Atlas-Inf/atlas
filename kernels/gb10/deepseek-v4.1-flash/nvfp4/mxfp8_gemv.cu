// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4.1-Flash: the glm-5.3-flash `mxfp8_gemv.cu` kernels verbatim,
// included (not copied) as qwen3.8-flash-next's are.
//
// The release stores every FP8 projection as E4M3 [n, k] with one UE8M0 scale
// per 32 x 32 block ([ceil(n/32), k/32]). The MXFP8 tiers read a scale per row
// and 32 columns ([n, k/32]), so the loader repeats each grid row for the 32
// weight rows it covers (`weight_map::fp8_block_scales_to_rows`) and decode
// reads the checkpoint's own values. E4M3 times a power of two is exact in
// BF16, so the tensor-core operands are the checkpoint's weights unrounded.
// V4's own FP8 kernels assume 128 x 128 blocks with f32 scales.
#include "../../glm-5.3-flash/nvfp4/mxfp8_gemv.cu"
