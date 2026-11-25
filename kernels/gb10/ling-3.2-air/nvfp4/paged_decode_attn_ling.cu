// SPDX-License-Identifier: AGPL-3.0-only

// Keep the logical attention width while reading the shared RoPE tail once per token.
#include "ling_cache_layout.cuh"
#define HDIM 192
#define paged_decode_attn paged_decode_attn_ling
#include "../../common/paged_decode_attn.cu"
