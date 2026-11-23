// SPDX-License-Identifier: AGPL-3.0-only

// Ling MLA stores the absorbed K and padded V rows at qk_head_dim=192.  The
// shared paged-decode implementation is parameterised by HDIM at compile
// time; compiling this translation unit separately keeps the ordinary 256-wide
// module intact while making every lane's vector and shared-memory geometry
// exact for Ling.
#define HDIM 192
#define paged_decode_attn paged_decode_attn_ling
#include "../../common/paged_decode_attn.cu"
