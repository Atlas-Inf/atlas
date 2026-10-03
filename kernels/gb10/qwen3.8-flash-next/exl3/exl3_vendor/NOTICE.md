# Vendored ExLlamaV3 device code

The `.cuh` files in this directory are adapted from ExLlamaV3 by turboderp-org:

    https://github.com/turboderp-org/exllamav3  (commit 6b84a21)

    codebook.cuh        exllamav3_ext/quant/codebook.cuh (trellis codebooks, incl. mul1 decode_3inst<2>)
    exl3_dq.cuh         exllamav3_ext/quant/exl3_dq.cuh (trellis state extraction into MMA fragments)
    reconstruct_tile.cuh exllamav3_ext/quant/reconstruct.cu (reconstruct_tile, lines 12-85)
    hadamard_inner.cuh  exllamav3_ext/quant/hadamard_inner.cuh (128-wide warp Hadamard; adapted:
                        unused variants deleted, see the header's Modifications line)
    compat.cuh          exllamav3_ext/compat.cuh (verbatim; no longer included by the vendored headers)
    half4.cuh           exllamav3_ext/util.cuh lines 8-18 (the half4 struct only)
    ptx_frag.cuh        exllamav3_ext/ptx.cuh (fragment types, lines 1-16)
    half_uint16.cuh     exllamav3_ext/util.cuh (half_uint16 union, lines 83-91)

The files under `coopmk/` are adapted from the vcruz305/exllamav3 fork of ExLlamaV3 (same MIT license and
copyright; CoopMK commits 9aa5329 / 047ce72, coop kernels from turboderp 58d4d73):

    https://github.com/vcruz305/exllamav3  (commit 047ce72)

    coopmk/moe_coop.cuh          exllamav3_ext/quant/exl3_moe_coop.cuh (MoeCoopParams + defines only)
    coopmk/gemv_regs.cuh         exllamav3_ext/quant/exl3_gemv_kernel.cuh lines 32-155 (exl3_gemv_ns helpers)
    coopmk/moe_coop_kernel.cuh   exllamav3_ext/quant/exl3_moe_coop_kernel.cuh lines 32-629 (shared device code)
    coopmk/moe_coopmk_kernel.cuh exllamav3_ext/quant/exl3_moe_coopmk_kernel.cuh (coopmk_a/b bodies)

Each header lists its modifications (includes, __host__ __device__ on the constexpr helpers, the stage kernels
as __device__ bodies wrapped by extern "C" entry points in ../exl3_coopmk.cu).

Only device code is vendored; the torch/ATen host launchers are replaced by Atlas extern "C" entry points
(../exl3.cu) and Rust launchers. Each file keeps the AGPL SPDX line Atlas requires plus this MIT notice.
The rest of Atlas remains AGPL-3.0-only.

MIT License

Copyright (c) 2025 Turboderp

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