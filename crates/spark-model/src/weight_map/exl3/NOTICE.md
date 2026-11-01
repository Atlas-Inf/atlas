The Rust code in this directory is a transcription of the EXL3 trellis decode
published in ExLlamaV3 by turboderp-org:

    https://github.com/turboderp-org/exllamav3  (commit 6b84a21)

    exllamav3/exllamav3_ext/cpu/moe_mul1.cpp   state extraction, tile permutation
    exllamav3/exllamav3_ext/quant/codebook.cuh mul1 codebook (decode_3inst<2>)
    exllamav3/exllamav3/modules/quant/exl3.py  weight assembly (get_weight_tensor)

Rust files carrying that provenance:

    crates/spark-model/src/weight_map/exl3/mod.rs
    crates/spark-model/src/weight_map/exl3/cpu_ref.rs
    crates/spark-model/src/weight_map/exl3/tests.rs

so the MIT notice travels with it. The rest of Atlas remains AGPL-3.0-only.

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
