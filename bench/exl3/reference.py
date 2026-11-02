"""Independent NumPy reference for EXL3 (ExLlamaV3) trellis weights, mul1 codebook.

Transcribed from exllamav3 (MIT, turboderp-org/exllamav3 @ 6b84a21):
  * state extraction  -- exllamav3_ext/cpu/moe_mul1.cpp `decode_state_scalar`
  * mul1 decode       -- exllamav3_ext/quant/codebook.cuh `decode_3inst<2>` (fp16 fma)
  * tile permutation  -- exllamav3_ext/cpu/moe_mul1.cpp `make_tc_perm` / `make_tc_perm_inv`
  * weight assembly   -- exllamav3/modules/quant/exl3.py `get_weight_tensor`

Slow and explicit on purpose: this is the oracle the Rust port is tested against,
not a serving path.
"""

import numpy as np

MUL1_MULT = 0x83DCD12D
K_INV = np.frombuffer(np.uint16(0x1EEE).tobytes(), np.float16)[0]
K_BIAS = np.frombuffer(np.uint16(0xC931).tobytes(), np.float16)[0]


def mul1_decode(states):
    """fp16 value of each uint16 trellis state, bit-exact with `decode_3inst<2>`.

    The GPU forms h = 1024 + bytesum(state * M) as an exact fp16 (0x6400 + sum) and
    computes one `__hfma(h, k_inv, k_bias)`: the product and sum are exact in f64
    for these magnitudes, so rounding once to fp16 reproduces the fused op.
    """
    s = np.asarray(states, dtype=np.uint64)
    x = (s * MUL1_MULT) & 0xFFFFFFFF
    bsum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    h = (1024 + bsum).astype(np.float64)
    return (h * np.float64(K_INV) + np.float64(K_BIAS)).astype(np.float16)


def tc_perm():
    """Position within the 16x16 tile (row*16+col) of the t-th decoded state."""
    p = np.empty(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = t // 4
        for j, (r, c) in enumerate([(r, c0) for r in rows] + [(r, c0 + 8) for r in rows]):
            p[t * 8 + j] = r * 16 + c
    return p


def tc_perm_inv():
    inv = np.empty(256, dtype=np.int64)
    inv[tc_perm()] = np.arange(256)
    return inv


def tile_states(packed, bits):
    """uint16 states [..., 256] of packed tiles [..., 16*bits] (int16 or uint16)."""
    w16 = np.ascontiguousarray(packed).view(np.uint16).astype(np.uint64)
    if w16.shape[-1] != 16 * bits:
        raise ValueError(f"tile has {w16.shape[-1]} words, expected {16 * bits}")
    words32 = bits * 256 // 32
    # load_u32_: little-endian pair of u16 words
    w32 = w16[..., 0::2] | (w16[..., 1::2] << 16)
    t = np.arange(256)
    b0 = t * bits + bits - 16 + 256 * bits
    b1 = b0 + 16
    shift = ((b1 - 1) // 32 + 1) * 32 - b1
    hi = w32[..., (b0 // 32) % words32]
    lo = w32[..., ((b1 - 1) // 32) % words32]
    merged = (hi << np.uint64(32)) | lo
    return ((merged >> shift.astype(np.uint64)) & 0xFFFF).astype(np.uint16)


def inner_weight(trellis):
    """Decoded W_inner, fp16 [in, out], from trellis [in/16, out/16, 16*bits]."""
    ti, tn, words = trellis.shape
    if words % 16:
        raise ValueError("trellis last dim must be 16*bits")
    bits = words // 16
    vals = mul1_decode(tile_states(trellis, bits))  # [ti, tn, 256] in state order
    tiles = vals[..., tc_perm_inv()].reshape(ti, tn, 16, 16)
    return tiles.transpose(0, 2, 1, 3).reshape(ti * 16, tn * 16)


def hadamard128():
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(128.0)


def full_weight(trellis, suh, svh):
    """W = diag(suh) . H128 . W_inner . H128 . diag(svh), f64 [in, out]."""
    w = inner_weight(trellis).astype(np.float64)
    k, n = w.shape
    h = hadamard128()
    w = np.einsum("ij,bjn->bin", h, w.reshape(k // 128, 128, n)).reshape(k, n)
    w *= np.asarray(suh, np.float64)[:, None]
    w = (w.reshape(k, n // 128, 128) @ h).reshape(k, n)
    w *= np.asarray(svh, np.float64)[None, :]
    return w


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
