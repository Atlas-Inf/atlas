# SPDX-License-Identifier: AGPL-3.0-only
"""FP8 parity evaluator for the native GB10 port."""
import torch
import triton
import triton.language as tl


@triton.jit
def _block_mm(X, SX, W, SW, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 128)
    acc = tl.full((16, 32), 0, tl.float32)
    for g in range(K // 128):
        x = tl.load(X + rows[:, None]*K + g*128 + inner[None, :],
                    mask=rows[:, None] < M, other=0.0)
        w = tl.load(W + cols[:, None]*K + g*128 + inner[None, :], mask=cols[:, None] < N, other=0.0)
        xs = tl.load(SX + rows*(K//128) + g, mask=rows < M, other=0.0)
        ws = tl.load(SW + (cols//128)*(K//128) + g, mask=cols < N, other=0.0).to(tl.float32)
        part = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        acc += part * (xs[:, None] * ws[None, :])
    tl.store(Y + rows[:, None]*N + cols[None, :], acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


def project(x, scale, w, weight_scales):
    m, n, k = x.shape[0], w.shape[0], w.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _block_mm[(triton.cdiv(m,16), triton.cdiv(n, 32))](x.contiguous(), scale.contiguous(), w.contiguous(),
                                         weight_scales.contiguous(), out, m, n, k)
    return out
