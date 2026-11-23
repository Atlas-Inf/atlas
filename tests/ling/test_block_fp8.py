# SPDX-License-Identifier: AGPL-3.0-only
import unittest
import torch
from fp8_weights import block_fp8


class StrictBlockFP8(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_dynamic_activation_and_per_tile_scales(self):
        g = torch.Generator().manual_seed(42)
        for m, n, k in [(1, 768, 2560), (17, 576, 2560), (4, 2560, 768)]:
            x = torch.randn(m, k, generator=g).bfloat16()
            w = torch.randn(n, k, generator=g).to(torch.float8_e4m3fn)
            s = torch.rand((n+127)//128, k//128, generator=g)
            expected = block_fp8(x, w, s)
            actual = block_fp8(x.cuda(), w.cuda(), s.cuda()).cpu()
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.02)


if __name__ == '__main__':
    unittest.main()
