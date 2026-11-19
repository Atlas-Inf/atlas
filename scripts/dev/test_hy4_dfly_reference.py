# SPDX-License-Identifier: AGPL-3.0-only

import types
import unittest

import torch

from hy4_dfly_reference import BF, TorchOps, draft


class DFlyReferenceTests(unittest.TestCase):
    def test_invalid_count_and_topology(self):
        for count, world in [(0, 1), (-1, 1), (6, 1), (1, 0), (1, 3)]:
            with self.subTest(count=count, world=world), self.assertRaises(ValueError):
                draft(None, None, 0, None, [], [], count, world)

    def test_empty_cache_list_and_hidden_shape(self):
        checkpoint = types.SimpleNamespace(config={"text_config": {"hidden_size": 8}})
        for shape in [(1, 8), (0, 8), (8,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                draft(checkpoint, None, 0, torch.empty(shape), [], [], 1, 1)

    def test_invalid_bonus_and_anchor(self):
        checkpoint = types.SimpleNamespace(config={"text_config": {"hidden_size": 8, "vocab_size": 10}})
        for rows, bonus in [(1, 0), (2, -1), (2, 10)]:
            with self.subTest(rows=rows, bonus=bonus), self.assertRaises(ValueError):
                draft(checkpoint, None, bonus, torch.empty((1, 8)),
                      [torch.empty((rows, 576))] * 5, [None] * 5, 1, 1)

    def test_bf16_fusion_materializes_product(self):
        ops = TorchOps()
        anchor = torch.tensor([[1.0, -1.0]], dtype=BF)
        projected = torch.tensor([[1.0078125, 0.333984375]], dtype=BF)
        gate = torch.tensor([0.333984375, 0.69921875], dtype=BF)
        expected = anchor + (projected.float() * gate.float()).to(BF)
        self.assertTrue(torch.equal(ops.fuse(anchor, projected, gate), expected))

    def test_empty_elementwise_buffers(self):
        empty = torch.empty((0, 8), dtype=BF)
        ops = TorchOps()
        self.assertEqual(ops.add(empty, empty).shape, empty.shape)
        self.assertEqual(ops.silu(empty, empty).shape, empty.shape)


if __name__ == "__main__":
    unittest.main()
