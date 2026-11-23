#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the Ling attention/MLP oracle."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ling_attention import (  # noqa: E402
    LingConfig,
    MLAState,
    depthwise_causal_conv,
    kda,
    mla,
    routed_shared_swiglu,
    swiglu,
)


class TinyConfig(LingConfig):
    def __init__(self, **overrides):
        values = dict(
            hidden_size=8,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=None,
            kv_lora_rank=3,
            qk_nope_head_dim=2,
            qk_rope_head_dim=4,
            v_head_dim=2,
            head_dim=4,
            rope_theta=1_000.0,
            rms_norm_eps=1.0e-6,
            layer_group_size=2,
            short_conv_kernel_size=4,
            no_kda_lora=True,
            kda_safe_gate=True,
            kda_lower_bound=-5.0,
            gated_attention_proj_granularity_type="head_wise",
            rope_interleave=True,
            num_shared_experts=1,
            expert_swiglu_limit_list=(4.0,),
            share_expert_swiglu_limit_list=(5.0,),
        )
        values.update(overrides)
        super().__init__(**values)


def matrix_adapter(weights):
    def linear(name, x):
        return x @ weights[name].to(device=x.device, dtype=x.dtype).T

    return linear


class LingAttentionOracleTest(unittest.TestCase):
    def test_pinned_config_shape_and_schedule(self):
        config_path = os.environ.get("LING_CONFIG")
        if not config_path:
            self.skipTest("set LING_CONFIG to the pinned Ling config for this check")
        config = LingConfig.from_json(config_path)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 32)
        self.assertIsNone(config.q_lora_rank)
        self.assertEqual(config.qk_nope_head_dim, 128)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertEqual(config.short_conv_kernel_size, 4)
        self.assertEqual(config.kda_lower_bound, -5.0)
        self.assertTrue(config.rope_interleave)
        self.assertEqual(sum(config.is_mla_layer(i) for i in range(42)), 7)
        self.assertEqual(sum(not config.is_mla_layer(i) for i in range(42)), 35)
        self.assertIsNone(config.expert_swiglu_limit(34))
        self.assertEqual(config.expert_swiglu_limit(35), 4.0)
        self.assertEqual(config.shared_swiglu_limit(34), 5.0)
        self.assertEqual(config.shared_swiglu_limit(40), 7.0)

    def test_four_wide_conv_matches_left_padded_correlation(self):
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result, state = depthwise_causal_conv(x, weight)
        expected_acc = torch.tensor([4.0, 11.0, 20.0, 30.0])
        torch.testing.assert_close(result[:, 0], F.silu(expected_acc))
        torch.testing.assert_close(state[:, 0], torch.tensor([1.0]))
        torch.testing.assert_close(state[:, 1], torch.tensor([2.0]))
        torch.testing.assert_close(state[:, 2], torch.tensor([3.0]))
        torch.testing.assert_close(state[:, 3], torch.tensor([4.0]))

    def test_kda_prefill_and_decode_share_conv_and_recurrent_state(self):
        torch.manual_seed(11)
        config = TinyConfig()
        hidden = config.hidden_size
        heads = config.num_attention_heads
        dim = config.head_dim
        channels = heads * dim
        weights = {
            name: torch.randn(width, hidden) / hidden**0.5
            for name, width in (
                ("attn.q_proj", channels),
                ("attn.k_proj", channels),
                ("attn.v_proj", channels),
                ("attn.f_proj", channels),
                ("attn.b_proj", heads),
                ("attn.g_proj", channels),
                ("attn.o_proj", hidden),
            )
        }
        tensors = {
            "attn.q_conv1d.weight": torch.randn(channels, 4),
            "attn.k_conv1d.weight": torch.randn(channels, 4),
            "attn.v_conv1d.weight": torch.randn(channels, 4),
            "attn.A_log": torch.randn(heads),
            "attn.dt_bias": torch.randn(heads, dim),
            "attn.o_norm.weight": torch.ones(dim),
        }
        linear = matrix_adapter(weights)
        tensor = tensors.__getitem__
        x = torch.randn(7, hidden)

        whole, whole_state = kda(x, "attn", linear, tensor, None, config)
        first, state = kda(x[:3], "attn", linear, tensor, None, config)
        rest, split_state = kda(x[3:], "attn", linear, tensor, state, config)

        torch.testing.assert_close(torch.cat((first, rest)), whole, rtol=1e-5, atol=1e-5)
        for left, right in zip(
            (whole_state.conv_q, whole_state.conv_k, whole_state.conv_v, whole_state.h),
            (split_state.conv_q, split_state.conv_k, split_state.conv_v, split_state.h),
        ):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)

    def test_mla_interleaved_rope_and_kv_cache_across_decode(self):
        torch.manual_seed(23)
        config = TinyConfig()
        heads = config.num_attention_heads
        qk = config.qk_head_dim
        weights = {
            "attn.q_proj": torch.randn(heads * qk, config.hidden_size),
            "attn.kv_a_proj_with_mqa": torch.randn(
                config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size
            ),
            "attn.kv_b_proj": torch.randn(
                heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank
            ),
            "attn.g_proj": torch.randn(heads, config.hidden_size),
            "attn.dense": torch.randn(config.hidden_size, heads * config.v_head_dim),
        }
        tensors = {"attn.kv_a_layernorm.weight": torch.ones(config.kv_lora_rank)}
        linear = matrix_adapter(weights)
        tensor = tensors.__getitem__
        x = torch.randn(5, config.hidden_size)
        positions = torch.arange(5)

        whole, whole_state = mla(x, positions, "attn", linear, tensor, None, config)
        first, state = mla(x[:2], positions[:2], "attn", linear, tensor, None, config)
        rest, split_state = mla(x[2:], positions[2:], "attn", linear, tensor, state, config)

        self.assertIsInstance(whole_state, MLAState)
        self.assertEqual(tuple(whole_state.key.shape), (5, heads, qk))
        self.assertEqual(tuple(whole_state.value.shape), (5, heads, config.v_head_dim))
        torch.testing.assert_close(torch.cat((first, rest)), whole, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(split_state.key, whole_state.key, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(split_state.value, whole_state.value, rtol=1e-5, atol=1e-5)

    def test_swiglu_and_routed_shared_limits_are_layer_specific(self):
        config = TinyConfig()
        x = torch.tensor([[1.0, -1.0]])

        def linear(name, values):
            if name.endswith("gate_proj"):
                return torch.tensor([[10.0, -10.0]], device=values.device)
            if name.endswith("up_proj"):
                return torch.tensor([[10.0, -10.0]], device=values.device)
            if name.endswith("down_proj"):
                return values
            raise AssertionError(name)

        clipped = swiglu(x, "experts.0", linear, limit=4.0)
        gate = F.silu(torch.tensor(10.0)).clamp(max=4.0)
        up = torch.tensor(10.0).clamp(min=-4.0, max=4.0)
        expected = torch.tensor([[gate * up, F.silu(torch.tensor(-10.0)) * -4.0]])
        torch.testing.assert_close(clipped, expected)

        result = routed_shared_swiglu(
            x,
            "attn.mlp",
            linear,
            lambda name: torch.empty(0),
            0,
            torch.tensor([[0]]),
            torch.tensor([[1.0]]),
            config,
        )
        # The routed expert and unweighted shared expert have distinct clips.
        shared_gate = F.silu(torch.tensor(10.0)).clamp(max=5.0)
        shared_up = torch.tensor(10.0).clamp(min=-5.0, max=5.0)
        shared_expected = torch.tensor(
            [[shared_gate * shared_up, F.silu(torch.tensor(-10.0)) * -5.0]]
        )
        torch.testing.assert_close(result, expected + shared_expected)


if __name__ == "__main__":
    unittest.main()
