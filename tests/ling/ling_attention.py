#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Small CPU oracle for Ling-3.2-air's KDA, MLA, and SwiGLU paths.

The oracle deliberately keeps checkpoint I/O outside the math.  A caller gives
it two adapters:

``linear(name, x)``
    Apply the named projection to a ``[tokens, hidden]`` tensor.  The adapter
    may dispatch BF16 and FP8 weights differently.

``tensor(name)``
    Return a non-linear tensor such as a convolution weight, RMSNorm weight,
    ``A_log``, or ``dt_bias``.

``prefix`` is prepended to every local name, so the same functions work for a
checkpoint layer namespace such as ``model.layers.6.self_attn`` or for the
short names used by the unit tests.  The returned states are explicit and can
be passed to the next call for decode.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F


Linear = Callable[[str, torch.Tensor], torch.Tensor]
Tensor = Callable[[str], torch.Tensor]


@dataclass(frozen=True)
class LingConfig:
    """Text-side fields used by the native attention and MLP port.

    Defaults are the current pinned Ling-3.2-air values. Tests use the same
    class with smaller dimensions to keep CPU fixtures cheap.
    """

    hidden_size: int = 2560
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    head_dim: int = 128
    rope_theta: float = 6_000_000.0
    rms_norm_eps: float = 1.0e-6
    layer_group_size: int = 6
    short_conv_kernel_size: int = 4
    no_kda_lora: bool = True
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    gated_attention_proj_granularity_type: str = "head_wise"
    rope_interleave: bool = True
    num_shared_experts: int | None = 1
    moe_shared_expert_intermediate_size: int = 768
    expert_swiglu_limit_list: tuple[float, ...] = ()
    share_expert_swiglu_limit_list: tuple[float, ...] = ()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_mla_layer(self, layer_idx: int) -> bool:
        """Match ``BailingMoeV3DecoderLayer``'s 5:1 KDA/MLA schedule."""
        return (
            (layer_idx + 1) % self.layer_group_size == 0
            or layer_idx >= (self.num_hidden_layers // self.layer_group_size)
            * self.layer_group_size
        )

    @staticmethod
    def _limit(values: Sequence[float], layer_idx: int) -> float | None:
        if not 0 <= layer_idx < len(values):
            return None
        value = float(values[layer_idx])
        return None if value == 0.0 else value

    def expert_swiglu_limit(self, layer_idx: int) -> float | None:
        return self._limit(self.expert_swiglu_limit_list, layer_idx)

    def shared_swiglu_limit(self, layer_idx: int) -> float | None:
        return self._limit(self.share_expert_swiglu_limit_list, layer_idx)

    @classmethod
    def from_json(cls, source: str | Path | Mapping[str, Any]) -> "LingConfig":
        """Load the nested ``text_config`` from the pinned model config."""
        if isinstance(source, (str, Path)):
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            payload = dict(source)
        text = dict(payload.get("text_config", payload))
        values: dict[str, Any] = {}
        for field in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "head_dim",
            "rope_theta",
            "rms_norm_eps",
            "layer_group_size",
            "short_conv_kernel_size",
            "no_kda_lora",
            "kda_safe_gate",
            "kda_lower_bound",
            "gated_attention_proj_granularity_type",
            "num_shared_experts",
            "moe_shared_expert_intermediate_size",
        ):
            if field in text:
                values[field] = text[field]
        # rope_interleave is a configuration-class default, omitted from the
        # JSON for this checkpoint; the pinned BailingMoeV3 config defaults it
        # to True.
        values["rope_interleave"] = text.get("rope_interleave", True)
        values["expert_swiglu_limit_list"] = tuple(
            text.get("expert_swiglu_limit_list", ()) or ()
        )
        values["share_expert_swiglu_limit_list"] = tuple(
            text.get("share_expert_swiglu_limit_list", ()) or ()
        )
        return cls(**values)


@dataclass
class KDAState:
    """Persistent KDA state after the last processed token.

    ``conv_*`` use Atlas's ``[channels, 4]`` sliding-window layout. ``h`` is
    Atlas's row-major ``[heads, key_dim, value_dim]`` FP32 state.
    """

    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor
    h: torch.Tensor


@dataclass
class MLAState:
    """Persistent MLA key/value cache, with sequence dimension first."""

    key: torch.Tensor
    value: torch.Tensor


def _config(config: LingConfig | Mapping[str, Any]) -> LingConfig:
    if isinstance(config, LingConfig):
        return config
    return LingConfig.from_json(config)


def _name(prefix: str, local: str) -> str:
    return f"{prefix}.{local}" if prefix else local


def _as_float(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def _tensor(tensor: Tensor, name: str, device: torch.device) -> torch.Tensor:
    value = tensor(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"tensor({name!r}) returned {type(value).__name__}, not a Tensor")
    return _as_float(value, device)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    result = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps) * weight
    return result.to(dtype=x.dtype)


def depthwise_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    kernel_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atlas's four-wide depthwise causal convolution plus SiLU.

    ``weight`` is ``[channels, kernel]`` (a singleton middle dimension is
    accepted), and ``state`` is the pre-token sliding window.  Each token is
    rounded only by the caller's projection/engine adapter; the oracle keeps
    the recurrence in FP32 like Atlas's KDA state.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [tokens, channels], got {tuple(x.shape)}")
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0]
    if weight.ndim != 2 or weight.shape[1] != kernel_size:
        raise ValueError(
            f"depthwise weight must be [channels, {kernel_size}], got {tuple(weight.shape)}"
        )
    channels = x.shape[-1]
    if weight.shape[0] != channels:
        raise ValueError(f"weight channels {weight.shape[0]} != input channels {channels}")
    if state is None:
        window = torch.zeros((channels, kernel_size), dtype=torch.float32, device=x.device)
    else:
        if tuple(state.shape) != (channels, kernel_size):
            raise ValueError(
                f"conv state must be {(channels, kernel_size)}, got {tuple(state.shape)}"
            )
        window = state.to(device=x.device, dtype=torch.float32).clone()
    weight32 = weight.to(device=x.device, dtype=torch.float32)
    bias32 = None if bias is None else bias.to(device=x.device, dtype=torch.float32)
    output = torch.empty(
        (x.shape[0], channels), dtype=x.dtype if x.is_floating_point() else torch.float32,
        device=x.device,
    )
    for token in range(x.shape[0]):
        window[:, :-1] = window[:, 1:].clone()
        window[:, -1] = x[token].to(dtype=torch.float32)
        acc = (window * weight32).sum(dim=-1)
        if bias32 is not None:
            acc = acc + bias32
        output[token] = F.silu(acc).to(dtype=output.dtype)
    return output, window


def _new_kda_state(x: torch.Tensor, config: LingConfig) -> KDAState:
    channels = config.num_attention_heads * config.head_dim
    zeros_conv = torch.zeros(
        (channels, config.short_conv_kernel_size), dtype=torch.float32, device=x.device
    )
    h = torch.zeros(
        (config.num_attention_heads, config.head_dim, config.head_dim),
        dtype=torch.float32,
        device=x.device,
    )
    return KDAState(zeros_conv.clone(), zeros_conv.clone(), zeros_conv, h)


def kda(
    x: torch.Tensor,
    prefix: str,
    linear: Linear,
    tensor: Tensor,
    state: KDAState | None,
    config: LingConfig | Mapping[str, Any],
) -> tuple[torch.Tensor, KDAState]:
    """Evaluate one Ling KDA layer and return output plus its new cache state.

    The recurrence follows ``atlas_kda_recurrent_body`` exactly: normalize Q/K
    per token, use the per-element ``A_log``/``dt_bias`` decay with lower bound
    ``-5``, decay the state before the delta correction, and read output from
    the updated state.  The 32-head pinned model uses ``q_lora_rank=None`` and
    ``no_kda_lora=True``; unsupported alternate branches fail loudly.
    """
    cfg = _config(config)
    if x.ndim != 2 or x.shape[-1] != cfg.hidden_size:
        raise ValueError(f"x must be [tokens, {cfg.hidden_size}], got {tuple(x.shape)}")
    if cfg.q_lora_rank is not None or not cfg.no_kda_lora:
        raise ValueError("this oracle is pinned to q_lora_rank=None and no_kda_lora=True")
    if cfg.num_attention_heads != cfg.num_key_value_heads:
        raise ValueError("Ling-3.2-air KDA oracle requires equal Q/K/V head counts")
    if cfg.short_conv_kernel_size != 4:
        raise ValueError("the Ling-3.2-air KDA path requires a four-wide convolution")
    if state is None:
        live = _new_kda_state(x, cfg)
    else:
        live = state

    q_raw = linear(_name(prefix, "q_proj"), x)
    k_raw = linear(_name(prefix, "k_proj"), x)
    v_raw = linear(_name(prefix, "v_proj"), x)
    conv_q_weight = _tensor(tensor, _name(prefix, "q_conv1d.weight"), x.device)
    conv_k_weight = _tensor(tensor, _name(prefix, "k_conv1d.weight"), x.device)
    conv_v_weight = _tensor(tensor, _name(prefix, "v_conv1d.weight"), x.device)
    conv_q, conv_q_state = depthwise_causal_conv(
        q_raw, conv_q_weight, live.conv_q, kernel_size=cfg.short_conv_kernel_size
    )
    conv_k, conv_k_state = depthwise_causal_conv(
        k_raw, conv_k_weight, live.conv_k, kernel_size=cfg.short_conv_kernel_size
    )
    conv_v, conv_v_state = depthwise_causal_conv(
        v_raw, conv_v_weight, live.conv_v, kernel_size=cfg.short_conv_kernel_size
    )

    tokens = x.shape[0]
    heads = cfg.num_attention_heads
    dim = cfg.head_dim
    q = conv_q.reshape(tokens, heads, dim).float()
    k = conv_k.reshape(tokens, heads, dim).float()
    v = conv_v.reshape(tokens, heads, dim).float()
    q_inv = torch.rsqrt(q.square().sum(dim=-1) + 1.0e-6) * dim**-0.5
    k_inv = torch.rsqrt(k.square().sum(dim=-1) + 1.0e-6)
    q_norm = q * q_inv.unsqueeze(-1)
    k_norm = k * k_inv.unsqueeze(-1)

    raw_gate = linear(_name(prefix, "f_proj"), x).reshape(tokens, heads, dim)
    raw_beta = linear(_name(prefix, "b_proj"), x).reshape(tokens, heads)
    a_log = _tensor(tensor, _name(prefix, "A_log"), x.device).reshape(heads)
    dt_bias = _tensor(tensor, _name(prefix, "dt_bias"), x.device).reshape(heads, dim)
    a = torch.exp(a_log).view(1, heads, 1)
    log_decay = cfg.kda_lower_bound / (
        1.0 + torch.exp(-a * (raw_gate.float() + dt_bias.view(1, heads, dim)))
    )
    decay = torch.exp(log_decay)
    beta = torch.sigmoid(raw_beta.float())

    h = live.h.to(device=x.device, dtype=torch.float32).clone()
    if tuple(h.shape) != (heads, dim, dim):
        raise ValueError(f"KDA h state must be {(heads, dim, dim)}, got {tuple(h.shape)}")
    output = torch.empty(
        (tokens, heads, dim), dtype=conv_q.dtype, device=x.device
    )
    for token in range(tokens):
        # Atlas stores H as [key_dim, value_dim] and applies per-key decay.
        h_decayed = h * decay[token].unsqueeze(-1)
        dot_k = torch.einsum("hkv,hk->hv", h_decayed, k_norm[token])
        delta = (v[token] - dot_k) * beta[token].unsqueeze(-1)
        h = h_decayed + k_norm[token].unsqueeze(-1) * delta.unsqueeze(1)
        output[token] = torch.einsum("hkv,hk->hv", h, q_norm[token]).to(
            dtype=output.dtype
        )

    gate_out = linear(_name(prefix, "g_proj"), x).reshape(tokens, heads, dim).float()
    o_norm_weight = _tensor(tensor, _name(prefix, "o_norm.weight"), x.device)
    inv_rms = torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
    gated = output * inv_rms * o_norm_weight.view(1, 1, dim)
    gated = gated * torch.sigmoid(gate_out)
    # FusedRMSNormGated writes the recurrent boundary in the convolution's
    # dtype (BF16 in the deployable path) before o_proj consumes it.
    gated = gated.to(dtype=conv_q.dtype)
    projected = linear(_name(prefix, "o_proj"), gated.reshape(tokens, heads * dim))

    new_state = KDAState(conv_q_state, conv_k_state, conv_v_state, h)
    return projected, new_state


def _rotary_interleaved(
    x: torch.Tensor, positions: torch.Tensor, rope_theta: float
) -> torch.Tensor:
    """Apply Ling's even/odd (interleaved) rotary pairing."""
    rope_dim = x.shape[-1]
    if rope_dim % 2:
        raise ValueError("interleaved rotary dimension must be even")
    frequencies = torch.arange(
        0, rope_dim, 2, dtype=torch.float32, device=x.device
    ) / rope_dim
    inv_freq = torch.pow(torch.tensor(rope_theta, device=x.device), -frequencies)
    angle = positions.to(dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    cos = angle.cos().unsqueeze(1)
    sin = angle.sin().unsqueeze(1)
    pairs = x.reshape(x.shape[0], x.shape[1], rope_dim // 2, 2).float()
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return rotated.reshape_as(x).to(dtype=x.dtype)


def mla(
    x: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    prefix: str,
    linear: Linear,
    tensor: Tensor,
    state: MLAState | None,
    config: LingConfig | Mapping[str, Any],
) -> tuple[torch.Tensor, MLAState]:
    """Evaluate one gated MLA layer with an append-only KV cache.

    ``positions`` are one-dimensional text positions for the current rows.
    The pinned VL model's three M-RoPE axes are identical for text-only rows,
    so this is the exact text path while keeping the API convenient for native
    prefill/decode tests.
    """
    cfg = _config(config)
    if x.ndim != 2 or x.shape[-1] != cfg.hidden_size:
        raise ValueError(f"x must be [tokens, {cfg.hidden_size}], got {tuple(x.shape)}")
    if cfg.q_lora_rank is not None:
        raise ValueError("this oracle is pinned to q_lora_rank=None")
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("num_attention_heads must divide num_key_value_heads")
    if not cfg.rope_interleave:
        raise ValueError("the pinned Ling config requires interleaved rotary pairing")
    positions_t = torch.as_tensor(positions, dtype=torch.long, device=x.device).reshape(-1)
    if positions_t.numel() != x.shape[0]:
        raise ValueError("positions must contain one value per input row")

    tokens = x.shape[0]
    heads = cfg.num_attention_heads
    q = linear(_name(prefix, "q_proj"), x).reshape(tokens, heads, cfg.qk_head_dim)
    kv_a = linear(_name(prefix, "kv_a_proj_with_mqa"), x)
    kv_pass, k_rot = torch.split(
        kv_a, [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1
    )
    kv_norm_weight = _tensor(tensor, _name(prefix, "kv_a_layernorm.weight"), x.device)
    kv_norm = _rms_norm(kv_pass, kv_norm_weight, cfg.rms_norm_eps)
    kv_b = linear(_name(prefix, "kv_b_proj"), kv_norm).reshape(
        tokens, heads, cfg.qk_nope_head_dim + cfg.v_head_dim
    )
    k_pass, value = torch.split(
        kv_b, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1
    )
    q_pass, q_rot = torch.split(
        q, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1
    )
    q_rot = _rotary_interleaved(q_rot, positions_t, cfg.rope_theta)
    k_rot = _rotary_interleaved(k_rot.unsqueeze(1), positions_t, cfg.rope_theta)
    k_rot = k_rot.expand(-1, heads, -1)
    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_pass, k_rot), dim=-1)

    if state is None:
        past_key = torch.empty(
            (0, heads, cfg.qk_head_dim), dtype=key.dtype, device=x.device
        )
        past_value = torch.empty(
            (0, heads, cfg.v_head_dim), dtype=value.dtype, device=x.device
        )
    else:
        past_key = state.key.to(device=x.device)
        past_value = state.value.to(device=x.device)
        if past_key.ndim != 3 or past_key.shape[1:] != (heads, cfg.qk_head_dim):
            raise ValueError("MLA key cache has the wrong shape")
        if past_value.ndim != 3 or past_value.shape[1:] != (heads, cfg.v_head_dim):
            raise ValueError("MLA value cache has the wrong shape")
    past_tokens = past_key.shape[0]
    all_key = torch.cat((past_key, key), dim=0)
    all_value = torch.cat((past_value, value), dim=0)

    scale = cfg.qk_head_dim**-0.5
    scores = torch.einsum("thd,shd->ths", query.float(), all_key.float()) * scale
    allowed = torch.arange(all_key.shape[0], device=x.device).view(1, -1)
    allowed = allowed <= (past_tokens + torch.arange(tokens, device=x.device)).view(-1, 1)
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
    attention = torch.einsum(
        "ths,shv->thv", probabilities.float(), all_value.float()
    ).to(dtype=value.dtype)

    if cfg.gated_attention_proj_granularity_type == "head_wise":
        gate = torch.sigmoid(
            linear(_name(prefix, "g_proj"), x).reshape(tokens, heads).float()
        ).to(dtype=attention.dtype)
        attention = attention * gate.unsqueeze(-1)
    elif cfg.gated_attention_proj_granularity_type == "element_wise":
        gate = torch.sigmoid(
            linear(_name(prefix, "g_proj"), x).reshape(tokens, heads, cfg.v_head_dim).float()
        ).to(dtype=attention.dtype)
        attention = attention * gate
    elif cfg.gated_attention_proj_granularity_type is not None:
        raise ValueError(
            f"unsupported gated_attention_proj_granularity_type={cfg.gated_attention_proj_granularity_type!r}"
        )
    flat = attention.reshape(tokens, heads * cfg.v_head_dim)
    # HF names this projection ``dense`` (MLA), while KDA's projection is
    # ``o_proj``.  Keeping the names separate prevents a silent state-dict
    # mismatch when both attention families share a layer adapter.
    output = linear(_name(prefix, "dense"), flat)
    return output, MLAState(all_key, all_value)


def swiglu(
    x: torch.Tensor,
    prefix: str,
    linear: Linear,
    limit: float | None = None,
) -> torch.Tensor:
    """BailingMoeV3's gated MLP, including its asymmetric clip."""
    gate = F.silu(linear(_name(prefix, "gate_proj"), x))
    up = linear(_name(prefix, "up_proj"), x)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return linear(_name(prefix, "down_proj"), gate * up)


def routed_shared_swiglu(
    x: torch.Tensor,
    prefix: str,
    linear: Linear,
    tensor: Tensor,
    layer_idx: int,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    config: LingConfig | Mapping[str, Any],
) -> torch.Tensor:
    """Apply routed experts and the unweighted shared expert.

    Router selection is intentionally an input: Atlas owns that separate
    operation.  This function mirrors the HF block's per-layer routed/shared
    SwiGLU clipping and weighted top-k sum, which is the part needed by the
    native forward oracle.
    """
    cfg = _config(config)
    if x.ndim != 2:
        raise ValueError("x must be [tokens, hidden]")
    if expert_ids.ndim != 2 or expert_weights.shape != expert_ids.shape:
        raise ValueError("expert_ids and expert_weights must both be [tokens, top_k]")
    if expert_ids.shape[0] != x.shape[0]:
        raise ValueError("expert rows must match x")
    output = torch.zeros_like(x, dtype=torch.float32)
    routed_limit = cfg.expert_swiglu_limit(layer_idx)
    for slot in range(expert_ids.shape[1]):
        ids = expert_ids[:, slot].to(dtype=torch.long)
        weights = expert_weights[:, slot].float()
        for expert in torch.unique(ids).tolist():
            rows = torch.nonzero(ids == expert, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            expert_prefix = _name(prefix, f"experts.{int(expert)}")
            values = swiglu(x.index_select(0, rows), expert_prefix, linear, routed_limit)
            output.index_add_(
                0, rows,
                values.float() * weights.index_select(0, rows).unsqueeze(-1),
            )

    # HF's moe_infer narrows the routed accumulation back to the module
    # boundary dtype before adding the unweighted shared expert. Keeping the
    # weighted sum in FP32 through `y + shared` subtly changes BF16 outputs.
    output = output.to(dtype=x.dtype)
    if cfg.num_shared_experts:
        shared_limit = cfg.shared_swiglu_limit(layer_idx)
        shared_prefix = _name(prefix, "shared_experts")
        shared = swiglu(x, shared_prefix, linear, shared_limit).to(dtype=x.dtype)
        output = output + shared
    return output
