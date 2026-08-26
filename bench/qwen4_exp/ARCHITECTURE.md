# Qwen3.8-Flash-Next (`qwen4_exp`) — the three unported forward mechanisms

Transcribed from `ref/modeling_qwen4_exp.py` (`transformers` main, vendored
here the way `bench/ngram_ref/` vendors LongCat's). Line numbers refer to that
file.

The model **loads** today (see Avarok #753); it does not serve because these
three are unimplemented. Everything below is the reference's math, not an
inference from tensor names — which matters, because two of the three had
plausible-but-wrong readings available.

---

## 1. mHC — multi-hyperconnection residual (`Qwen4ExpTextGatedResidual`, L940)

**The residual stream is `hc_count * hidden_size` = 4 × 2560 = 10240 wide.**
Every block reads it down to one 2560 stream, runs, and injects back.

```python
hyper_input_normed = hc_norm(hyper_input)                 # grouped RMSNorm, group=hidden
w = silu(input_mix_weight_down(hyper_input_normed) / hc_count)   # 10240 -> 320
w = sigmoid(input_mix_weight_up(w))                              # 320 -> 10240
w = w.unflatten(-1, (hc_count, hidden))
mixed_input = (w * hyper_input_normed.unflatten(-1, (hc_count, hidden))).mean(dim=-2)
injection_weights = 2 * sigmoid(block_inject_weight(hyper_input_normed) / hc_count)  # -> [hc_count]
```

and the decoder layer wraps each of attention and MLP with it (L1222–L1244):

```python
hidden, hyper_input, inj = attn_hyper_connection(residual)   # residual [T,10240] -> hidden [T,2560]
hidden = attn(hidden)
residual = hyper_input + (hidden.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
#          residual[t, s*H + d] = hyper_input[t, s*H + d] + hidden[t,d] * inj[t,s]

hidden, hyper_input, inj = mlp_hyper_connection(residual)
hidden = mlp(hidden)
residual = hyper_input + (hidden.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
```

At the end of the stack `hyper_connection_mixer` (L1330, `use_combine=False`)
runs the same collapse and returns `mixed_input` only — **that is the final
norm**, which is why the checkpoint has no `model.norm.weight`.

### Why Atlas's existing mHC kernels are the wrong ones

`ops/hyper_connection.rs` is DeepSeek-V4's: its `hc_pre` mixes with a
**Sinkhorn-normalized** matrix over `hc_fn`/`hc_scale`/`hc_base`. The stream
layout `[T, hc_mult, H]` and `config.hc_mult = 4` carry across unchanged, but
the mixing math does not. Symlinking `hyper_connection.cu` from
`deepseek-v4-flash` makes the startup audit pass and the model generate —
**confidently, and wrong**. Two new entry points are needed:

- `hc_pre_lowrank` — grouped RMSNorm, down/SiLU/up/sigmoid, per-stream
  elementwise gate, mean over streams, plus the `2*sigmoid(...)` injection
  vector
- `hc_post_lowrank` — `residual[t, s*H+d] += hidden[t,d] * inj[t,s]`

`hc_expand` / `hc_head` should be reusable as-is (pure broadcast/collapse).

> `hc_norm` is a **grouped** RMSNorm: `group_size = hidden_size`, so the four
> 2560-wide streams normalize INDEPENDENTLY inside the 10240 vector. A single
> RMS over all 10240 is a different function and will not fail loudly.

---

## 2. PLE — n-gram injection (`Qwen4ExpTextPLELayer`, L1117)

Its own docstring: *"PLE projects each token's concatenated n-gram embedding
to a shared value and one key per residual stream. The normalized stream
activations gate those values, then a dilated depthwise convolution adds local
lexical context."*

**This confirms the cross-attention reading and refutes the additive-embedding
one** — which is the failure mode that cost real time on LongCat (a plain
gather where a fused embedding belongs produces fluent, wrong text).

```python
embeddings   = ple_embedding(input_ids)                   # [T, 2560] = concat 16 heads x 160
key_normed   = norm_key(key_proj(embeddings)).unflatten(-1, (hc, H))   # [T,4,2560]
value        = value_proj(embeddings)                                   # [T,2560]
query_normed = norm_query(hidden_states).unflatten(-1, (hc, H))         # [T,4,2560]

gate = (key_normed * query_normed).sum(-1, keepdim=True) / sqrt(H)      # [T,4,1]
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()                  # signed sqrt
gated_value = sigmoid(gate) * value.unsqueeze(-2)                       # [T,4,2560]

gated_value_normed = norm_conv(gated_value.flatten(-2))
output = gated_value.flatten(-2) + silu(conv1d(gated_value_normed))     # [T,10240]
```

and the layer adds it to the residual BEFORE the attention hyper-connection
(L1218): `hidden_states = hidden_states + self.ple(...)`.

Notes that bite:
- the **signed square root** on the gate is not a normalization anyone would
  guess; omit it and the gate distribution is wrong but finite
- `conv1d` is **depthwise** (`groups = hc_hidden_size = 10240`),
  `kernel_size = ple_conv_kernel_size = 4`, and **dilated by
  `dilation = ngram_size = 3`** → state length `(4-1)*3 = 9`
- it runs on ONE layer (`ple_layer_ids = [2]`), and its 320M-row table is
  served off NVMe by the row cache from PR #746

---

## 3. QSA indexer (`Qwen4ExpTextQSAIndexer`, L611)

On the 12 full-attention layers only; `indexer_budget = 2048`,
`compress_ratio = 4`, 4 heads × 128, `index_qk_proj [640, 2560]` fused as
q(4×128) + k(1×128). Atlas has DeepSeek-V4's CSA machinery
(`index_n_heads` / `index_head_dim` / `index_topk` / `compress_ratios`, plus
`csa_compress` and `prefill_attn_compressed`); the open question is whether
the selection semantics match. Read L611 onward before wiring.

---

## Status against the startup audit

The 6 unresolved lookups that currently block serving map exactly onto
sections 1 and 3:

```
hyper_connection::hc_pre / hc_post / hc_expand / hc_head   -> section 1
csa_compress::csa_compress                                 -> section 3
prefill_attn_compressed::prefill_attn_compressed           -> section 3
```

Neither `--dangerously-allow-unresolved-kernel-lookups` nor symlinking
DeepSeek's kernels is a route to serving: the first dispatches handle 0, the
second dispatches the wrong math. Both produce output that looks fine.
