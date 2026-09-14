"""Dump HF's Qwen4ExpTextAttention output at real weights, with cos/sin."""
import json, sys, pathlib, torch
from safetensors.torch import load_file
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
raw = json.loads((ckpt / "config.json").read_text())
cfg = Qwen4ExpTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
cfg._attn_implementation = "eager"
w = load_file(ckpt / "model.safetensors")
torch.manual_seed(77)

print("layer_types:", getattr(cfg, "layer_types", None))
layer = next((i for i, t in enumerate(getattr(cfg, "layer_types", []) or []) if t == "full_attention"), 3)
pre = f"model.language_model.layers.{layer}.self_attn."
sd = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
mod = Qwen4ExpTextAttention(cfg, layer).eval()
missing, unexpected = mod.load_state_dict(sd, strict=False)
print("layer", layer, "missing:", missing, "unexpected:", unexpected)

seq = 6
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
x = torch.randn(1, seq, cfg.hidden_size) * 0.5
cos = torch.randn(1, seq, rot) * 0.3 + 0.9
sin = torch.randn(1, seq, rot) * 0.3
# Causal float mask, eager convention.
mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1).view(1, 1, seq, seq)
# The indexer is a no-op at this length; neutralise it so this isolates attention.
class _NoIndexer(torch.nn.Module):
    """The indexer is a verified no-op below its budget; return an all-zero
    additive mask so this fixture isolates attention itself."""
    def forward(self, *a, **k):
        return torch.zeros(1, 1, seq, seq)
mod.indexer = _NoIndexer()
with torch.no_grad():
    y, _ = mod(x, (cos, sin), attention_mask=mask, past_key_values=None)
print(f"out{tuple(y.shape)} mean={y.mean():.6f} std={y.std():.6f}")

out.write_text(json.dumps({
    "layer": layer, "rotary_dim": rot, "seq": seq,
    "hidden": x[0].flatten().tolist(),
    "cos": cos[0].flatten().tolist(), "sin": sin[0].flatten().tolist(),
    "output": y[0].flatten().tolist(),
}))
print("wrote", out)
