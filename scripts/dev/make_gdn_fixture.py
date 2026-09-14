"""Dump HF's Qwen4ExpTextGatedDeltaNet output at real weights."""
import json, sys, pathlib, torch
from safetensors.torch import load_file
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
raw = json.loads((ckpt / "config.json").read_text())
cfg = Qwen4ExpTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
w = load_file(ckpt / "model.safetensors")
torch.manual_seed(101)

layer = 0
pre = f"model.language_model.layers.{layer}.linear_attn."
sd = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
mod = Qwen4ExpTextGatedDeltaNet(cfg, layer).eval()
missing, unexpected = mod.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
print("gate activation:", mod.norm.activation)

seq = 7
x = torch.randn(1, seq, cfg.hidden_size) * 0.5
with torch.no_grad():
    y = mod(x, cache_params=None, attention_mask=None)
print(f"out{tuple(y.shape)} mean={y.mean():.6f} std={y.std():.6f}")

out.write_text(json.dumps({
    "layer": layer, "seq": seq,
    "sigmoid_gate": mod.norm.activation == "sigmoid",
    "hidden": x[0].flatten().tolist(),
    "output": y[0].flatten().tolist(),
}))
print("wrote", out)
