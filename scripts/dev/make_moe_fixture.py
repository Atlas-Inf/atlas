"""Dump HF's Qwen4ExpTextSparseMoeBlock output at real weights."""
import json, sys, pathlib, torch
from safetensors.torch import load_file
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
raw = json.loads((ckpt / "config.json").read_text())
cfg = Qwen4ExpTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
w = load_file(ckpt / "model.safetensors")
torch.manual_seed(31)

layer = 0
pre = f"model.language_model.layers.{layer}.mlp."
sd = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
# Stack the per-expert projections into HF's runtime layout.
n = cfg.num_experts
gate = torch.stack([sd.pop(f"experts.{i}.gate_proj.weight") for i in range(n)])
up = torch.stack([sd.pop(f"experts.{i}.up_proj.weight") for i in range(n)])
sd["experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
sd["experts.down_proj"] = torch.stack([sd.pop(f"experts.{i}.down_proj.weight") for i in range(n)])

mod = Qwen4ExpTextSparseMoeBlock(cfg).eval()
missing, unexpected = mod.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)

x = torch.randn(1, 6, cfg.hidden_size) * 0.6
with torch.no_grad():
    y = mod(x)
    _, rw, sel = mod.gate(x.view(-1, cfg.hidden_size))
print(f"out{tuple(y.shape)} mean={y.mean():.6f} std={y.std():.6f}")
print("token0 experts:", sel[0].tolist(), "weights:", [round(v,5) for v in rw[0].tolist()])

out.write_text(json.dumps({
    "num_experts": cfg.num_experts, "top_k": cfg.num_experts_per_tok,
    "intermediate": cfg.moe_intermediate_size,
    "shared_intermediate": cfg.shared_expert_intermediate_size,
    "norm_topk_prob": bool(cfg.norm_topk_prob), "layer": layer,
    "input": x[0].flatten().tolist(), "output": y[0].flatten().tolist(),
}))
print("wrote", out)
