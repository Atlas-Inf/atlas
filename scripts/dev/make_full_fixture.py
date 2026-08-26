"""Can HF instantiate and run the FULL tiny qwen4_exp model?

If yes, a Rust CPU forward can be validated against real logits end to end,
and then pointed at the real checkpoint to produce actual text.
"""
import json, sys, pathlib, torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from safetensors.torch import load_file

ckpt = pathlib.Path(sys.argv[1])
cfg = Qwen4ExpConfig(**json.loads((ckpt / "config.json").read_text()))
torch.manual_seed(3)
model = Qwen4ExpForConditionalGeneration(cfg).eval()
sd = load_file(ckpt / "model.safetensors")
# The sharded n-gram table concatenates into one embedding at load.
shards = sorted((k for k in sd if ".ngram_embedding.shard_" in k),
                key=lambda k: (k.split(".shard_")[0], int(k.split(".shard_")[1].split(".")[0])))
if shards:
    pre = shards[0].split(".shard_")[0]
    sd[pre + ".weight"] = torch.cat([sd.pop(k) for k in shards], dim=0)
    sd.pop(pre + ".weight_scale", None)
# HF's runtime holds experts STACKED (gate_up_proj [E, 2*inter, hidden],
# down_proj [E, hidden, inter]); the per-expert split only exists on disk,
# because quantizers work per nn.Linear. Do the same concat HF's loader does.
import re as _re
per_expert = [k for k in sd if _re.search(r"\.mlp\.experts\.\d+\.", k)]
groups = {}
for k in per_expert:
    base, rest = k.split(".experts.")
    idx, proj = rest.split(".", 1)
    groups.setdefault(base, {}).setdefault(proj.replace(".weight", ""), {})[int(idx)] = sd[k]
for base, projs in groups.items():
    n = len(projs["gate_proj"])
    gate = torch.stack([projs["gate_proj"][i] for i in range(n)])
    up = torch.stack([projs["up_proj"][i] for i in range(n)])
    down = torch.stack([projs["down_proj"][i] for i in range(n)])
    sd[f"{base}.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)
    sd[f"{base}.experts.down_proj"] = down
for k in per_expert:
    sd.pop(k)
print(f"stacked {len(groups)} expert groups")

# Vision is out of scope here -- the server serves this model text-only, and
# the tiny generator's merger dims are its own problem, not the LM's.
sd = {k: v for k, v in sd.items() if not k.startswith("model.visual.")}
missing, unexpected = model.load_state_dict(sd, strict=False)
real_missing = [m for m in missing if "ngram_heads" not in m and "layer_multipliers" not in m and not m.startswith("model.visual.")]
print("missing:", real_missing[:6], f"({len(real_missing)} total)")
print("unexpected:", unexpected[:6], f"({len(unexpected)} total)")

ids = torch.tensor([[11, 42, 7, 300, 5]])
with torch.no_grad():
    out = model(input_ids=ids)
print("logits", tuple(out.logits.shape))
print("logits[0,-1,:6] =", [round(v, 5) for v in out.logits[0, -1, :6].tolist()])
print("argmax per position =", out.logits[0].argmax(-1).tolist())

# Also dump the rotary, so a reference can be checked against it in isolation.
rot = model.model.language_model.rotary_emb
pos = torch.arange(ids.shape[1]).view(1, -1)
with torch.no_grad():
    cos, sin = rot(torch.zeros(1, ids.shape[1], cfg.text_config.hidden_size), pos)
import pathlib as _p
_p.Path(sys.argv[2] if len(sys.argv) > 2 else "full_fixture.json").write_text(json.dumps({
    "input_ids": ids[0].tolist(),
    "logits": out.logits[0].flatten().tolist(),
    "argmax": out.logits[0].argmax(-1).tolist(),
    "cos": cos[0].flatten().tolist(),
    "sin": sin[0].flatten().tolist(),
    "rotary_dim": cos.shape[-1],
}))
print("wrote fixture, cos shape", tuple(cos.shape))
