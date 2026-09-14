"""Run HF's real Qwen4ExpTextPLELayer and dump inputs/outputs as fixtures.

The tiny checkpoint is small enough to instantiate the layer for real, weights
and all, so this is the reference implementation's own output rather than
anything we derived.
"""
import json, sys, pathlib, torch
from safetensors.torch import load_file
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
raw = json.loads((ckpt / "config.json").read_text())
cfg = Qwen4ExpTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
torch.manual_seed(11)

layer_idx = cfg.ple_layer_ids[0] - 1              # one-indexed in config
ple = Qwen4ExpTextPLELayer(cfg, layer_idx, 0).eval()

w = load_file(ckpt / "model.safetensors")
pre = f"model.language_model.layers.{layer_idx}.ple."
sd = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
# The sharded table concatenates on dim 0 into one embedding weight.
shards = sorted((k for k in sd if k.startswith("ple_embedding.ngram_embedding.shard_")),
                key=lambda k: int(k.split("shard_")[1].split(".")[0]))
sd["ple_embedding.ngram_embedding.weight"] = torch.cat([sd.pop(k) for k in shards], dim=0)
sd.pop("ple_embedding.ngram_embedding.weight_scale", None)
missing, unexpected = ple.load_state_dict(sd, strict=False)
print("missing:", [m for m in missing if "ngram_heads" not in m and "layer_multipliers" not in m])
print("unexpected:", unexpected)

hc_hidden = cfg.hidden_size * cfg.hc_count
cases = []
for name, seq in (("short", 3), ("medium", 7), ("long", 16)):
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    hidden = torch.randn(1, seq, hc_hidden) * 0.5
    with torch.no_grad():
        y = ple(hidden, ids, past_key_values=None, conv_mask=None)
    cases.append({
        "name": name,
        "input_ids": ids[0].tolist(),
        "hidden_states": hidden[0].flatten().tolist(),
        "output": y[0].flatten().tolist(),
        "seq_len": seq,
    })
    print(f"  {name}: seq={seq} out={tuple(y.shape)} mean={y.mean():.6f} std={y.std():.6f}")

out.write_text(json.dumps({
    "hidden_size": cfg.hidden_size, "hc_count": cfg.hc_count,
    "ple_embed_dim": cfg.ple_embed_dim, "rms_norm_eps": cfg.rms_norm_eps,
    "ple_conv_kernel_size": cfg.ple_conv_kernel_size, "ngram_size": cfg.ngram_size,
    "cases": cases,
}))
print("wrote", out)
