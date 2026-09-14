"""Dump HF's Qwen4ExpTextGatedResidual outputs at real weights."""
import json, sys, pathlib, torch
from safetensors.torch import load_file
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
raw = json.loads((ckpt / "config.json").read_text())
cfg = Qwen4ExpTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
w = load_file(ckpt / "model.safetensors")
torch.manual_seed(23)
wide = cfg.hidden_size * cfg.hc_count
cases = []

for label, prefix, combine in (
    ("layer0_attn", "model.language_model.layers.0.attn_hyper_connection.", True),
    ("layer3_mlp",  "model.language_model.layers.3.mlp_hyper_connection.",  True),
    ("trunk_mixer", "model.language_model.hyper_connection_mixer.",          False),
):
    mod = Qwen4ExpTextGatedResidual(cfg, use_combine=combine).eval()
    sd = {k[len(prefix):]: v for k, v in w.items() if k.startswith(prefix)}
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    assert not unexpected, (label, unexpected)
    x = torch.randn(1, 4, wide) * 0.7
    with torch.no_grad():
        y = mod(x)
    if combine:
        mixed, _, inject = y
        cases.append({"name": label, "combine": True, "input": x[0].flatten().tolist(),
                      "mixed": mixed[0].flatten().tolist(), "injection": inject[0].flatten().tolist()})
        print(f"  {label}: mixed{tuple(mixed.shape)} inject{tuple(inject.shape)} "
              f"inject_mean={inject.mean():.4f}")
    else:
        cases.append({"name": label, "combine": False, "input": x[0].flatten().tolist(),
                      "mixed": y[0].flatten().tolist(), "injection": []})
        print(f"  {label}: mixed{tuple(y.shape)}")

out.write_text(json.dumps({"hc_lowrank": cfg.hc_lowrank, "cases": cases}))
print("wrote", out)
