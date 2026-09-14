"""Does the QSA indexer restrict anything below its budget?

block_topk = indexer_budget // indexer_compress_ratio, and the selection takes
min(block_topk, num_complete_blocks). If a sequence has no more complete blocks
than that, every block is selected and the mask cannot mask anything -- the
indexer is exactly a no-op and dense attention is numerically identical.

This checks that claim against HF's own module instead of reading it off the
source, at lengths either side of the threshold.
"""
import json, sys, pathlib, torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ckpt = pathlib.Path(sys.argv[1])
raw = json.loads((ckpt / "config.json").read_text())
tc = {k: v for k, v in raw["text_config"].items() if k != "model_type"}
# Probe the PUBLISHED indexer sizing, not the tiny one, since the threshold is
# what we care about. Everything else comes from the checkpoint under test.
for key, val in (("indexer_budget", 2048), ("indexer_compress_ratio", 4),
                 ("indexer_n_heads", 4), ("indexer_kv_heads", 1),
                 ("indexer_head_dim", 128)):
    tc[key] = val
cfg = Qwen4ExpTextConfig(**tc)
torch.manual_seed(5)

budget, ratio = cfg.indexer_budget, cfg.indexer_compress_ratio
block_topk = budget // ratio
threshold = block_topk * ratio
print(f"budget={budget} ratio={ratio} block_topk={block_topk} -> no-op at kv <= {threshold + ratio - 1}")

idx = Qwen4ExpTextQSAIndexer(cfg, 0).eval()
head_dim = cfg.indexer_head_dim

for seq in (16, 512, threshold, threshold + ratio, threshold + 4 * ratio):
    hidden = torch.randn(1, seq, cfg.hidden_size) * 0.4
    cos = torch.randn(1, seq, head_dim); sin = torch.randn(1, seq, head_dim)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).view(1, 1, seq, seq)
    with torch.no_grad():
        mask = idx(hidden, (cos, sin), causal, None)
    # A no-op means: every causally-visible token is still visible.
    visible_before = causal[0, 0]
    visible_after = mask[0, 0]
    restricted = (visible_before & ~visible_after).sum().item()
    print(f"  seq {seq:>5}: tokens masked out by the indexer = {restricted}"
          f"   ({'NO-OP' if restricted == 0 else 'RESTRICTS'})")
