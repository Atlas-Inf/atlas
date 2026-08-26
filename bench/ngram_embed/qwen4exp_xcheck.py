"""Diff Atlas's qwen4_exp n-gram ids against HuggingFace's own module.

Run:  python bench/ngram_embed/qwen4exp_xcheck.py
Needs Python >= 3.10, `torch` (CPU is fine) and `transformers` >= 5.8.0.dev0.
No GPU and no model weights.

HF is the authority here: the Rust was written by reading its algorithm, so a
transcription error in the XOR mix or the shift semantics would not be caught by
any test Atlas writes about itself. This runs the real
`Qwen4ExpTextNGramEmbedding.forward` and diffs the ids it would gather with.

The embedding tensor is 320_001_536 x 160 (~51 GB), so nn.Embedding is stubbed
during construction -- the ids are computed before the lookup and are all we
need. Every buffer that feeds the ids (layer_multipliers, head vocab sizes,
head offsets) is built by the real __init__, untouched.
"""
import json, pathlib, random, subprocess, sys, tempfile
from unittest.mock import patch
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = ROOT / "test_data" / "qwen4_exp_flash_next_config.json"
TMP = tempfile.mkdtemp(prefix="qwen4exp-xcheck-")

raw = json.loads(CFG.read_text())
tc = {k: v for k, v in raw["text_config"].items() if k != "model_type"}
config = Qwen4ExpTextConfig(**tc)
EOS = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
print(f"eos={EOS} ngram_size={config.ngram_size} heads_per_ngram={config.heads_per_ngram} "
      f"seed={config.seed} base={config.ngram_vocab_size_base}")

class StubEmbedding(torch.nn.Module):
    def __init__(self, num, dim, **kw):
        super().__init__()
        self.dim = dim
        self.weight = torch.zeros(1, dim)
        self.captured = None
    def forward(self, ids):
        self.captured = ids
        return torch.zeros(*ids.shape, self.dim)

with patch.object(torch.nn, "Embedding", StubEmbedding):
    mod = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer_idx=1, ple_layer_index=0)

print("HF layer_multipliers      :", mod.layer_multipliers.tolist())
print("HF ngram_heads_vocab_sizes:", mod.ngram_heads_vocab_sizes.tolist())
print("HF ngram_heads_offsets[:4]:", mod.ngram_heads_offsets.tolist()[:4])
print("HF ngram_heads            :", mod.ngram_heads)

# Fixed edge cases plus random streams; seeded so a failure is reproducible.
random.seed(20260826)
streams = [
    [11, 523, 9001, 44, 130000, 7, 88, 4, 1, 2, 3],
    [EOS, 5, EOS, EOS, 9, 13, 21, EOS],
    [0, 1, 248319, EOS, 77, 12345, 99, 248318],
    [EOS],
    [7] * 12,
    [100, 200, 300, EOS, 400, 500, EOS, 600, 700, 800, 900, 1000],
    [248319, 248318, 248317, 1, 0, EOS, 2, 3],
]
for _ in range(25):
    n = random.randint(1, 24)
    streams.append([random.choice([EOS, EOS] + list(range(config.vocab_size)))
                    if random.random() < 0.15 else random.randrange(config.vocab_size)
                    for _ in range(n)])

ctx_len = config.ngram_size - 1
extended = [[EOS] * ctx_len + s for s in streams]
streams_file = pathlib.Path(TMP) / "streams_ext.json"
streams_file.write_text(json.dumps(extended))
def dump_atlas_ids(streams_file):
    """Run the Rust side. atlas-core defaults to the `cuda` feature, which
    needs nvcc; fall back to `metal` so this is runnable on a machine that has
    no CUDA toolchain (the ids are pure integer math either way)."""
    base = ["cargo", "run", "-q", "--release", "-p", "atlas-core",
            "--example", "qwen4exp_ngram_ids"]
    for extra in ([], ["--no-default-features", "--features", "metal"]):
        done = subprocess.run(base + extra + ["--", str(streams_file)],
                              cwd=ROOT, capture_output=True, text=True)
        if done.returncode == 0:
            return json.loads(done.stdout)
        last = done.stderr
    raise SystemExit(f"cargo run failed:\n{last}")


atlas = dump_atlas_ids(streams_file)

bad = 0
for i, s in enumerate(streams):
    ids = torch.tensor([s], dtype=torch.long)
    mod(ids, past_key_values=None)
    hf = mod.ngram_embedding.captured[0]          # [seq, heads]
    got = torch.tensor(atlas[i], dtype=torch.long).T[-len(s):]   # [seq, heads]
    if hf.shape != got.shape:
        print(f"  stream {i}: SHAPE {tuple(hf.shape)} vs {tuple(got.shape)}"); bad += 1; continue
    if not torch.equal(hf, got):
        d = (hf != got).nonzero()
        print(f"  stream {i}: {len(d)} differing ids, first at {d[0].tolist()}: "
              f"hf={hf[tuple(d[0])].item()} atlas={got[tuple(d[0])].item()}")
        bad += 1

total = sum(len(s) for s in streams) * mod.ngram_heads
print(f"\n{len(streams)} streams, {total} ids compared -> "
      + ("ALL MATCH" if bad == 0 else f"{bad} STREAMS DIFFER"))
sys.exit(1 if bad else 0)
