#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# stream_bf16.sh DIR -- cut the GDN in_proj_qkv / in_proj_z / out_proj tensors (BF16) out of
# Qwen/Qwen3.8-27B at a pinned revision without holding the checkpoint: one shard at a time,
# download, check its sha256 against the LFS etag, keep the GDN tensors as DIR/gdn-<shard>,
# delete the shard. Peak disk is one shard (~3 GB) plus the cuts (11.1 GB in all). Needs
# curl and a python with torch + safetensors (PY, default: python3).
set -u
R=37561816e66b5b9153b1b64c10968c18c1e6f4a0
D=$1; PY=${PY:-python3}
mkdir -p "$D" && cd "$D"
base=https://huggingface.co/Qwen/Qwen3.8-27B/resolve/$R
[ -f model.safetensors.index.json ] || curl -sfL -o model.safetensors.index.json $base/model.safetensors.index.json
want="'.linear_attn.' in k and any(p in k for p in ('in_proj_qkv.', 'in_proj_z.', 'out_proj.'))"
for f in $($PY -c "
import json
m = json.load(open('model.safetensors.index.json'))['weight_map']
print(' '.join(sorted({f for k, f in m.items() if $want})))"); do
  [ -f gdn-$f ] && continue
  etag=$(curl -sIL $base/$f | grep -i "^x-linked-etag" | tail -1 | tr -d '"\r' | awk '{print $2}')
  curl -sfL --retry 3 -o $f.part $base/$f && mv $f.part $f || { echo "download failed: $f"; exit 1; }
  [ "$(sha256sum $f | cut -d' ' -f1)" = "$etag" ] || { echo "sha256 mismatch: $f"; rm -f $f; exit 1; }
  $PY -c "
from safetensors import safe_open
from safetensors.torch import save_file
f = safe_open('$f', 'pt')
save_file({k: f.get_tensor(k).contiguous() for k in f.keys() if $want}, 'gdn-$f.tmp',
          metadata={'source': 'Qwen/Qwen3.8-27B@$R/$f'})" && mv gdn-$f.tmp gdn-$f
  rm -f $f
  echo "done $f"
done
