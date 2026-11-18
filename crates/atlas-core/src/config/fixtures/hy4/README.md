# Hy4-flash CPU contract fixtures

Registration only: there is deliberately no Hy4 forward implementation.

## Sources

* `config.json`: supplied `atakans/Hy4-flash-NVFP4` config (multimodal wrapper,
  nested `HYV4FlashForCausalLM` text model).
* `generation_config.json`: Tencent HF generation config. HF model API snapshot
  reports revision `d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`.
  `top_k=-1` means disabled; Atlas's registry representation is `top_k=0`.
* `index-contract.json`: exact names for layer 0 (dense+DSA), layer 1
  (latent MoE+DSA+Engram) and layer 2 (latent MoE+SWA), plus embedding,
  final norm and HC head. Experts 0 and 511 are retained to cover both ends
  of the per-tensor expert namespace. Counts cover all 42 layers.
  Includes SHA256 of the original index; no shard data is vendored.
* `tensor-contract.json`: expected checkpoint storage dtypes and shapes for
  those exact names. Source: config, ModelOpt metadata, checkpoint README and
  vLLM PR #68296 head `af127016aa1e14f1e597b546e14f5fdfbd05db11`
  (`hy_v4_flash/nvidia/{attention,moe,engram}.py` and inherited HYV4 HC).
  Observed header entries supersede these expectations; only the five
  names in `provenance.unverified_names` remain unverified.
* `header-observations.json`: exact storage dtypes/shapes for 208 of the
  213 representatives from eight currently present shards. Each observation
  is tied to its source shard and checked against the supplied index mapping.

## Header capture status

Read-only SSH to `spark@192.168.68.74`, directory
`~/models/hy4/Hy4-flash-NVFP4`, initially found no shards; the final capture
found **eight of fourteen shards**, covering all layer-type representatives.
No services were started/stopped and no tensor payloads were read.
The scanner reads only 8 bytes for the little-endian u64 header
length, then exactly that many JSON bytes (bounded to 64 MiB); never read
payload or download shards. Preserve source shard/revision with observations.

Headers establish FP32 storage for HC parameters, attention sinks, routing
correction bias and selected indexer tensors, despite the general BF16
description in the README. Global/input NVFP4 scales are rank-0 `[]`.
The frozen Engram remap has 120818 entries, not the padded model vocabulary.
The five unobserved tensors (lm_head, final norm and HC head) must not be
treated as header evidence; their source shard is not yet available.

## Extending the templates to all layers

DSA: 0,1,5,9,13,17,21,25,29,33,37,41; all others SWA.
Only layer 0 has the dense MLP; all others have 512 individually stored
experts, each with gate/up/down and four NVFP4 fields per projection.
Use layer 1's MoE template and layer 2's attention template independently:
Engram and attention kinds are not coupled.

Engram configuration uses **1-based** IDs 2,12,22,32; checkpoint layers
are 1,11,21,31. Sum each layer's 16 prime vocabularies for its table rows:
164994380, 164999008, 165004086, 165008942 respectively, each of width96.
Replace both table and per-row scale dimensions together. Key/value/norm
projection shapes are identical at all four sites.
DFly and visual namespaces are intentionally outside this first text-backbone
contract. They must not be silently admitted by a future loader.

## CPU checks

`ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p atlas-core --lib config::hy4`

The tests cover index-name coverage, per-layer expert counts, representative
NVFP4 packing, strict parser errors, layer maps, Engram IDs and explicit empty
Engram configuration. They do not establish numerical/model parity.
