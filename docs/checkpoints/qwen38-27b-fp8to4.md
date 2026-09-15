# qwen38-27b-fp8to4 — derived-checkpoint fingerprint sidecar

Provenance record for the FP8→NVFP4 requantized derivative of
`nvidia/Qwen3.8-27B-NVFP4`. Lives in the repo so the checkpoint on a bench
host can be verified byte-for-byte against what was measured.

## Production

- Source: `nvidia/Qwen3.8-27B-NVFP4` snapshot
  `dbb8f445b3145f8a4c18ddc769f032d57d32867c` (HF cache on AzeezStrix).
- Tool: `scripts/requant_fp8_to_nvfp4_np.py` (numpy/vectorized; the scalar
  `requant_fp8_to_nvfp4.py` is ~hours on this checkpoint).
- Transform, per FP8 projection `[N,K]` with scalar F32 `weight_scale`:
  per-16-element block scale `s = amax/6` encoded E4M3, elements rounded to
  nearest E2M1 (sign bit 3), consecutive-pair nibble packing (element 2b
  low nibble) → `weight U8 [N,K/2]` + `weight_scale F8_E4M3 [N,K/16]` +
  `weight_scale_2 F32 [] = 1.0` + `input_scale` preserved. NVFP4 tensors
  and non-projection tensors pass through unchanged.
- Output: `HaloLoom/workspace/qwen38-27b-fp8to4/` on AzeezStrix
  (3 shards + index + copied configs/tokenizer files).

## Fingerprint

- Run record (ST-995, n=995 golden draw): `run-1789313837819983248.json`
  (replicated `run-1789298778720183887.json`) — overall 82.71 /
  normalized 79.58 vs the shipped checkpoint's 83.22 / 79.02 on the same
  binary/recipe → within the 0.4 noise band.
- Serve recipe: see `kernels/strix-hip/qwen3.8-27b/BENCH.toml` nvidia
  entry note.
- Serve binary: `spark_spec` sha256
  `e4fbc84f2f23105259a076d956d1f9653b8b5d565e000239c07ec2e5dd36963f`.
- Verify a copy before benching: `safetensors` headers must show
  `linear_attn.in_proj_qkv/in_proj_z/out_proj` and `self_attn.{q,k,v,o}_proj`
  as `U8 [N,K/2]` with `weight_scale F8_E4M3 [N,K/16]`; `mlp.*` stays
  U8/F8 NVFP4; `mtp.*` stays BF16 (excluded).
