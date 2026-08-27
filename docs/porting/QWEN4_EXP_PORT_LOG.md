# qwen4_exp serving port — running log

Working branch: `feat/qwen4exp-serve` (off `feat/ngram-embed` @ 2bcda1dc).
Goal: bring rsafier's #754 serving work onto this branch, keep this branch's
stronger artifacts, and have `qwen3.8-flash-next` / NVFP4 ready to test the
moment the GB10 is reachable again.

`reiner` is OFF the NetBird tailnet (peer `gx10-e3a3`, 100.105.184.175, Idle,
no WireGuard handshake, 0 B transferred). **Nothing in this log was measured on
hardware here.** Every claim is either (a) quoted from #754/#13 with its
source, or (b) verified by the local gate below.

## The local gate (no GPU, no nvcc)

The whole CUDA code path type-checks on an Apple-silicon box:

```sh
export CUDARC_CUDA_VERSION=13000   # stops cudarc's build.rs probing nvcc
export ATLAS_SKIP_BUILD=1          # build.rs writes a type-checkable PTX stub

cargo check --workspace --lib --bins           --no-default-features --features cuda
cargo check -p atlas-core -p spark-model -p spark-runtime -p spark-server \
            --all-targets                       --no-default-features --features cuda
cargo check -p atlas-core -p spark-model       --no-default-features --features metal
cargo test  -p atlas-core --lib                --no-default-features --features cuda
cargo fmt --all -- --check
cargo clippy -p atlas-core -p spark-model -p spark-runtime -p spark-server \
            --all-targets                       --no-default-features --features cuda
```

`spark-storage`'s io_uring / GDS examples and integration tests are Linux-only
and are outside the gate; that is pre-existing and unrelated to this port.

Three darwin arms were added to make this possible (commit `6d3c6a1d` +
follow-up): `posix_fallocate`, `posix_fadvise` and `O_DIRECT` have no macOS
equivalent. Production NVMe tiers stay Linux-only; no Linux behaviour changes.

## Status

| stage | state |
|---|---|
| 0. local gate established | DONE — 5/5 pass at `d75075f7` |
| 1. merge #754 + #746 | DONE — 7 conflicts resolved by inspection, 265 files |
| 2. kernel tree integrity | DONE — 5 symlinks resolve, no module collision |
| 3. config surface reconciled | DONE — one parser, both field families, 158 tests pass |
| 4. clippy clean | in progress |
| 5. microtest extended to the SERVING kernels | TODO — highest-value next item |
| 6. recipe + serve wiring reviewed | TODO |
| 7. `--generate N` off-by-one | TODO (reviewer-reported on #13) |
| 8. docs/CHANGELOG | TODO |

## Decisions taken (and why)

### Kept from THIS branch
* **`output_gate_type` string, not a bool.** `qwen3_ssm/init.rs` reads the raw
  value and refuses anything but `silu`/`sigmoid`. #754 passes a pre-computed
  `gdn_norm_sigmoid` bool, so an unknown activation silently takes the family
  default. Same swap, stricter failure mode.
* **The sigmoid twins live in `common/rms_norm.cu`** (module `norm`), not in a
  per-target shadow. #754's `gated_norm_sigmoid.cu` was byte-identical to ours
  and its own header warns that a fix to the SiLU originals "should be
  re-derived here". Dropped; every target now inherits the twins.
* **`weight_manifest` + `qwen4exp_preflight`** — 296,142 tensors, 0/0/0 in
  2.81 s and 372 MB RSS. #754 has only `ns_audit.py`, a dev script. Ours runs
  at load time.
* **`atlas_core::qwen4exp_reference`** — the CPU oracle, per-block 1.6e-7 to
  8.0e-7 against HF at real weights, full forward token-identical. This is what
  every GPU kernel is measured against.
* **`Qwen4ExpNgram`** — derives the multipliers/primes/offsets from config and
  asserts equality with the shipped buffers. #754 reads the buffers directly.
  Keeping both means the derivation and the checkpoint check each other.
* **`demand_paged_patterns`** — the never-resident rule, validated on two GB10s
  (78.19 GB pre-flight, ~90.4 GB peak, 47.68 GiB table excluded).
* **`split_ngram_parts`** as the field name — it is the key the checkpoint uses.
* **`finalize_config`** on the qwen4_exp path — #754's parser skipped it, so an
  NVFP4 checkpoint parsed as unquantized.
* **`cap_thinking_at_max_tokens` / `enable_loop_watchdog`** in `[behavior]`.

### Taken from #754
* **Layer placement.** mHC / PLE / QSA hang off the existing `qwen3_ssm` and
  `qwen3_attention` layers rather than a standalone `Qwen4ExpLayer`. That is
  what carries paged attention, CUDA graphs, prefix caching and C>1. Our
  scaffold layer is retired; its kernels and ops wrappers stay as oracles.
* **The four serving kernels**: `hyper_connection.cu` (FP32 highway, fused
  prefill path + 3-launch decode split), `ple.cu` (FP32 end to end),
  `qsa_indexer.cu` (decode + prefill selection), and the five per-file
  symlinks into `qwen3.6-35b-a3b`.
* **`[expected_absent]`** — what takes the fail-closed startup audit from 6
  unresolved kernels to 0. Per-file symlinks replace our whole-target
  `kernel_source` alias.
* **The measured `[behavior]` and sampling presets**, including all three
  sampled-quality fixes (`use_sampling_presets_for_core` in `[behavior]` —
  after a `[sampling.*]` header TOML swallows it as a preset key;
  `min_reasoning_floor_tokens = 0`; `honor_eos_inside_thinking = true`).
* **`norm_topk_prob = true`** — both branches found this independently.
* **`NgramRowCache::open_segmented`** — the 128 PLE shards are NOT contiguous
  (26.4 GB span across a 102.4 GB table), so one base offset reads
  wrong-but-valid rows silently.
* **The parser's stricter surface**: `weight_prefix`, model_type
  normalization, eos_token_id arrays, mandatory `layer_types`.

### Open in BOTH branches
* MTP — refused at pre-flight (issue #753 item I).
* Stacked expert layout — unreached by both published releases.
* Prefix-cache re-ingest for QSA raw indexer keys.
* Thinking-body quality at the card's temp 1.0 after `norm_topk_prob`
  (rsafier's last open thread on #754).

## Traps carried across from #754's comments (each cost a debug cycle there)

1. `hc_expand` must seed at MODEL layer 0, not `attn_layer_idx == 0` — on a
   3:1 interleave those differ by three layers.
2. `hc_head` fires at model layer 47; a `12 == 48` check never fires, and the
   mixer IS the final norm.
3. The GDN gated norm gates with **sigmoid** — wrong activation on 36 of 48
   layers, cos 0.80 before / 0.999990 after.
4. `norm_topk_prob` defaults TRUE in the reference and is OMITTED from the
   checkpoint json; serde's `false` cost 0.33–1.62 nats/token.
5. The terminal dense-BF16 GEMM arm must route to cuBLASLt — prefill
   283 → 747 tok/s.
6. The fused mHC collapse starves at `grid=[1]` on decode — split it:
   4.4 → 16.5 tok/s.
7. All QSA mid-path D2H must use `copy_d2h_on_stream`; `copy_d2h` orders
   against the DEFAULT stream while layers compute on the worker stream, and
   every parity test passed anyway because tests run on the default stream.
8. PLE's host hash must be hoisted out of the CUDA-graph capture region, and
   the batched path must consult the per-layer graph veto too.
9. An always-on debug diag (`diag_norm`: synchronize + copy_d2h, both errors
   swallowed) silently invalidated graph capture. Now behind
   `ATLAS_DIAG_V4_ALL_LAYERS`.
10. The engine's prefill chunk is **8193**, not 8192 — `ATLAS_PLE_MAX_TOKENS`
    must clear it (recipe carries ~9500 for >8K prompts).
11. Raw-row logit dumps are `model.vocab_size()` = 248077 wide, not the
    config's 248320.

## Kill switches / diagnostics available after this merge

`ATLAS_QSA_DISABLE`, `ATLAS_QSA_NO_PREFILL_SELECT`,
`ATLAS_QWEN4EXP_NO_HC_GEMM`, `ATLAS_DEBUG_NO_GRAPH`, `ATLAS_NO_GDN_FLA`,
`ATLAS_GDN_FLASHINFER`, `ATLAS_QWEN4EXP_DUMP`, `ATLAS_QWEN4EXP_PREFILL_PROF`,
`ATLAS_DUMP_HYPER_RMS`, `ATLAS_DUMP_LOGITS_PATH`, `ATLAS_PLE_MAX_TOKENS`,
`ATLAS_DIAG_V4_ALL_LAYERS`, `ATLAS_QWEN4EXP_NO_PLE`.

## First run on the box, when it is back

```sh
# 1. the gate that needs no checkpoint
cargo run --release -p spark-model --example qwen4exp_grouped_norm_microtest \
      --no-default-features --features cuda
# 2. the checkpoint is described exactly
cargo run --release -p atlas-core --example qwen4exp_preflight -- <ckpt>
# 3. serve
./serve_qwen4exp_tui.sh    # ATLAS_PLE_MAX_TOKENS >= 9500 for >8K prompts
```
