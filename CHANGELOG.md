# Changelog

All notable changes to Atlas are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For per-release deep dives — kernel-level wins, the engineering history
behind specific subsystems — see the
[Atlas Spark Journey](docs/ATLAS_SPARK_JOURNEY.md).

## [Unreleased]

### Added
- `spark benchmark <list|run|history>` — the dashboard's benchmark suite as a
  headless subcommand, driving the same executor. Machine-readable output on
  stdout, progress on stderr; exit codes separate a broken harness (1) from a
  failed gate (2).
- `--version`, sourced from the packaged version so a build cannot report a
  version it was not packaged as.
- **N-gram scaled embeddings, LongCat flavour: id core and validity envelope**
  (`NgramDims`, `ngram_ids`) — groundwork for the family whose embedding tables
  are far larger than their backbones. Row ids are a polynomial rolling hash
  over token ids only, with document-boundary resets, checked bit-exact against
  a dependency-free reference (`bench/ngram_embed/`) at real LongCat-Flash-Lite
  dimensions. A config declaring the trio is validated, not assumed: table
  counts that would exceed the u32 the gather kernels index with, a
  `hidden_size` that does not divide by the table count, an accumulator that
  would wrap u64, and partially-declared trios are all refused at parse.
- **N-gram hashed embeddings, `qwen4_exp` flavour** (`Qwen4ExpNgram`) — the
  mechanism Qwen3.8-Flash-Next actually uses, which is a **different algorithm**
  from the LongCat one above rather than a parameterisation of it. It mixes by
  XOR of `shift_d · m_d` where the multipliers are seeded SplitMix64 draws, its
  per-head table sizes are consecutive PRIMES above a base, and its heads share
  one concatenated table addressed through per-head offsets. Feeding a LongCat
  id into a `qwen4_exp` table reads an unrelated row, so the two are separate
  types and a test asserts they do not coincide.

  Every derived quantity is reproduced bit-exact from `config.json` alone and
  pinned against the published `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint: the
  `layer_multipliers`, `ngram_heads_vocab_sizes` and `ngram_heads_offsets`
  buffers, and the `[2_500_012, 160] x 128` shard geometry. Two traps are
  encoded rather than commented — `ple_layer_ids` is ONE-indexed (`[2]` is
  decoder layer 1, which is where the checkpoint stores `layers.1.ple.*`), and
  `seed` is ABSENT from that `config.json`, so it defaults to 1234 instead of
  zero; a zero seed draws the wrong multipliers and hashes every token to an
  unrelated row without failing.

  Cross-checked against the reference rather than only against ourselves:
  `bench/ngram_embed/qwen4exp_xcheck.py` runs the real
  `Qwen4ExpTextNGramEmbedding.forward` and diffs its row ids against
  `cargo run -p atlas-core --example qwen4exp_ngram_ids`. 5408/5408 ids match
  over 32 streams. No GPU and no weights needed — the 51 GB embedding tensor is
  stubbed, every buffer that feeds the ids is not.
- **`qwen4_exp` weight manifest** (`atlas_core::weight_manifest`) — what a
  checkpoint must contain, derived from its config, so a loader's first job is
  written down separately from the loading and can be checked without the
  weights. Diffed against the published `Qwen3.8-Flash-Next-FP8`
  `model.safetensors.index.json`: of its 152,089 tensors, 333 are
  `model.visual.*` and 75,264 are FP8 `weight_scale_inv` siblings; the manifest
  covers the remaining **76,492 with zero missing and zero unexpected**, every
  scale attaches to a routed-expert weight it expects, and 1,653 shapes read
  from real safetensors headers match. `scripts/dev/verify_qwen4_exp_manifest.py`
  reproduces this against a checkpoint directory or a bare index.

  It earned its keep immediately, catching three wrong widths in the tiny
  development checkpoint (hyper-connections are `hc_count * hidden` wide, not
  `hidden`; `q_proj` is 2x for the gate; the indexer's `index_qk_proj` is
  `(n_heads + kv_heads) * head_dim`).

  Quantization is described too, and **both published releases now match
  exactly**: `Qwen/…-FP8` at 151,756 tensors and `RadixArk/…-NVFP4` at 296,142,
  each with zero missing and zero unexpected against their published indexes.
  NVFP4 repacks the weight as well as adding siblings — `[2560, 640]` is stored
  U8 `[2560, 320]`, two FP4 values per byte.

  Three things had to be handled that the tensor names do not reveal:
  `modules_to_not_convert` was going unparsed (HF's FP8 spelling, 943 literal
  entries); ModelOpt's ignore globs need `*` to span dots while HF's list has no
  globs at all; and the 128 n-gram shards are FP8 behind a single shared
  `weight_scale`, so they take no per-block sibling despite being absent from
  the ignore list. Expert layout is a fourth: HF stores `gate_up_proj` stacked,
  quantizers split it per `nn.Linear`, and `Qwen4ExpLayout` expresses both.
- **`qwen4_exp` config parsing.** The published `Qwen3.8-Flash-Next-FP8`
  `config.json` now parses (vendored whole into `test_data/`). Two of its
  defaults are absent rather than stated and both fail silently: `norm_topk_prob`
  (HF true, serde false — skips top-K renormalisation) and `seed` (HF 1234, a
  zero would hash every n-gram token to an unrelated row).

  Parsing is not serving — and serving landed separately, below.

- **Qwen3.8-Flash-Next (`qwen4_exp`) SERVES on a single GB10.** The port from
  Avarok #753/#754 (rsafier) merged onto this branch, with its #746 base: the
  low-rank multi-hyperconnection highway on all 48 layers, PLE n-gram
  injection at model layer 1 served by row off NVMe, the QSA indexer selecting
  on decode and on prefill, the 512-expert MoE, gated-Q attention on the
  production paged path, the ViT tower, CUDA graph capture, prefix caching and
  C>1 concurrency. Measured on a GB10: decode 16.5 tok/s, prefill 747 tok/s at
  a 2191-token prompt, ~90.4 GB peak with ~27.8 GB free.

  What makes it fit is one exclusion: the 47.7 GiB n-gram table is never
  resident and is read by row (`NgramRowCache::open_segmented`, because the 128
  shards are not contiguous — they span 26.4 GB of a 102.4 GB table, so a
  single base offset would read wrong-but-valid rows silently). Without it the
  same run refuses at 163.64 GB.

  Refused rather than half-wired: MTP (`--speculative` fails at pre-flight) and
  the stacked expert layout, which neither published release uses.

  Six defects in this family are silent by construction, so each one is pinned
  by a test rather than a comment: the GDN output gate is a **sigmoid**, not
  SiLU (wrong on 36 of 48 layers, cos 0.80 before / 0.999990 after);
  `norm_topk_prob` defaults TRUE in the reference and is omitted from the
  checkpoint (0.33–1.62 nats/token); `ple_layer_ids` are ONE-indexed; the PLE
  conv is dilated by `ngram_size`, so its state is 9 steps, not 3; the
  hyper-connection norms are offset-from-1 and GROUPED per stream; and the
  stream reduction is a mean, not a sum.

- **GPU parity for the qwen4_exp serving kernels against the in-process CPU
  oracle** (`ops/qwen4exp_oracle_tests.rs`). The existing gates need a 126 GiB
  checkpoint, `transformers` and generated fixture bins before they say
  anything; these drive the same kernels against
  `atlas_core::qwen4exp_reference` — itself checked against HuggingFace at real
  weights — so a clean checkout can gate the mHC collapse (at T=1/64/96, which
  is what selects the split and GEMM paths), the trunk entry (sentinel-checked),
  the residual fold and the whole PLE chain with one command and no downloads.
  Every check carries a control that must fail.

### Fixed
- **The mHC highway is FP32 for both families.** `hc_streams`' element size was
  branched per family — f32 for DeepSeek-V4's `hc_mult`, bf16 for `hc_count` —
  which was correct while this tree's own bf16 kernels read it and wrong for
  the kernels that now serve, every one of which declares the buffer `float*`.
  Sizing it bf16 under them does not crash: it reads the wrong half of every
  value, on all 48 layers. Now unconditional, with all three config fillings
  pinned to the same size.
- **`qwen4exp_forward --generate N` appended N+1 tokens**, not N — reported and
  reproduced twice on a GB10 during independent validation. `--generate 0`
  keeps its own meaning: one pass, print the argmax, diff the fixture, append
  nothing.
- **The qwen4_exp parser skipped `finalize_config`**, which is where
  `quantization_config` is read off the top level of `config.json`. ModelOpt
  writes it beside `text_config`, so serde on `text_config` alone never saw it
  and an NVFP4 checkpoint parsed as unquantized.
- **Benchmark runs no longer overwrite each other.** History files were named by
  whole seconds, so two runs of the same benchmark within the same second
  silently destroyed the first. Records are now keyed by nanosecond with an
  explicit collision guard.
- Run history records the parameters, target, source and version alongside the
  result. Previously only the result was stored, so a number could not be
  attributed to a configuration or reproduced. Pre-existing files still load.

### Added

- DeepSeek-V4-Flash support on GB10: native MXFP4 (E8M0) routed-expert
  loading (transcode-free — no MXFP4→BF16→NVFP4 double-quant) plus the
  Phase-K E8M0 GEMM kernels, end-to-end. (#293)
- `/v1/completions` legacy-API parity: `echo`, integer `logprobs` (four
  parallel-array `CompletionLogprobs` block), `n`, `stream_options`, and
  accepted-but-ignored `user`/`suffix`/`best_of`; prompt-position logprob
  collection during prefill. (#291)
- Native U8 NVFP4 loading for pre-quantized checkpoints. (#257)
- Holo-3.1-35B-A3B / Holo-3.1-0.8B / Ornith-1.0-9B model support on GB10
  (sm_121): hybrid Gated-DeltaNet + full-attention + (256-expert MoE | dense
  FFN) + Qwen3-VL vision tower. Brings CUTLASS Sm120 NVFP4 grouped MoE, FLA
  chunked-scan GDN prefill + wmma DV-block decode, cuBLASLt/CUTLASS attention
  projections, kernel-batched co-dispatch prefill, radix-KV + Marconi
  SSM-snapshot prefix caching, and self-relative auto KV budget. (#203)
- GEMM-based Qwen3-VL ViT attention kernel (tensor-core SDPA replacing the
  warp-per-query kernel) + tensor-core ViT block GEMMs + batched multi-image
  forward — ~2× image-request TTFT on GB10. (#202)

### Fixed

- SSM snapshot eviction is now recency-only: the hit-weighted score was
  pinning fossil anchors and inflating warm TTFT; the pure-LRU/winner-only
  policy restores warm-TTFT parity with llama.cpp. (3d8130d0)
- 35B agentic-wall recipe: SSM tail-protect brings webserver_ok
  Σ(wall_time) from 2765s to 1364s (<1500s gate). (#278)
- Weight-only NVFP4 (W4A16) checkpoints now load. llm-compressor
  `nvfp4-pack-quantized` with `input_activations: None` ships no static
  activation scale; the loader previously required `input_global_scale` and
  failed (e.g. `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4`). The
  field is loaded-but-unused (activations are quantized dynamically), so it is
  now optional. W4A4/W4A8 checkpoints are unaffected. (#203)
- `--gpu-memory-utilization` now enforces a hard ceiling on total GPU
  memory (weights + buffers + KV cache + reserves), matching the vLLM /
  sparkrun convention.  Previously the fraction was applied only to
  post-weight free memory, causing the KV cache to over-allocate by
  20-27 GB when values below the ~0.88 default were used.  This blocked
  multi-service co-residency on shared-memory systems (e.g. DGX Spark
  GB10).  The flag now behaves as documented: `0.50` on a 120 GB device
  caps Atlas at ~60 GB total.  (#180)

## [0.1.0] — 2026-05-06

Initial public release. Atlas is a pure-Rust LLM inference engine
targeting NVIDIA GB10 (DGX Spark, SM121) with twelve hand-tuned
(Hardware × Model × Quantization) targets.

### Added

- Pure-Rust runtime — no Python, no PyTorch — for hybrid Attention +
  SSM/GDN/Mamba-2 architectures with NVFP4 / FP8 / BF16 quantization.
- 35 hyperoptimized CUDA kernels per target, compiled to PTX and
  embedded in the binary at build time. Multi-model image dispatches
  the right kernel set at startup from `config.json`.
- OpenAI- and Anthropic-compatible HTTP API (`/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, `/v1/models`, `/v1/conversations`,
  `/tokenize`, `/detokenize`, `/health`, `/metrics`).
- Tool calling with grammar-constrained decoding (Hermes,
  Qwen3-Coder, Mistral, MiniMax-XML formats).
- MTP speculative decoding (K=2 pipelined verify), self-speculative
  layer-skipping, and N-gram speculative decoding.
- Prefix caching: radix-tree (RadixAttention) + SSM snapshot cache
  (Marconi-style). 10× warm-cache TTFT reduction.
- KV cache dtypes: BF16, FP8, NVFP4, turbo3, turbo4. Optional
  per-layer high-precision overlay (`--kv-high-precision-layers`).
- Multi-GPU expert parallelism (EP=2 over RoCEv2) for models that
  exceed a single GB10's weight budget (122B-class, MiniMax M2.7).
- Vision encoder (Qwen3-VL, Qwen3.6 ViT).
- High-speed NVMe KV swap (sliding-window, io_uring) for
  long-context decoding past the HBM cap.
- Bearer-token authentication (`--require-auth` +
  `--auth-tokens-file`), constant-time validated. Default bind is
  `127.0.0.1`; `--bind 0.0.0.0` warns when used.
- Twelve supported (GB10, model, quant) targets across Qwen3.5 /
  Qwen3.6 / Qwen3-Next / Qwen3-VL / Gemma-4 / Mistral-Small-4 /
  MiniMax-M2.7 / Nemotron-H families.
- mdBook documentation at `book/src/`, rustdoc at `target/doc/`,
  Docker image `azeezish/atlas-gb10:latest`.

### Engineering notes

For the kernel-level perf history — long-context regression sweeps,
the parking_lot migration, the libcuda + libnccl CI stubs, the
multi-stage scheduler refactor — see
[`docs/ATLAS_SPARK_JOURNEY.md`](docs/ATLAS_SPARK_JOURNEY.md) and the
[`book/`](book/) chapters under `deep-dives/`.

[Unreleased]: https://github.com/Atlas-Inf/atlas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Atlas-Inf/atlas/releases/tag/v0.1.0
