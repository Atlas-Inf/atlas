# Atlas native sparse GPU checks

These ignored tests run the actual `AtlasCudaBackend`, checked Rust `Plan`, and
production `dispatch::execute` against the embedded GLM CUDA target. They do not
load a sparse-prefill shared library or model weights. They require the
nondefault `cuda,gpu-examples` features; ordinary CPU CI does not run them.

Build the GLM target with the normal per-source compiler flags, with
`ATLAS_SKIP_BUILD` unset. Build the test executable separately from execution:

```sh
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 \
  cargo test --locked -p spark-model --lib --features gpu-examples --no-run
```

Run only in an exclusive GB10 maintenance window, with external host-memory and
GPU-idle checks. Cap the process/container at 4 GiB RAM, no swap, two CPUs and a
300-second timeout. The fixture itself uses approximately 407 MiB of device
storage including redzones; the CUDA context and compiled modules are additional.
The test requires at least 4 GiB of GPU headroom. Run one test process at a time.

Use the emitted executable path and these filters with
`--ignored --nocapture --test-threads=1`:

- `analytic_sparse_attention_through_atlas_dispatch`: zero Q with zero, constant,
  and one-hot KV; all BF16 output values compared to analytic expectations. LSE
  compared to `log2(selected_count)` with a fixed eight-FP32-ULP test policy.
- `sampled_scaffold_comparison_through_atlas_dispatch`: explicit sampled fixture;
  reports deviations from the historical source-built scaffold for eight
  row/head pairs. These numbers are diagnostics, not independent math validation
  or a tuned acceptance gate.

Both tests independently check Q padding, E4M3 KV bytes/scales, sparse metadata,
input immutability, and every redzone. A CPU-only scalar reference test covers
E4M3 representable values, midpoint ties, signed zero and finite saturation.
No ignored GPU test has been run merely by compiling its executable.

## Preparing the sampled fixture

The audit sample bundle contains original BF16 query/selected-KV operands and
stored output/LSE, rather than pre-quantized data from the implementation under
test. Convert it on CPU using Torch and safetensors as **test preparation tools**:

```sh
python3 scripts/dev/glm_sparse_native/convert-sampled-fixture.py \
  /explicit/path/native-attention-samples.pt /explicit/path/samples.safetensors
export ATLAS_GLM_SPARSE_TEST_FIXTURE=/explicit/path/samples.safetensors
```

The converter uses `torch.load(weights_only=True, map_location="cpu")`, validates
all names/shapes/dtypes and finite values, and never overwrites an output. The
Rust reader limits the file to 20 MiB, checks the safetensors format marker and
exact tensor schema, and never interprets pickle or loads a model checkpoint.

The test reconstructs eight disjoint 2,048-token main selections in a 2,048-row
fixture, preserving each selected value's order and the original causal tail
positions. Shared tail values must agree byte for byte. Only the sampled
row/head outputs are comparable with the stored scaffold; other rows establish
padding, metadata, and memory-guard coverage. Sampled comparisons do not qualify
long context, capture/verification, full-model quality, or performance.
