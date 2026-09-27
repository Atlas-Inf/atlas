# EXL3 tools (Qwen3.8-Flash-Next)

Offline helpers for Atlas's EXL3 (exllamav3 trellis) support. None of them run at serve time.

**Build note:** the EXL3 kernels live in their own kernel target, `kernels/gb10/qwen3.8-flash-next/exl3/` (composed on `nvfp4/`). Build the server with `ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=exl3`, or `ATLAS_TARGET_QUANT='*'` for both quants. An nvfp4-only build refuses an EXL3 checkpoint at startup.

## Before serving: `convert_ngram_bf16.py` (interim)

**Serving an EXL3 checkpoint currently needs this step.**
- Atlas's PLE loader reads the n-gram table as BF16 row shards (`<ple prefix>.ngram_embedding.shard_<i>.weight`).
- An EXL3 checkpoint ships that table only in trellis form, in `ngram_embedding.safetensors`.

Run the converter once per model directory:

```
python3 bench/exl3/convert_ngram_bf16.py /path/to/exl3-model --workers 16
```

What to expect:
- **About 96 GiB of new files.** It writes `ngram_bf16_0.safetensors` through `ngram_bf16_7.safetensors` into the model directory. For Qwen3.8-Flash-Next that is about 96 GiB, decoded from the 37 GiB `ngram_embedding.safetensors` of the 4.05 bpw build. The BF16 table's size depends on the model, not the bpw. Check free disk space first.
- **No serve flags needed.** For an EXL3 checkpoint, Atlas adds every `*.safetensors` file that `model.safetensors.index.json` doesn't list to its weight map (`crates/spark-runtime/src/weights/side_files.rs`). The table is then read from NVMe at serve time, not held in GPU memory.
- **Safe to interrupt.** Each file is written as `<name>.partial` and renamed only when complete, so an interrupted run never leaves a truncated file under the final name. A re-run starts over.
- **Dependencies:** Python 3 and NumPy. `--files` (default 8) sets the number of output files; `--rows-per-job` (default 250,000) sets the work unit per worker.

**Interim.** Delete this script, this section and the `ngram_bf16_*` side files once M7 lands, when Atlas decodes EXL3 n-gram rows natively. It is tracked in [#65](https://github.com/Atlas-Inf/atlas/issues/65).

## Test data and references

| File | What it is |
|---|---|
| `reference.py` | Independent NumPy reference for the EXL3 trellis decode (mul1 codebook), transcribed from exllamav3 @ `6b84a21` |
| `st_io.py` | Minimal safetensors readers: local memmap and HTTP range |
| `make_fixtures.py` → `fixtures.json` | Cross-language fixtures for the Rust EXL3 CPU reference decoder's tests |
| `make_gpu_testdata.py` | Fetches whole EXL3 linears (bits 3–6) into one safetensors file for the `#[ignore]` GPU parity tests (`ATLAS_EXL3_TEST_DATA`) |
| `xcheck_fp8.py` | Checks `reference.py` against Qwen's FP8 release; reports only, commits nothing |
| `LICENSE` | MIT notice for the code transcribed from exllamav3; the rest of Atlas stays AGPL-3.0-only |
