# Hy4 CPU equation fixtures

`torch_fixture.json` is the independent CPU PyTorch equation snapshot used by
`torch_tests.rs`. Regenerate it from any working directory with:

```sh
python3 /path/to/atlas/scripts/dev/hy4/generate_fixture.py
```

The harness requires PyTorch; Rust tests do not require Python or PyTorch.
The pinned equation reference is recorded in `../hy4_reference.rs`.
This snapshot does not establish parity with vLLM's fused GPU implementations.
