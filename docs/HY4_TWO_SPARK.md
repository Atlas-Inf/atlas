# Hy4-flash NVFP4: first eager TP2/EP2 run

**Proposed operator commands, not an executed deployment.** No Spark was
accessed during implementation. PTX compilation is not forward-pass or
collective validation. Start with one request, short text and greedy decoding.
The scalar correctness-first kernels will be slow; this is not a throughput
configuration.

## Preconditions

* Same source commit and ARM64 Docker image on both Sparks; complete local
  `~/models/hy4/Hy4-flash-NVFP4` on **both** hosts, including tokenizer files
  and all indexed shards. The script mounts the checkpoint read-only.
  `HY4_MODEL_DIR` and `HY4_IMAGE` can override those paths/tags.
* The local index contains all 253640 tensor names required by the text
  loader. Cutover header reads on both Sparks verified all 14 shard lengths
  and the five previously unobserved final norm/head tensors. This NVFP4
  checkpoint stores the LM head and HC-head parameters as FP32; final norm
  is BF16. Their shape/dtype is checked at real load.
* `.75` is rank 0, direct-link `192.168.200.1`; `.74` is rank 1,
  direct-link `192.168.200.2`. Confirm the interface/HCA names with the
  operator before launching. The names and RDMA/NCCL settings come from
  `~/scratch/hy4/glm53-r{0,1}-inspect.json`.
* The old GLM containers/front end remain stopped as described in
  `~/scratch/hy4/RESTORE.md`. These commands neither remove nor start them.
  Hy4 uses new container names, ports 18995/18996 and rendezvous port 29611.
* Enough RAM for TP-local Engram FP8 rows, locally owned whole experts,
  replicated attention/HC/head weights, loader staging, transposed attention
  weights and eagerly allocated caches. Real peak residency is unmeasured.
  Missing/malformed tensors and insufficient capacity fail rather than using
  synthetic weights or skipping a layer.

## Build on each Spark

From the same checkout on **each** Spark (or transfer the identical built
ARM64 image yourself):

```bash
cd ~/repos/atlas
git fetch origin devin/1795013508-hy4-flash
git switch devin/1795013508-hy4-flash
git pull --ff-only origin devin/1795013508-hy4-flash
docker build --no-cache -f docker/gb10/hy4-flash.Dockerfile \
  --build-arg ATLAS_GIT_SHA="$(git rev-parse HEAD)" \
  -t atlas-hy4-flash:nvfp4 .
```

This model-specific Dockerfile compiles only `gb10/hy4-flash/nvfp4` and
installs NCCL development files in the build stage as well as runtime NCCL
(2.28.9 for CUDA 13.0). The builder and runtime upgrade the base image's held
NCCL packages together to the same pinned version. Host packages are unchanged.

## Launch commands

On **`.75` / rank 0**, in one terminal:

```bash
cd ~/repos/atlas
bash scripts/dev/hy4/launch-tp2.sh 0 --execute
docker logs --follow atlas-hy4-r0
```

On **`.74` / rank 1**, in another terminal, promptly after rank 0 starts:

```bash
cd ~/repos/atlas
bash scripts/dev/hy4/launch-tp2.sh 1 --execute
docker logs --follow atlas-hy4-r1
```

Omitting `--execute` only prints the fully expanded Docker command. The
script never SSHs, starts/stops other services, downloads weights or builds
an image. A pre-existing `atlas-hy4-rN` container is an explicit Docker error;
inspect it instead of automatically replacing it.

## Smoke request, after both ranks report ready

From `.75` only:

```bash
curl --fail-with-body http://127.0.0.1:18995/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash-atlas","messages":[{"role":"user","content":"Reply with the word hello."}],"temperature":0,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

The API binds loopback; reuse the operator-managed front end only after this
smoke test, not automatically. Rank 1 is a collective worker, not a second
independent endpoint. Do not change one rank's topology or RDMA environment
without changing the other.

## Deliberate limits

* Text only. The dedicated model runs accepted tokens eagerly; it has no
  graph capture, DFly, DFlash, MTP, self-speculation, LoRA or vision support.
  Unsupported factory modes fail. No speculative state is committed.
* TP and EP are independently explicit (`2` each). Attention shards heads;
  Engram shards complete hash heads and reduces key/value projections; MoE
  owns complete experts. Business logic uses `spark_comm::CommBackend`.
* BF16 KV cache only, unquantized checkpoint LM head with FP32 accumulation
  (this NVFP4 checkpoint stores FP32 weights), unquantized BF16 attention projections.
  Do not copy GLM's FP8-cache or head-quantization flags.
* No prefix-cache reuse or NVMe swapping in this launch. Per-request token
  history carries bigram/trigram lookback across chunked prefill and decode;
  allocation/free isolates histories. Prefix replay recomputes, not reuses
  an incomplete Engram lookback snapshot.
* The HF chat template preserves native `:opensource` markers in prompts,
  with MiniJinja-compatible constant marker construction and Atlas's
  disabled-effort spelling mapped to HF `no_think`. Decode maps native
  atomic reasoning/tool markers to existing parsers. Hy4's parser has no
  constrained-decoding grammar; do not assume strict tool-choice enforcement.
* CPU oracles and mock lifecycle/ABI tests do not establish CUDA numerical
  parity, real checkpoint payload validity, NCCL/RDMA correctness, memory
  fit, model quality or performance. Those are the purpose of the first
  authorized Spark run, not results claimed here.
