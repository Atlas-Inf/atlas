<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# GLM-5.3-Flash NVFP4 on two DGX Sparks

Build and run native Atlas with either `nvidia/GLM-5.3-Flash-NVFP4` or
`nvidia/GLM-5.3-Flash-FP4-Local`, **TP2/EP2**, FP8 `fp8_g128` target KV and FP32
persistent KDA state. FP4-Local describes the weight checkpoint; the target
KV cache remains FP8 for both choices.
Both ranks run the same image and settings over RoCE. DFlash2 is an explicit
option requiring separately obtained draft weights; the commands do not enable
native MTP or require a converted NVIDIA checkpoint.

The initial limit is **32,768 tokens per session and 32,768 aggregate KV tokens
with one plain-decode slot, or shared by four slots with DFlash2**, including
generated tokens. The actual negotiated pool can be smaller. This is a bounded
starting configuration, not a maximum-context claim. [Validation receipts](../../../../docs/porting/glm53-validation/README.md)
identify the measured revisions and profiles; the no-drafter option below is not
a claim of fresh output parity with the qualified DFlash profile.
The recipe explicitly reserves a 4100-row workspace with
`ATLAS_MAX_BATCH_TOKENS=4100`. The native sparse prefill path needs that capacity
even for one plain-decode slot; a 4096-token chunk plus one slot would otherwise
allocate only 4097 rows. With DFlash2, keep all four configured slots even when
benchmarking one active client.

## Build from this checkout

Stop existing model services on both Sparks before building or launching.
The measured setup started with idle GPUs, at least 96 GiB of host memory
available on each Spark, and no swap in use. Check these conditions again
before launching the pair.

On an ARM64 Spark, from the repository root:

```bash
docker build --no-cache \
  --build-arg ATLAS_GIT_SHA="$(git rev-parse HEAD)" \
  -f docker/gb10/glm-5.3-flash/nvfp4/Dockerfile \
  -t atlas-glm53-flash:local .
```

Use this same image on both hosts (build the same commit or transfer it with
`docker save` / `docker load`). The Dockerfile builds Atlas's native KDA and
sparse-MLA paths and includes their notices. No SparkGLM checkout, FlashInfer
package or external prefill library is needed. Both hosts need NVIDIA Container
Toolkit, a working RoCE link and sufficient free disk for the image build.

## Choose the checkpoint

Use the same complete checkpoint on both hosts:

| Checkpoint | Host directory example |
| --- | --- |
| Existing `nvidia/GLM-5.3-Flash-NVFP4` | `/absolute/path/to/GLM-5.3-Flash-NVFP4` |
| New `nvidia/GLM-5.3-Flash-FP4-Local` | `/absolute/path/to/GLM-5.3-Flash-FP4-Local` |

For FP4-Local, pin revision
[`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5).
`MODEL_DIR` must point to the checkpoint root containing its config, tokenizer,
safetensors index and complete nested `tensors/` directory. Preserve all paths
referenced by the index; mounting `tensors/` alone or flattening its files is
insufficient. The container sees the root at `/model` and tensor files beneath
`/model/tensors/`. No conversion or different kernel target is required. See
[FP4-Local loading and checkpoint terms](../../../../docs/porting/GLM_5_3_FLASH_FP4_LOCAL.md).

## Prepare both hosts

Use Bash in this repository's root on each Spark. Set these values for the
actual host; `HEAD_FABRIC_IP` is the head's **inter-Spark fabric address**, not
its HTTP/VPN address. `HCA` is the host's comma-separated RoCE device rails.
`MODEL_DIR` must contain the complete NVIDIA checkpoint. Snapshot symlinks must
resolve inside the mounted directory; use a materialized download if needed.

```bash
MODEL_DIR=/absolute/path/to/GLM-5.3-Flash-NVFP4
# For FP4-Local, use its complete pinned checkpoint root instead:
# MODEL_DIR=/absolute/path/to/GLM-5.3-Flash-FP4-Local
HEAD_FABRIC_IP=10.10.10.1
IFACE=enp1s0f0np0
HCA=rocep1s0f0,rocep1s0f1

# Explicit environment files are applied in this order on BOTH ranks.
DOCKER_ARGS=(
  --network=host --gpus=all --ipc=host --workdir=/
  --device=/dev/infiniband --ulimit=memlock=-1:-1 --cap-add=IPC_LOCK
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly"
  --env-file scripts/dev/glm53_native/qualification.env
  --env-file scripts/dev/glm53_native/rc2-candidate.env
  --env-file scripts/dev/glm53_native/rc4-candidate.env
  -e ATLAS_KV_MAX_BLOCKS=2048
  -e ATLAS_MAX_BATCH_TOKENS=4100
  -e "NCCL_SOCKET_IFNAME=$IFACE" -e "GLOO_SOCKET_IFNAME=$IFACE"
  -e "NCCL_IB_HCA=$HCA" -e "ATLAS_RDMA_RAILS=$HCA"
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_CROSS_NIC=0 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_IB_RETRY_CNT=7 -e NCCL_IB_TIMEOUT=22
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_BUFFSIZE=33554432
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_NCHANNELS=1
  -e NCCL_DMABUF_ENABLE=0 -e NCCL_NET_GDR_C2C=0 -e NCCL_NET_GDR_LEVEL=0
  -e NCCL_DEBUG=WARN -e RUST_LOG=info
)
SERVE_ARGS=(
  --kernel-target=glm-5.3-flash --world-size=2 --tp-size=2 --ep-size=2
  --master-addr="$HEAD_FABRIC_IP" --master-port=29610 --gpu-ordinal=0
  --max-seq-len=32768 --max-prefill-tokens=4096
  --block-size=16
  --kv-cache-dtype=fp8_g128 --ssm-h-dtype=f32 --ssm-rollback-mode=records
  --gpu-memory-utilization=0.88 --oom-guard-mb=12288
  --ssm-cache-slots=0 --ssm-checkpoint-interval=0 --enable-prefix-caching=false
  --swap-space-gb=0 --no-auto-swap --no-tui --lm-head-dtype=bf16
  --request-timeout=0 --tool-call-parser=poolside_v1
  --bind=127.0.0.1 --model-name=glm-5.3-flash
)
# Ordinary decode: override the inherited DFlash-specific switches together.
DRAFT_ENV=(-e ATLAS_GLM_DFLASH=0 -e ATLAS_GLM_MTP_LONG_CONTEXT=0
           -e ATLAS_DFLASH_COPY_DRAFTS=0)
DECODE_ARGS=(--max-batch-size=1 --max-num-seqs=1)
```

Plain decode uses one active/admitted sequence at 32K with FP8 KV. Merely
setting `ATLAS_GLM_C4_DECODE=1` does not enable this same profile at C4: that
separate lane requires BF16 KV and a smaller context, plus its other policy
settings. Use the DFlash2 option below for this recipe's four-slot profile.

These environment files retain explicit disabled experimental settings and the
conservative stream-ordered launch policy. They enable native KDA and sparse
prefill; KDA has FP32 state storage but BF16 internal recurrence. Keep the same
model and algorithm settings on both ranks. Host interface names and paths can
differ. Use the image's bundled libraries rather than host `LD_LIBRARY_PATH`.

### Optional native prefill optimizations

The four new prefill switches default **off** with either checkpoint. To test
all four, append this block after defining `DOCKER_ARGS` on **both ranks**:

```bash
DOCKER_ARGS+=(
  -e ATLAS_GLM_KDA_SIDE_PROJ_LT=1
  -e ATLAS_GLM_KDA_CONV_PLANES=1
  -e ATLAS_GLM_HC_PREFILL_FUSE2=1
  -e ATLAS_GLM_SP_GATHER_OVERLAP=1
)
```

Keep the resolved settings identical across ranks. Gather overlap uses four
blocks for `1`, or accepts `2` through `8` as an explicit block count; start
with `1`. Omitting this optional block leaves the existing recipe unchanged.
These paths adapt Jason McCartney / Enntity's work; see the
[prefill eligibility, source credits and validation requirements](../../../../docs/porting/GLM_5_3_FLASH_PREFILL.md).

### Optional DFlash2

To select the DFlash profile, replace the last two array assignments above with
this block on **both** hosts before launching. The target weights and chosen
DFlash2 weights have separate terms; Atlas does not bundle either checkpoint.

```bash
DRAFT_DIR=/absolute/path/to/compatible-GLM-5.3-Flash-DFlash2
DRAFT_ENV=(
  --mount "type=bind,src=$DRAFT_DIR,dst=/draft,readonly"
  -e ATLAS_GLM_DFLASH=1 -e ATLAS_GLM_MTP_LONG_CONTEXT=1
  -e ATLAS_DFLASH_COPY_DRAFTS=1
)
DECODE_ARGS=(--max-batch-size=4 --max-num-seqs=4
             --dflash --draft-model=/draft --dflash-gamma=8
             --dflash-window-size=2048 --mtp-gate=force)
```

The inherited draft optimizations remain those of the full profile. The
historical `MTP_LONG_CONTEXT` environment name and `--mtp-gate` flag also control
this DFlash lane; they do not select native MTP. Its 2048-token draft window does
not truncate the target's context. Do not pass `--speculative` for this recipe.

## Launch the pair

Stop any previous model on both GPUs first. Run the worker command, then the head
command in their prepared shells. Port 29610 must be reachable over the fabric.

**Worker / rank 1:**

```bash
docker run --name atlas-glm53-r1 "${DOCKER_ARGS[@]}" "${DRAFT_ENV[@]}" \
  atlas-glm53-flash:local serve /model "${SERVE_ARGS[@]}" \
  "${DECODE_ARGS[@]}" --rank=1 --port=18994
```

**Head / rank 0:**

```bash
docker run --name atlas-glm53-r0 "${DOCKER_ARGS[@]}" "${DRAFT_ENV[@]}" \
  atlas-glm53-flash:local serve /model "${SERVE_ARGS[@]}" \
  "${DECODE_ARGS[@]}" --rank=0 --port=18993
```

Only rank 0 serves clients, at `http://127.0.0.1:18993/v1`. In another head shell,
wait for both ranks to finish loading, then make a bounded request:

```bash
curl --fail-with-body http://127.0.0.1:18993/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":64}'
```

For network/VPN access, apply the [network exposure and authentication
settings](../../../../QUICKSTART.md#network-exposure) to rank 0. The fabric
address still belongs to the inter-rank transport. Stop both containers with
`docker stop atlas-glm53-r0` on the head and `docker stop atlas-glm53-r1` on the
worker; remove these stopped containers before reusing their names.

## Increasing capacity

The cap is `ATLAS_KV_MAX_BLOCKS × 16`, shared across all active sessions and
resident prefixes. Sharding does not create a second logical context pool to
add to the first. Inspect startup's actual allocated blocks and both hosts'
free memory after warmup. The CLI OOM reserve protects loading; it is not a
runtime host-memory watchdog.

After bounded requests pass, a separate capacity experiment can set
`ATLAS_KV_MAX_BLOCKS=8192` in `DOCKER_ARGS` on both hosts for a nominal 131,072-token
aggregate cap while retaining 32,768 per session. This change requires its own
memory and workload validation on the current build. The model metadata limit
is 1,048,576 positions per session, including output; that is not a promise the
chosen configuration will fit or retrieve reliably. See the [recorded context
experiments and limitations](../../../../docs/porting/GLM_5_3_FLASH_RC4.md#context-capacity)
before larger changes. Keep optional display-memory allocation and NVMe prefix
tiering outside this initial recipe.

[Source credits and licensing boundaries](../../../../docs/porting/GLM_5_3_FLASH.md#provenance-credits-and-licensing-boundaries)
remain part of the integration; numerical and historical results are scoped to
their recorded sources and settings.

## GPU results for this update

The existing NVFP4 checkpoint passed a fresh 32K-pool comparison: all 11
benchmark outputs matched with the four prefill switches off versus on.
Measured approximately 10K-token chat TTFT improved by 2.3–2.4%, while decode
was essentially unchanged. Core CUDA, tools/JSON and cancellation checks passed.
See [settings, results and limits](../../../../docs/porting/GLM_5_3_FLASH_PREFILL_RESULTS.md).

FP4-Local also passed six checkpoint GPU checks and both 32K-pool serving
profiles, including positive vision probes. All 13 compared answer/reasoning
pairs matched between prefill settings. These bounded results do not establish
a new maximum context.
A separate FP4-Local smoke test passed one 65,536-token random-input request
with a 98,304 session limit, 131,072 aggregate cap and four configured slots;
only one request was active. Above 32K, sparse attention uses Atlas's existing
fallback. See the [64K result and limits](../../../../docs/porting/GLM_5_3_FLASH_PREFILL_RESULTS.md#one-64k-random-input-request).
