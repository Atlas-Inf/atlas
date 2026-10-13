<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Native GLM-5.3 qualification recipe

This is an explicit environment file plus ordinary Atlas CLI commands, not a
launcher or converter. Build Atlas's GLM target from this repository, including
its native KDA operator and sparse-MLA kernels. Use the original NVIDIA
`GLM-5.3-Flash-NVFP4` snapshot and a separately obtained compatible DFlash2 draft
checkpoint. No other project's checkout, runtime adapter, shared prefill
library or deployment recipe is required. Review the chosen weights' terms
separately; this repository does not distribute weights.

The settings reproduce the native implementation tested at `451bbc21`, including
FP8 latent KV, FP32 persistent KDA state, DFlash and four sequence slots. Native
KDA uses BF16 internally despite FP32 state storage. Keep all explicit zeroes;
these disable other experimental paths. This is a qualification configuration,
not a claim that every supported model or flag combination has been validated.

## Build and host prerequisites

Use the [GLM Dockerfile](../../../docker/gb10/glm-5.3-flash/nvfp4/Dockerfile) or
its equivalent source build: CUDA 13.0, the committed Rust toolchain/lockfile,
CUTLASS `cf064d2e6bad2886238ac565b3b49007764f4939`, target
`ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4`.
For a host binary, install its CUDA/NCCL/RDMA shared dependencies through the
system loader or set host-specific library search paths outside this recipe.
NCCL must provide the symmetric-memory APIs used by Atlas (>=2.28).

Record the actual Atlas revision, image/binary digest, checkpoint revisions,
NVIDIA driver, CUDA and NCCL versions, network devices and link topology beside
each result. Matching flags does not make different driver/NCCL/fabric versions
numerically or operationally equivalent. The reference used two GB10/DGX Sparks
with direct RoCE; network tuning below must fit the actual fabric.

## Prepare each rank's shell

Use one terminal on each Spark. A clean shell prevents unrelated `ATLAS_*` or
NCCL settings from silently changing the recipe. Run this first, then configure
the variables in that shell (the command itself does not start a model):

```bash
env -i HOME="$HOME" PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash --noprofile --norc
```

Set these to actual paths/devices on **each** host before the next block:

- `ATLAS_ROOT`: this Atlas checkout, containing `jinja-templates/` and this recipe.
- `BINARY`: absolute path to that host's compiled native `spark` executable.
- `MODEL`: absolute path to the original NVIDIA checkpoint snapshot.
- `DRAFT`: absolute path to the separately selected DFlash2 checkpoint.
- `FABRIC`: the head/rank 0 address on the inter-Spark fabric, identical on both ranks.
- `IFACE`: this host's fabric network interface for NCCL/Gloo bootstrap.
- `HCA`: this host's comma-separated RoCE device rails; also used by Atlas RDMA.

Do not substitute the HTTP/VPN address for the inter-Spark fabric address unless
that interface actually provides the intended fabric. The local HTTP endpoint
below deliberately binds loopback; change exposure separately after validation.

Run this common setup on both ranks after setting those variables:

```bash
set -euo pipefail
: "${ATLAS_ROOT:?Set the Atlas checkout path}"
: "${BINARY:?Set the absolute native spark binary path}"
: "${MODEL:?Set the original NVIDIA checkpoint path}"
: "${DRAFT:?Set the selected DFlash2 checkpoint path}"
: "${FABRIC:?Set the head fabric IP address}"
: "${IFACE:?Set this host's fabric interface}"
: "${HCA:?Set this host's RoCE device rails}"
cd "$ATLAS_ROOT"
set -a
. scripts/dev/glm53_native/qualification.env
set +a
SESSION_TOKENS=32768
export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_IB_HCA="$HCA" ATLAS_RDMA_RAILS="$HCA"
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_CROSS_NIC=0
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_TIMEOUT=22
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=33554432
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_MAX_NCHANNELS=2
export NCCL_MIN_NCHANNELS=1
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_C2C=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=WARN
export RUST_LOG=info
```

The working directory matters: Atlas must discover its own GLM chat template in
`jinja-templates/`. The environment file contains no weight conversion or
machine-specific library paths. Set the same model algorithm values on both
ranks; interface names and checkpoint paths may differ by host.

## Start the pair

Start rank 1, then rank 0 in their prepared shells. These are ordinary standalone
Atlas commands. Port 29610 must be reachable on the fabric; local API ports are
18993/18994. The head serves requests at `http://127.0.0.1:18993/v1`.

Rank 1 / worker:

```bash
"$BINARY" serve "$MODEL" \
  --kernel-target=glm-5.3-flash \
  --world-size=2 \
  --tp-size=2 \
  --ep-size=2 \
  --max-prefill-tokens=4096 \
  --gpu-memory-utilization=0.88 \
  --oom-guard-mb=12288 \
  --ssm-h-dtype=f32 \
  --ssm-rollback-mode=records \
  --block-size=16 \
  --ssm-cache-slots=0 \
  --ssm-checkpoint-interval=0 \
  --enable-prefix-caching=false \
  --gpu-ordinal=0 \
  --swap-space-gb=0 \
  --no-auto-swap \
  --no-tui \
  --lm-head-dtype=bf16 \
  --request-timeout=0 \
  --tool-call-parser=poolside_v1 \
  --master-port=29610 \
  --bind=127.0.0.1 \
  --model-name=glm-5.3-flash-local-port \
  --max-batch-size=4 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8_g128 \
  --dflash \
  --dflash-gamma=8 \
  --dflash-window-size=2048 \
  --mtp-gate=force \
  --master-addr="$FABRIC" \
  --draft-model="$DRAFT" \
  --max-seq-len="$SESSION_TOKENS" \
  --rank=1 \
  --port=18994
```

Rank 0 / head:

```bash
"$BINARY" serve "$MODEL" \
  --kernel-target=glm-5.3-flash \
  --world-size=2 \
  --tp-size=2 \
  --ep-size=2 \
  --max-prefill-tokens=4096 \
  --gpu-memory-utilization=0.88 \
  --oom-guard-mb=12288 \
  --ssm-h-dtype=f32 \
  --ssm-rollback-mode=records \
  --block-size=16 \
  --ssm-cache-slots=0 \
  --ssm-checkpoint-interval=0 \
  --enable-prefix-caching=false \
  --gpu-ordinal=0 \
  --swap-space-gb=0 \
  --no-auto-swap \
  --no-tui \
  --lm-head-dtype=bf16 \
  --request-timeout=0 \
  --tool-call-parser=poolside_v1 \
  --master-port=29610 \
  --bind=127.0.0.1 \
  --model-name=glm-5.3-flash-local-port \
  --max-batch-size=4 \
  --max-num-seqs=4 \
  --kv-cache-dtype=fp8_g128 \
  --dflash \
  --dflash-gamma=8 \
  --dflash-window-size=2048 \
  --mtp-gate=force \
  --master-addr="$FABRIC" \
  --draft-model="$DRAFT" \
  --max-seq-len="$SESSION_TOKENS" \
  --rank=0 \
  --port=18993
```

The initial per-session limit is 32768 tokens, including generated output.
`ATLAS_KV_MAX_BLOCKS=8192` with block 16 caps the shared physical pool at 131072
KV tokens across sessions. It is an upper bound, not a guaranteed allocation:
Atlas also respects memory availability/utilization and the rank-minimum pool.
Four configured slots do not reserve four full contexts automatically. Inspect
startup's actual allocated block count and both hosts' free memory.

The explicit memory settings (`--gpu-memory-utilization=0.88`,
`--oom-guard-mb=12288`, no swap) reproduce the tested launch. Keep the guard and
monitor both hosts. Begin with bounded smoke requests and verify actual memory
headroom before increasing context; never interpret the cap as permission to
consume the entire machine. Do not load another model alongside this test.

## Qualified 524288-token override

After the 32K smoke and memory checks pass, stop the test pair cleanly and repeat
the common setup in fresh shells. **After** sourcing `qualification.env`, set
these two overrides on both ranks, then use the same rank commands:

```bash
export ATLAS_KV_MAX_BLOCKS=32768
SESSION_TOKENS=524288
```

These are the only algorithm/CLI context changes between the audited 32K and
524K launches. The native sparse operator's admission remains bounded and Atlas
uses its other sparse path outside that domain; this is not evidence that every
native sparse call itself handles 524K. The aggregate physical cap is 524288
tokens across all four slots, not 524288 per slot. Confirm the actual allocated
pool before submitting a near-limit request.

Native code 451bbc21 completed a single 523232-input-token retrieval request with
20/20 buried records recovered, two counterbalanced image smoke requests and
14 cancellation/state-lifecycle cases. These establish the tested cases only:
a single retrieval pass is not broad long-context quality validation, four
slots are not a four-times-full-context test, and the image smoke is not a full
vision benchmark. Preserve complete request/response and hardware evidence when
reproducing; benchmark throughput varies with workload and draft acceptance.

No optional `*_LIBRARY` variable is used. Historical external bridges have been
removed; source records and notices remain in
[the attribution archive](../../../third_party/nvidia_sparse_mla_sm120/history/README.md).
See the [validation report](../../../docs/porting/glm53-validation/README.md) for
matched content benchmarks, output differences and the tested limits.
