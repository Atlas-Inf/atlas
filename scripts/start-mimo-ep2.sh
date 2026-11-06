#!/usr/bin/env bash
# Multi-node MiMo-V2.6-Flash bring-up on 2x GB10 (EP=2), text backbone.
#
# The checkpoint is ~173 GB (native MXFP4 experts, FP8 attention), so each
# rank holds half of the 256 experts (~89 GB). Phase 1 limits:
#   * --kv-cache-dtype bf16 only (the QK192/V128 attention kernels read a
#     BF16 paged cache; the loader rejects other KV dtypes)
#   * --speculative is optional (SPECULATIVE=1; the 3 MTP modules run on rank 0)
#   * text only (vision / audio towers are skipped at load)
#   * --max-batch-size 1 by default (MAX_BATCH); raise once single-stream output is clean
#
# The image must be built from a tree that contains kernels/gb10/mimo-v2.6-flash
# (docker/gb10/Dockerfile builds every target, ATLAS_TARGET_MODEL=*) so the
# MiMo attention kernels and the E8M0 MoE GEMMs are compiled in.
set -euo pipefail

MODEL="${1:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
IMAGE="${IMAGE:-atlas-gb10:mimo-ep2}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PORT="${PORT:-8888}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
KV_DTYPE="${KV_DTYPE:-bf16}"
MAX_BATCH="${MAX_BATCH:-1}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-2}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
SPECULATIVE="${SPECULATIVE:-0}"
NUM_DRAFTS="${NUM_DRAFTS:-3}"
if [ "$SPECULATIVE" = "1" ]; then
  EXTRA_FLAGS="--speculative --num-drafts $NUM_DRAFTS $EXTRA_FLAGS"
fi

echo "=== Atlas MiMo-V2.6-Flash EP=2 bring-up (RDMA-enabled) ==="
echo "Model:   $MODEL"
echo "Image:   $IMAGE"
echo "Head:    $HEAD_IP (rank 0, HTTP on $PORT)"
echo "Worker:  $WORKER_IP (rank 1)"
echo "GPU mem: $GPU_MEM_UTIL    Max seq: $MAX_SEQ_LEN"
echo ""

# Clean old containers
echo "Cleaning up old containers..."
sudo docker rm -f atlas-mimo-ep0 2>/dev/null || true
ssh "$WORKER_IP" "sudo docker rm -f atlas-mimo-ep1 2>/dev/null || true"

# RDMA device and capability flags. SYS_NICE is required by io_uring's
# IORING_SETUP_SQPOLL (kernel ≥ 5.13) used by --high-speed-swap; the
# default Docker seccomp profile blocks io_uring_* syscalls so we run
# unconfined for the storage path.
RDMA_FLAGS="--device=/dev/infiniband --cap-add=IPC_LOCK --cap-add=SYS_NICE --ulimit memlock=-1 --security-opt seccomp=unconfined"

# NCCL env — same as Qwen3.5-122B EP=2 (RoCEv2 / IPv4, GB10-safe)
NCCL_ENV="\
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_IB_RETRY_CNT=7 \
  -e NCCL_NET_GDR_LEVEL=0 \
  -e NCCL_NET_GDR_C2C=0 \
  -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_CUMEM_HOST_ENABLE=0 \
  -e NCCL_PROTO=Simple \
  -e NCCL_ALGO=Ring \
  -e NCCL_BUFFSIZE=33554432 \
  -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_DEBUG_SUBSYS=INIT,NET"

# Start rank 0 (head) — HTTP server + scheduler
echo "Starting rank 0 on $HEAD_IP..."
sudo docker run -d \
  --name atlas-mimo-ep0 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  ${ATLAS_PROFILE_FIRST:+-e ATLAS_PROFILE_FIRST=$ATLAS_PROFILE_FIRST} \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" serve "$MODEL" \
    --rank 0 \
    --world-size 2 \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    --bind 0.0.0.0 \
    --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size "$MAX_BATCH" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-prefix-caching \
    $EXTRA_FLAGS \
    --oom-guard-mb 512

# Start rank 1 (worker)
echo "Starting rank 1 on $WORKER_IP..."
ssh "$WORKER_IP" "sudo docker run -d \
  --name atlas-mimo-ep1 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  ${ATLAS_PROFILE_FIRST:+-e ATLAS_PROFILE_FIRST=$ATLAS_PROFILE_FIRST} \
  -v \"\${HOME}/.cache/huggingface:/root/.cache/huggingface\" \
  $IMAGE serve $MODEL \
    --rank 1 \
    --world-size 2 \
    --tp-size $TP_SIZE \
    --ep-size $EP_SIZE \
    --master-addr $HEAD_IP \
    --master-port $MASTER_PORT \
    --port 0 \
    --max-seq-len $MAX_SEQ_LEN \
    --max-batch-size $MAX_BATCH \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --kv-cache-dtype $KV_DTYPE \
    --enable-prefix-caching \
    $EXTRA_FLAGS \
    --oom-guard-mb 512"

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: sudo docker logs -f atlas-mimo-ep0"
echo "Monitor rank 1: ssh $WORKER_IP 'sudo docker logs -f atlas-mimo-ep1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"
