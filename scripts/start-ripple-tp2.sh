#!/usr/bin/env bash
# Ripple-65B dense TP=2 bring-up on 2× GB10 (RoCEv2).
#
# Megatron TP over attention + dense FFN: Q/K/V/G and gate/up column-parallel,
# O and down row-parallel; each rank runs ~23 GB/token instead of ~46. The
# DFlash2 drafter stays replicated on both ranks (dflash_loader, tp_size=1).
#
# A/B against the single-node Strict launch: same checkpoint, same flags,
# minus --tp-size/--rank/--world-size/--master-*.
set -euo pipefail

MODEL="${1:-ricursiveAI/Ripple-65B-Preview-NVFP4}"
IMAGE="${IMAGE:-atlas-gb10:ripple-tp2}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
MASTER_PORT="29500"
PORT="8888"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"
KV_DTYPE="${KV_DTYPE:-bf16}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-1}"
# DFlash2 speculative decode on by default — the prose-vs-code acceptance
# swing is the headline A/B metric. DFLASH=0 gives the AR-decode floor.
DFLASH="${DFLASH:-1}"
DFLASH_WINDOW="${DFLASH_WINDOW:-16384}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

DFLASH_FLAGS=""
if [ "$DFLASH" = "1" ]; then
  # draft_model/gamma fall through from MODEL.toml [dflash].
  DFLASH_FLAGS="--dflash --dflash-window-size $DFLASH_WINDOW"
fi

echo "=== Atlas Ripple-65B TP=2 bring-up (RDMA-enabled) ==="
echo "Model:   $MODEL"
echo "Image:   $IMAGE"
echo "Head:    $HEAD_IP (rank 0, HTTP on $PORT)"
echo "Worker:  $WORKER_IP (rank 1)"
echo "TP/EP:   $TP_SIZE/$EP_SIZE    DFlash: $DFLASH    GPU mem: $GPU_MEM_UTIL"
echo ""

echo "Cleaning up old containers..."
sudo docker rm -f atlas-ripple-tp0 2>/dev/null || true
ssh "$WORKER_IP" "sudo docker rm -f atlas-ripple-tp1 2>/dev/null || true"

# RDMA device + io_uring flags — same as start-minimax-ep2.sh.
RDMA_FLAGS="--device=/dev/infiniband --cap-add=IPC_LOCK --cap-add=SYS_NICE --ulimit memlock=-1 --security-opt seccomp=unconfined"

# NCCL env — same validated RoCEv2 / IPv4 set as MiniMax M2.7 EP=2 and
# Qwen3.5-122B EP=2. TP's per-layer latency-bound all-reduces stress the
# same collective path EP's exchanges do.
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

# Rank 0 (head) — HTTP server + scheduler
echo "Starting rank 0 on $HEAD_IP..."
sudo docker run -d \
  --name atlas-ripple-tp0 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" serve "$MODEL" \
    --rank 0 \
    --world-size 2 \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --master-addr "$HEAD_IP" \
    --master-port "$MASTER_PORT" \
    --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --enable-prefix-caching \
    $DFLASH_FLAGS \
    $EXTRA_FLAGS \
    --oom-guard-mb 512

# Rank 1 (worker) — no HTTP listener
echo "Starting rank 1 on $WORKER_IP..."
ssh "$WORKER_IP" "sudo docker run -d \
  --name atlas-ripple-tp1 \
  --gpus all \
  --ipc=host \
  --network host \
  $RDMA_FLAGS \
  $NCCL_ENV \
  -e RUST_LOG=info \
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
    --max-batch-size 1 \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --kv-cache-dtype $KV_DTYPE \
    --enable-prefix-caching \
    $DFLASH_FLAGS \
    $EXTRA_FLAGS \
    --oom-guard-mb 512"

echo ""
echo "=== Both ranks starting ==="
echo "Monitor rank 0: sudo docker logs -f atlas-ripple-tp0"
echo "Monitor rank 1: ssh $WORKER_IP 'sudo docker logs -f atlas-ripple-tp1'"
echo "API endpoint:   http://$HEAD_IP:$PORT/v1/chat/completions"
