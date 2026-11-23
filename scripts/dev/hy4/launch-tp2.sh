#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

rank="${1:?usage: launch-tp2.sh 0|1 [--execute]}"
mode="${2:---print}"
case "$rank" in 0|1) ;; *) echo "rank must be 0 or 1" >&2; exit 2;; esac
case "$mode" in --execute|--print) ;; *) echo "expected --execute or --print" >&2; exit 2;; esac
model="${HY4_MODEL_DIR:-$HOME/models/hy4/Hy4-flash-NVFP4}"
image="${HY4_IMAGE:-atlas-hy4-flash:nvfp4}"
port=$((18995+rank))

cmd=(docker run --detach --name "atlas-hy4-r$rank" --restart=no
    --gpus all --network host --ipc host --cap-add IPC_LOCK
    --ulimit memlock=-1 --device /dev/infiniband
    --mount "type=bind,src=$model,dst=/model,readonly"
    -e ATLAS_FAST_LOAD=1 -e ATLAS_DEBUG_NO_GRAPH=1
    -e ATLAS_EP_PROTOCOL=v2 -e ATLAS_RDMA_ALLREDUCE=1
    -e ATLAS_RDMA_RAILS=rocep1s0f0,roceP2p1s0f0
    -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
    -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_DISABLE=0
    -e NCCL_IB_RETRY_CNT=7 -e NCCL_PROTO=Simple -e NCCL_BUFFSIZE=33554432
    -e NCCL_NET_GDR_LEVEL=0 -e NCCL_IB_ROCE_VERSION_NUM=2
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_NVLS_ENABLE=0
    -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=2 -e NCCL_CROSS_NIC=0
    -e NCCL_ALGO=Ring -e NCCL_NET_GDR_C2C=0 -e NCCL_DEBUG=WARN
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0
    -e NCCL_NET=IB -e NCCL_IB_TIMEOUT=22 -e NCCL_DMABUF_ENABLE=0
    "$image" serve /model --kernel-target=hy4-flash
    --world-size=2 --tp-size=2 --ep-size=2 "--rank=$rank"
    --master-addr=192.168.200.1 --master-port=29611 --gpu-ordinal=0
    --max-seq-len="${HY4_MAX_SEQ_LEN:-4096}" --max-prefill-tokens="${HY4_MAX_PREFILL_TOKENS:-256}"
    --max-batch-size=1 --max-num-seqs=1 --block-size=16
    --kv-cache-dtype=bf16 --attn-proj-dtype=bf16 --lm-head-dtype=default
    --gpu-memory-utilization=0.88 --oom-guard-mb=12288
    --enable-prefix-caching=false --swap-space-gb=0 --no-auto-swap
    --ssm-cache-slots=0 --ssm-checkpoint-interval=0
    --no-tui --request-timeout=0 --tool-call-parser=hy4
    --bind=127.0.0.1 "--port=$port" --model-name=hy4-flash-atlas)

printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "$mode" == --execute ]]; then
    [[ "$(uname -m)" == aarch64 ]] || { echo "GB10 ARM64 host required" >&2; exit 2; }
    [[ -f "$model/model.safetensors.index.json" ]] || { echo "checkpoint index missing" >&2; exit 2; }
    exec "${cmd[@]}"
fi
