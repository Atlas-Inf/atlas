# Atlas Spark — Qwen3.8-Flash-Next NVFP4 on GB10
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.8-flash-next/nvfp4/Dockerfile -t atlas-q4e .
#
# Run (the checkpoint must be a LOCAL bind mount — see "NVMe, not NFS" below):
#   docker run --gpus all --ipc=host -p 8889:8889 \
#     -v /path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4:/ckpt:ro \
#     atlas-q4e serve \
#       --model-from-path /ckpt \
#       --model-name qwen4exp \
#       --kernel-target qwen3.8-flash-next \
#       --bind 0.0.0.0 --port 8889 \
#       --max-seq-len 8192 \
#       --kv-cache-dtype bf16 \
#       --gpu-memory-utilization 0.90 \
#       --fast-load-prefetch-shards \
#       --enable-prefix-caching
#
# ── Four flags that are NOT optional on this model ────────────────
#
# --kernel-target qwen3.8-flash-next
#     Selects the 177 kernels this architecture needs (mHC highway, PLE tower,
#     QSA indexer, the sigmoid GDN gated-norm twins). Without it the startup
#     kernel audit fails closed rather than serving something wrong.
#
# --kv-cache-dtype bf16
#     The checkpoint ships NO k_scale/v_scale, so the FP8 KV default falls back
#     to scale 1.0 and silently clips BF16 into E4M3's [-448, 448]. That costs
#     coherence at long context, where it is hardest to attribute. Matches
#     `kv_cache_dtype: bf16` in the vendored recipe.
#
# --gpu-memory-utilization 0.90
#     BF16 KV needs the headroom. At 0.80 the model loads and then fails with
#     "No memory left for KV cache" (93 GB pre-KV + 6.6 GB reserve vs a 95.7 GB
#     budget). 0.90 leaves ~7.5 GB, which is 20512 blocks x 12 layers — only
#     the 12 full-attention layers hold KV; the 36 GDN layers hold none.
#
# ATLAS_PLE_MAX_TOKENS (set below)
#     Sizes the PLE scratch. It defaults to 2048, and a 2191-token prompt
#     already exceeds that. Set at or above --max-seq-len; above 8192 the
#     engine's prefill chunk is 8193, so use 9500.
#
# ── NVMe, not NFS ────────────────────────────────────────────────
#
# The PLE n-gram table is 51.2 GB of F8_E4M3 across 128 shards in 10 files and
# is DEMAND-PAGED off disk by row, never resident: a token touches 16 rows
# (~2.6 KB) regardless of sequence length, read with O_DIRECT into a pinned,
# GPU-addressable arena. That makes the mount's random-read latency a direct
# per-token cost. Bind-mount local NVMe. A network mount will serve correct
# tokens slowly, which reads as a model problem rather than a storage one.
#
# On-disk resident is ~78 GB with ~90 GB peak, so a 128 GB GB10 fits it with
# the table paged rather than loaded.

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=qwen3.8-flash-next
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      (echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" && exit 1)

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY LICENSE /LICENSE
COPY README.md /README.md

ENV RUST_LOG=info
# See the note above: the default of 2048 is exceeded by a 2191-token prompt.
ENV ATLAS_PLE_MAX_TOKENS=8192
EXPOSE 8889

ENTRYPOINT ["spark"]
