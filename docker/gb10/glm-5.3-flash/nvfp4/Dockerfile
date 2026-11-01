# Atlas Spark — GLM-5.3-Flash NVFP4 on GB10 (two Sparks, TP2 + EP2)
#
# Build from the repository root:
#   docker build -f docker/gb10/glm-5.3-flash/nvfp4/Dockerfile -t atlas-glm53-flash .
#
# The weights (~180 GB) need two Sparks at minimum (EP2); see
# docs/porting/GLM_5_3_FLASH.md.

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

# The grouped NVFP4 MoE path compiles CUTLASS objects in spark-runtime/build.rs
# when CUTLASS_HOME is set. CUTLASS is header-only; pin it for reproducibility.
ARG CUTLASS_SHA=cf064d2e6bad2886238ac565b3b49007764f4939

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev \
      libibverbs-dev librdmacm-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV RUSTUP_TOOLCHAIN=stable
ENV CUTLASS_HOME=/opt/cutlass

RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git ${CUTLASS_HOME} && \
    git -C ${CUTLASS_HOME} checkout ${CUTLASS_SHA}

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
# atlas-plugin embeds the quick-speed prompt fixtures with include_str!.
COPY tests/fixtures/ tests/fixtures/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=glm-5.3-flash
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
ARG ATLAS_GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${ATLAS_GIT_SHA}"

# NCCL >= 2.28 (symmetric memory windows) and RDMA userspace for RoCE between Sparks.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
