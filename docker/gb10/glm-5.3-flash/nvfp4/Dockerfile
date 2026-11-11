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
ARG CARGO_BUILD_JOBS=2
ARG ATLAS_HIPCC_WORKERS=2
ARG NVCC_THREADS=2
ARG ATLAS_GIT_SHA=unknown

RUN apt-get update -qq && \
    apt-get install -y -qq --allow-change-held-packages curl build-essential pkg-config git cmake libclang-dev \
      libibverbs-dev librdmacm-dev binutils libnccl-dev libnccl2 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    { dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: builder NCCL $NCCL_VER < 2.28" >&2; exit 1; }; }

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV CUTLASS_HOME=/opt/cutlass
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}
ENV ATLAS_HIPCC_WORKERS=${ATLAS_HIPCC_WORKERS}
ENV NVCC_THREADS=${NVCC_THREADS}

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
# Sparse MLA device headers are compiled by Atlas's kernel target. KDA's three
# MIT device headers and owned host launcher are already in crates/ above.
# Neither accelerator uses an external model-specific library or host dispatcher.
COPY third_party/nvidia_sparse_mla_sm120/ third_party/nvidia_sparse_mla_sm120/
COPY LICENSE ./LICENSE
# Includes the full MIT notice for the adapted knapcio confidence tables.
COPY docs/glm-prior-art.md /opt/atlas/notices/glm-prior-art.md

# Preserve upstream notices and exact imported-source provenance in the image.
RUN mkdir -p /opt/atlas/notices/flashkda /opt/atlas/notices/nvidia-sparse-mla /opt/atlas/notices/cutlass && \
    cp crates/spark-runtime/cuda/glm_kda/vendor/LICENSE \
       crates/spark-runtime/cuda/glm_kda/vendor/NOTICE \
       crates/spark-runtime/cuda/glm_kda/vendor/provenance.json /opt/atlas/notices/flashkda/ && \
    cp third_party/nvidia_sparse_mla_sm120/LICENSE \
       third_party/nvidia_sparse_mla_sm120/provenance.json /opt/atlas/notices/nvidia-sparse-mla/ && \
    cp ${CUTLASS_HOME}/LICENSE.txt /opt/atlas/notices/cutlass/LICENSE.txt && \
    printf '%s\n' "${CUTLASS_SHA}" > /opt/atlas/notices/cutlass/revision.txt

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=glm-5.3-flash
ENV ATLAS_TARGET_QUANT=nvfp4

# rustup honors the repository's rust-toolchain.toml (1.93.1); do not override
# it with a moving stable toolchain. Keep the committed dependency lockfile.
RUN cargo build --locked --release -p spark-server

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
COPY --from=builder /opt/atlas/notices/ /opt/atlas/notices/
COPY LICENSE /LICENSE

ENV RUST_LOG=info
# Both native accelerators remain opt-in through their Atlas execution flags.
# No *_LIBRARY path or separate prefill shared library is used.
EXPOSE 8888

ENTRYPOINT ["spark"]
