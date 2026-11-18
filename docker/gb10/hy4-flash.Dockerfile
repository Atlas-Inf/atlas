# Hy4-only GB10 build. NCCL development files are needed at link time too.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    curl build-essential pkg-config git cmake libclang-dev libibverbs-dev libnccl-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}" CUDA_HOME=/usr/local/cuda
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN rustup toolchain install
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
COPY third_party/ third_party/
COPY tests/fixtures/ tests/fixtures/
ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=hy4-flash ATLAS_TARGET_QUANT=nvfp4
RUN cargo build --locked --release -p spark-server

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3
ARG ATLAS_GIT_SHA=unknown
LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${ATLAS_GIT_SHA}"
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    --allow-change-held-packages libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
    libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/* && \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' libnccl2)" ge "2.28"
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=builder /build/third_party/ /licenses/third_party/
COPY LICENSE /LICENSE
ENV RUST_LOG=info
ENTRYPOINT ["spark"]
