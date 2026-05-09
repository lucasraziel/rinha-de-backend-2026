## Multi-stage build for the Rinha 2026 api binary.
##
## Stage 1 builds against linux/amd64 with `target-cpu=x86-64-v3` so the SIMD
## intrinsics (AVX2 + F16C) are baked in unconditionally. Stage 2 ships the
## stripped binary plus the pre-processed dataset on top of distroless cc.
##
## We commit data/references.bin to git instead of running `prep` here:
## - keeps the build deterministic (no streaming JSON parse on every build)
## - keeps the build small (50MB resources/references.json.gz never enters the layer cache)
## - cuts ~30s off every CI rebuild

FROM --platform=linux/amd64 rust:1-slim-bookworm AS builder
WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config lld \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C link-arg=-fuse-ld=lld"

# Cache the dependency graph layer.
COPY rust-toolchain.toml Cargo.toml ./
COPY .cargo .cargo
COPY crates ./crates
COPY resources ./resources

RUN cargo build --release --bin api -p api \
    && strip target/release/api

# distroless cc-debian12 (root, not nonroot) — needed so the api can bind
# its UDS socket inside the volume mounted at /run/uds without the volume
# being pre-chowned. We `chmod 0666` the socket after bind so nginx can talk
# to us regardless of its own uid.
FROM --platform=linux/amd64 gcr.io/distroless/cc-debian12
COPY --from=builder /build/target/release/api /api
COPY data/references.bin /data/references.bin
ENV DATASET_PATH=/data/references.bin
EXPOSE 9000
ENTRYPOINT ["/api"]
