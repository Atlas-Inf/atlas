# Atlas confidential-computing runtime validation

## Status and terminology

This document uses **CC** to mean NVIDIA GPU Confidential Computing. `CCD` is
not a CUDA or NVIDIA Confidential Containers mode name and is treated as a
terminology error unless a separate requirement defines it.

The procedure below is a qualification plan, not evidence that Atlas has
already passed on CC hardware. A deployment is supported only after its exact
hardware and software fingerprint has a completed result record.

## Hardware boundary

NVIDIA's current Confidential Containers supported-platform table lists these
GPU targets:

| GPU | Validated passthrough |
|---|---|
| H100 | Single GPU |
| H200 | Single GPU |
| H100/H200 Protected PCIe | Multi-GPU |
| B200 | Single and multi-GPU |
| HGX B300 | Single and multi-GPU |
| RTX PRO 6000 Blackwell Server Edition | Single GPU |

DGX Spark/GB10 is absent from the supported CC table and is therefore an
**unsupported, non-CC test target**. A normal DGX Spark run cannot be cited as
CC validation.

The validated host combinations are AMD SEV-SNP on AMD Genoa/Milan and Intel
TDX on Intel Emerald Rapids/Granite Rapids. At the time this document was
written, NVIDIA specifies Ubuntu 25.10 or 26.04, kernel 6.17 or newer,
Kubernetes 1.32 or newer, containerd 2.3.x, Kata Containers 4.0.0, and NVIDIA
GPU Operator 26.3.1 or newer. Re-check the live supported-platform page before
each qualification campaign; a result must record the versions actually used.

Sources:

- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/supported-platforms.html>
- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/prerequisites.html>

## Host and cluster prerequisites

Before installing Atlas, verify all of the following:

1. Hardware virtualization and ACS are enabled in firmware.
2. IOMMU is enabled (`amd_iommu=on` or `intel_iommu=on`).
3. The host has no NVIDIA driver bound to the passthrough GPU; `vfio-pci` owns
   it and the guest driver is managed by the GPU Operator.
4. The node uses the `vm-passthrough` GPU workload configuration.
5. Kata installs the matching confidential runtime class:
   `kata-qemu-nvidia-gpu-snp` or `kata-qemu-nvidia-gpu-tdx`.
6. All GPUs on a node enter the same CC mode. Partial-node CC configuration is
   unsupported.
7. Hopper multi-GPU uses `ppcie`; Blackwell uses `on`.

Configure and verify a node:

```bash
export NODE_NAME='<node-name>'
kubectl label node "$NODE_NAME" nvidia.com/gpu.workload.config=vm-passthrough --overwrite
kubectl label node "$NODE_NAME" nvidia.com/cc.mode=on --overwrite

until [ "$(kubectl get node "$NODE_NAME" \
  -o jsonpath='{.metadata.labels.nvidia\.com/cc\.ready\.state}')" = true ]; do
  echo 'waiting for CC mode to converge'
  sleep 5
done

kubectl get node "$NODE_NAME" -o json | \
  jq '.metadata.labels | with_entries(select(.key | startswith("nvidia.com/cc")))'
```

For Hopper multi-GPU, replace `on` with `ppcie`. The required terminal state is
`cc.mode.state == cc.mode` and `cc.ready.state == true`.

Installation details and the current Helm values are maintained by NVIDIA:

- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/confidential-containers-deploy.html>
- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/configure-cc-mode.html>

## Workload and attestation gate

A single-GPU qualification pod must use the appropriate confidential Kata
runtime and request a passthrough GPU:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: atlas-cc
spec:
  runtimeClassName: kata-qemu-nvidia-gpu-snp # use -tdx on Intel TDX
  restartPolicy: Never
  containers:
    - name: atlas
      image: <qualified-atlas-image-by-digest>
      resources:
        limits:
          nvidia.com/pgpu: "1"
          memory: 64Gi
```

Pin the Atlas image by digest, not a mutable tag. Attach a Kata agent security
policy for production. The policy and image measurement must be part of the
attested evidence accepted by Trustee before model decryption credentials are
released.

CC mode alone is not attestation. The qualification record must include:

- CPU TEE evidence and policy verdict;
- GPU evidence and policy verdict;
- guest image/kernel/firmware reference values;
- Kata agent policy hash;
- Trustee/KBS policy hash;
- proof that a protected test resource was released only after a successful
  combined CPU and GPU verdict.

NVIDIA's Trustee quickstart validates connectivity only and explicitly does not
produce hardware evidence. Use an end-to-end attested workload for the final
gate:

- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/attestation.html>
- <https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/configure-workloads.html>

## Atlas correctness matrix

Run each row with the same checkpoint revision, kernel build, quantization,
request corpus, seed, and serve configuration. Graph modes must expose their
engagement and fallback counters through `/metrics`.

| Execution path | Required comparison |
|---|---|
| Eager | Non-CC eager reference |
| Decode graph | CC eager and non-CC decode graph |
| K=2/K=3/K=4 verification graphs | CC eager verification and non-CC graph |
| Batched/ragged verification | Same slot/depth schedule in eager and graph modes |
| Prefill graph | Dense, MoE, and hybrid-SSM eager output at every bucket boundary |
| DFlash/D-Spark | Token output, acceptance histogram, accepted-token length |
| Graph prewarm | Cold capture key set equals post-restart recreated key set |

For the checked-in real-model prefill differential gate:

```bash
export ATLAS_GRAPH_PREFILL_MODEL_DIRS='<dense>,<moe>,<hybrid-ssm>'
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test -p spark-server --release \
  prefill_graph_matches_eager_for_model_matrix -- --ignored --nocapture
```

Record both successful graph engagement and every eager fallback reason. A pass
with zero graph captures is not a graph test.

## CC-specific feature probes

The following probes are mandatory because CC changes DMA, tooling, and device
exposure. `Unknown` is not equivalent to supported.

| Feature | Qualification requirement |
|---|---|
| Stream capture/replay | Capture, replay, eviction, recapture, and teardown under CC |
| Pinned host allocation | Probe `cuMemAllocHost` and preserve a pageable fallback |
| Host registration | Probe the CUDA capability; never assume `cuMemHostRegister` works |
| Host-mapped result buffers | Verify mapping or use the existing D2H fallback |
| JIT-loaded modules | Load every PTX module, capture its kernels, restart, and repeat |
| KV offload | Validate file I/O, staging lifetime, and encrypted storage policy |
| Graph DOT export | Confirm policy permits topology export and that output contains no secrets |
| Profiling | Record which Nsight/CUPTI facilities the selected CC stack permits |
| Cancellation/EOS | Exercise graph and bounded-loop early termination |
| OOM/eviction | Force cache churn without exceeding the confidential VM memory limit |

Older Hopper CC release notes report unsupported pinned-host and host-register
APIs in SPT mode. Treat that as a required runtime probe rather than assuming
newer stacks behave identically. Atlas must remain correct with pageable host
staging even when the optimization is unavailable.

## Performance comparison

Measure CC and non-CC as a controlled one-variable comparison. Every record
must include commit, full serve flags, checkpoint revision, kernel target,
driver/CUDA/firmware, TEE/runtime versions, prompt corpus, ISL/OSL, seed,
repetition count, and raw run-record paths.

Required metrics:

- TTFT for short, medium, and long prompts;
- inter-token latency and aggregate throughput;
- graph capture, replay, and instantiation latency;
- graph memory and total process memory;
- launch count and eager fallback counts;
- speculative acceptance rate and accepted-token length.

Use at least three repetitions for a reported comparison. Do not publish a
performance delta until the graph-engagement counters prove that the intended
path ran in both arms.

## Support classification

- **Production-ready:** no CC target until a fingerprinted campaign completes.
- **Experimental:** Atlas eager and ordinary CUDA graph paths on an NVIDIA-listed
  CC platform using the procedure above.
- **Unsupported:** DGX Spark/GB10 as a CC target, partial-node CC mode, stale
  graph/prewarm manifests, unauthenticated mode-only deployments, and any
  feature whose CC probe is not recorded.
