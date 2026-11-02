# NVIDIA sparse MLA source subset

These 17 unmodified headers are pinned by `provenance.json` to FlashInfer commit
`8eccd0c1352165302840c0e19066bc42d36dbd7a`. NVIDIA's BSD-3-Clause copyright and
license notices are retained in every file. They require the CUDA toolkit and
standard C/C++ headers only; FlashInfer is not a build or runtime dependency.

Atlas compiles its own GLM-specific CUDA entry point against the device template
in `prefill_kernel.cuh`. The multi-model host dispatcher is not imported.

Prior integration work: Enntity / Jason McCartney's SparkGLM source recipe
`84310dc3` established the pinned NVIDIA kernel and main/tail composition;
Reiner Schmidt / Mango-kid's Atlas supplied the model and kernel runtime. The
Atlas-owned prep/split/merge sources retain their original AGPL attribution.
See `docs/glm-prior-art.md` and `third_party/nvidia_sparse_mla_sm120/history/SOURCE_MAP.json`
for the historical scaffold. No opaque binary object is shipped or consumed.
