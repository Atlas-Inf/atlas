# Historical sparse-prefill provenance

These records describe the comparison scaffold from which Atlas's native
prep/split/merge operators were adapted. They are retained verbatim for source
attribution; references to a supported builder or current build describe that
historical snapshot, not the present deployment.

The external bridge sources and build scripts were removed after native GPU
qualification. They remain available in Atlas commit `3c991613462c79730e2936a7b0f97add40f74370`:

The commands use its source-equivalent review revision from the
[history map](../../../docs/porting/glm53-validation/history-map.json).

```bash
git show b09ee1b6903931296e5a63d922150fb7cac272ca:scripts/dev/glm_sparse_native/README.md
git show b09ee1b6903931296e5a63d922150fb7cac272ca:scripts/build_glm_prefill/sources.sh
```

`SOURCE_MAP.json` records the earlier experiment inputs and their identity
limits. `NATIVE-BRIDGE-NOTICE.txt` preserves the NVIDIA notice and subsequent
source-build provenance. Historical cached objects are not shipped or consumed.
The [current native inventory](../provenance.json) identifies the source that
actually builds into Atlas. [Prior art](../../../docs/glm-prior-art.md) records
the contributors and distinguishes copied material from design references.
