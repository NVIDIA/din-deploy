# Review guidelines

DIN Deploy provides practical examples for exporting models to ONNX and running
local inference on client PCs. Prioritize straightforward setup, understandable
code, inference correctness, and useful performance.

- Keep model-specific exporters and runtime code together. Put shared
  functionality in `common/` when multiple samples need it.
- Preserve supported Windows and Linux builds, including supported ARM64
  configurations. Preserve CPU execution where supported while optimizing CUDA
  and TensorRT RTX paths.
- Keep exporter outputs and runtime expectations consistent: filenames, tensor
  shapes, dtypes, preprocessing, tokenizers, and metadata.
- Prefer focused changes and simple implementations. Suggest abstractions when
  they solve a concrete maintenance problem.
- Keep CI efficient without silently skipping relevant tests. Missing dependency
  declarations must broaden execution. Export-cache inputs must cover the
  sources and settings that affect the model. See `docs/model-ci.md`.
- Update model documentation when setup, commands, dependencies, or supported
  behavior changes.
- Prioritize demonstrable bugs, compatibility regressions, incorrect outputs,
  and significant performance problems. Explain each finding's trigger and
  impact. Avoid speculative warnings, cosmetic preferences, and requests to
  rewrite unrelated code.
- Treat these principles as guidance; identify tradeoffs rather than enforcing
  blanket rules. Follow the contributor sign-off requirements in CONTRIBUTING.md.
