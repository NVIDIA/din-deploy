# DIN Deploy: repository guidance for coding agents

## Goal

Build practical, high-performance local inference samples that others can understand,
reuse, and integrate into commercial applications. Keep this a sample repository,
not a framework: as small as possible and as complex as necessary, without sacrificing
correctness, functionality, or performance.

Apply these rules to new and changed code; do not rewrite unrelated samples to enforce them.

## Portable models and execution providers

- Export standard ONNX operators. Do not introduce ONNX Runtime contrib operators
  (for example `com.microsoft::*`), ATen fallbacks, custom operator libraries, or
  vendor-specific graph operators. Use standard ONNX decompositions instead.
- Inspect exported graphs, including subgraphs and local functions, for nonstandard
  operators; ONNX checker success alone does not demonstrate EP compatibility.
- Keep graphs and core algorithms EP-independent; EP-internal fusion is allowed.
  Maintain a usable CPU path and isolate optional backend-specific optimizations.
- Target Windows and Linux on x64 and ARM64. Hardware acceleration depends on the
  device and driver, not just the OS/CPU architecture; identify those requirements.
- Validate supported EPs and precisions explicitly; standard ONNX does not guarantee
  kernel support. Do not silently move expensive operations to CPU or claim that
  accelerator-only precisions work there. Provide a compatible configuration or
  explain the limitation.

## Performance and memory ownership

- Optimize measured bottlenecks in latency, throughput, memory, and startup/compilation.
- Keep intermediate tensors on their device through preprocessing and inference.
  Use device-buffer binding and reusable allocations where supported. Avoid host
  round trips and per-inference allocation in hot paths.
- Make memory location, ownership, layout/stride, and lifetime explicit at API
  boundaries. Keep buffers alive until asynchronous consumers finish. Bound queues
  and caches.
- Reuse the application's CUDA context for cooperating components where supported.
  Do not create extra contexts or processes casually; document any required boundary.
- Prefer stream/event dependencies over CPU waits and device-wide synchronization.
  Preserve producer/consumer dependencies when removing waits.
- GPU-resident is not synonymous with zero-copy or wait-free. Account for device
  copies and preprocessing when evaluating performance.

## Minimal code and useful reuse

- Reuse existing export, runtime, and I/O helpers. Extract small shared functions
  for concrete repetition; keep model-specific behavior near its model. Avoid
  speculative abstractions, registries, and frameworks for hypothetical reuse.
- Simplify control flow and remove obsolete code. Do not shorten code at the expense
  of readability, buffer reuse, or asynchronous execution.
- Follow surrounding conventions and repository formatting, including `.clang-format`.
- Preserve error causes; do not present failures or unsupported settings as success.

## Licensing and commercial reuse

- Keep code contributions compatible with this repository's Apache-2.0 license and
  commercial reuse. Prefer permissive dependencies with clear redistribution terms.
- Review licenses for new code, transitive dependencies, model weights, and bundled
  assets. Do not assume that open source or public availability permits every use.
- Discuss downstream impact with the user before introducing non-commercial
  restrictions or copyleft obligations. Flag unclear terms; do not claim unverified
  commercial compatibility or assume model weights share the code's license.
- Preserve required attribution and license notices, and document redistribution
  obligations. Consider relevant patent and SDK terms separately from code licenses.

## Validation and working practices

- Read the relevant README, build configuration, and CONTRIBUTING.md. Preserve
  unrelated user/agent changes. Use the existing project environment and build paths.
- Run focused validation: export parity and graph checks for model changes,
  numerical/task-quality checks for inference changes, and regression checks for
  state, timing, cancellation, and ownership bugs.
- Benchmark performance changes before/after with matching hardware, EP, precision,
  input, and warmup. Measure completed GPU work, not just submission time. Inspect
  copies/waits when claiming to reduce them; a smoke test is not performance evidence.
- Separate cold startup from steady-state timings. End-to-end timings include all
  enabled steps. Label model-only timings separately and state the RTF convention.
- Build/test the affected available targets. Report untested platform/EP combinations
  and unavailable prerequisites honestly. Do not add redundant tests or expand
  testing without a concrete reason.
- Update nearby documentation when behavior, requirements, or commands change.
  Report what changed, what was verified, and material limitations concisely.
- Commit/push only when requested; follow CONTRIBUTING.md's sign-off requirement.
