# Model CI

CI has one planner and one reusable export/inference workflow. Model-specific
commands are data; adding a model does not require editing workflow logic.

## Two kinds of configuration

- `.github/model-tests/*.json` describes **how** to export and test a model family:
  CMake targets, model variants, dependency installation, exporter command,
  required output files, and CPU smoke-test command. Every recipe is discovered
  automatically. The first CMake target is the executable used for inference.
- `.github/model-ci.json` optionally describes **when it is safe to skip** work:
  export inputs, runtime/test inputs, reusable dependency groups, and explicitly
  ignored files. A recipe without complete dependency rules runs conservatively.

The execution recipe is necessary: CI cannot infer model IDs, CLI arguments,
credentials, or hardware requirements from a new C++ executable. Dependency
optimization is optional.

## Behavior

| Change | Native build | Export | Inference |
| --- | --- | --- | --- |
| Model exporter or its declared dependencies | Model targets | New fingerprint; reuse only an exact match | Affected model variants |
| Model runtime | Model targets | Reuse matching export, or create one on cache miss | Affected model variants |
| Shared runtime | All consuming targets | Reuse matching exports | All consuming models |
| Test asset | Consuming targets | Reuse matching exports | Consuming models |
| README or `.coderabbit.yaml` only | Skip | Skip | Skip |
| Unknown file | Full build | Conservative fresh exports | All available recipes |
| Missing model dependency rules | That model always runs on non-exempt changes | Fresh export if export rules are missing | That model |
| CI control logic or dependency registry | Full build | Fingerprint determines reuse | All models |
| Manual run or unavailable Git base | Full build | Fresh exports | All models |

Draft PRs skip the pipeline. Marking a PR ready starts it; returning it to draft
cancels an in-progress run through workflow concurrency. Lightweight planner tests
run on every non-draft PR, including documentation changes. C++ formatting runs
when native source or `.clang-format` changes.

All native platforms use the same selected CMake targets. CMake/Ninja resolves
their actual compilation and linking dependencies. Linux x64 and Windows x64
smoke tests consume build artifacts from **the same workflow run and checkout**;
there is no polling for a separate Build run. ARM64 keeps its build coverage.
No inference-test success or native binary cache is introduced here.

## Conservative fallback

Unowned changed files trigger all known recipes and a full build. Literal CMake
executable declarations outside the root helper file are independently scanned:
an executable absent from the recipes/build-only declarations forces a full
build and all available smoke tests on non-exempt changes. The job summary lists
those executables and calls out the missing inference recipe. This is deliberately
not a general-purpose CMake parser; unusual dynamically declared targets should
get a recipe or an explicit build-only declaration.

A new recipe with no entry in the dependency registry is always selected. It does
not reuse an earlier run's export. Missing or invalid dependency configuration
falls back to all recipes. Invalid **execution** recipes fail visibly: continuing
without knowing how to execute the test would hide missing coverage.

FLUX and the base ONNX sample retain their existing build-only hosted coverage,
with explicit reasons in the registry. GPU/local-asset CTest tests are still listed,
not executed on CPU hosted runners. A full-build fallback cannot manufacture the
hardware or model artifacts needed to run those tests.

## Export reuse

Export names include a fingerprint of tracked export input blob IDs, export
recipe, model variant/options, and shared exporter runner/workflow. Runtime-only
changes do not invalidate an export. Unowned tracked files are also fingerprinted,
so their changes cannot accidentally restore an old export in a later run.
Unknown dependencies use a run/attempt-specific key to prevent cross-run reuse.

Only unexpired artifacts from successful allowed runs (default branch, same PR,
or allowed non-PR source branch) are reused. A missing/expired artifact produces a
new export. Precision fallback is preserved: try the requested precision, then
its configured fallback if export fails. Smoke tests still have to pass.

The existing dependency installation recipes use some floating package/model
versions. Cached exports represent the versions downloaded when they were made,
not a continuous check for upstream updates. Change `export.cache_epoch` to force
refreshes; pin dependency/model revisions in recipes when reproducibility requires
it. A complete package lock and model revision pinning are separate follow-up work.

## Adding a model

1. Add its normal CMake executable target and exporter.
2. Add `.github/model-tests/<family>.json`, using an existing recipe as an example.
   Commands are argument arrays, executed without a shell. Available placeholders
   include `{python}`, `{model_id}`, `{output}`, `{precision}`, `{dtype}`, `{slug}`,
   and variant fields. Smoke tests additionally receive `{executable}`.
3. Run `python -m unittest discover -s .github/tests -v`. The new recipe is now
   discoverable and runs without dependency optimization.
4. Optionally add `models.<family>.export` and `.runtime` input patterns to
   `.github/model-ci.json`. Reference shared groups rather than copying them.
   Include transitive Python helpers, preprocessing, tokenizer code, build
   configuration, and test assets. These declarations are correctness contracts.
5. Inspect the CI selection summary and confirm the new variant passes on both
   runtime platforms before relying on selective execution.

Patterns use Python `fnmatch`: `*` matches across directory separators. Renames
are treated as a deletion plus an addition. Unknown ownership always broadens
work. Add ignore patterns only for files known not to affect builds or tests.

The first run after this migration creates new export artifacts because the
fingerprint format changed. Existing required-check rules referring to the old
standalone model workflows must be updated for the consolidated workflow.
