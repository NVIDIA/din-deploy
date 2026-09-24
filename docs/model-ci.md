# Model CI

Model recipes in `.github/model-tests/*.json` define exports, CMake targets, and
CPU smoke tests. Optional dependency rules in `.github/model-ci.json` determine
which models need to run. Adding a model requires no workflow changes.

## What runs

| Change | CI behavior |
| --- | --- |
| Exporter or export dependencies | Build and test affected models; reuse only matching exports |
| Runtime or test inputs | Build and test affected models using cached exports |
| Shared code | Run all affected consumers |
| Documentation or `.coderabbit.yaml` only | Lightweight checks only |
| Unknown files or CI control logic | Full build and all model recipes |

Draft PRs skip CI. Missing dependency rules make a model run conservatively;
missing export rules prevent reuse across runs. Missing or expired artifacts are
re-exported. Unknown files also force fresh exports.

Unregistered CMake executables trigger a full build and all available recipes;
new inference tests still need an execution recipe. Linux and Windows x64 run
CPU smoke tests; ARM64, FLUX, and the base ONNX sample retain build-only coverage.

## Adding a model

1. Add its CMake target and exporter.
2. Copy an existing `.github/model-tests/<family>.json` recipe and adapt its
   variants, commands, and required outputs. The first target is the inference CLI.
3. Optionally declare export and runtime/test inputs in `.github/model-ci.json`,
   including shared dependencies. Without these rules, the recipe always runs on
   non-exempt changes. Patterns use `fnmatch`; `*` spans directories.
4. Run `python -m unittest discover -s .github/tests -v`, then verify the CI
   selection summary and model results on both runtime platforms.

Exports are fingerprinted from their declared inputs, recipe, and model options.
Bump `export.cache_epoch` to refresh floating upstream dependencies or weights;
pin versions when reproducibility is required.

When migrating, update required-check rules that reference the old standalone
model workflows. The first run creates exports under the new cache keys.
