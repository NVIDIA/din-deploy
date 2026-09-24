#!/usr/bin/env python3
"""Plan model CI from optional dependency declarations; uncertainty runs more work."""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = '.github/model-ci.json'
RECIPES = '.github/model-tests'
CONTROL = ['.github/scripts/**', '.github/workflows/**', '.github/tests/**', CONFIG]
NATIVE_SUFFIXES = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.cu', '.cuh'}


def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root)


def matches(path, patterns):
    # fnmatch intentionally lets * span directories. Both common/*.h and ** work.
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def read_config(root=ROOT):
    try:
        config = json.loads((root / CONFIG).read_text())
        if not isinstance(config, dict) or config.get('version') != 1:
            raise ValueError('unsupported dependency configuration version')
        for field in ('groups', 'models', 'build_only'):
            if not isinstance(config.get(field, {}), dict):
                raise ValueError(f'{field} must be an object')
        def patterns(value):
            if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
                raise ValueError('dependency patterns and group names must be string lists')
        patterns(config.get('ignore', []))
        for value in config.get('groups', {}).values():
            patterns(value)
        sections = list(config.get('build_only', {}).values())
        for model in config.get('models', {}).values():
            if not isinstance(model, dict):
                raise ValueError('model dependencies must be objects')
            sections.extend(model.values())
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError('dependency sections must be objects')
            patterns(section.get('paths', []))
            patterns(section.get('groups', []))
        return config
    except (OSError, ValueError) as error:
        print(f'::warning::Dependency optimization disabled: {error}')
        return {}


def recipes(root=ROOT):
    result = {}
    for path in sorted((root / RECIPES).glob('*.json')):
        recipe = json.loads(path.read_text())
        name = recipe['name']
        if name in result or path.stem != name or not re.fullmatch(r'[a-z0-9-]+', name):
            raise ValueError(f'invalid or duplicate recipe name: {path}')
        if not recipe['targets'] or not recipe['variants'] or not recipe['source_roots']:
            raise ValueError(f'incomplete execution recipe: {name}')
        slugs = [v['slug'] for v in recipe['variants']]
        if len(set(slugs)) != len(slugs) or any(not re.fullmatch(r'[a-z0-9.-]+', s) for s in slugs):
            raise ValueError(f'invalid or duplicate variant slug: {name}')
        for target in recipe['targets']:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', target):
                raise ValueError(f'invalid CMake target: {target}')
        for variant in recipe['variants']:
            if variant['precision'] not in ('fp16', 'fp32') or not variant['model_id']:
                raise ValueError(f'invalid export variant: {variant}')
        result[name] = recipe
    if not result:
        raise ValueError('No model execution recipes found; refusing to report empty coverage')
    return result


def inputs(section, config):
    if not isinstance(section, dict):
        return None
    patterns = section.get('paths', []).copy()
    for group in section.get('groups', []):
        if group not in config.get('groups', {}):
            return None
        patterns.extend(config['groups'][group])
    return patterns or None


def discover_unregistered(config, entries, root=ROOT):
    """Discover literal sample targets independently of the optimization registry.

    CMake remains responsible for actual build dependencies. Unknown/dynamic
    declarations get a full build, never a guessed partial target list.
    """
    known = {t for entry in entries.values() for t in entry['targets']}
    known.update(t for item in config.get('build_only', {}).values() for t in item['targets'])
    unknown = []
    tracked = git('ls-files', '-z', root=root).decode().split('\0')
    for path in tracked:
        if not path.endswith('CMakeLists.txt') or path == 'CMakeLists.txt':
            continue
        text = (root / path).read_text()
        text = re.sub(r'#[^\n]*', '', text)
        for target in re.findall(r'\badd_(?:din_)?executable\s*\(\s*([^\s)]+)', text, re.I):
            target = target.strip('"')
            if target not in known:
                unknown.append(f'{path}: {target}')
    return sorted(unknown)


def plan(files, config, entries, unknown_targets=()):
    reasons = []
    force_all = files is None or not config
    force_export = files is None or not config
    if files is None:
        reasons.append('No reliable change base (manual/initial run): run all')
    files = [p for p in files or [] if p]
    # Positive exemptions, not a blanket "all YAML is harmless" rule.
    relevant = [p for p in files if not matches(p, config.get('ignore', []))]
    if config and files and not relevant:
        return dict(build=False, targets=[], matrix={'include': []}, format=False,
                    reasons=['Only explicitly ignored documentation/review files changed'], unknown_targets=[])
    if unknown_targets:
        force_all = force_export = True
        reasons.append('Unregistered CMake targets: full build and all available smoke recipes')
    paths_by_model = {}
    for name in entries:
        dependency = config.get('models', {}).get(name, {})
        paths_by_model[name] = (inputs(dependency.get('export'), config), inputs(dependency.get('runtime'), config))
    owned = [p for pair in paths_by_model.values() for patterns in pair if patterns for p in patterns]
    for item in config.get('build_only', {}).values():
        owned.extend(inputs(item, config) or [])
    recipe_paths = [f'{RECIPES}/{name}.json' for name in entries]
    unknown_files = [p for p in relevant if not matches(p, owned + CONTROL + recipe_paths)]
    if unknown_files:
        force_all = force_export = True
        reasons.append('Unowned changes: ' + ', '.join(unknown_files))
    control_changed = any(matches(p, CONTROL) for p in relevant)
    selected, targets = [], set()
    full_build = force_all or control_changed
    for name, entry in entries.items():
        export_paths, runtime_paths = paths_by_model[name]
        undeclared = export_paths is None or runtime_paths is None
        changed = any(matches(p, (export_paths or []) + (runtime_paths or []) + [f'{RECIPES}/{name}.json']) for p in relevant)
        if force_all or control_changed or undeclared or changed:
            reasons.append(f'{name}: ' + ('missing dependency rules; run conservatively' if undeclared else 'affected or full validation'))
            targets.update(entry['targets'])
            for variant in entry['variants']:
                selected.append({'model': name, 'variant': variant['slug'], 'force_export': force_export or export_paths is None})
    for name, item in config.get('build_only', {}).items():
        patterns = inputs(item, config)
        if force_all or control_changed or patterns is None or any(matches(p, patterns) for p in relevant):
            targets.update(item['targets'])
            reasons.append(f'{name}: build only ({item["reason"]})')
    return dict(build=bool(targets) or full_build, targets=[] if full_build else sorted(targets),
                matrix={'include': selected}, format=files is None or any(Path(p).suffix in NATIVE_SUFFIXES or p == '.clang-format' for p in files),
                reasons=reasons, unknown_targets=list(unknown_targets))


def changed_files(event, root=ROOT):
    pr = event.get('pull_request')
    base = pr['base']['sha'] if pr else event.get('before')
    if not base or not base.strip('0'):
        return None
    try:
        git('cat-file', '-e', f'{base}^{{commit}}', root=root)
        if pr:
            base = git('merge-base', base, 'HEAD', root=root).decode().strip()
        return git('diff', '--no-renames', '--name-only', '-z', base, 'HEAD', root=root).decode().split('\0')
    except subprocess.CalledProcessError:
        return None


def export_key(name, variant, config, entries, root=ROOT, force=False):
    entry = entries[name]
    patterns = inputs(config.get('models', {}).get(name, {}).get('export'), config)
    fingerprint = {'recipe': entry['export'], 'variant': variant, 'inputs': patterns,
                   'environment': 'ubuntu-24.04-python-3.12-cpu'}
    # The runner and export workflow affect all export recipes, unlike runtime code.
    patterns = (patterns or []) + ['.github/scripts/model_ci.py', '.github/scripts/run_model_ci.py', '.github/workflows/model-test.yml']
    owned = CONTROL + config.get('ignore', []) + [RECIPES + '/**']
    for dependency in config.get('models', {}).values():
        for stage in ('export', 'runtime'):
            owned.extend(inputs(dependency.get(stage), config) or [])
    for component in config.get('build_only', {}).values():
        owned.extend(inputs(component, config) or [])
    sources = []
    for record in git('ls-files', '--stage', '-z', root=root).decode().split('\0'):
        if not record:
            continue
        metadata, path = record.split('\t', 1)
        if matches(path, patterns) or not matches(path, owned):
            sources.append([path, metadata])
    fingerprint['sources'] = sources
    if force or not inputs(config.get('models', {}).get(name, {}).get('export'), config):
        # Unknown export dependencies must not reuse an earlier run's model.
        fingerprint['uncertain_run'] = [os.getenv('GITHUB_RUN_ID', 'local'), os.getenv('GITHUB_RUN_ATTEMPT', '1'), git('rev-parse', 'HEAD', root=root).decode().strip()]
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:24]


def write_outputs(values):
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        for key, value in values.items():
            output.write(f'{key}={json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'build'])
    parser.add_argument('--preset')
    args = parser.parse_args()
    if args.mode == 'build':
        targets = json.loads(os.environ.get('BUILD_TARGETS', '[]'))
        command = ['cmake', '--build', '--preset', args.preset, '--parallel']
        if targets:
            command += ['--target', *targets]
        subprocess.run(command, check=True)
        return
    entries = recipes()
    config = read_config()
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    result = plan(changed_files(event), config, entries, discover_unregistered(config, entries))
    print(json.dumps(result, indent=2))
    write_outputs({'build': result['build'], 'targets': result['targets'], 'matrix': result['matrix'],
                   'models': bool(result['matrix']['include']), 'format': result['format']})
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
            summary.write('## CI selection\n\n' + '\n'.join('- ' + reason for reason in result['reasons']) + '\n')
            if result['unknown_targets']:
                summary.write('\nUnregistered executables are included in the full build. Add an execution recipe to run their inference tests:\n\n')
                summary.write('\n'.join('- ' + name for name in result['unknown_targets']) + '\n')


if __name__ == '__main__':
    main()
