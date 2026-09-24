#!/usr/bin/env python3
"""Execute model recipes without model-specific workflow branches."""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from model_ci import ROOT, export_key, read_config, recipes, write_outputs


def expand(command, values):
    return [argument.format_map(values) for argument in command]


def run(command, values, env=None):
    subprocess.run(expand(command, values), cwd=ROOT, env=env, check=True)


def artifact_name(slug, precision, key):
    return f'model-{slug}-{precision}-{key}'


def find_artifact(name):
    command = [sys.executable, str(ROOT / '.github/scripts/find_successful_artifact.py'),
               '--repository', os.environ['GITHUB_REPOSITORY'], '--artifact-name', name,
               '--default-branch', os.environ['DEFAULT_BRANCH'], '--event-name', os.environ['EVENT_NAME'],
               '--source-branch', os.environ['SOURCE_BRANCH'], '--pull-request', os.environ.get('PULL_REQUEST', ''), '--json']
    return json.loads(subprocess.check_output(command, text=True))


def export_model(entry, variant, values, key):
    definition = entry['export']
    env = dict(os.environ, **definition.get('env', {}))
    for command in definition['install']:
        run(command, values, env)
    output = Path(values['output'])
    precisions = [variant['precision']]
    if variant.get('fallback_precision'):
        precisions.append(variant['fallback_precision'])
    for index, precision in enumerate(precisions):
        current = dict(values, precision=precision, dtype={'fp16': 'float16', 'fp32': 'float32'}[precision])
        if output.exists():
            shutil.rmtree(output)
        try:
            run(definition['command'], current, env)
        except subprocess.CalledProcessError:
            if index == len(precisions) - 1:
                raise
            print(f'::warning::{precision} export failed; trying {precisions[index + 1]}', flush=True)
            continue
        missing = [name for name in definition['required_files'] if not (output / name).is_file()]
        if missing:
            if index == len(precisions) - 1:
                raise RuntimeError(f'Export did not produce required files: {missing}')
            print(f'::warning::{precision} export missing required files: {missing}; trying {precisions[index + 1]}', flush=True)
            continue
        manifest = {'model_id': variant['model_id'], 'precision': precision, 'source_hash': key,
                    'commit': os.getenv('GITHUB_SHA'), 'variant': variant}
        (output / 'ci-export-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        write_outputs({'name': artifact_name(variant['slug'], precision, key), 'run_id': os.environ['GITHUB_RUN_ID']})
        return


def smoke_test(entry, values):
    definition = entry['smoke']
    for source, destination in definition.get('decode_base64', {}).items():
        (ROOT / destination).write_bytes(base64.b64decode((ROOT / source).read_text()))
    for directory in definition.get('mkdir', []):
        (ROOT / directory.format_map(values)).mkdir(parents=True, exist_ok=True)
    executable = ROOT / 'runtime' / (entry['targets'][0] + ('.exe' if os.name == 'nt' else ''))
    if os.name != 'nt':
        executable.chmod(executable.stat().st_mode | 0o111)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(ROOT / 'runtime') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    run(definition['command'], dict(values, executable=str(executable)), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['resolve', 'export', 'smoke'])
    args = parser.parse_args()
    entries = recipes()
    entry = entries[os.environ['MODEL']]
    variant = next(v for v in entry['variants'] if v['slug'] == os.environ['VARIANT'])
    values = dict(variant, python=sys.executable, output=str(ROOT / 'artifacts/ci' / variant['slug']))
    if args.mode == 'smoke':
        smoke_test(entry, values)
        return
    key = export_key(entry['name'], variant, read_config(), entries, force=os.getenv('FORCE_EXPORT') == 'true')
    if args.mode == 'export':
        export_model(entry, variant, values, key)
        return
    for precision in dict.fromkeys([variant['precision'], variant.get('fallback_precision')]):
        if not precision:
            continue
        found = find_artifact(artifact_name(variant['slug'], precision, key))
        if found['found']:
            write_outputs(found)
            return
    write_outputs({'found': False, 'name': '', 'run_id': ''})


if __name__ == '__main__':
    main()
