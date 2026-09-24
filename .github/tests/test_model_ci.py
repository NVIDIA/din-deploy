import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import model_ci as ci
import run_model_ci as runner


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = ci.read_config()
        self.entries = {name: value for name, value in ci.recipes().items() if name in {"whisper", "parakeet", "nemotron", "sam2"}}

    def plan(self, *paths, **kwargs):
        return ci.plan(list(paths), self.config, self.entries, **kwargs)

    def selected(self, result):
        return {row['model'] for row in result['matrix']['include']}

    def test_review_config_and_docs_do_not_build_or_export(self):
        for path in ['.coderabbit.yaml', 'README.md', 'asr/rnnt/README.md']:
            with self.subTest(path=path):
                result = self.plan(path)
                self.assertFalse(result['build'])
                self.assertEqual(self.selected(result), set())

    def test_parakeet_runtime_selects_only_parakeet(self):
        result = self.plan('asr/rnnt/cpp/parakeet_tdt.cpp')
        self.assertEqual(self.selected(result), {'parakeet'})
        self.assertEqual(result['targets'], ['din_asr_parakeet_tdt_cli'])
        self.assertFalse(result['matrix']['include'][0]['force_export'])

    def test_nemotron_runtime_selects_only_nemotron(self):
        self.assertEqual(self.selected(self.plan('asr/rnnt/cpp/nemotron.cpp')), {'nemotron'})

    def test_export_change_also_selects_inference(self):
        self.assertEqual(self.selected(self.plan('asr/whisper/model_export/export_whisper.py')), {'whisper'})

    def test_shared_audio_runs_all_models(self):
        self.assertEqual(self.selected(self.plan('common/io/audio.cpp')), set(self.entries))

    def test_shared_export_selects_only_consumers(self):
        self.assertEqual(self.selected(self.plan('asr/rnnt/python/tools/nemo_preprocessor_export.py')), {'parakeet', 'nemotron'})

    def test_test_asset_selects_consumers_without_forcing_export(self):
        result = self.plan('assets/sample.wav')
        self.assertEqual(self.selected(result), {'whisper', 'parakeet', 'nemotron'})
        self.assertFalse(any(row['force_export'] for row in result['matrix']['include']))

    def test_unknown_file_runs_everything_and_forces_export(self):
        result = self.plan('new_component/kernel.cu')
        self.assertEqual(self.selected(result), set(self.entries))
        self.assertEqual(result['targets'], [])
        self.assertTrue(all(row['force_export'] for row in result['matrix']['include']))

    def test_unknown_yaml_is_not_silently_ignored(self):
        self.assertEqual(self.selected(self.plan('model-settings.yaml')), set(self.entries))

    def test_missing_dependency_entry_always_runs_existing_recipe(self):
        del self.config['models']['parakeet']
        result = self.plan('vision/sam2/cpp/main.cpp')
        self.assertIn('parakeet', self.selected(result))
        self.assertTrue(next(row['force_export'] for row in result['matrix']['include'] if row['model'] == 'parakeet'))

    def test_new_recipe_requires_no_planner_change(self):
        self.entries['new-model'] = copy.deepcopy(self.entries['parakeet'])
        self.entries['new-model']['targets'] = ['din_new_cli']
        self.entries['new-model']['variants'] = [{'slug': 'new-variant'}]
        result = self.plan('assets/sample.wav')
        self.assertIn({'model': 'new-model', 'variant': 'new-variant', 'force_export': True}, result['matrix']['include'])
        self.assertIn('din_new_cli', result['targets'])

    def test_unknown_group_runs_component_conservatively(self):
        self.config['models']['whisper']['export']['groups'] = ['missing']
        result = self.plan('asr/rnnt/cpp/parakeet_tdt.cpp')
        self.assertIn('whisper', self.selected(result))

    def test_unknown_target_forces_full_build(self):
        result = self.plan('assets/sample.wav', unknown_targets=['models/new/CMakeLists.txt: din_new'])
        self.assertEqual(result['targets'], [])
        self.assertTrue(result['build'])
        self.assertEqual(self.selected(result), set(self.entries))

    def test_manual_or_unavailable_base_runs_all(self):
        result = ci.plan(None, self.config, self.entries)
        self.assertTrue(result['build'])
        self.assertEqual(self.selected(result), set(self.entries))

    def test_missing_configuration_runs_all(self):
        result = ci.plan(['README.md'], {}, self.entries)
        self.assertTrue(result['build'])
        self.assertEqual(self.selected(result), set(self.entries))

    def test_workflow_change_validates_all_models(self):
        self.assertEqual(self.selected(self.plan('.github/workflows/ci.yml')), set(self.entries))

    def test_flux_changes_build_without_unrelated_inference(self):
        result = self.plan('image_gen/flux2/cpp/flux.cpp')
        self.assertEqual(result['targets'], ['din_flux2_cli'])
        self.assertEqual(self.selected(result), set())

    def test_variant_matrix_preserves_existing_coverage(self):
        result = ci.plan(None, self.config, self.entries)
        self.assertEqual(len(result['matrix']['include']), 12)


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'CI Test')
        self.git('config', 'user.email', 'ci@example.invalid')
        self.write('export.py', 'initial')
        self.write('runtime.cpp', 'initial')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial')
        self.base = self.git('rev-parse', 'HEAD').strip()
        self.config = {'models': {'demo': {'export': {'paths': ['export.py']}, 'runtime': {'paths': ['runtime.cpp']}}}}
        self.entries = {'demo': {'export': {'command': ['export.py']}}}
        self.variant = {'model_id': 'model/revision', 'precision': 'fp16'}

    def git(self, *args):
        return ci.git(*args, root=self.root).decode()

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def key(self, **kwargs):
        return ci.export_key('demo', self.variant, self.config, self.entries, root=self.root, **kwargs)

    def test_runtime_change_preserves_export_key(self):
        before = self.key()
        self.write('runtime.cpp', 'changed')
        self.git('add', '.')
        self.assertEqual(before, self.key())

    def test_exporter_edit_delete_and_rename_change_key(self):
        previous = self.key()
        self.write('export.py', 'changed')
        self.git('add', '.')
        self.assertNotEqual(previous, self.key())
        previous = self.key()
        self.git('mv', 'export.py', 'renamed.py')
        self.assertNotEqual(previous, self.key())
        previous = self.key()
        self.git('rm', '-f', 'renamed.py')
        self.assertNotEqual(previous, self.key())

    def test_unknown_file_stays_in_key_on_future_runs(self):
        before = self.key()
        self.write('unowned_helper.py', 'new export dependency')
        self.git('add', '.')
        self.assertNotEqual(before, self.key())

    def test_export_settings_change_key(self):
        before = self.key()
        self.entries['demo']['export']['install'] = ['changed dependency']
        self.assertNotEqual(before, self.key())
        before = self.key()
        self.variant['model_id'] = 'different/model'
        self.assertNotEqual(before, self.key())

    def test_fallback_cannot_reuse_previous_run(self):
        with patch.dict(os.environ, {'GITHUB_RUN_ID': '1'}):
            before = self.key(force=True)
        with patch.dict(os.environ, {'GITHUB_RUN_ID': '2'}):
            self.assertNotEqual(before, self.key(force=True))

    def test_deletion_and_rename_list_both_paths(self):
        self.git('mv', 'runtime.cpp', 'renamed.cpp')
        self.git('commit', '-qm', 'rename')
        changed = ci.changed_files({'before': self.base}, root=self.root)
        self.assertIn('runtime.cpp', changed)
        self.assertIn('renamed.cpp', changed)

    def test_pr_uses_merge_base(self):
        self.git('checkout', '-qb', 'base-update')
        self.write('base-only.cpp', 'base change')
        self.git('add', '.')
        self.git('commit', '-qm', 'base update')
        base_tip = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '-qb', 'feature', self.base)
        self.write('runtime.cpp', 'feature change')
        self.git('add', '.')
        self.git('commit', '-qm', 'feature')
        changed = ci.changed_files({'pull_request': {'base': {'sha': base_tip}}}, root=self.root)
        self.assertIn('runtime.cpp', changed)
        self.assertNotIn('base-only.cpp', changed)

    def test_unknown_target_discovery_is_independent_of_registry(self):
        self.write('new/CMakeLists.txt', 'add_din_executable(din_new_cli main.cpp)')
        self.git('add', '.')
        found = ci.discover_unregistered({}, {}, root=self.root)
        self.assertEqual(found, ['new/CMakeLists.txt: din_new_cli'])


class ExecutionTests(unittest.TestCase):
    def test_subprocess_failures_propagate(self):
        with patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['cli'])):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run(['{executable}', '--model-dir', '{output}'], {'executable': 'cli', 'output': 'model'})

    def test_command_arguments_do_not_go_through_shell(self):
        with patch.object(subprocess, 'run') as run:
            runner.run(['cli', '{output}'], {'output': 'directory with spaces'})
            self.assertEqual(run.call_args.args[0], ['cli', 'directory with spaces'])
            self.assertNotIn('shell', run.call_args.kwargs)

    def test_all_current_recipes_expand(self):
        for entry in ci.recipes().values():
            for variant in entry['variants']:
                values = dict(variant, python='python', output='model', executable='cli', dtype='float16')
                runner.expand(entry['export']['command'], values)
                runner.expand(entry['smoke']['command'], values)
                for command in entry['export']['install']:
                    runner.expand(command, values)


class ExportExecutionTests(unittest.TestCase):
    def test_export_retries_fallback_and_records_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model'
            outputs = Path(directory) / 'outputs'
            definition = {'export': {'install': [['install']], 'command': ['export', '{precision}'], 'required_files': ['model.onnx']}}
            variant = {'slug': 'example', 'model_id': 'org/model', 'precision': 'fp16', 'fallback_precision': 'fp32'}
            values = dict(variant, output=str(output))
            def execute(command, expanded, env):
                if command[0] == 'install':
                    return
                output.mkdir(parents=True)
                if expanded['precision'] == 'fp16':
                    (output / 'partial').write_text('partial export')
                    raise subprocess.CalledProcessError(1, command)
                self.assertFalse((output / 'partial').exists())
                (output / 'model.onnx').write_text('valid model')
            with patch.object(runner, 'run', side_effect=execute), patch.dict(os.environ, {'GITHUB_OUTPUT': str(outputs), 'GITHUB_RUN_ID': '123'}):
                runner.export_model(definition, variant, values, 'fingerprint')
            self.assertIn('name=model-example-fp32-fingerprint', outputs.read_text())
            self.assertEqual(json.loads((output / 'ci-export-manifest.json').read_text())['precision'], 'fp32')

    def test_successful_command_with_missing_artifacts_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = {'export': {'install': [], 'command': ['export'], 'required_files': ['missing.onnx']}}
            variant = {'precision': 'fp32'}
            with patch.object(runner, 'run'):
                with self.assertRaisesRegex(RuntimeError, 'required files'):
                    runner.export_model(entry, variant, {'output': str(Path(directory) / 'model')}, 'key')

    def test_missing_artifacts_retry_fallback_and_clear_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model'
            outputs = Path(directory) / 'outputs'
            entry = {'export': {'install': [], 'command': ['export'], 'required_files': ['model.onnx', 'metadata.json']}}
            variant = {'slug': 'example', 'model_id': 'org/model', 'precision': 'fp16', 'fallback_precision': 'fp32'}
            attempted = []

            def execute(command, values, env):
                attempted.append(values['precision'])
                output.mkdir(parents=True)
                if values['precision'] == 'fp16':
                    (output / 'partial').write_text('partial export')
                    (output / 'model.onnx').write_text('stale model')
                else:
                    self.assertFalse((output / 'partial').exists())
                    self.assertFalse((output / 'model.onnx').exists())
                    (output / 'model.onnx').write_text('valid model')
                    (output / 'metadata.json').write_text('valid metadata')

            with patch.object(runner, 'run', side_effect=execute), patch.dict(os.environ, {'GITHUB_OUTPUT': str(outputs), 'GITHUB_RUN_ID': '123'}):
                runner.export_model(entry, variant, {'output': str(output)}, 'fingerprint')
            self.assertEqual(attempted, ['fp16', 'fp32'])
            self.assertIn('name=model-example-fp32-fingerprint', outputs.read_text())
            self.assertEqual(json.loads((output / 'ci-export-manifest.json').read_text())['precision'], 'fp32')

    def test_missing_artifacts_after_fallback_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model'
            entry = {'export': {'install': [], 'command': ['export'], 'required_files': ['model.onnx']}}
            variant = {'precision': 'fp16', 'fallback_precision': 'fp32'}
            attempted = []

            def execute(command, values, env):
                attempted.append(values['precision'])
                output.mkdir(parents=True)

            with patch.object(runner, 'run', side_effect=execute), patch.object(runner, 'write_outputs') as write_outputs:
                with self.assertRaisesRegex(RuntimeError, 'required files.*model.onnx'):
                    runner.export_model(entry, variant, {'output': str(output)}, 'key')
            self.assertEqual(attempted, ['fp16', 'fp32'])
            self.assertFalse((output / 'ci-export-manifest.json').exists())
            write_outputs.assert_not_called()


class ConfigurationTests(unittest.TestCase):
    def test_malformed_optional_configuration_disables_optimization(self):
        for content in ['{', '[]', '{"version": 1, "models": {"demo": []}}', '{"version": 1, "groups": {"shared": "not-a-list"}}']:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / '.github').mkdir()
                (root / ci.CONFIG).write_text(content)
                self.assertEqual(ci.read_config(root), {})


if __name__ == '__main__':
    unittest.main()
