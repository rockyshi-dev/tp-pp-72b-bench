"""CPU-only launch tests: arm argv/env differences, numactl PATH, version and code checks,
API-server placement. No server, HTTP or GPU process."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tppbench import launch
from tppbench import spec
from tppbench.spec import ARMS, LADDER, PARALLEL_FLAGS, digest


def strip_parallel(argv, arm):
    flags = list(PARALLEL_FLAGS[arm])
    for i in range(len(argv) - len(flags) + 1):
        if argv[i:i + len(flags)] == flags:
            return argv[:i] + argv[i + len(flags):]
    raise AssertionError(f'{arm} flags missing')


def fake_numactl(root):
    prefix = Path(root) / 'numactl'
    (prefix / 'bin').mkdir(parents=True)
    binary = prefix / 'bin/numactl'
    binary.write_text('#!/bin/sh\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    (prefix / 'path-bin').mkdir()
    (prefix / 'path-bin/numactl').symlink_to('../bin/numactl')
    probe = Path(root) / 'probe.json'
    probe.write_text(json.dumps({'pass': True, 'mode': 'restricted', 'path_dir': str(prefix / 'path-bin'),
                                 'numactl': str(binary), 'numactl_sha256': digest(binary)}))
    return probe


class Argv(unittest.TestCase):
    def test_only_the_parallel_flags_differ_between_arms(self):
        for step in LADDER:
            stripped = [strip_parallel(launch.argv(arm, 8150, step), arm) for arm in ARMS]
            self.assertEqual(stripped[0], stripped[1])
            self.assertEqual(stripped[0], stripped[2])
        self.assertIn('-tp', launch.argv('TP4xPP2', 8150, 1))

    def test_declared_pool_ladder_and_binding(self):
        a1, a2, a3 = (launch.argv('PP8', 8150, s) for s in (1, 2, 3))
        self.assertEqual(a1[a1.index('--kv-cache-memory-bytes') + 1], '2415919104')
        self.assertNotIn('--max-model-len', a1)
        self.assertEqual(a2[a2.index('--kv-cache-memory-bytes') + 1:a2.index('--kv-cache-memory-bytes') + 4],
                         ['1342177280', '--max-model-len', '8192'])
        self.assertEqual(a3[a3.index('--kv-cache-memory-bytes') + 1], '805306368')
        i = a1.index('--numa-bind-nodes')
        self.assertEqual(a1[i - 1], '--numa-bind')
        self.assertEqual(a1[i + 1:i + 9], ['0', '0', '0', '0', '1', '1', '1', '1'])
        j = a1.index('--numa-bind-cpus')
        self.assertEqual(a1[j + 1:j + 9], ['0-21,52-73'] * 4 + ['26-43,78-95'] * 4)
        self.assertNotIn('--profiler-config', a1)
        self.assertEqual(a1[1:3], ['serve', str(spec.model_dir())])
        self.assertEqual(a1[0], str(launch.vllm_bin()))


class Environment(unittest.TestCase):
    def env(self, root, extra=None, **kind):
        numactl = launch.numactl_dir(fake_numactl(root))
        base = {'HF_HOME': '/models/hf', 'PATH': '/usr/bin:/bin', **(extra or {})}
        with patch.dict(os.environ, base, clear=True), patch.object(launch, 'check_vllm') as check:
            env = launch.environment({}, numactl, **kind)
        check.assert_called_once()
        return env

    def test_declared_differences_only(self):
        with tempfile.TemporaryDirectory() as root:
            default = self.env(root + '/a')
            transport = self.env(root + '/b', transport=True, nccl_dir='/tmp/nccl')
            settings = self.env(root + '/c', settings=True)
        for env in (default, transport, settings):
            self.assertEqual(env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
            self.assertTrue(env['PATH'].endswith('/path-bin:/usr/bin:/bin'))
        strip = lambda e: {k: v for k, v in e.items() if k != 'PATH'}
        self.assertEqual({k for k in strip(transport) if strip(transport).get(k) != strip(default).get(k)},
                         {'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_DEBUG_FILE'})
        self.assertEqual(transport['NCCL_DEBUG_FILE'], '/tmp/nccl/nccl.%h.%p.log')
        self.assertEqual(transport['NCCL_DEBUG_SUBSYS'], 'INIT,BOOTSTRAP,ENV,SHM')
        self.assertEqual({k for k in strip(settings) if strip(settings).get(k) != strip(default).get(k)},
                         {'VLLM_LOGGING_LEVEL'})

    def test_inherited_overrides_refused(self):
        for key in ('NCCL_P2P_LEVEL', 'VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC', 'LD_PRELOAD', 'CUDA_LAUNCH_BLOCKING',
                    'TORCH_NCCL_BLOCKING_WAIT', 'OMP_NUM_THREADS'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root, self.assertRaises(RuntimeError):
                self.env(root, {key: '1'})

    def test_numactl_receipt_checks(self):
        with tempfile.TemporaryDirectory() as root:
            probe = fake_numactl(root)
            data = json.loads(probe.read_text())
            for change in ({'pass': False}, {'mode': 'neither'}, {'numactl_sha256': '0' * 64}):
                probe.write_text(json.dumps({**data, **change}))
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    launch.numactl_dir(probe)
            probe.write_text(json.dumps(data))
            (Path(data['path_dir']) / 'python').write_text('')
            with self.assertRaises(RuntimeError):
                launch.numactl_dir(probe)


class VersionAndCode(unittest.TestCase):
    def test_check_vllm(self):
        with tempfile.TemporaryDirectory() as root:
            version = Path(root) / '_version.py'
            version.write_text("__version__ = '0.30.0+cu129'\n__commit__ = 'ced6857afa0e'\n")
            dist = SimpleNamespace(locate_file=lambda name: version)
            exe = Path(root) / 'bin/python'
            exe.parent.mkdir()
            (exe.parent / 'vllm').write_text('')
            with patch.object(launch.importlib.metadata, 'version', return_value='0.30.0+cu129'), \
                    patch.object(launch.importlib.metadata, 'distribution', return_value=dist), \
                    patch.object(launch.sys, 'executable', str(exe)):
                launch.check_vllm()
                version.write_text("__commit__ = 'abcdef0'\n")
                with self.assertRaisesRegex(RuntimeError, 'commit'):
                    launch.check_vllm()
            with patch.object(launch.importlib.metadata, 'version', return_value='0.30.0'), \
                    self.assertRaisesRegex(RuntimeError, 'CUDA 12.9'):
                launch.check_vllm()

    def test_code_files_cover_python_and_data(self):
        with tempfile.TemporaryDirectory() as root:
            code = Path(root)
            (code / 'runner.py').write_text('x')
            (code / 'gate_e_prompts.json').write_text('{}')
            (code / 'notes.txt').write_text('ignored')
            first = launch.code_files(code)
            self.assertEqual(sorted(first), ['gate_e_prompts.json', 'runner.py'])
            (code / 'gate_e_prompts.json').write_text('{"changed": 1}')
            self.assertNotEqual(launch.code_files(code), first)
        self.assertIn('gates.py', launch.code_files())


class ApiPlacement(unittest.TestCase):
    def server(self, mode, directory):
        server = launch.Server.__new__(launch.Server)
        server.placement_mode = mode
        server.process = SimpleNamespace(pid=4321)
        server.directory = Path(directory)
        return server

    def proc(self, root):
        for tid in ('4321', '4322'):
            (Path(root) / 'proc/4321/task' / tid).mkdir(parents=True)
        return Path(root) / 'proc'

    def test_fallback_moves_every_thread(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            run = lambda argv, **k: calls.append(argv) or SimpleNamespace(returncode=0, stdout='', stderr='')
            with patch.object(launch.os, 'sched_getaffinity', return_value={22, 23, 24, 25, 74, 75, 76, 77}):
                self.server('fallback', root).api_server_placement(run, self.proc(root))
            self.assertEqual(calls, [['taskset', '-a', '-p', '-c', '22-25,74-77', '4321']])

    def test_wrong_affinity_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(launch.os, 'sched_getaffinity', return_value=set(range(104))), \
                 self.assertRaises(launch.gates.EvidenceError):
                self.server('restricted', root).api_server_placement(proc=self.proc(root))


class Signatures(unittest.TestCase):
    def test_deviation_signatures(self):
        log = 'vLLM is using nccl==2.29.7\nWorker proc X died unexpectedly (exit code: -11)\ncuMemCreate'
        self.assertEqual(launch.startup_signature(log, 1, False), 'NCCL_CUMEM_HOST_ENABLE')
        hang = 'vLLM is using nccl==2.29.7\nnothing'
        self.assertEqual(launch.startup_signature(hang, None, True, 1800), 'NCCL_P2P_DISABLE')
        self.assertIsNone(launch.startup_signature(hang, None, True, 1799))
        self.assertIsNone(launch.startup_signature(hang + '\nLoading weights took 9 s', None, True, 1800))


class LockNames(unittest.TestCase):
    def test_run_lock_and_one_lock_per_gpu(self):
        import fcntl
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'run'
            spec.configure(data_dir=Path(root) / 'data', lock_dir=Path(root) / 'locks')
            try:
                locks = launch.Locks(out)
                names = sorted(Path(f.name).name for f in locks.files)
                self.assertEqual(names, sorted(['run.lock'] + [f'gpu{g}.lock' for g in range(8)]))
                with open(Path(root) / 'locks/gpu3.lock', 'a') as other, self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.close()
                with self.assertRaises(FileExistsError):      # a used run directory is never reused
                    launch.Locks(out)
            finally:
                spec._PATHS.clear()


class Redaction(unittest.TestCase):
    def test_secret_and_ssh_values_are_masked(self):
        env = {'HF_TOKEN': 'hf_abc', 'hf_token': 'x', 'OPENAI_API_KEY': 'sk-1', 'AWS_SECRET_ACCESS_KEY': 'a',
               'DB_PASSWORD': 'p', 'GOOGLE_APPLICATION_CREDENTIALS': '/k.json', 'SSH_CLIENT': '10.0.0.1 5 22',
               'ssh_auth_sock': '/tmp/s', 'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '0,1', 'HF_HOME': '/hf'}
        out = launch.redact_environment(env)
        self.assertEqual(set(out), set(env))                        # names kept, values masked
        for key in ('HF_TOKEN', 'hf_token', 'OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'DB_PASSWORD',
                    'GOOGLE_APPLICATION_CREDENTIALS', 'SSH_CLIENT', 'ssh_auth_sock'):
            self.assertEqual(out[key], '<redacted>', key)
        for key in ('PATH', 'CUDA_VISIBLE_DEVICES', 'HF_HOME'):
            self.assertEqual(out[key], env[key])
        self.assertEqual(env['HF_TOKEN'], 'hf_abc')                 # the server's environment is untouched

    def test_manifest_records_only_the_redacted_environment(self):
        source = (Path(launch.__file__)).read_text()
        self.assertIn("'environment': redact_environment(env)", source)
        self.assertNotIn("'environment': env", source)


class ExtraLocks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        spec.configure(data_dir=self.root / 'data', lock_dir=self.root / 'locks')

    def tearDown(self):
        spec._PATHS.clear()
        self.tmp.cleanup()

    def test_gpu_pattern_is_expanded_and_deduplicated(self):
        shared = self.root / 'shared'
        extra = [f'{shared}/gpu-{{gpu}}.lock', f'{shared}/gpu-{{gpu}}.lock', f'{shared}/node.lock',
                 f'{self.root}/locks/gpu{{gpu}}.lock']           # the last one equals the run's own names
        paths = launch.lock_paths(self.root / 'run', extra)
        names = [str(p) for p in paths]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 1 + 8 + 8 + 1)
        for g in range(8):
            self.assertIn(str(shared / f'gpu-{g}.lock'), names)
        import fcntl
        locks = launch.Locks(self.root / 'run', extra)
        try:
            with open(shared / 'gpu-5.lock', 'a') as other, self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            locks.close()

    def test_pattern_without_gpu_naming_an_own_lock_is_refused(self):
        for bad in (f'{self.root}/locks/gpu3.lock', f'{self.root}/run/run.lock'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "own lock files"):
                launch.lock_paths(self.root / 'run', [bad])
        with self.assertRaises(ValueError):
            launch.Locks(self.root / 'run2', [f'{self.root}/locks/gpu0.lock'])
        self.assertEqual(sorted(p.name for p in (self.root / 'run2').iterdir()), [])   # nothing held or left


if __name__ == '__main__':
    unittest.main()
