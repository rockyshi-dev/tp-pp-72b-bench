"""CPU-only tests of the touch-and-free step before every server start and the per-level
memory records. hostmem.run and launch.Server are mocked."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tppbench import gates
from tppbench import hostmem
from tppbench import launch
from tppbench import runner
from tppbench import telemetry
from tppbench.spec import HOST_MEMORY_NOTE, USABLE_FREE_AFTER_HELPER, cpu_list

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'tppbench'
OK, LOW = USABLE_FREE_AFTER_HELPER, USABLE_FREE_AFTER_HELPER - 1


def helper_rows(nodes, usable):
    return {n: {'node': n, 'pid': 1000 + n, 'seconds': 1.0, 'before': {'usable_free_bytes': 1},
                'after': {'usable_free_bytes': usable[n]}} for n in nodes}


class HostMemoryStep(unittest.TestCase):
    def step(self, outcomes):
        """outcomes: list of {node: usable} or a Refusal, one per hostmem.run call."""
        calls = []

        def run(nodes, directory, owned, cpus_of=None, **kw):
            calls.append({'nodes': nodes, 'directory': Path(directory), 'owned': owned, 'cpus_of': cpus_of})
            value = outcomes.pop(0)
            if isinstance(value, Exception):
                raise value
            return helper_rows(nodes, value)
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        start = Path(root.name) / 'pass1-step1-TP8'
        with patch.object(hostmem, 'run', side_effect=run):
            try:
                result = runner.host_memory_step(start)
                error = None
            except runner.HostMemoryStop as exc:
                result, error = None, exc
        record = json.loads((Path(root.name) / 'pass1-step1-TP8-host-memory/host-memory.json').read_text())
        return result, error, record, calls

    def test_both_nodes_in_parallel_bound_to_their_cores(self):
        result, error, record, calls = self.step([{0: OK, 1: OK}])
        self.assertIsNone(error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['nodes'], (0, 1))                  # one hostmem.run = both nodes in parallel
        self.assertIs(calls[0]['owned'], telemetry.HELPER_PIDS)
        self.assertEqual(calls[0]['cpus_of'], {0: cpu_list('0-21,52-73'), 1: cpu_list('26-43,78-95')})
        self.assertEqual((record['precondition'], record['gate'], record['note']),
                         ('met', False, HOST_MEMORY_NOTE))

    def test_repeat_once_on_the_failing_node_then_pass(self):
        result, error, record, calls = self.step([{0: LOW, 1: OK}, {0: OK}])
        self.assertIsNone(error)
        self.assertEqual([c['nodes'] for c in calls], [(0, 1), (0,)])
        self.assertEqual([c['directory'].name for c in calls], ['helper-1', 'helper-2'])
        self.assertEqual([(r['helper'], r['node'], r['precondition_met']) for r in record['runs']],
                         [(1, 0, False), (1, 1, True), (2, 0, True)])

    def test_repeat_then_stop(self):
        result, error, record, calls = self.step([{0: OK, 1: LOW}, {1: LOW}])
        self.assertIsInstance(error, runner.HostMemoryStop)
        self.assertEqual([c['nodes'] for c in calls], [(0, 1), (1,)])
        self.assertEqual((record['stopped'], record['gate_result'], record['start_test_failure'],
                          record['failing_nodes']), (True, False, False, [1]))

    def test_refusal_stops_at_once(self):
        refusal = hostmem.Refusal('node 1: reclaimable 1 bytes <= 44000000000 bytes', {1: {'refused': True}})
        result, error, record, calls = self.step([refusal])
        self.assertIsInstance(error, runner.HostMemoryStop)
        self.assertEqual(len(calls), 1)
        self.assertEqual((record['gate_result'], record['refusal']), (False, {'1': {'refused': True}}))

    def test_refusal_on_repeat_stops(self):
        refusal = hostmem.Refusal('node 0 refused', {0: {'refused': True}})
        _, error, record, calls = self.step([{0: LOW, 1: OK}, refusal])
        self.assertIsInstance(error, runner.HostMemoryStop)
        self.assertEqual(len(calls), 2)


class Sentinel(BaseException):
    """Stands in for a server start: raised by the fake launch.Server."""


class EveryStart(unittest.TestCase):
    """The helper runs before every server start of every phase."""

    def args(self, root):
        bank = {'requests': [{'id': i, 'messages': []} for i in range(64)],
                'banks': {'burst': {'measured': list(range(64))}, '1:0': {'measured': list(range(30))}},
                'gate_e_non_english': {'prompts': []}}
        return SimpleNamespace(topology={'client_cpus': [44], 'runner_cpus': [96]}, locks=None, deviations={},
                               port=8150, numactl={'mode': 'restricted'}, bank=bank,
                               study={'step': 1, 'arms': ['TP8', 'PP8']}, output=Path(root))

    def phases(self, args, root):
        root = Path(root)
        yield 'start-test', lambda: runner.start_test(args, 'TP8', 1, root / 'pass1-step1-TP8')
        yield 'settings', lambda: runner.settings(args, root / 'settings')
        yield 'transport', lambda: runner.transport(args, root / 'transport')
        yield 'timed', lambda: runner.timed_part(args, 'TP8', root / 'round-1-attempt-0/TP8', (1, 8, 32), 0)

    def test_helper_precedes_the_server_in_each_phase(self):
        for name in ('start-test', 'settings', 'transport', 'timed'):
            with self.subTest(phase=name), tempfile.TemporaryDirectory() as root:
                order = []

                def server(arm, directory, *a, **k):
                    order.append(('server', Path(directory)))
                    raise Sentinel()
                args = self.args(root)
                with patch.object(launch, 'preflight', side_effect=lambda port: order.append(('preflight', port))), \
                        patch.object(runner, 'host_memory_step',
                                     side_effect=lambda d: order.append(('helper', Path(d)))), \
                        patch.object(launch, 'Server', side_effect=server):
                    run = dict(self.phases(args, root))[name]
                    with self.assertRaises(Sentinel):
                        run()
                self.assertEqual([k for k, _ in order], ['preflight', 'helper', 'server'])
                self.assertEqual(order[1][1], order[2][1])            # same start directory

    def test_stop_prevents_the_server_in_each_phase(self):
        for name in ('start-test', 'settings', 'transport', 'timed'):
            with self.subTest(phase=name), tempfile.TemporaryDirectory() as root:
                args = self.args(root)
                with patch.object(launch, 'preflight', return_value={}), \
                        patch.object(runner, 'host_memory_step', side_effect=runner.HostMemoryStop('low')), \
                        patch.object(launch, 'Server', side_effect=AssertionError('server started')) as server:
                    with self.assertRaises(runner.HostMemoryStop):
                        dict(self.phases(args, root))[name]()
                server.assert_not_called()
                self.assertFalse(any(Path(root).rglob('start-test.json')))   # not a start-test failure

    def test_open_server_is_the_only_start(self):
        sources = {p.name: p.read_text() for p in PACKAGE.glob('*.py')}
        starts = {n: len(re.findall(r'launch\.Server\(', s)) for n, s in sources.items()}
        self.assertEqual({n: k for n, k in starts.items() if k}, {'runner.py': 1})
        body = sources['runner.py'].split('def open_server', 1)[1].split('\ndef ', 1)[0]
        self.assertLess(body.index('launch.preflight('), body.index('host_memory_step('))
        self.assertLess(body.index('host_memory_step('), body.index('launch.Server('))

    def test_foreign_compute_process_stops_before_the_helper(self):
        foreign = {'gpus': [], 'compute_apps': [{'pid': 4242, 'uuid': 'GPU-x', 'name': 'python'}]}
        for name in ('start-test', 'settings', 'transport', 'timed'):
            with self.subTest(phase=name), tempfile.TemporaryDirectory() as root:
                args = self.args(root)
                with patch.object(launch.telemetry, 'gpu_snapshot', return_value=foreign), \
                        patch.object(launch, 'check_port') as port, \
                        patch.object(hostmem, 'run', side_effect=AssertionError('helper ran')) as helper, \
                        patch.object(launch, 'Server', side_effect=AssertionError('server started')) as server:
                    with self.assertRaisesRegex(RuntimeError, 'Foreign compute process'):
                        dict(self.phases(args, root))[name]()
                helper.assert_not_called()
                server.assert_not_called()
                port.assert_not_called()
                self.assertFalse(any(Path(root).rglob('*-host-memory')))

    def test_busy_port_stops_before_the_helper(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root)
            with patch.object(launch.telemetry, 'gpu_snapshot', return_value={'gpus': [], 'compute_apps': []}), \
                    patch.object(launch, 'check_port', side_effect=OSError(98, 'Address already in use')), \
                    patch.object(hostmem, 'run', side_effect=AssertionError('helper ran')) as helper:
                with self.assertRaises(OSError):
                    runner.start_test(args, 'TP8', 1, Path(root) / 'pass1-step1-TP8')
            helper.assert_not_called()


class StopNeverMovesLadder(unittest.TestCase):
    def test_stop_propagates_without_a_ladder_step(self):
        calls = []

        def start_test(args, arm, step, directory):
            calls.append((arm, step))
            if (arm, step) == ('TP4xPP2', 1):
                raise runner.HostMemoryStop('node 0 low')
            return True, {'arm': arm, 'step': step, 'pass': True}
        with tempfile.TemporaryDirectory() as root, \
                patch.object(runner, 'start_test', side_effect=start_test), \
                patch.object(gates, 'ladder_step', side_effect=AssertionError('ladder moved')) as step:
            with self.assertRaises(runner.HostMemoryStop):
                runner.ladder(None, Path(root))
            attempts = json.loads((Path(root) / 'ladder-attempts.json').read_text())
        step.assert_not_called()
        self.assertEqual(calls, [('TP8', 1), ('TP4xPP2', 1)])
        self.assertEqual(attempts, [{'arm': 'TP8', 'step': 1, 'pass': True}])   # the stop is not an attempt

    def test_real_start_test_raises_instead_of_failing(self):
        args = EveryStart().args('/nonexistent')
        with tempfile.TemporaryDirectory() as root, patch.object(launch, 'preflight', return_value={}), \
                patch.object(runner, 'host_memory_step', side_effect=runner.HostMemoryStop('refused')):
            with self.assertRaises(runner.HostMemoryStop):
                runner.start_test(args, 'PP8', 1, Path(root) / 'pass1-step1-PP8')
            self.assertEqual(list(Path(root).iterdir()), [])


class LevelRecords(unittest.TestCase):
    VMSTAT_1 = 'nr_free_pages 5\npgscan_kswapd 100\npgscan_direct 7\nallocstall_normal 2\nallocstall_movable 1\n'
    VMSTAT_2 = 'nr_free_pages 4\npgscan_kswapd 160\npgscan_direct 7\nallocstall_normal 5\nallocstall_movable 1\n'

    def test_before_and_delta(self):
        texts = [self.VMSTAT_1, self.VMSTAT_2]
        vmstat = lambda: runner.read_vmstat(texts.pop(0))
        snap = lambda node: {'node': node, 'usable_free_bytes': 30 * 10**9 + node}
        with tempfile.TemporaryDirectory() as root:
            record = runner.level_memory_before(root, snapshot=snap, vmstat=vmstat)
            self.assertEqual(record['usable_free_bytes_before'], {'0': 30 * 10**9, '1': 30 * 10**9 + 1})
            runner.level_memory_after(root, record, vmstat=vmstat)
            saved = json.loads((Path(root) / 'host-memory-level.json').read_text())
        self.assertEqual(saved['vmstat_delta'], {'allocstall_movable': 0, 'allocstall_normal': 3,
                                                 'pgscan_direct': 0, 'pgscan_kswapd': 60})
        self.assertFalse(saved['gate'])

    def test_real_vmstat_fixture(self):
        counters = runner.reclaim_counters(runner.read_vmstat((HERE / 'fixtures/hostmem/vmstat.txt').read_text()))
        self.assertEqual(list(counters)[:2], ['pgscan_kswapd', 'pgscan_direct'])
        self.assertTrue(all(isinstance(v, int) for v in counters.values()))
        self.assertTrue(any(k.startswith('allocstall_') for k in counters))

    def test_failed_reads_are_recorded_not_raised(self):
        def snap(node):
            raise ValueError('no Normal zone')

        def vmstat():
            raise OSError('no vmstat')
        with tempfile.TemporaryDirectory() as root:
            record = runner.level_memory_before(root, snapshot=snap, vmstat=vmstat)
            runner.level_memory_after(root, record, vmstat=vmstat)
        self.assertEqual(set(record['node_errors']), {'0', '1'})
        self.assertIn('vmstat_error', record)
        self.assertIn('vmstat_error_after', record)

    def test_run_level_records_around_the_level(self):
        order = []

        class Monitor:
            def __init__(self, root, directory, cpus):
                self.path = Path(directory) / 'telemetry.jsonl'
                self.path.write_text('')

            def sample(self):
                return {'monotonic_ns': 0}

            boundary = sample

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with tempfile.TemporaryDirectory() as root:
            server_dir = Path(root) / 'server'
            server_dir.mkdir()
            (server_dir / 'metrics-startup.txt').write_text('')
            server = SimpleNamespace(log_text=lambda: '', directory=server_dir, workers=[], url='http://x',
                                     process=SimpleNamespace(pid=1, poll=lambda: None))
            phase = lambda url, rows, c, cpus, d: order.append(('requests', Path(d).name)) or [{}] * len(rows)
            with patch.object(runner, 'level_memory_before',
                              side_effect=lambda d: order.append('memory-before') or {}), \
                    patch.object(runner, 'level_memory_after',
                                 side_effect=lambda d, r: order.append('memory-after') or r), \
                    patch.object(runner.client, 'phase', side_effect=phase), \
                    patch.object(runner.gates, 'request_valid', return_value=True), \
                    patch.object(runner.telemetry, 'placement', return_value={}), \
                    patch.object(runner.telemetry, 'Monitor', Monitor), \
                    patch.object(runner, 'metrics', side_effect=lambda s, d, n: order.append(n) or ''), \
                    patch.object(runner, 'check_m', return_value={'pass': True}), \
                    patch.object(runner.gates, 'level', return_value={'window_ns': (0, 1)}), \
                    patch.object(runner.gates, 'binding', return_value={}), \
                    patch.object(runner.gates, 'placement', return_value={'pass': True}), \
                    patch.object(runner.gates, 'quiet', return_value={'pass': True}):
                runner.run_level(server, [{}] * 16, [{'id': i} for i in range(80)], 8, {'client_cpus': [44], 'runner_cpus': [96]},
                                 Path(root) / 'c8', 0, timed=True)
        self.assertEqual(order, ['memory-before', ('requests', 'warmup'), 'metrics-before.txt',
                                 ('requests', 'measured'), 'metrics-after.txt', 'memory-after'])


class OwnedHelper(unittest.TestCase):
    def test_live_helper_pid_is_owned(self):
        table = {10: {'ppid': 1, 'start': 1, 'ticks': 5, 'child_ticks': 0, 'state': 'R'},
                 77: {'ppid': 1, 'start': 2, 'ticks': 9, 'child_ticks': 0, 'state': 'R'}}
        with patch.object(telemetry, 'HELPER_PIDS', {77, 999}):
            rows = telemetry.owned_rows(10, table, affinity=lambda pid: {0, 1}, threads=lambda pid: {})
        self.assertEqual(set(rows), {'10', '77'})
        with patch.object(telemetry, 'HELPER_PIDS', set()):
            self.assertEqual(set(telemetry.owned_rows(10, table, affinity=lambda pid: {0}, threads=lambda p: {})),
                             {'10'})


if __name__ == '__main__':
    unittest.main()
