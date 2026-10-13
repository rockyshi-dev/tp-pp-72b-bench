"""CPU-only tests of the phase drivers with mocked starts: rotation, gate M
reruns, gate K equality and T/C retries in the timed loop, the ladder driver,
and the client's split and gate E requests."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tppbench import client
from tppbench import runner


def cells(levels, cap=58976, **flags):
    out = {}
    for c in levels:
        cell = {'K_capacity': True, 'K_preemption': True, 'kv_capacity_tokens': cap, 'T': True, 'C': True,
                'G': True, 'N': True, 'Q': True, 'rerun_level': False, 'ttft_s': 1., 'token_gap_s': 1.,
                'output_tokens_per_second': 1.}
        cell.update(flags)
        cell['rerun_level'] = not (cell['T'] and cell['C'])
        out[str(c)] = cell
    return out


class Timed(unittest.TestCase):
    def test_errored_boundary_sample_through_the_real_path_reruns_the_round(self):
        # Real run_level ->
        # telemetry.Monitor.boundary() -> gates.quiet -> timed() round loop. The first
        # enclosing sample of the study raises: that level's Q fails with 'telemetry
        # sample error', the round reruns, retry-action.json carries the reasons.
        from tests import fixtures_lines as fx
        clock = {'calls': 0}

        def sample(self):
            clock['calls'] += 1
            if clock['calls'] == 1:
                raise RuntimeError('Cannot observe all eight cards')
            return fx.sample(0 if clock['calls'] % 2 == 1 else 2, {}, {})  # odd: before, even: after

        def run(self):
            self.path.write_text(json.dumps(fx.sample(1, {}, {})) + '\n')

        def level(rows, before, after, startup, count, needed):
            return {'K_capacity': True, 'K_preemption': True, 'kv_capacity_tokens': 1, 'T': True, 'C': True,
                    'G': True, 'rerun_level': False, 'ttft_s': 1., 'token_gap_s': 1.,
                    'output_tokens_per_second': 1., 'window_ns': [500_000_000, 1_500_000_000]}

        with tempfile.TemporaryDirectory() as root:
            server_dir = Path(root) / 'server'
            server_dir.mkdir()
            (server_dir / 'metrics-startup.txt').write_text('')
            server = SimpleNamespace(log_text=lambda: '', directory=server_dir, workers=[], url='http://x',
                                     process=SimpleNamespace(pid=1, poll=lambda: None))
            topology = {'client_cpus': [44], 'runner_cpus': {0}}

            def part(args, arm, directory, levels, retry):
                directory.mkdir(parents=True)
                return {str(c): runner.run_level(server, [], [], c, topology, directory / f'c{c}', 0, timed=True)
                        for c in levels}, None
            args = SimpleNamespace(study={'step': 1, 'arms': ['TP8', 'TP4xPP2', 'PP8']})
            with patch.object(runner, 'timed_part_with_m', side_effect=part), \
                    patch.object(runner, 'level_memory_before', return_value={}), \
                    patch.object(runner, 'level_memory_after', side_effect=lambda d, r: r), \
                    patch.object(runner.client, 'phase', return_value=[]), \
                    patch.object(runner.telemetry, 'placement', return_value={}), \
                    patch.object(runner.telemetry.Monitor, 'sample', sample), \
                    patch.object(runner.telemetry.Monitor, 'run', run), \
                    patch.object(runner, 'metrics', return_value=''), \
                    patch.object(runner, 'check_m', return_value={'pass': True}), \
                    patch.object(runner.gates, 'level', side_effect=level), \
                    patch.object(runner.gates, 'binding', return_value={}), \
                    patch.object(runner.gates, 'placement', return_value={'pass': True}):
                rounds, terminal, _ = runner.timed(args, Path(root))
            action = json.loads((Path(root) / 'round-1-attempt-0' / 'retry-action.json').read_text())
        self.assertIsNone(terminal)
        self.assertEqual(len(rounds), 5)
        self.assertEqual((rounds[0]['round'], rounds[0]['attempt']), (1, 1))
        self.assertEqual(action['action'], 'rerun_round')
        self.assertEqual(len(action['q_failures']), 1)
        self.assertEqual(action['q_failures'][0]['reasons'], ['telemetry sample error'])
        self.assertTrue(action['q_failures'][0]['directory'].endswith('round-1-attempt-0/TP8/c1'))

    def run_timed(self, part):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(study={'step': 1, 'arms': ['TP8', 'TP4xPP2', 'PP8']})
            with patch.object(runner, 'timed_part_with_m', side_effect=part):
                return runner.timed(args, Path(root))

    def test_rotation_and_acceptance(self):
        calls = []

        def part(args, arm, directory, levels, retry):
            calls.append((directory.parent.name, arm))
            return cells(levels), None
        rounds, terminal, _ = self.run_timed(part)
        self.assertIsNone(terminal)
        self.assertEqual(len(rounds), 5)
        orders = [[a for d, a in calls if d == f'round-{k}-attempt-0'] for k in range(1, 6)]
        self.assertEqual(orders, [['TP8', 'TP4xPP2', 'PP8'], ['TP4xPP2', 'PP8', 'TP8'], ['PP8', 'TP8', 'TP4xPP2'],
                                  ['TP8', 'TP4xPP2', 'PP8'], ['TP4xPP2', 'PP8', 'TP8']])

    def test_capacity_inequality_fails_k_in_the_round(self):
        def part(args, arm, directory, levels, retry):
            return cells(levels, cap=58960 if arm == 'PP8' else 58976), None
        rounds, _, _ = self.run_timed(part)
        self.assertFalse(any(cell['K'] for r in rounds for lv in r['levels'].values() for cell in lv.values()))

    def test_t_retry_uses_bank_1_in_every_arm(self):
        retries = []

        def part(args, arm, directory, levels, retry):
            if retry:
                retries.append((arm, tuple(levels)))
                return cells(levels), None
            base = cells(levels)
            if arm == 'PP8':
                base['8'] = cells([8], T=False)['8']      # PP8 drops > 2% at c = 8
            return base, None
        rounds, _, _ = self.run_timed(part)
        self.assertEqual(sorted(set(retries)), [('PP8', (8,)), ('TP4xPP2', (8,)), ('TP8', (8,))])
        self.assertTrue(all(r['levels']['8']['PP8']['T'] for r in rounds))

    def test_gate_m_terminal(self):
        def part(args, arm, directory, levels, retry):
            if arm == 'PP8':
                return None, {'arm': 'PP8', 'first': 'oom', 'second': 'oom'}
            return cells(levels), None
        rounds, terminal, failure = self.run_timed(part)
        self.assertEqual((rounds, terminal, failure['arm']), ([], 'M', 'PP8'))

    def test_q_failure_reruns_then_not_measurable(self):
        def part(args, arm, directory, levels, retry):
            return cells(levels, Q=False), None
        rounds, terminal, _ = self.run_timed(part)
        self.assertEqual((rounds, terminal), ([], 'Q'))


class GateMRerun(unittest.TestCase):
    def test_one_rerun_then_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            outcomes = [runner.GateM('first'), {'1': {}}]

            def part(*a):
                value = outcomes.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
            with patch.object(runner, 'timed_part', side_effect=part):
                self.assertEqual(runner.timed_part_with_m(None, 'TP8', Path(root) / 'TP8', (1,), 0),
                                 ({'1': {}}, None))
            with patch.object(runner, 'timed_part', side_effect=runner.GateM('again')):
                result, failure = runner.timed_part_with_m(None, 'TP8', Path(root) / 'TP8', (1,), 0)
            self.assertIsNone(result)
            self.assertEqual(failure['second'], 'again')


class LadderDriver(unittest.TestCase):
    def drive(self, fails):
        calls = []

        def start_test(args, arm, step, directory):
            calls.append((arm, step))
            return (arm, step) not in fails, {'arm': arm, 'step': step}
        with tempfile.TemporaryDirectory() as root, patch.object(runner, 'start_test', side_effect=start_test):
            return runner.ladder(None, Path(root)), calls

    def test_pp8_infeasible_then_retest_from_step_1(self):
        result, calls = self.drive({('PP8', 1), ('PP8', 2), ('PP8', 3)})
        self.assertEqual((result['result'], result['step'], result['arms'], result['infeasible']),
                         ('done', 1, ['TP8', 'TP4xPP2'], ['PP8']))
        self.assertEqual(calls[-2:], [('TP8', 1), ('TP4xPP2', 1)])
        self.assertEqual(len(calls), 11)

    def test_step_2(self):
        result, calls = self.drive({('PP8', 1)})
        self.assertEqual((result['result'], result['step']), ('done', 2))

    def test_gate_s(self):
        fails = {(a, s) for a in ('TP8', 'PP8') for s in (1, 2, 3)}
        result, _ = self.drive(fails)
        self.assertEqual((result['result'], result['arms']), ('not-measurable', ['TP4xPP2']))


class Client(unittest.TestCase):
    def test_split(self):
        prompts = list(range(64))
        plan = client.split(prompts, 16, list(range(44, 52)))
        self.assertEqual([len(s) for s, _ in plan], [8, 8])
        self.assertEqual([c for _, c in plan], [[44, 45, 46, 47], [48, 49, 50, 51]])
        self.assertEqual(plan[1][0][0], prompts[8::16])
        self.assertEqual(len(client.split(prompts, 32, list(range(8)))), 4)
        with self.assertRaises(ValueError):
            client.split(prompts, 4, list(range(8)))

    def test_gate_e_requests_sequential_and_declared_fields(self):
        sent = []

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(request, timeout):
            sent.append(json.loads(request.data))
            self.assertEqual(request.headers.get('Connection'), 'close')
            return Response(b'{"ok": 1}')
        with tempfile.TemporaryDirectory() as root:
            rows = client.gate_e_requests('http://x', [{'id': 'a', 'messages': []}, {'id': 'b', 'messages': []}],
                                          Path(root) / 'e', opener=opener)
        self.assertEqual([r['response'] for r in rows], [{'ok': 1}, {'ok': 1}])
        body = sent[0]
        self.assertEqual({k: body[k] for k in ('temperature', 'max_tokens', 'min_tokens', 'logprobs',
                                               'top_logprobs', 'stream')},
                         {'temperature': 0, 'max_tokens': 128, 'min_tokens': 128, 'logprobs': True,
                          'top_logprobs': 5, 'stream': False})


class Prerequisites(unittest.TestCase):
    def test_model_ready(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            ready = root / 'MODEL_READY.json'
            good = {'repo': runner.MODEL, 'revision': runner.REVISION, 'snapshot': str(root / 'snap')}
            ready.write_text(json.dumps(good))
            self.assertEqual(runner.model_ready(ready, root / 'snap'), good)
            with self.assertRaisesRegex(RuntimeError, 'verified'):
                runner.model_ready(ready, root / 'other')
            for change in ({'revision': '0' * 40}, {'repo': 'Qwen/Qwen2.5-7B-Instruct'}):
                ready.write_text(json.dumps({**good, **change}))
                with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    runner.model_ready(ready, root / 'snap')
            ready.write_text(json.dumps({k: v for k, v in good.items() if k != 'snapshot'}))
            with self.assertRaises(RuntimeError):
                runner.model_ready(ready, root / 'snap')

    def prior(self, root, code):
        run = Path(root) / 'start'
        run.mkdir()
        runner.write_json(run / 'run-manifest.json', {'code_sha256': code, 'phase': 'start-test',
                                                      'deviations': {}, 'prompts_sha256': 'p'})
        runner.write_json(run / 'ladder.json', {'result': 'done'})
        evidence = {str(run / n): runner.digest(run / n) for n in ('run-manifest.json', 'ladder.json')}
        runner.write_json(run / 'receipt.json', {'phase': 'start-test', 'pass': True, 'deviations': {},
                                                 'prompts_sha256': 'p', 'evidence_sha256': evidence})
        return run / 'receipt.json'

    def test_check_prior_code_hash(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = self.prior(root, {'runner.py': 'a'})
            self.assertTrue(runner.check_prior(receipt, 'start-test', {}, 'p', {'runner.py': 'a'})['pass'])
            for code in ({'runner.py': 'b'}, {'runner.py': 'a', 'gates.py': 'c'}, {}):
                with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, 'different code'):
                    runner.check_prior(receipt, 'start-test', {}, 'p', code)
            with self.assertRaisesRegex(RuntimeError, 'different prompts'):
                runner.check_prior(receipt, 'start-test', {}, 'q', {'runner.py': 'a'})
            (Path(root) / 'start/ladder.json').write_text('{"result": "changed"}')
            with self.assertRaisesRegex(RuntimeError, 'evidence changed'):
                runner.check_prior(receipt, 'start-test', {}, 'p', {'runner.py': 'a'})


if __name__ == '__main__':
    unittest.main()


class StudyFrom(unittest.TestCase):
    def test_gate_e_leaving_one_arm_is_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            start, settings = Path(root, 'start'), Path(root, 'settings')
            start.mkdir(); settings.mkdir()
            (start / 'ladder.json').write_text(json.dumps({'result': 'done', 'step': 1, 'infeasible': [],
                                                            'arms': ['TP8', 'TP4xPP2', 'PP8']}))
            (settings / 'gate-e.json').write_text(json.dumps({'premise_pass': True, 'excluded': ['TP8', 'PP8']}))
            args = SimpleNamespace(phase='transport', start_receipt=start / 'receipt.json',
                                   settings_receipt=settings / 'receipt.json')
            with self.assertRaises(RuntimeError):
                runner.study_from(args)
            (settings / 'gate-e.json').write_text(json.dumps({'premise_pass': True, 'excluded': ['TP8']}))
            self.assertEqual(runner.study_from(args)['arms'], ['TP4xPP2', 'PP8'])

    def test_failures_are_classified_in_ladder_output(self):
        result = {'attempts': [{'arm': 'PP8', 'step': 1, 'pass': False, 'failure_class': 'oom'},
                               {'arm': 'TP8', 'step': 1, 'pass': True}]}
        self.assertEqual(runner.failures_of(result), [{'arm': 'PP8', 'step': 1, 'failure_class': 'oom'}])
