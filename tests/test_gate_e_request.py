"""CPU-only tests of gate E's request body (repetition_penalty 1.0;
the premise check before gate E is read; a premise failure stops the study at
the settings phase)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import io
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tppbench import client
from tppbench import gates
from tppbench import launch
from tppbench import runner
from tests import fixtures_lines as fx
from tppbench import telemetry
from tppbench.spec import STOP_TOKEN_IDS

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'tppbench'
EOS, EOT = STOP_TOKEN_IDS


class Payloads(unittest.TestCase):
    def test_gate_e_has_penalty_one_and_timed_payload_does_not(self):
        body = client.gate_e_payload([{'role': 'user', 'content': 'x'}])
        self.assertIn('repetition_penalty', body)
        self.assertEqual(body['repetition_penalty'], 1.0)
        self.assertIsInstance(body['repetition_penalty'], float)
        self.assertNotIn('repetition_penalty', client.payload({'messages': []}))
        # Everything else in the gate E request is unchanged.
        self.assertEqual({k: v for k, v in body.items() if k not in ('repetition_penalty', 'messages', 'model')},
                         {'max_tokens': 128, 'min_tokens': 128, 'temperature': 0, 'stream': False,
                          'logprobs': True, 'top_logprobs': 5, 'return_tokens_as_token_ids': True})

    def test_sent_gate_e_body_carries_the_penalty(self):
        sent = []

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(request, timeout):
            sent.append(json.loads(request.data))
            return Response(b'{}')
        with tempfile.TemporaryDirectory() as root:
            client.gate_e_requests('http://x', [{'id': 'a', 'messages': []}], Path(root) / 'e', opener=opener)
        self.assertEqual(sent[0]['repetition_penalty'], 1.0)

    def test_penalty_set_only_in_gate_e_payload(self):
        # Start-test, transport and timed requests all go through client.payload
        # (client.phase -> client.request); no other source sets a repetition penalty.
        sources = {p.name: p.read_text() for p in PACKAGE.glob('*.py')}
        users = {n for n, s in sources.items() if re.search(r'[\'"]repetition_penalty[\'"]|GATE_E_REPETITION_PENALTY', s)}
        self.assertEqual(users, {'client.py', 'spec.py'})
        text = sources['client.py']
        gate_e_body = text.split('def gate_e_payload', 1)[1].split('\ndef ', 1)[0]
        rest = text.replace(gate_e_body, '')
        self.assertNotIn('repetition_penalty', rest)
        self.assertEqual(rest.count('GATE_E_REPETITION_PENALTY'), 1)        # the import line only
        self.assertIn("json=payload(prompt)", text)


class Premise(unittest.TestCase):
    base = list(range(100, 228))

    def tokens(self, response):
        return gates.gate_e_tokens(response)

    def check(self, response):
        return gates.premise({'p': self.tokens(response)})

    def test_argmax_everywhere_passes(self):
        record = self.check(fx.gate_e_response(self.base, top={5: {105: -0.2, 7: -1.0, 8: -2.0}}))
        self.assertEqual(record, {'positions_checked': 128, 'failures': [], 'stop_masked': 0, 'tied_top': 0})

    def test_non_argmax_fails(self):
        response = fx.gate_e_response(self.base, top={9: {109: -0.9, 42: -0.4}})
        record = self.check(response)
        self.assertEqual(record['failures'], [{'prompt': 'p', 'position': 9, 'emitted_token': 109,
                                               'emitted_logprob': -0.9, 'max_non_stop_logprob': -0.4,
                                               'reason': 'emitted logprob is not the largest non-stop value'}])

    def test_no_tolerance(self):
        response = fx.gate_e_response(self.base, top={9: {109: -0.4, 42: -0.4 + 1e-12}})
        self.assertEqual(len(self.check(response)['failures']), 1)

    def test_stop_masked_position_passes(self):
        # min_tokens masks the stop ids; the raw top is a stop id, the emitted token is the best non-stop.
        record = self.check(fx.gate_e_response(self.base, top={3: {EOS: -0.1, 103: -2.5, EOT: -3.0, 9: -4.0},
                                                               4: {EOT: -0.2, 104: -0.3}}))
        self.assertEqual((record['failures'], record['stop_masked']), ([], 2))
        # A stop-masked position whose emitted token is not the best non-stop one still fails.
        record = self.check(fx.gate_e_response(self.base, top={3: {EOS: -0.1, 103: -2.5, 9: -2.0}}))
        self.assertEqual([(f['position'], f['max_non_stop_logprob']) for f in record['failures']], [(3, -2.0)])
        self.assertEqual(record['stop_masked'], 1)

    def test_tie_passes_and_is_counted(self):
        record = self.check(fx.gate_e_response(self.base, top={0: {100: -0.6875, 55: -0.6875, 56: -3.0},
                                                               1: {77: -0.5, 101: -0.5}}))
        self.assertEqual((record['failures'], record['tied_top']), ([], 2))

    def test_missing_top_logprobs_fails(self):
        response = fx.gate_e_response(self.base)
        response['choices'][0]['logprobs']['content'][7]['top_logprobs'] = []
        record = self.check(response)
        self.assertEqual([(f['position'], f['reason']) for f in record['failures']],
                         [(7, 'logprob or top_logprobs missing')])
        response = fx.gate_e_response(self.base)
        del response['choices'][0]['logprobs']['content'][8]['top_logprobs']
        self.assertEqual([f['position'] for f in self.check(response)['failures']], [8])

    def test_missing_logprob_fails(self):
        response = fx.gate_e_response(self.base)
        del response['choices'][0]['logprobs']['content'][2]['logprob']
        self.assertEqual([f['position'] for f in self.check(response)['failures']], [2])
        response = fx.gate_e_response(self.base)
        response['choices'][0]['logprobs']['content'][2]['logprob'] = None
        self.assertEqual([f['position'] for f in self.check(response)['failures']], [2])

    def test_response_without_logprobs_fails(self):
        for logprobs in (None, {'content': []}, {'content': None}):
            bare = {'choices': [{'index': 0, 'logprobs': logprobs}], 'usage': {'completion_tokens': 128}}
            record = gates.premise({'p': gates.gate_e_tokens(bare)})
            self.assertEqual(record['positions_checked'], 0)
            self.assertEqual([(f['prompt'], f['position'], f['reason']) for f in record['failures']],
                             [('p', None, 'response without logprobs')])

    def test_only_stop_ids_in_top_fails(self):
        response = fx.gate_e_response(self.base, top={1: {EOS: -0.1, EOT: -0.2}})
        response['choices'][0]['logprobs']['content'][1]['logprob'] = -0.1
        self.assertEqual([f['reason'] for f in self.check(response)['failures']],
                         ['no non-stop token in top_logprobs'])

    def test_default_penalty_response_fails_where_the_penalty_cut(self):
        # Recorded response (fixtures/gate_e/SOURCE.txt): penalty 1.05 applied, one
        # emission below its own top raw logprob, two tied tops, no stop-masked position.
        response = json.loads((HERE / 'fixtures/gate_e/response_penalty_default.json').read_text())
        record = gates.premise({'c1-20': gates.gate_e_tokens(response)})
        self.assertEqual((record['positions_checked'], record['stop_masked'], record['tied_top']), (128, 0, 2))
        self.assertEqual([(f['prompt'], f['position'], f['emitted_token']) for f in record['failures']],
                         [('c1-20', 3, 697)])
        self.assertLess(record['failures'][0]['emitted_logprob'], record['failures'][0]['max_non_stop_logprob'])


class GateE(unittest.TestCase):
    base = list(range(100, 228))

    def test_premise_failure_applies_no_exclusion(self):
        good = fx.gate_e_response(self.base)
        other = fx.gate_e_response(self.base[:5] + [999] + self.base[6:])       # would exclude TP8
        cut = fx.gate_e_response(self.base, top={5: {105: -0.9, 999: -0.4}})
        result = gates.gate_e({'TP8': {'p': other, 'q': good}, 'TP4xPP2': {'p': good, 'q': good},
                               'PP8': {'p': cut, 'q': good}})
        self.assertIs(result['premise_pass'], False)
        self.assertIsNone(result['excluded'])
        self.assertIsNone(result['corrupted_pairs'])
        self.assertIsNone(result['pairs'])
        self.assertIn('TP8|TP4xPP2', result['pairs_information_only'])
        self.assertEqual(result['premise']['PP8']['failures'][0]['position'], 5)
        self.assertEqual([result['premise'][a]['failures'] for a in ('TP8', 'TP4xPP2')], [[], []])
        self.assertEqual(result['premise']['TP8']['positions_checked'], 256)
        json.dumps(result, allow_nan=False)

    def test_premise_pass_keeps_section_6(self):
        good = fx.gate_e_response(self.base)
        other = fx.gate_e_response(self.base[:5] + [999] + self.base[6:])
        result = gates.gate_e({'TP8': {'p': other}, 'TP4xPP2': {'p': good}, 'PP8': {'p': good}})
        self.assertIs(result['premise_pass'], True)
        self.assertEqual(result['excluded'], ['TP8'])
        self.assertNotIn('pairs_information_only', result)
        self.assertEqual(set(result['premise']), {'TP8', 'TP4xPP2', 'PP8'})
        self.assertEqual(result['stop_token_ids'], [151645, 151643])

    def test_premise_checked_with_one_arm(self):
        cut = fx.gate_e_response(self.base, top={5: {105: -0.9, 999: -0.4}})
        result = gates.gate_e({'TP8': {'p': cut}})
        self.assertIs(result['premise_pass'], False)
        self.assertIsNone(result['excluded'])


def study():
    return {'step': 1, 'feasible': ['TP8', 'TP4xPP2', 'PP8'], 'infeasible': [], 'excluded': [],
            'arms': ['TP8', 'TP4xPP2', 'PP8'], 'terminal_gate': None}


FAILED = {'arms': ['TP8', 'TP4xPP2', 'PP8'], 'premise_pass': False, 'excluded': None, 'pairs': None,
          'corrupted_pairs': None, 'premise': {}, 'pairs_information_only': {}}


class SettingsPhase(unittest.TestCase):
    def test_settings_study_on_premise_failure(self):
        with tempfile.TemporaryDirectory() as root:
            passed = runner.settings_study(SimpleNamespace(study=study()), Path(root), FAILED)
            data = json.loads(Path(root, 'study.json').read_text())
            self.assertFalse(Path(root, 'decision-for-review.json').exists())
        self.assertIs(passed, False)
        self.assertIsNone(data['excluded'])
        self.assertIs(data['gate_e_premise_pass'], False)
        self.assertIs(data['gate_e_premise_requires_review'], True)
        self.assertIs(data['gate_e_requires_review'], True)

    def test_settings_study_on_premise_pass(self):
        with tempfile.TemporaryDirectory() as root:
            passed = runner.settings_study(SimpleNamespace(study=study()), Path(root),
                                           {'premise_pass': True, 'excluded': []})
            data = json.loads(Path(root, 'study.json').read_text())
        self.assertIs(passed, True)
        self.assertEqual((data['excluded'], data['gate_e_premise_pass'], data['gate_e_requires_review']),
                         ([], True, False))
        with tempfile.TemporaryDirectory() as root:
            passed = runner.settings_study(SimpleNamespace(study=study()), Path(root),
                                           {'premise_pass': True, 'excluded': ['TP8', 'PP8']})
            data = json.loads(Path(root, 'study.json').read_text())
        self.assertIs(passed, False)
        self.assertEqual((data['terminal_gate'], data['gate_e_requires_review']), ('E', True))

    def test_main_writes_failing_receipt_on_premise_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'setup').mkdir()
            model = root / 'model'
            (root / 'setup/MODEL_READY.json').write_text(json.dumps({'revision': runner.REVISION,
                                                                     'repo': runner.MODEL, 'snapshot': str(model)}))
            prompts = root / 'prompts.json'
            prompts.write_text('{}')
            output = root / 'settings-02'
            output.mkdir()

            def settings(args, out):
                runner.write_json(out / 'gate-e.json', FAILED)
                return FAILED
            args = SimpleNamespace(output=output, data_dir=root, model=model, lock_dir=None, prompts=prompts,
                                   probe_receipt=None, start_receipt=root / 'start/receipt.json',
                                   settings_receipt=None, transport_receipt=None, deviation_receipt=None, port=8150)
            with patch.object(launch, 'numactl_dir', return_value={'mode': 'restricted'}), \
                    patch.object(launch, 'deviations_from_receipt', return_value={}), \
                    patch.object(launch, 'Locks', return_value=MagicMock()), \
                    patch.object(runner, 'load_prompts', return_value={}), \
                    patch.object(runner, 'check_prior', return_value={}), \
                    patch.object(runner, 'study_from', return_value=study()), \
                    patch.object(telemetry, 'topology', return_value={'runner_cpus': [0]}), \
                    patch.object(runner.os, 'sched_setaffinity'), \
                    patch.object(runner, 'settings', side_effect=settings):
                self.assertIs(runner.main('settings', args), False)
            receipt = json.loads((output / 'receipt.json').read_text())
            data = json.loads((output / 'study.json').read_text())
            manifest = json.loads((output / 'run-manifest.json').read_text())
            self.assertIs(receipt['pass'], False)
            self.assertEqual(receipt['phase'], 'settings')
            self.assertIn(str(output / 'gate-e.json'), receipt['evidence_sha256'])
            self.assertIs(data['gate_e_premise_requires_review'], True)
            self.assertEqual(manifest['code_sha256'], launch.code_files())
            self.assertFalse((output / 'INCOMPLETE.json').exists())

    def test_no_later_phase_reads_a_failed_premise(self):
        with tempfile.TemporaryDirectory() as root:
            start, settings = Path(root, 'start'), Path(root, 'settings')
            start.mkdir(); settings.mkdir()
            (start / 'ladder.json').write_text(json.dumps({'result': 'done', 'step': 1, 'infeasible': [],
                                                            'arms': ['TP8', 'TP4xPP2', 'PP8']}))
            args = SimpleNamespace(phase='transport', start_receipt=start / 'receipt.json',
                                   settings_receipt=settings / 'receipt.json')
            for gate_e in (FAILED, {'excluded': []}, {'premise_pass': True, 'excluded': None}):
                (settings / 'gate-e.json').write_text(json.dumps(gate_e))
                with self.subTest(gate_e=gate_e), self.assertRaisesRegex(RuntimeError, 'premise'):
                    runner.study_from(args)
            (settings / 'gate-e.json').write_text(json.dumps({'premise_pass': True, 'excluded': []}))
            self.assertEqual(runner.study_from(args)['arms'], ['TP8', 'TP4xPP2', 'PP8'])

    def test_check_prior_refuses_a_failing_settings_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            receipt = Path(root, 'receipt.json')
            receipt.write_text(json.dumps({'phase': 'settings', 'pass': False, 'deviations': {},
                                           'prompts_sha256': 'x', 'evidence_sha256': {}}))
            with self.assertRaisesRegex(RuntimeError, 'passing settings prerequisite'):
                runner.check_prior(receipt, 'settings', {}, 'x', {'a.py': 'y'})


if __name__ == '__main__':
    unittest.main()
