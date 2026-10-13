"""CPU-only tests of `report`: a synthetic timed run directory in the layout the timed phase
writes, recomputed from its per-request records and /metrics snapshots."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tppbench import gates, report
from tppbench.__main__ import main as _cli
from tppbench.spec import MEASURED, capacity_needed
from tests import fixtures_lines as fx

ARMS3 = ['TP8', 'TP4xPP2', 'PP8']


def cli(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return _cli(argv)

# (TTFT s, token gap s) per arm; PP8's c = 8 TTFT varies by round.
VALUES = {'TP8': (5.28, 0.0405), 'TP4xPP2': (3.62, 0.0522), 'PP8': (2.19, 0.159)}
PP8_C8 = (2.19, 2.31, 2.31, 2.88, 3.00)


def requests(c, ttft, gap):
    """c parallel streams, each sending its share of MEASURED[c] requests back to back."""
    n, rows = MEASURED[c], []
    duration = int((ttft + 127 * gap) * 1e9) + 1_000_000
    for i in range(n):
        stream, slot = i % c, i // c
        send = 10**12 + slot * duration + stream
        content = [send + int(ttft * 1e9) + j * int(gap * 1e9) for j in range(128)]
        rows.append({'id': 1000 * c + i, 'status': 200, 'done': True, 'error': None, 'finish_reason': 'length',
                     'usage': {'prompt_tokens': 1536, 'completion_tokens': 128}, 'send_ns': send,
                     'end_ns': content[-1] + 500_000, 'content_ns': content, 'contents': ['x'] * 128})
    return rows


def fake_part(retry_case):
    """A stand-in for runner.timed_part: writes one arm's part of a round in the timed phase's
    layout (per-request records and /metrics snapshots) and returns its cells, so that the
    real runner.timed loop does the round combination, the level retries and acceptance.
    With `retry_case`, round 2's first c = 1 attempt has one TP8 request with the wrong prompt
    length (gate T fails: a level retry in every arm) and two PP8 requests that arrived as 64
    chunks (gate G fails while T and C pass, so the replacement's G is forced false)."""
    from tppbench import client

    def part(args, arm, directory, step_levels, retry):
        k = int(directory.parent.name.split('-')[1])
        directory.mkdir(parents=True)
        (directory / 'metrics-startup.txt').write_text(fx.cache_metric() + '\n')
        cells = {}
        for c in step_levels:
            ttft, gap = VALUES[arm]
            if arm == 'PP8' and c == 8:
                ttft = PP8_C8[k - 1]
            rows = requests(c, ttft, gap)
            if retry_case and k == 2 and c == 1 and retry == 0:
                if arm == 'TP8':
                    rows[3]['usage']['prompt_tokens'] = 1535
                if arm == 'PP8':
                    for row in rows[:2]:
                        row['content_ns'], row['contents'] = row['content_ns'][:64], row['contents'][:64]
            d = directory / f'c{c}'
            (d / 'measured').mkdir(parents=True)
            with (d / 'measured/client-0.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            (d / 'metrics-before.txt').write_text(fx.metrics() + '\n')
            (d / 'metrics-after.txt').write_text(fx.metrics(q=len(rows) * 1536) + '\n')
            cell = gates.level([client.summarize(json.loads(json.dumps(r))) for r in rows], fx.metrics(),
                               fx.metrics(q=len(rows) * 1536), fx.cache_metric(), len(rows), capacity_needed(1))
            cell.update(N=True, Q=True, M=True, prompt_ids=[r['id'] for r in rows], directory=str(d),
                        quiet={'foreign_cpu_fraction': {'0': 0.001, '1': 0.0044},
                               'telemetry_gaps': {'max_interval_s': 1.67}})
            cells[str(c)] = cell
        return cells
    return part


def write_run(root, rounds_count=5, retry_case=False):
    """A timed run made by the real runner.timed loop with fake parts; its stored cell
    directories are then rewritten to another machine's path."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from tppbench import runner
    run = Path(root) / 'timed-01'
    run.mkdir(parents=True)
    study = {'step': 1, 'feasible': ARMS3, 'infeasible': [], 'excluded': [], 'arms': ARMS3, 'terminal_gate': None}
    with patch.object(runner, 'timed_part', side_effect=fake_part(retry_case)):
        accepted, terminal, failure = runner.timed(SimpleNamespace(study=study), run)
    assert terminal is None and failure is None
    text = json.dumps(accepted[:rounds_count]).replace(str(run), '/elsewhere/runs/timed-01')
    (run / 'accepted-rounds.json').write_text(text)
    if rounds_count == 5:
        (run / 'decision-for-review.json').write_text(json.dumps(gates.decision(study, accepted)))
    else:
        (run / 'run-manifest.json').write_text(json.dumps({'study': study}))
    return run


class Report(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_recomputes_tables_without_warnings(self):
        rep = report.report(write_run(self.tmp.name))
        self.assertIsNone(rep['error'])
        self.assertEqual(rep['warnings'], [])
        d = rep['decision']
        self.assertEqual(d['table_a']['row'], 'A2')
        self.assertTrue(d['table_a']['throughput_clause_contradicted'])   # TP8 better on throughput
        self.assertEqual(d['table_b']['row'], 'B3')
        self.assertEqual(d['table_b']['cell_winners']['ttft_s:c32'], 'PP8')
        self.assertEqual(d['table_b']['cell_winners']['token_gap_s:c1'], 'TP8')
        reading = {(r['c'], r['metric']): r for r in rep['reading']}
        self.assertEqual(reading[(8, 'ttft_s')]['arms']['PP8']['text'], '2.19-3.00')   # five-round range
        self.assertTrue(reading[(8, 'ttft_s')]['arms']['PP8']['range_shown'])
        self.assertEqual(reading[(1, 'ttft_s')]['arms']['TP8']['text'], '5.28')
        self.assertEqual(reading[(32, 'token_gap_s')]['arms']['PP8']['text'], '159.0')   # ms
        self.assertNotIn((1, 'output_tokens_per_second'), reading)
        tput = reading[(8, 'output_tokens_per_second')]['arms']['TP8']['value']
        self.assertAlmostEqual(tput, 8 * 128 / (5.28 + 127 * 0.0405 + 0.001), delta=0.2)
        self.assertEqual(rep['gates']['pass'], {g: 45 for g in ('K', 'T', 'C', 'G', 'N', 'Q', 'M')})
        self.assertEqual(rep['gates']['kv_capacity_tokens'], [58976])
        self.assertEqual(rep['gates']['foreign_cpu_fraction_max'], 0.0044)
        text = report.text_report(rep)
        self.assertIn('Table A (TP8 against PP8): row A2', text)
        self.assertIn('Table B (cell winners): row B3', text)
        self.assertIn('Every recomputed value, gate and verdict equals the stored one.', text)
        # Throughput TP8 better and Table B row B3: not the expected outcome, with the reasons.
        self.assertIn('Matches the expected outcome stated in the README: no (TP8|PP8 output_tokens_per_second:c8 '
                      'is TP8 better, expected PP8 better', text)
        matching = dict(rep, decision={**d, 'against_expectation': False, 'expectation_check': []})
        self.assertIn('Matches the expected outcome stated in the README: yes', report.text_report(matching))

    def test_a_changed_stored_value_is_a_warning(self):
        run = write_run(self.tmp.name)
        accepted = json.loads((run / 'accepted-rounds.json').read_text())
        accepted[2]['levels']['8']['TP8']['ttft_s'] *= 1.01
        accepted[0]['levels']['1']['PP8']['G'] = False
        (run / 'accepted-rounds.json').write_text(json.dumps(accepted))
        decision = json.loads((run / 'decision-for-review.json').read_text())
        decision['table_b']['row'] = 'B2'
        (run / 'decision-for-review.json').write_text(json.dumps(decision))
        rep = report.report(run)
        joined = '\n'.join(rep['warnings'])
        self.assertIn('round 3 c=8 TP8: ttft_s recomputed', joined)
        self.assertIn('round 1 c=1 PP8: gate G recomputed True, stored False', joined)
        self.assertIn('table_b row recomputed B3, stored B2', joined)
        self.assertEqual(cli(['report', str(run)]), 1)

    def test_tampered_raw_record_changes_the_recomputation(self):
        run = write_run(self.tmp.name)
        path = run / 'round-4-attempt-0/TP4xPP2/c1/measured/client-0.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            row['usage']['prompt_tokens'] = 1535           # every request fails gate T
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        rep = report.report(run)
        self.assertTrue(any('round 4 c=1 TP4xPP2: gate T recomputed False' in w for w in rep['warnings']))

    def test_incomplete_run_prints_no_tables(self):
        run = write_run(self.tmp.name, rounds_count=4)
        rep = report.report(run)
        self.assertIn('not a complete run', rep['error'])
        self.assertEqual(cli(['report', str(run)]), 2)

    def test_level_retry_is_combined_as_the_run_combines_it(self):
        run = write_run(self.tmp.name, retry_case=True)
        accepted = json.loads((run / 'accepted-rounds.json').read_text())
        self.assertTrue(accepted[1]['levels']['1']['TP8']['directory'].endswith('TP8-level-retry/c1'))
        self.assertFalse(accepted[1]['levels']['1']['PP8']['G'])          # forced false by the run
        rep = report.report(run)
        self.assertEqual(rep['warnings'], [])
        stored = json.loads((run / 'decision-for-review.json').read_text())
        self.assertEqual(rep['decision']['pair_verdicts'], stored['pair_verdicts'])
        self.assertEqual(rep['decision']['table_b'], stored['table_b'])
        self.assertNotIn('token_gap_s:c1', rep['decision']['table_a']['verdicts'])   # gap cell removed
        self.assertIn('ttft_s:c1', rep['decision']['table_a']['verdicts'])
        # The replacement alone passes G; only the run's rule removes the cell.
        alone = report.recompute_dir(run / 'round-2-attempt-0/PP8-level-retry/c1', 1, 1)
        self.assertTrue(alone['G'])

    def test_statistic_option(self):
        run = write_run(self.tmp.name)
        mean = report.report(run, 'mean')
        reading = {(r['c'], r['metric']): r for r in mean['reading']}
        self.assertAlmostEqual(reading[(8, 'ttft_s')]['arms']['PP8']['value'], sum(PP8_C8) / 5, places=3)
        self.assertIn('(mean over the five rounds', report.text_report(mean))
        self.assertIn('(median over the five rounds', report.text_report(report.report(run)))
        self.assertEqual(cli(['report', str(run), '--statistic', 'mean']), 0)

    def test_cell_directory_from_another_machine(self):
        self.assertEqual(report.cell_directory('/here/run', '/there/x/timed-01/round-2-attempt-1/PP8-level-retry/c8'),
                         Path('/here/run/round-2-attempt-1/PP8-level-retry/c8'))
        with self.assertRaises(ValueError):
            report.cell_directory('/here', '/there/c8')


if __name__ == '__main__':
    unittest.main()
