"""CPU-only tests of the decision (tables A and B), the ladder rule and retries."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import unittest

from tppbench import gates
from tests import fixtures_lines as fx

ARMS3 = ['TP8', 'TP4xPP2', 'PP8']
EQUAL = (1., 1., 1.)


def study(arms=ARMS3, step=1, terminal=None, excluded=()):
    return {'step': step, 'arms': list(arms), 'infeasible': [a for a in ARMS3 if a not in arms and a not in excluded],
            'excluded': list(excluded), 'terminal_gate': terminal}


def decide(values, arms=ARMS3, step=1, override=None, **kwargs):
    levels = {1: (1, 8, 32), 2: (1, 8, 16), 3: (1, 8)}[step]
    return gates.decision(study(arms, step, **kwargs), fx.rounds(values, levels, override))


class Verdicts(unittest.TestCase):
    def test_bounds(self):
        for ratio, expected in [(0.89, 'X better'), (.9, 'no material difference'),
                                (1.1, 'no material difference'), (1.11, 'Y better')]:
            self.assertEqual(gates.pair_verdict([ratio] * 5, 'X', 'Y'), expected)
        self.assertEqual(gates.pair_verdict([.8] * 4 + [.9], 'X', 'Y'), 'unresolved')
        with self.assertRaises(gates.EvidenceError):
            gates.pair_verdict([.8] * 4, 'X', 'Y')

    def test_orientation_of_throughput(self):
        # TP8 has twice PP8's throughput: ratio PP8/TP8 = 0.5 -> TP8 better.
        result = decide({'TP8': (1., 1., 2.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})
        self.assertEqual(result['table_a']['verdicts']['output_tokens_per_second:c8'], 'TP8 better')
        self.assertNotIn('output_tokens_per_second:c1', result['table_a']['verdicts'])
        self.assertEqual(result['table_a']['verdicts']['ttft_s:c1'], 'no material difference')


    def test_deciding_levels(self):
        # Throughput decides at every valid level but c = 1.
        from tppbench.spec import levels
        tput = {step: gates.deciding_levels('output_tokens_per_second', levels(step)) for step in (1, 2, 3)}
        self.assertEqual(tput, {1: [8, 32], 2: [8, 16], 3: [8]})
        for metric in ('ttft_s', 'token_gap_s'):
            self.assertEqual(gates.deciding_levels(metric, levels(2)), [1, 8, 16])


class TableA(unittest.TestCase):
    def test_every_row(self):
        rows = {
            'A0': decide({'TP8': EQUAL, 'TP4xPP2': EQUAL, 'PP8': EQUAL}, terminal='P'),
            'A1': decide({'TP8': EQUAL, 'TP4xPP2': EQUAL}, arms=['TP8', 'TP4xPP2']),
            # TP8 lower gap, PP8 lower TTFT and higher throughput: the expectation.
            'A2': decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, .75, .75), 'PP8': EQUAL}),
            'A3': decide({'TP8': (lambda k: (.5 if k else 1., 1., 1.)), 'TP4xPP2': EQUAL, 'PP8': EQUAL}),
            'A4': decide({'TP8': EQUAL, 'TP4xPP2': EQUAL, 'PP8': EQUAL}),
            'A5': decide({'TP8': (.5, 1., 1.), 'TP4xPP2': EQUAL, 'PP8': EQUAL}),
            'A6': decide({'TP8': (2., 1., 1.), 'TP4xPP2': EQUAL, 'PP8': EQUAL}),
        }
        for row, result in rows.items():
            with self.subTest(row=row):
                self.assertEqual(result['table_a']['row'], row)

    def test_a0_for_fewer_than_two_valid_levels_and_terminal_gates(self):
        bad_k = lambda k, c, arm: {'K': c == 32}                      # only c = 32 valid
        result = decide({'TP8': (2., .5, .5), 'TP4xPP2': EQUAL, 'PP8': EQUAL}, override=bad_k)
        self.assertEqual(result['table_a']['row'], 'A0')
        self.assertEqual(result['valid_levels_tp8_pp8'], 1)
        for gate in ('S', 'N', 'Q'):
            self.assertEqual(decide({'TP8': EQUAL, 'PP8': EQUAL}, arms=['TP8', 'PP8'], terminal=gate)
                             ['table_a']['row'], 'A0')

    def test_row_order(self):
        # A0 before A1: the study ended at gate S and PP8 is not feasible.
        self.assertEqual(decide({'TP8': EQUAL, 'TP4xPP2': EQUAL}, arms=['TP8', 'TP4xPP2'], terminal='S')
                         ['table_a']['row'], 'A0')
        # A2 before A3: a trade-off plus an unresolved cell is A2.
        values = {'TP8': {1: (2., .5, 1.), 8: (lambda k: (.5 if k else 1., 1., 1.)), 32: EQUAL},
                  'TP4xPP2': {c: EQUAL for c in (1, 8, 32)}, 'PP8': {c: EQUAL for c in (1, 8, 32)}}
        result = decide(values)
        self.assertIn('unresolved', result['table_a']['verdicts'].values())
        self.assertEqual(result['table_a']['row'], 'A2')

    def test_clauses(self):
        a2 = decide({'TP8': (2., .5, .5), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']
        self.assertFalse(a2['throughput_clause_contradicted'])
        a2 = decide({'TP8': (2., .5, 2.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']
        self.assertTrue(a2['throughput_clause_contradicted'])
        self.assertTrue(decide({'TP8': (.5, 1., 2.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']['docs_correction'])
        self.assertFalse(decide({'TP8': (.5, 1., 1.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']['docs_correction'])
        self.assertTrue(decide({'TP8': (1., 1., .5), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']['confirms_docs'])
        self.assertFalse(decide({'TP8': (2., 1., 1.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})['table_a']['confirms_docs'])

    def test_gap_cells_removed_by_g_in_any_arm(self):
        no_g = lambda k, c, arm: {'G': not (c == 8 and arm == 'TP4xPP2')}
        result = decide({'TP8': (1., .5, 1.), 'TP4xPP2': EQUAL, 'PP8': EQUAL}, override=no_g)
        self.assertNotIn('token_gap_s:c8', result['table_a']['verdicts'])
        self.assertIn('ttft_s:c8', result['table_a']['verdicts'])

    def test_expectation_and_review(self):
        expected = decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, .75, .75), 'PP8': EQUAL})
        self.assertEqual((expected['table_a']['row'], expected['table_b']['row']), ('A2', 'B3'))
        self.assertFalse(expected['against_expectation'])
        self.assertFalse(expected['requires_review'])
        flipped = decide({'TP8': (.5, 2., 2.), 'TP4xPP2': EQUAL, 'PP8': EQUAL})
        self.assertTrue(flipped['against_expectation'])
        self.assertTrue(flipped['requires_review'])
        excluded = decide({'TP8': (2., .5, .5), 'PP8': EQUAL}, arms=['TP8', 'PP8'], excluded=['TP4xPP2'])
        self.assertTrue(excluded['requires_review'])

    def test_gate_m_stops_for_review(self):
        result = gates.decision(study(terminal='M'), [])
        self.assertIsNone(result['table_a'])
        self.assertTrue(result['requires_review'])

    def test_lower_ladder_steps(self):
        step3 = decide({'TP8': (2., .5, .5), 'TP4xPP2': EQUAL, 'PP8': EQUAL}, step=3)
        self.assertEqual(sorted(step3['table_a']['verdicts']),
                         ['output_tokens_per_second:c8', 'token_gap_s:c1', 'token_gap_s:c8', 'ttft_s:c1', 'ttft_s:c8'])


class TableB(unittest.TestCase):
    def test_every_row(self):
        rows = {
            'B0': decide({'TP8': EQUAL, 'PP8': EQUAL}, arms=['TP8', 'PP8']),
            # Hybrid lowest TTFT everywhere; pure arms win nothing.
            'B1': decide({'TP8': EQUAL, 'TP4xPP2': (.5, 1., 1.), 'PP8': EQUAL}),
            # Hybrid best TTFT, TP8 best gap.
            'B2': decide({'TP8': (1., .5, 1.), 'TP4xPP2': (.5, 1., 1.), 'PP8': EQUAL}),
            'B3': decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, .75, .75), 'PP8': EQUAL}),
            'B4': decide({'TP8': EQUAL, 'TP4xPP2': EQUAL, 'PP8': EQUAL}),
        }
        for row, result in rows.items():
            with self.subTest(row=row):
                self.assertEqual(result['table_b']['row'], row)

    def test_b0_for_fewer_than_two_valid_levels(self):
        bad_t = lambda k, c, arm: {'T': not (arm == 'TP4xPP2' and c in (1, 8))}
        result = decide({'TP8': EQUAL, 'TP4xPP2': (.5, 1., 1.), 'PP8': EQUAL}, override=bad_t)
        self.assertEqual(result['valid_levels_all_arms'], 1)
        self.assertEqual(result['table_b']['row'], 'B0')
        # Table A is read independently over the pair's own valid levels.
        self.assertEqual(result['valid_levels_tp8_pp8'], 3)

    def test_winner_needs_better_against_every_other_arm(self):
        # Hybrid beats TP8 but ties PP8 on TTFT: no winner in that cell.
        result = decide({'TP8': (2., 1., 1.), 'TP4xPP2': (1., 1., 1.), 'PP8': (1., 1., 1.)})
        self.assertIsNone(result['table_b']['cell_winners']['ttft_s:c1'])
        self.assertEqual(result['table_b']['row'], 'B4')

    def test_row_order(self):
        # B1 before B2 needs no pure winner anywhere; one pure win makes it B2.
        mixed = decide({'TP8': {1: (1., .5, 1.), 8: EQUAL, 32: EQUAL},
                        'TP4xPP2': {c: (.5, 1., 1.) for c in (1, 8, 32)},
                        'PP8': {c: EQUAL for c in (1, 8, 32)}})
        self.assertEqual(mixed['table_b']['row'], 'B2')


class Ladder(unittest.TestCase):
    def test_all_pass(self):
        self.assertEqual(gates.ladder_step(ARMS3, 1, dict.fromkeys(ARMS3, True), {}), ('done', 1))

    def test_one_failure_moves_every_arm(self):
        history = {}
        self.assertEqual(gates.ladder_step(ARMS3, 1, {'TP8': True, 'TP4xPP2': True, 'PP8': False}, history),
                         ('next', 2))
        self.assertEqual(gates.ladder_step(ARMS3, 2, dict.fromkeys(ARMS3, True), history), ('done', 2))

    def test_infeasible_restart_and_not_measurable(self):
        history = {}
        fail_pp8 = {'TP8': True, 'TP4xPP2': True, 'PP8': False}
        gates.ladder_step(ARMS3, 1, fail_pp8, history)
        gates.ladder_step(ARMS3, 2, fail_pp8, history)
        self.assertEqual(gates.ladder_step(ARMS3, 3, fail_pp8, history), ('restart', ['PP8']))
        history = {}
        both = {'TP8': False, 'PP8': False, 'TP4xPP2': True}
        gates.ladder_step(ARMS3, 1, both, history)
        gates.ladder_step(ARMS3, 2, both, history)
        self.assertEqual(gates.ladder_step(ARMS3, 3, both, history), ('not-measurable', ['TP8', 'PP8']))

    def test_undefined_case_stops(self):
        history = {}
        gates.ladder_step(ARMS3, 1, {'TP8': False, 'TP4xPP2': True, 'PP8': True}, history)
        gates.ladder_step(ARMS3, 2, {'TP8': True, 'TP4xPP2': False, 'PP8': True}, history)
        action, _ = gates.ladder_step(ARMS3, 3, {'TP8': True, 'TP4xPP2': True, 'PP8': False}, history)
        self.assertEqual(action, 'stop')

    def test_needs_every_arm(self):
        with self.assertRaises(gates.EvidenceError):
            gates.ladder_step(ARMS3, 1, {'TP8': True}, {})


class Retries(unittest.TestCase):
    def test_caps(self):
        state = gates.RoundRetries()
        self.assertEqual([state.failed(False, True) for _ in range(4)],
                         ['rerun_round', 'rerun_round', 'replace_round', 'NOT-MEASURABLE'])
        state = gates.RoundRetries()
        self.assertEqual([state.failed(True, False) for _ in range(3)],
                         ['rerun_round', 'rerun_round', 'NOT-MEASURABLE'])
        shared = gates.StudyRetries()
        first = gates.RoundRetries(shared)
        [first.failed(False, True) for _ in range(3)]
        second = gates.RoundRetries(shared)
        self.assertEqual([second.failed(False, True) for _ in range(3)],
                         ['rerun_round', 'rerun_round', 'NOT-MEASURABLE'])


if __name__ == '__main__':
    unittest.main()


class Expectation(unittest.TestCase):
    def test_missing_cell_and_gap_order(self):
        expected = decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, .75, .75), 'PP8': EQUAL})
        self.assertFalse(expected['against_expectation'], expected['expectation_check'])
        # Gap cell at c = 8 removed by gate G: a missing expected cell is against the expectation.
        no_g = lambda k, c, arm: {'G': c != 8}
        result = decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, .75, .75), 'PP8': EQUAL}, override=no_g)
        self.assertEqual((result['table_a']['row'], result['table_b']['row']), ('A2', 'B3'))
        self.assertTrue(result['against_expectation'])
        self.assertTrue(result['requires_review'])
        # Hybrid gap not between the pure arms (equal to PP8): ordering fails.
        result = decide({'TP8': (2., .5, .5), 'TP4xPP2': (1.5, 1., .75), 'PP8': EQUAL})
        self.assertTrue(any('gap order' in p for p in result['expectation_check']))
        self.assertTrue(result['against_expectation'])
