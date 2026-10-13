"""CPU-only gate tests (P, N, Q, K/T/C/G, S/M lines, E). No server, HTTP or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import unittest

from tppbench import gates
from tests import fixtures_lines as fx
from tppbench.spec import GPUS, REQUEST_TOKENS


class GateP(unittest.TestCase):
    def check(self, arm, log, workers=fx.WORKERS):
        files, server_log = fx.split_log(log)
        return gates.transport(files, arm, workers, server_log)

    def test_real_formats_pass_in_every_arm(self):
        for arm in ('TP8', 'TP4xPP2', 'PP8'):
            with self.subTest(arm=arm):
                result = self.check(arm, fx.nccl_log(arm))
                self.assertTrue(result['pass'], result['checks'])
                self.assertEqual(len(result['mmap_host_buffers_evidence_only']), 8)

    def test_trailing_potential_list_is_never_read(self):
        # The default line's potential list names CUSTOM and SYMM_MEM; only the first list counts.
        result = self.check('TP8', fx.nccl_log('TP8'))
        self.assertTrue(result['checks']['allreduce_no_custom_or_symm_mem_enabled'])

    def test_zero_matches_fail(self):
        lines = fx.nccl_log('TP8').splitlines()
        cases = {
            'no channel lines': [x for x in lines if 'Channel' not in x],
            'no check p2p lines': [x for x in lines if 'Check P2P Type' not in x],
            'no allreduce line': [x for x in lines if "group 'tp:0'" not in x],
        }
        for name, kept in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self.check('TP8', '\n'.join(kept))['pass'])
        self.assertFalse(self.check('PP8', '')['pass'])

    def test_every_minimum_match_is_required(self):
        pp = fx.nccl_log('PP8')
        # Remove the only 3 -> 4 line (the cross-socket boundary).
        result = self.check('PP8', '\n'.join(x for x in pp.splitlines() if '3[3] -> 4[4]' not in x))
        self.assertFalse(result['checks']['every_stage_boundary_has_shm_line'])
        hybrid = fx.nccl_log('TP4xPP2')
        result = self.check('TP4xPP2', '\n'.join(x for x in hybrid.splitlines() if '[2] -> 1[6]' not in x))
        self.assertFalse(result['stage_boundaries']['2->6'])
        # A TP rank (worker pid) with no SHM line of its own.
        result = self.check('TP8', '\n'.join(x for x in fx.nccl_log('TP8').splitlines()
                                             if not (x.startswith('host1:1005:') and 'Channel' in x)))
        self.assertFalse(result['checks']['shm_line_from_every_tp_rank'])
        # A worker whose only Check P2P Type line reports 1.
        log = fx.nccl_log('PP8').replace('host1:1007:1007 [7] NCCL INFO Check P2P Type isAllDirectP2p 0',
                                         'host1:1007:1007 [7] NCCL INFO Check P2P Type isAllDirectP2p 1')
        self.assertFalse(self.check('PP8', log)['checks']['check_p2p_type_zero_from_every_worker'])

    def test_forbidden_transports_and_backends(self):
        log = fx.nccl_log('TP8')
        p2p = log.replace('Channel 00 : 0[0] -> 1[1] via SHM/direct/direct', 'Channel 00/1 : 0[0] -> 1[1] via P2P/CUMEM')
        result = self.check('TP8', p2p)
        self.assertFalse(result['checks']['every_channel_shm'])
        self.assertFalse(result['checks']['no_via_p2p_or_net'])
        net = log.replace('Channel 00 : 0[0] -> 1[1] via SHM/direct/direct',
                          'Channel 00/0 : 0[0] -> 1[1] [send] via NET/Socket/0')
        self.assertFalse(self.check('TP8', net)['checks']['no_via_p2p_or_net'])
        custom = log.replace("Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'",
                             "Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'")
        result = self.check('TP8', custom)
        self.assertFalse(result['checks']['allreduce_tp_group_enabled_is_pynccl_only'])
        self.assertFalse(result['checks']['allreduce_no_custom_or_symm_mem_enabled'])
        symm = fx.nccl_log('PP8').replace("Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'pp:0'",
                                          "Using ['NCCL_SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch "
                                          "order) for group 'pp:0'")
        self.assertFalse(self.check('PP8', symm)['pass'])

    def test_pp8_needs_no_tp_line_and_needs_eight_workers(self):
        self.assertTrue(self.check('PP8', fx.nccl_log('PP8', allreduce=False))['pass'])
        with self.assertRaises(gates.EvidenceError):
            self.check('PP8', fx.nccl_log('PP8'), {g: p for g, p in fx.WORKERS.items() if g < 7})

    def test_every_info_line_is_kept_with_its_pid(self):
        rows = gates.nccl_info_lines(fx.nccl_log('TP8'))
        self.assertEqual({r['pid'] for r in rows}, set(fx.WORKERS.values()))


class VllmLines(unittest.TestCase):
    def test_kv_reserved_requires_lines_with_the_pinned_value(self):
        eight = lambda *a: '\n'.join([fx.kv_line(*a)] * 8)
        self.assertTrue(gates.kv_reserved(eight(), 2415919104)['pass'])
        self.assertTrue(gates.kv_reserved(eight('1.25'), 1342177280)['pass'])
        self.assertTrue(gates.kv_reserved(eight('0.75'), 805306368)['pass'])
        self.assertFalse(gates.kv_reserved('', 2415919104)['pass'])
        self.assertFalse(gates.kv_reserved(eight('2.0'), 2415919104)['pass'])
        # Exactly one line per worker.
        self.assertFalse(gates.kv_reserved('\n'.join([fx.kv_line()] * 7), 2415919104)['pass'])
        self.assertFalse(gates.kv_reserved('\n'.join([fx.kv_line()] * 9), 2415919104)['pass'])

    def test_binding_lines(self):
        self.assertTrue(gates.binding(fx.binding_log())['pass'])
        self.assertFalse(gates.binding('')['pass'])
        wrong = {g: '0-25,52-77' for g in GPUS}
        self.assertFalse(gates.binding(fx.binding_log(cpus=wrong))['pass'])
        self.assertFalse(gates.binding(fx.binding_log(nodes=[0] * 8))['pass'])
        seven = '\n'.join(fx.binding_log().splitlines()[1:])
        self.assertFalse(gates.binding(seven)['pass'])
        rejected = ("WARNING [numa_utils.py:497] numactl args '--physcpubind=0-21,52-73 --membind=0' rejected; "
                    "falling back to '--physcpubind=0-21,52-73'. Add --cap-add SYS_NICE for full NUMA binding.")
        self.assertFalse(gates.binding(fx.binding_log(extra=rejected))['pass'])

    def test_fatal_lines(self):
        for line in ('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB',
                     'RuntimeError: CUDA error: an illegal memory access was encountered',
                     'Worker proc VllmWorker-3 died unexpectedly (exit code: -9), shutting down executor.',
                     'EngineCore encountered an issue. See stack trace (above) for the root cause.'):
            self.assertEqual(gates.fatal_lines('ok\n' + line), [line])
        self.assertEqual(gates.fatal_lines(fx.kv_line() + '\n' + fx.binding_log()), [])


class Levels(unittest.TestCase):
    def level(self, rows, q=30*1536, h=0, p=0, blocks=3686, expected=30, needed=32 * REQUEST_TOKENS):
        return gates.level(rows, fx.metrics(), fx.metrics(q, h, p), fx.cache_metric(blocks), expected, needed)

    def test_t_c_g_thresholds(self):
        data = [fx.request(i) for i in range(30)]
        self.assertTrue(self.level(data)['T'])
        data[0]['usage']['prompt_tokens'] = 1535
        result = self.level(data)
        self.assertFalse(result['T'])
        self.assertTrue(result['rerun_level'])
        rows = [fx.request(i) for i in range(160)]
        for i in range(8):
            rows[i] = fx.request(i, chunks=64)
        self.assertTrue(self.level(rows, q=160*1536, expected=160)['G'])     # 152/160 = 95%
        rows[9] = fx.request(9, chunks=64)
        self.assertFalse(self.level(rows, q=160*1536, expected=160)['G'])
        self.assertFalse(self.level([fx.request(i) for i in range(30)], q=100, h=1)['C'])

    def test_k_capacity_and_preemption(self):
        rows = [fx.request(i) for i in range(30)]
        self.assertTrue(self.level(rows)['K'])                       # 58,976 >= 53,248
        self.assertFalse(self.level(rows, blocks=3327)['K'])         # 53,232 < 53,248
        self.assertFalse(self.level(rows, p=1)['K'])

    def test_k_equality_across_arms_of_a_round(self):
        cells = {'TP8': {'1': {'K_capacity': True, 'K_preemption': True, 'kv_capacity_tokens': 58976}},
                 'PP8': {'1': {'K_capacity': True, 'K_preemption': True, 'kv_capacity_tokens': 58960}}}
        self.assertEqual(gates.round_capacity_k(cells), [58960, 58976])
        self.assertFalse(cells['TP8']['1']['K'])
        cells['PP8']['1']['kv_capacity_tokens'] = 58976
        gates.round_capacity_k(cells)
        self.assertTrue(cells['PP8']['1']['K'])

    def test_missing_counters_fail_closed(self):
        with self.assertRaises(gates.EvidenceError):
            gates.level([fx.request(i) for i in range(30)], '', fx.metrics(1), fx.cache_metric(), 30, 1)


class GateN(unittest.TestCase):
    def test_placement(self):
        binding = gates.binding(fx.binding_log())
        self.assertTrue(gates.placement(fx.placement_rows(), fx.WORKERS, binding)['pass'])
        self.assertFalse(gates.placement(fx.placement_rows(.94), fx.WORKERS, binding)['pass'])
        cpus = {g: '0-25,52-77' if g < 4 else '26-43,78-95' for g in GPUS}
        result = gates.placement(fx.placement_rows(cpus=cpus), fx.WORKERS, binding)
        self.assertFalse(result['pass'])
        self.assertFalse(result['binding_checks_pass'])
        self.assertFalse(gates.placement(fx.placement_rows(), fx.WORKERS, gates.binding(''))['pass'])
        rows = fx.placement_rows()
        rows[str(fx.WORKERS[3])] = {'error': 'gone'}
        self.assertFalse(gates.placement(rows, fx.WORKERS, binding)['pass'])
        # A memory share below 95% fails N but not the start test's binding checks.
        result = gates.placement(fx.placement_rows(.5), fx.WORKERS, binding)
        self.assertTrue(result['binding_checks_pass'])


def owned(ticks, affinity, start=1, child=0, threads=None):
    row = {'start': start, 'ticks': ticks, 'child_ticks': child, 'state': 'S', 'affinity': affinity}
    if threads is not None:
        row['threads'] = threads
    return row


def interpolate(a, b, step=2):
    """Samples every `step` seconds between a (t = 0) and b, scaling counters linearly."""
    import copy
    end = b['monotonic_ns'] // 1_000_000_000
    out = [a]
    for t in range(step, end, step):
        f = t / end
        x = copy.deepcopy(b)
        x['monotonic_ns'] = t * 1_000_000_000
        for c, row in x['cpu'].items():
            row['busy'] = int(a['cpu'][c]['busy'] + f * (b['cpu'][c]['busy'] - a['cpu'][c]['busy']))
            row['total'] = 100 * t
        for pid, row in x['owned'].items():
            row['ticks'] = int(f * row['ticks'])
            for thread in row.get('threads', {}).values():
                thread['ticks'] = int(f * thread['ticks'])
        x['compute_apps'] = []
        x['gpus'] = copy.deepcopy(a['gpus'])
        out.append(x)
    return out + [b]


class GateQ(unittest.TestCase):
    def window(self, extra_busy=0, engine_cpu=10, born=False, foreign_app=False):
        # 10 s level; workers on socket 0 run 500 ticks, the engine core spans both
        # sockets and runs 300 ticks on a socket-0 CPU, the client 400 on socket 1.
        a_owned = {'1000': owned(0, '0-21,52-73'), '2000': owned(0, '44-51'),
                   '900': owned(0, '0-103', threads={'900': {'ticks': 0, 'start': 1, 'processor': engine_cpu}})}
        b_owned = {'1000': owned(500, '0-21,52-73'), '2000': owned(400, '44-51'),
                   '900': owned(300, '0-103', threads={'900': {'ticks': 300, 'start': 1, 'processor': engine_cpu}})}
        if born:
            b_owned['3000'] = owned(200, '96-103')
        busy0 = {c: 0 for c in fx.ALL}
        busy1 = {c: 0 for c in fx.ALL}
        busy1[0], busy1[1], busy1[engine_cpu] = 400, 100, busy1.get(engine_cpu, 0) + 300
        busy1[44] = 400
        busy1[96] = 200 if born else 0
        busy1[2] = extra_busy
        apps = [{'pid': 4242, 'uuid': 'x'}] if foreign_app else []
        return interpolate(fx.sample(0, busy0, a_owned), fx.sample(10, busy1, b_owned, apps))

    def test_quiet_with_engine_core_attribution(self):
        result = gates.quiet(self.window(), 0, 10_000_000_000)
        self.assertTrue(result['pass'], result)
        self.assertEqual(result['foreign_cpu_fraction'], {'0': 0.0, '1': 0.0})
        # Without per-thread credit the engine core would look foreign on socket 0.
        self.assertGreater(result['foreign_cpu_fraction_without_spanning_credit']['0'], 0)

    def test_foreign_cpu_per_socket(self):
        # socket 0 has 52 CPUs x 1000 ticks; 2% = 1040 ticks.
        self.assertTrue(gates.quiet(self.window(extra_busy=1040), 0, 10**10)['pass'])
        self.assertFalse(gates.quiet(self.window(extra_busy=1041), 0, 10**10)['pass'])

    def test_process_born_inside_window_is_owned(self):
        self.assertTrue(gates.quiet(self.window(born=True), 0, 10**10)['pass'])

    def test_engine_core_on_socket_1_is_credited_there(self):
        result = gates.quiet(self.window(engine_cpu=30), 0, 10**10)
        self.assertEqual(result['foreign_cpu_fraction']['1'], 0.0)

    def test_foreign_gpu_process_and_clock(self):
        self.assertFalse(gates.quiet(self.window(foreign_app=True), 0, 10**10)['pass'])
        samples = self.window()
        for s in samples[:2]:
            s['gpus'][3]['clock_mhz'] = 100.0     # two of six samples below 90% of the median
        result = gates.quiet(samples, 0, 10**10)
        self.assertAlmostEqual(result['low_clock_fraction']['3'], 2 / 6)
        self.assertFalse(result['pass'])

    def test_missing_telemetry_fails_closed(self):
        with self.assertRaises(gates.EvidenceError):
            gates.quiet(self.window()[:1], 0, 10**10)
        with self.assertRaises(gates.EvidenceError):
            gates.quiet(self.window(), 0, 11 * 10**9)

    def test_mid_window_gap_is_a_record_and_keeps_the_no_credit_rule(self):
        # Gate Q has no gap condition; the gap is recorded, and an interval over 2.5 s
        # gets no engine-core CPU credit.
        samples = self.window()          # t = 0, 2, 4, 6, 8, 10 s
        del samples[2]                   # one 4 s interval (2 -> 6 s)
        result = gates.quiet(samples, 0, 10**10)
        self.assertTrue(result['pass'])
        self.assertEqual(result['reasons'], [])
        gaps = result['telemetry_gaps']
        self.assertIs(gaps['gate'], False)
        self.assertEqual(gaps['over_threshold'][0], {'from_monotonic_ns': 2 * 10**9, 'to_monotonic_ns': 6 * 10**9,
                                                     'interval_s': 4.0})
        self.assertIsNone(result['level_scale'])
        # 120 of the engine core's 300 ticks fall in that interval: no credit, foreign.
        self.assertAlmostEqual(result['foreign_cpu_fraction']['0'], 120 / 52000, places=6)
        dense = gates.quiet(self.window(), 0, 10**10)
        self.assertTrue(dense['pass'])
        self.assertEqual(dense['foreign_cpu_fraction']['0'], 0.0)

    def test_edge_gap_uses_the_level_scaled_fraction(self):
        # Level 3-10 s; the samples are t = 0, 4, 6, 8, 10: the edge interval 0 -> 4 s is
        # over 2.5 s, so the foreign fraction is also taken over the level's 7 s.
        def thin(extra_busy):
            samples = self.window(extra_busy=extra_busy)
            del samples[1]
            return samples
        result = gates.quiet(thin(0), 3 * 10**9, 10**10)
        self.assertTrue(result['pass'])
        self.assertAlmostEqual(result['level_scale'], .7)
        self.assertAlmostEqual(result['foreign_cpu_fraction_unscaled']['0'], 120 / 52000, places=6)
        self.assertAlmostEqual(result['foreign_cpu_fraction_level_scaled']['0'], 120 / 52000 / .7, places=6)
        self.assertEqual(result['foreign_cpu_fraction']['0'], result['foreign_cpu_fraction_level_scaled']['0'])
        # 900 extra ticks: 1020/52000 = 1.96% unscaled would pass; over the level 2.8% fails.
        diluted = gates.quiet(thin(900), 3 * 10**9, 10**10)
        self.assertLess(diluted['foreign_cpu_fraction_unscaled']['0'], .02)
        self.assertFalse(diluted['pass'])
        self.assertEqual(diluted['reasons'], [])
        # A dense window is unchanged: no scaling, the same 900 ticks pass.
        dense = gates.quiet(self.window(extra_busy=900), 0, 10**10)
        self.assertTrue(dense['pass'])
        self.assertIsNone(dense['level_scale'])

    def test_errored_sample_fails_q_instead_of_aborting(self):
        # An errored sample fails gate Q for the level.
        samples = self.window()
        samples[2] = {'monotonic_ns': samples[2]['monotonic_ns'], 'error': 'nvidia-smi timed out'}
        result = gates.quiet(samples, 0, 10**10)
        self.assertFalse(result['pass'])
        # Dropping the errored t = 4 s sample leaves a 4 s interval: recorded, not a reason.
        self.assertEqual(result['reasons'], ['telemetry sample error'])
        self.assertEqual(result['telemetry_gaps']['intervals_over_threshold'], 1)
        self.assertEqual(result['telemetry_sample_errors'], ['nvidia-smi timed out'])
        self.assertIn('foreign_cpu_fraction', result)
        samples[0] = {'monotonic_ns': 0, 'error': 'boom'}       # the enclosing sample failed
        result = gates.quiet(samples, 0, 10**10)
        self.assertFalse(result['pass'])
        self.assertIn('not_computable', result)
        from tppbench import telemetry
        monitor = telemetry.Monitor.__new__(telemetry.Monitor)
        monitor.sample = lambda: (_ for _ in ()).throw(RuntimeError('Cannot observe all eight cards'))
        self.assertEqual(monitor.boundary()['error'], 'Cannot observe all eight cards')


class GateE(unittest.TestCase):
    base = list(range(100, 228))

    def resp(self, tokens=None, top=None):
        return fx.gate_e_response(tokens or self.base, top)

    def test_pairwise_rule(self):
        identical = gates.compare_outputs(gates.gate_e_tokens(self.resp()), gates.gate_e_tokens(self.resp()))
        self.assertEqual(identical['status'], 'identical')
        other = self.base[:10] + [999] + self.base[11:]
        x = self.resp(top={10: {110: -0.5, 999: -0.9}})
        y = self.resp(other, top={10: {999: -0.6, 110: -0.7}})
        result = gates.compare_outputs(gates.gate_e_tokens(x), gates.gate_e_tokens(y))
        self.assertEqual((result['status'], result['position']), ('benign', 10))
        # A margin of exactly 1.0 nat is not "less than 1.0".
        x = self.resp(top={10: {110: -0.5, 999: -1.5}})
        self.assertEqual(gates.compare_outputs(gates.gate_e_tokens(x), gates.gate_e_tokens(y))['status'],
                         'corruption')
        # The other arm's token absent from the top 5.
        x = self.resp(top={10: {110: -0.5, 7: -0.9}})
        self.assertEqual(gates.compare_outputs(gates.gate_e_tokens(x), gates.gate_e_tokens(y))['status'],
                         'corruption')
        # No top_logprobs at the first differing position.
        x = self.resp()
        x['choices'][0]['logprobs']['content'][10]['top_logprobs'] = []
        self.assertEqual(gates.compare_outputs(gates.gate_e_tokens(x), gates.gate_e_tokens(y))['status'],
                         'corruption')

    def test_tokens_are_ids_and_responses_complete(self):
        bad = self.resp()
        bad['choices'][0]['logprobs']['content'][0]['token'] = 'hello'
        with self.assertRaises(gates.EvidenceError):
            gates.gate_e_tokens(bad)
        short = self.resp()
        short['usage']['completion_tokens'] = 127
        with self.assertRaises(gates.EvidenceError):
            gates.gate_e_tokens(short)

    def test_exclusions(self):
        A, B, C = 'TP8', 'TP4xPP2', 'PP8'
        self.assertEqual(gates.exclusions([A, B, C], []), [])
        self.assertEqual(gates.exclusions([A, B, C], [(A, B), (A, C)]), [A])        # attributable
        self.assertEqual(gates.exclusions([A, B, C], [(A, B)]), [A, B])             # C benign with both
        self.assertEqual(gates.exclusions([A, B, C], [(A, B), (A, C), (B, C)]), [A, B, C])
        self.assertEqual(gates.exclusions([A, C], [(A, C)]), [A, C])

    def test_gate_e_over_three_arms(self):
        other = self.base[:5] + [999] + self.base[6:]
        responses = {'TP8': {'p': self.resp(other)}, 'TP4xPP2': {'p': self.resp()}, 'PP8': {'p': self.resp()}}
        result = gates.gate_e(responses)
        self.assertEqual(result['excluded'], ['TP8'])
        self.assertEqual(result['corrupted_pairs'], [['TP8', 'TP4xPP2'], ['TP8', 'PP8']])


if __name__ == '__main__':
    unittest.main()


class TransportAndFailureEvidence(unittest.TestCase):
    def test_gate_p_needs_a_file_from_every_worker(self):
        files, server_log = fx.split_log(fx.nccl_log('PP8'))
        del files[fx.WORKERS[5]]
        result = gates.transport(files, 'PP8', fx.WORKERS, server_log)
        self.assertFalse(result['checks']['nccl_file_from_every_worker'])
        self.assertEqual(result['workers_without_nccl_file'], [5])

    def test_gate_p_reads_files_not_the_interleaved_stdout(self):
        # NCCL lines that reached only the shared server log do not count.
        result = gates.transport({}, 'TP8', fx.WORKERS, fx.nccl_log('TP8'))
        self.assertFalse(result['pass'])
        self.assertFalse(result['checks']['channel_lines_present'])
        files, server_log = fx.split_log(fx.nccl_log('TP8'))
        files[fx.WORKERS[0]] += files[fx.WORKERS[1]]          # a foreign pid's line in pid 1000's file
        with self.assertRaises(gates.EvidenceError):
            gates.transport(files, 'TP8', fx.WORKERS, server_log)

    def test_nccl_files_pid_from_name(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'nccl.host1.1003.log').write_text('x\n')
            self.assertEqual(gates.nccl_files(root), {1003: 'x\n'})
            Path(root, 'nccl.host1.abc.log').write_text('')
            with self.assertRaises(gates.EvidenceError):
                gates.nccl_files(root)

    def test_failure_class_and_oom_lines(self):
        oom = ('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total '
               'capacity of 23.99 GiB of which 1.10 GiB is free.')
        self.assertEqual(gates.failure_class('x\n' + oom, True), 'oom')
        self.assertEqual(gates.oom_lines('x\n' + oom), [oom])
        self.assertEqual(gates.failure_class('Worker proc VllmWorker-1 died unexpectedly', False), 'engine_death')
        self.assertEqual(gates.failure_class('nothing fatal', True), 'engine_death')
        self.assertEqual(gates.failure_class('nothing fatal', False), 'other')

    def test_gate_e_without_logprobs_and_missing_token(self):
        good = fx.gate_e_response(list(range(128)))
        bare = {'choices': [{'index': 0, 'logprobs': None}], 'usage': {'completion_tokens': 128}}
        self.assertIsNone(gates.gate_e_tokens(bare))
        self.assertEqual(gates.compare_outputs(gates.gate_e_tokens(good), None)['status'], 'corruption')
        # A response without logprobs fails the premise check, so no
        # exclusion is applied; the pair comparison is kept for information only.
        result = gates.gate_e({'TP8': {'p': bare}, 'TP4xPP2': {'p': good}, 'PP8': {'p': good}})
        self.assertIs(result['premise_pass'], False)
        self.assertIsNone(result['excluded'])
        self.assertEqual(result['pairs_information_only']['TP8|PP8']['prompts']['p']['status'], 'corruption')
        broken = fx.gate_e_response(list(range(128)))
        del broken['choices'][0]['logprobs']['content'][3]['token']
        with self.assertRaises(gates.EvidenceError):
            gates.gate_e_tokens(broken)
        broken = fx.gate_e_response(list(range(128)))
        del broken['choices'][0]['logprobs']['content'][3]['top_logprobs'][0]['token']
        with self.assertRaises(gates.EvidenceError):
            gates.gate_e_tokens(broken)
