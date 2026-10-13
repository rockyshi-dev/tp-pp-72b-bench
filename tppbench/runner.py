"""The four phases: start tests with the KV-pool ladder, configuration starts with gate E,
the transport record (gate P), and five timed rounds with the decision calculation.

Before every server start of every phase (all go through open_server) the touch-and-free
helper runs on both NUMA nodes in parallel, with the per-node precondition of usable free
memory >= 20e9 bytes after it (one repeat on a failing node). A refusal or an unmet
precondition raises HostMemoryStop before the server starts: the phase stops for review;
it is not a gate result and not a start-test failure, so the ladder never sees it. Each
level also records each node's usable free memory before it and the level's /proc/vmstat
reclaim-counter deltas (records, not gates).

Gate E's requests carry repetition_penalty 1.0 (client.gate_e_payload only). Before gate E
is read, the premise check (gates.premise) runs in every arm; if it fails, gate-e.json has
premise_pass false and excluded null, study.json flags it for review, the settings receipt
is written with pass false and no later phase starts (check_prior needs pass true;
study_from needs premise_pass true).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import client
from . import gates
from . import hostmem
from . import launch
from . import spec
from . import telemetry
from .spec import (ARMS, BANK_LEVELS, BANK_MEASURED, BURST_REQUESTS, COMMIT, GATE_E_C1_PROMPTS,
                   HOST_MEMORY_NOTE, HOST_MEMORY_NODES, MEASURED, MODEL, PROMPT_TOKENS, REVISION,
                   ROUNDS, TP_ARMS, TRANSPORT_REQUESTS, USABLE_FREE_AFTER_HELPER, WORKER_CPUS, capacity_needed,
                   cpu_list, digest, levels, output_path, round_order, write_json)


class GateM(RuntimeError):
    """Gate M: out-of-memory or CUDA error, or engine death, during a timed level."""


class DeviationSignature(RuntimeError):
    """A default start failed with a declared deviation's signature: stop for review."""


class HostMemoryStop(RuntimeError):
    """The touch-and-free helper refused on a node, or a node's usable free memory
    stayed below 20e9 bytes after the helper and one repeat. Raised before the
    server starts; the phase stops for review. Not a gate result and not a
    start-test failure: it never moves the ladder or counts toward INFEASIBLE."""


# ---------------------------------------------------------------- host memory

# /proc/vmstat counters are machine-wide (sums over both NUMA nodes).
VMSTAT_RECORDED = ('pgscan_kswapd', 'pgscan_direct')
VMSTAT_PREFIX_RECORDED = 'allocstall_'
VMSTAT_NOTE = '/proc/vmstat counters are machine-wide sums over both NUMA nodes; reclaim by other users shows too'


def host_memory_dir(directory):
    # launch.Server creates the start directory with exist_ok=False, so the
    # helper's evidence sits beside it: <start dir>-host-memory.
    directory = Path(directory)
    return directory.parent / f'{directory.name}-host-memory'


def host_memory_stop(out, record, reason, **extra):
    record.update(stopped=True, gate_result=False, start_test_failure=False, reason=reason,
                  action='phase stopped for review before the server started', **extra)
    write_json(out / 'host-memory.json', record)
    raise HostMemoryStop(reason)


def host_memory_step(directory):
    """Before one server start: the touch-and-free helper (oom_score_adj 1000; refusal
    unless reclaimable > 44e9 bytes) on both nodes in
    parallel, each bound to its node (memory MPOL_BIND, CPUs the node's worker
    cores, idle before the start), writing and freeing 40e9 bytes. Precondition:
    each node's usable free memory (Normal free minus low watermark) >= 20e9
    bytes after it; a failing node gets one repeat; else HostMemoryStop. Helper
    pids are in telemetry.HELPER_PIDS while they live."""
    out = host_memory_dir(directory)
    out.mkdir(parents=True, exist_ok=False)
    record = {'note': HOST_MEMORY_NOTE, 'nodes': list(HOST_MEMORY_NODES),
              'precondition_usable_free_bytes': USABLE_FREE_AFTER_HELPER, 'gate': False, 'runs': [],
              'units': 'decimal bytes', 'start_wall_ns': time.time_ns()}
    pending = list(HOST_MEMORY_NODES)
    for number in (1, 2):
        try:
            rows = hostmem.run(tuple(pending), out / f'helper-{number}', telemetry.HELPER_PIDS,
                               cpus_of={n: cpu_list(WORKER_CPUS[n]) for n in pending})
        except hostmem.Refusal as exc:
            host_memory_stop(out, record, f'helper refused (run {number}): {exc}', refusal=exc.record)
        failing = []
        for node in pending:
            row = rows[node]
            usable = row['after']['usable_free_bytes']
            met = usable >= USABLE_FREE_AFTER_HELPER
            record['runs'].append({'helper': number, 'node': node, 'pid': row['pid'], 'seconds': row['seconds'],
                                   'before': row['before'], 'after': row['after'], 'precondition_met': met})
            if not met:
                failing.append(node)
        if not failing:
            record.update(precondition='met', finish_wall_ns=time.time_ns())
            write_json(out / 'host-memory.json', record)
            return record
        pending = failing
    host_memory_stop(out, record, f'usable free memory below {USABLE_FREE_AFTER_HELPER} bytes on node(s) '
                                  f'{pending} after the helper and one repeat', failing_nodes=pending)


def read_vmstat(text=None):
    text = Path('/proc/vmstat').read_text() if text is None else text
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def reclaim_counters(vmstat):
    """pgscan_kswapd, pgscan_direct and every allocstall_*; a missing named counter is None."""
    keys = list(VMSTAT_RECORDED) + sorted(k for k in vmstat if k.startswith(VMSTAT_PREFIX_RECORDED))
    return {k: vmstat.get(k) for k in keys}


def level_memory_before(directory, snapshot=hostmem.snapshot, vmstat=read_vmstat):
    """Record only: a failed read is written down, never raised."""
    record = {'gate': False, 'note': VMSTAT_NOTE,
              'monotonic_ns_before': time.monotonic_ns(), 'usable_free_bytes_before': {}, 'nodes_before': {}}
    for node in HOST_MEMORY_NODES:
        try:
            reading = snapshot(node)
            record['nodes_before'][str(node)] = reading
            record['usable_free_bytes_before'][str(node)] = reading['usable_free_bytes']
        except (OSError, ValueError) as exc:
            record.setdefault('node_errors', {})[str(node)] = f'{type(exc).__name__}: {exc}'
    try:
        record['vmstat_before'] = reclaim_counters(vmstat())
    except OSError as exc:
        record['vmstat_error'] = f'{type(exc).__name__}: {exc}'
    write_json(Path(directory) / 'host-memory-level.json', record)
    return record


def level_memory_after(directory, record, vmstat=read_vmstat):
    try:
        after = reclaim_counters(vmstat())
    except OSError as exc:
        record['vmstat_error_after'] = f'{type(exc).__name__}: {exc}'
    else:
        before = record.get('vmstat_before') or {}
        record.update(vmstat_after=after, monotonic_ns_after=time.monotonic_ns(),
                      vmstat_delta={k: (after[k] - before[k] if isinstance(after.get(k), int)
                                        and isinstance(before.get(k), int) else None)
                                    for k in sorted(set(before) | set(after))})
    write_json(Path(directory) / 'host-memory-level.json', record)
    return record


# ---------------------------------------------------------------- prompts

def gate_e_reference(path=None):
    """The five gate E prompts as stored in the prompt file (messages built from the
    package's gate_e_prompts.json)."""
    from .construct import GATE_E_FILE, messages
    data = json.loads(Path(path or GATE_E_FILE).read_text())
    return [{'id': p['id'], 'language': p['language'], 'messages': messages(p['text'])} for p in data['prompts']]


def load_prompts(path):
    bank = json.loads(Path(path).read_text())
    if (bank['model'], bank['model_revision'], bank['source_commit']) != (MODEL, REVISION, COMMIT):
        raise RuntimeError('Wrong prompt provenance')
    rows = bank['requests']
    expected = 2 * sum(BANK_MEASURED[c] + 2 * c for c in BANK_LEVELS) + BURST_REQUESTS
    if len(rows) != expected or len({r['first_text_token'] for r in rows}) != expected:
        raise RuntimeError('Prompt bank lacks distinct first text tokens')
    if any(r['id'] != i or len(r['token_ids']) != PROMPT_TOKENS or r['prompt_tokens'] != PROMPT_TOKENS
           for i, r in enumerate(rows)):
        raise RuntimeError('Prompt bank token count or identity mismatch')
    if len({tuple(r['token_ids'][:16]) for r in rows}) != expected:
        raise RuntimeError('Shared initial 16-token blocks')
    used = []
    for retry in (0, 1):
        for c in BANK_LEVELS:
            item = bank['banks'][f'{c}:{retry}']
            if len(item['warmup']) != 2 * c or len(item['measured']) != BANK_MEASURED[c]:
                raise RuntimeError('Wrong bank size')
            used += item['warmup'] + item['measured']
    used += bank['banks']['burst']['measured']
    if sorted(used) != list(range(expected)):
        raise RuntimeError('Overlapping prompt banks')
    gate_e = bank['gate_e_non_english']
    if ([{k: q.get(k) for k in ('id', 'language', 'messages')} for q in gate_e['prompts']] != gate_e_reference()
            or len(gate_e['prompts']) != 5):
        raise RuntimeError('Gate E prompts differ from the hashed file')
    return bank


def measured_count(c):
    n = MEASURED[c]
    if n is None:
        raise RuntimeError(f'No declared measured count at c = {c}; refusing to run')
    return n


def level_rows(bank, c, retry):
    ids = bank['banks'][f'{c}:{retry}']
    rows = bank['requests']
    return [rows[i] for i in ids['warmup']], [rows[i] for i in ids['measured'][:measured_count(c)]]


def gate_e_items(bank):
    rows = bank['requests']
    c1 = [{'id': f'c1-{rows[i]["id"]}', 'messages': rows[i]['messages']}
          for i in bank['banks']['1:0']['measured'][:GATE_E_C1_PROMPTS]]
    return c1 + [{'id': f'ne-{p["id"]}', 'messages': p['messages']} for p in bank['gate_e_non_english']['prompts']]


# ---------------------------------------------------------------- one level

def metrics(server, directory, name):
    text = launch.http(server.url + '/metrics')
    (directory / name).write_text(text)
    gates.metric_samples(text, 'vllm:num_requests_waiting')
    return text


def check_m(server, offset):
    """Gate M evidence for the log written since `offset` and the engine state."""
    log = server.log_text()[offset:]
    fatal = gates.fatal_lines(log)
    alive = server.process.poll() is None
    healthy = False
    if alive:
        try:
            launch.http(server.url + '/health', timeout=10)
            healthy = True
        except Exception:
            healthy = False
    return {'pass': alive and healthy and not fatal, 'alive': alive, 'healthy': healthy, 'fatal_lines': fatal}


def run_level(server, warm, measured, c, topology, directory, needed, timed):
    """Warm-up, placement evidence, metrics, monitored measured requests and the
    level's gates. With `timed`, an engine death or fatal line is gate M."""
    directory.mkdir()
    memory_record = level_memory_before(directory)       # host-memory record, before the level
    offset = len(server.log_text())
    try:
        if warm:
            result = client.phase(server.url, warm, c, topology['client_cpus'], directory / 'warmup')
            if not all(gates.request_valid(r) for r in result):
                raise launch.StartFailure('Warm-up incomplete or wrong token count')
        memory = telemetry.placement(server.process.pid, server.workers, directory / 'numa') if warm else None
        before = metrics(server, directory, 'metrics-before.txt')
        monitor = telemetry.Monitor(os.getpid(), directory, topology['runner_cpus'])
        boundary_before = monitor.boundary()
        with monitor:
            rows = client.phase(server.url, measured, c, topology['client_cpus'], directory / 'measured')
        boundary_after = monitor.boundary()
        if memory is None:      # the start-test burst has no warm-up: read placement after it
            memory = telemetry.placement(server.process.pid, server.workers, directory / 'numa')
        after = metrics(server, directory, 'metrics-after.txt')
        level_memory_after(directory, memory_record)
    except Exception as exc:
        m = check_m(server, offset)
        write_json(directory / 'gate-m.json', {**m, 'error': f'{type(exc).__name__}: {exc}'})
        if timed and not m['pass']:
            raise GateM(f'Gate M at c={c}: {m}') from exc
        raise
    m = check_m(server, offset)
    write_json(directory / 'gate-m.json', m)
    if timed and not m['pass']:
        raise GateM(f'Gate M at c={c}: {m}')
    write_json(directory / 'placement.json', memory)
    samples = [boundary_before, *(json.loads(x) for x in monitor.path.read_text().splitlines()), boundary_after]
    write_json(directory / 'telemetry-with-boundaries.json', samples)
    startup_metrics = (server.directory / 'metrics-startup.txt').read_text()
    result = gates.level(rows, before, after, startup_metrics, len(measured), needed)
    binding = gates.binding(server.log_text())
    result['placement'] = gates.placement(memory, server.workers, binding)
    result['N'] = result['placement']['pass']
    result['quiet'] = gates.quiet(samples, *result['window_ns'])
    result['Q'] = result['quiet']['pass']
    result['M'] = m['pass']
    result.update(directory=str(directory), prompt_ids=[r['id'] for r in measured])
    write_json(directory / 'gate-calculation.json', result)
    return result


# ---------------------------------------------------------------- one start

def open_server(args, arm, directory, kind, step):
    """Every server start of every phase. The foreign-process and port checks
    first (no helper while another job holds a card), then the touch-and-free
    step; HostMemoryStop propagates before any server process exists."""
    launch.preflight(args.port)
    host_memory_step(directory)
    server = launch.Server(arm, directory, args.topology, args.locks, args.deviations, args.port, kind, step,
                           args.numactl)
    try:
        server.ready()
        startup = metrics(server, directory, 'metrics-startup.txt')
        log = server.log_text()
        record = launch.start_record(log, startup, step)
        write_json(directory / 'start-record.json', record)
    except BaseException as exc:
        # The server exists although open_server did not return it; it rides on the
        # exception for callers that need it.
        try:
            exc.server = server
        except AttributeError:
            pass
        raise
    return server, startup, log, record


def start_test(args, arm, step, directory):
    """Start test at one ladder step; returns (passed, record)."""
    server = None
    try:
        server, startup, log, record = open_server(args, arm, directory, 'start-test', step)
        c = max(levels(step))
        burst = [args.bank['requests'][i] for i in args.bank['banks']['burst']['measured']]
        # The burst reaches the peak KV occupancy and activation size of the
        # timed levels; an engine death or a fatal line in it raises GateM.
        level = run_level(server, [], burst, c, args.topology, directory / f'burst-c{c}',
                          capacity_needed(step), timed=True)
        log = server.log_text()
        allreduce = gates.allreduce_check(log, arm)
        checks = {'all_burst_requests_valid': level['dropped'] == 0,
                  'engine_alive': level['M'],
                  'no_cuda_error_or_oom_line': gates.fatal_lines(log) == [],
                  'binding_checks': level['placement']['binding_checks_pass'],
                  'kv_pool_reserved_as_pinned': record['kv_reserved']['pass']}
        if arm in TP_ARMS:
            checks['allreduce_line_present'] = allreduce['checks'].get('tp_group_line_present', False)
        passed = all(checks.values())
        outcome = {'arm': arm, 'step': step, 'pass': passed, 'checks': checks,
                   'failure_class': None if passed else gates.failure_class(log, False),
                   'free_mib_after_start': json.loads((directory / 'ready-cards.json').read_text())
                   ['free_mib_after_start'],
                   'kv_capacity_tokens': record['kv_capacity_tokens'], 'burst_gates_record_only':
                   {k: level[k] for k in ('K', 'T', 'C', 'G', 'N', 'Q', 'M')}, 'allreduce': allreduce}
        write_json(directory / 'start-test.json', outcome)
        server.close_completed()
        return passed, outcome
    except (launch.StartFailure, GateM) as exc:
        failure = directory / 'start-failure.json'
        signature = None
        if failure.exists():
            data = json.loads(failure.read_text())
            signature = launch.startup_signature(Path(data['log']).read_text(errors='replace'), data['returncode'],
                                                 data['timed_out'], data.get('nccl_observed_seconds', 0))
        if signature and not args.deviations:
            raise DeviationSignature(f'{arm} step {step}: default start failed with signature {signature}') from exc
        if server is None:
            raise
        # Evidence before any shutdown.
        log = server.log_text()
        exited = server.process.poll() is not None
        try:
            snapshot = telemetry.gpu_snapshot()
        except Exception as snap_exc:
            snapshot = {'error': str(snap_exc)}
        write_json(directory / 'failure-evidence.json',
                   {'failure_class': gates.failure_class(log, exited), 'server_exited': exited,
                    'oom_lines': gates.oom_lines(log), 'fatal_lines': gates.fatal_lines(log),
                    'nvidia_smi': snapshot})
        server.settle_after_failure()     # ManualAttention propagates: the ladder stops
        outcome = {'arm': arm, 'step': step, 'pass': False, 'error': f'{type(exc).__name__}: {exc}',
                   'failure_class': gates.failure_class(log, exited)}
        if (directory / 'ready-cards.json').exists():
            outcome['free_mib_after_start'] = json.loads((directory / 'ready-cards.json').read_text())[
                'free_mib_after_start']
        write_json(directory / 'start-test.json', outcome)
        return False, outcome
    except BaseException as exc:
        if server is not None and not server.completed:
            server.retain(str(exc))
        raise


def ladder(args, output):
    """One ladder step for all arms; a failing arm moves all arms
    to the next step; an arm failing at every step is INFEASIBLE and the rest
    are re-tested from step 1."""
    active, infeasible, attempts, passes = list(ARMS), [], [], 0
    while True:
        passes += 1
        history, step = {}, 1
        while True:
            results = {}
            for arm in active:
                passed, outcome = start_test(args, arm, step, output / f'pass{passes}-step{step}-{arm}')
                results[arm] = passed
                attempts.append(outcome)
                write_json(output / 'ladder-attempts.json', attempts)
            action, value = gates.ladder_step(active, step, results, history)
            if action != 'next':
                break
            step = value
        if action == 'done':
            return {'result': 'done', 'step': value, 'arms': active, 'infeasible': infeasible,
                    'attempts': attempts}
        if action in ('restart', 'not-measurable'):
            infeasible += value
            active = [a for a in active if a not in value]
            if action == 'restart':
                continue
            return {'result': action, 'step': None, 'arms': active, 'infeasible': infeasible,
                    'reason': 'Gate S: fewer than two feasible arms', 'attempts': attempts}
        return {'result': action, 'step': None, 'arms': active, 'infeasible': infeasible,
                'reason': value, 'attempts': attempts}


# ---------------------------------------------------------------- configuration, transport

def settings(args, output):
    responses = {}
    for arm in args.study['arms']:
        directory = output / arm
        server = None
        try:
            server, startup, log, record = open_server(args, arm, directory, 'settings', args.study['step'])
            fields = launch.startup_fields(log, startup, arm)
            write_json(directory / 'startup-fields.json', fields)
            if fields['missing_fields']:
                raise gates.EvidenceError('DEBUG settings evidence missing: ' + ', '.join(fields['missing_fields']))
            if not record['kv_reserved']['pass']:
                raise gates.EvidenceError('Pinned KV pool not reserved as declared')
            rows = client.gate_e_requests(server.url, gate_e_items(args.bank), directory / 'gate-e')
            bad = [r['id'] for r in rows if r['error'] or r.get('status') != 200]
            if bad:
                raise gates.EvidenceError(f'Gate E requests failed: {bad}')
            responses[arm] = {r['id']: r['response'] for r in rows}
            server.close_completed()
        except BaseException as exc:
            if server is not None and not server.completed:
                server.retain(str(exc))
            raise
    result = gates.gate_e(responses)
    write_json(output / 'gate-e.json', result)
    return result


def transport(args, output):
    results = {}
    for arm in args.study['arms']:
        directory = output / arm
        server = None
        try:
            server, startup, log, record = open_server(args, arm, directory, 'transport', args.study['step'])
            rows = [args.bank['requests'][i] for i in args.bank['banks']['burst']['measured'][:TRANSPORT_REQUESTS]]
            answered = client.phase(server.url, rows, TRANSPORT_REQUESTS, args.topology['client_cpus'],
                                    directory / 'requests')
            if not all(gates.request_valid(r) for r in answered):
                raise gates.EvidenceError('Transport-check request failed')
            server.close_completed()
            log = server.log_text()
            files = gates.nccl_files(server.nccl_dir)
            with (directory / 'nccl-info-lines.jsonl').open('x') as dest:
                for pid, text in sorted(files.items()):
                    for row in gates.nccl_info_lines(text):
                        dest.write(json.dumps({**row, 'file_pid': pid}) + '\n')
            results[arm] = gates.transport(files, arm, server.workers, log)
            write_json(directory / 'transport.json', results[arm])
        except BaseException as exc:
            if server is not None and not server.completed:
                server.retain(str(exc))
            raise
    return results


# ---------------------------------------------------------------- timed rounds

def timed_part(args, arm, directory, step_levels, retry):
    """One arm's part of a round: start, levels in order, stop."""
    server = None
    try:
        server, startup, log, record = open_server(args, arm, directory, 'timed', args.study['step'])
        if not record['kv_reserved']['pass']:
            raise gates.EvidenceError('Pinned KV pool not reserved as declared')
        cells = {}
        for c in step_levels:
            warm, measured = level_rows(args.bank, c, retry)
            cells[str(c)] = run_level(server, warm, measured, c, args.topology, directory / f'c{c}',
                                      capacity_needed(args.study['step']), timed=True)
        server.close_completed()
        return cells
    except GateM:
        server.settle_after_failure()       # ManualAttention propagates and stops everything
        raise
    except BaseException as exc:
        if server is not None and not server.completed:
            server.retain(str(exc))
        raise


def timed_part_with_m(args, arm, directory, step_levels, retry):
    """Gate M: rerun that arm's part of the round once; a second failure ends
    the study for review (the arm becomes INFEASIBLE at this step)."""
    try:
        return timed_part(args, arm, directory, step_levels, retry), None
    except GateM as first:
        write_json(directory.parent / f'{directory.name}-gate-m-first.json', {'error': str(first)})
        try:
            return timed_part(args, arm, directory.parent / (directory.name + '-m-rerun'), step_levels, retry), None
        except GateM as second:
            return None, {'arm': arm, 'first': str(first), 'second': str(second)}


def timed(args, output):
    arms, step_levels = args.study['arms'], levels(args.study['step'])
    for c in step_levels:
        measured_count(c)
    rounds = []
    study_retries = gates.StudyRetries()
    for number in range(1, ROUNDS + 1):
        retry_state = gates.RoundRetries(study_retries)
        attempt = 0
        while True:
            directory = output / f'round-{number}-attempt-{attempt}'
            directory.mkdir()
            order = round_order(number, arms)
            parts = {}
            for name in order:
                cells, failure = timed_part_with_m(args, name, directory / name, step_levels, 0)
                if failure:
                    write_json(directory / 'gate-m-terminal.json', failure)
                    return rounds, 'M', failure
                parts[name] = cells
            gates.round_capacity_k(parts)
            history = [cell for cells in parts.values() for cell in cells.values()]
            retry_levels = [c for c in step_levels
                            if all(parts[a][str(c)]['K'] for a in arms)
                            and any(parts[a][str(c)]['rerun_level'] for a in arms)]
            if retry_levels:
                more_all = {}
                for name in order:
                    more, failure = timed_part_with_m(args, name, directory / (name + '-level-retry'),
                                                      retry_levels, 1)
                    if failure:
                        write_json(directory / 'gate-m-terminal.json', failure)
                        return rounds, 'M', failure
                    more_all[name] = more
                gates.round_capacity_k({**{a: parts[a] for a in arms},
                                        **{a + '-retry': more_all[a] for a in arms}})
                for name in arms:
                    history.extend(more_all[name].values())
                    for c, replacement in more_all[name].items():
                        original = parts[name][c]
                        replacement['K'] = original['K'] and replacement['K']
                        if original['T'] and original['C'] and not original['G']:
                            replacement['G'] = False
                    parts[name].update(more_all[name])
            paired = {'round': number, 'attempt': attempt, 'order': list(order),
                      'levels': {str(c): {name: parts[name][str(c)] for name in arms} for c in step_levels}}
            n_ok = all(cell['N'] for cell in history)
            q_ok = all(cell['Q'] for cell in history)
            paired.update(N=n_ok, Q=q_ok)
            write_json(directory / 'paired.json', paired)
            if n_ok and q_ok:
                rounds.append(paired)
                write_json(output / 'accepted-rounds.json', rounds)
                break
            action = retry_state.failed(n_ok, q_ok)
            # The gate-Q reasons of every failing cell.
            q_reasons = [{'directory': cell.get('directory'), 'reasons': (cell.get('quiet') or {}).get('reasons', []),
                          'not_computable': (cell.get('quiet') or {}).get('not_computable')}
                         for cell in history if not cell['Q']]
            write_json(directory / 'retry-action.json', {'action': action, **retry_state.receipt(),
                                                         'q_failures': q_reasons})
            if action == 'NOT-MEASURABLE':
                return rounds, retry_state.terminal_gate, None
            attempt += 1
    return rounds, None, None


# ---------------------------------------------------------------- prerequisites

PHASE_EVIDENCE = {'start-test': ('ladder.json',), 'settings': ('gate-e.json', 'study.json'),
                  'transport': ('study.json',), 'timed': ('accepted-rounds.json', 'decision-for-review.json')}


def check_prior(path, kind, deviations, prompts_sha, code_sha256):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not data.get('pass') or data.get('phase') != kind or data.get('deviations') != deviations:
        raise RuntimeError(f'Missing passing {kind} prerequisite with the same startup conditions')
    if data.get('prompts_sha256') != prompts_sha:
        raise RuntimeError('Prerequisite uses different prompts')
    evidence = data.get('evidence_sha256', {})
    required = [path.parent / 'run-manifest.json', *(path.parent / n for n in PHASE_EVIDENCE[kind])]
    if any(str(p) not in evidence for p in required):
        raise RuntimeError('Prerequisite is missing required hashed evidence')
    for filename, sha in evidence.items():
        if not Path(filename).resolve().is_relative_to(path.parent):
            raise RuntimeError('Prerequisite evidence is outside its run directory')
        if digest(filename) != sha:
            raise RuntimeError('Prerequisite evidence changed')
    manifest = json.loads(required[0].read_text())
    if not code_sha256 or manifest.get('code_sha256') != code_sha256:
        raise RuntimeError('Prerequisite was produced by different code')
    if (manifest.get('phase'), manifest.get('deviations'), manifest.get('prompts_sha256')) != (kind, deviations,
                                                                                             prompts_sha):
        raise RuntimeError('Prerequisite run manifest differs from receipt')
    return data


def study_from(args):
    """Ladder result (start-test receipt) plus gate E exclusions (settings receipt).
    A gate-e.json without premise_pass true stops every later phase."""
    ladder_data = json.loads((Path(args.start_receipt).resolve().parent / 'ladder.json').read_text())
    if ladder_data['result'] != 'done':
        raise RuntimeError('Start tests did not settle on a ladder step')
    study = {'step': ladder_data['step'], 'feasible': ladder_data['arms'],
             'infeasible': ladder_data['infeasible'], 'excluded': [], 'arms': ladder_data['arms'],
             'terminal_gate': None}
    if args.phase != 'settings':
        gate_e = json.loads((Path(args.settings_receipt).resolve().parent / 'gate-e.json').read_text())
        if gate_e.get('premise_pass') is not True or not isinstance(gate_e.get('excluded'), list):
            raise RuntimeError('Gate E premise check did not pass; no later phase runs')
        study['excluded'] = gate_e['excluded']
        study['arms'] = [a for a in ARMS if a in study['feasible'] and a not in gate_e['excluded']]
        if len(study['arms']) < 2:
            raise RuntimeError('Gate E left fewer than two arms (terminal E); no later phase runs')
    return study


def failures_of(result):
    """Every failed start test with its class (oom / engine_death / other)."""
    return [{'arm': a['arm'], 'step': a['step'], 'failure_class': a.get('failure_class')}
            for a in result['attempts'] if not a.get('pass')]


def settings_study(args, output, result):
    """Write study.json from gate E's result; return the settings receipt's pass.

    If the premise check failed in any arm, no gate E exclusion is applied (excluded
    None), study.json flags the premise failure for review,
    and the receipt does not pass, so check_prior refuses every later phase."""
    if result.get('premise_pass') is not True:
        write_json(output / 'study.json', {**args.study, 'excluded': None, 'gate_e_premise_pass': False,
                                           'gate_e_premise_requires_review': True, 'gate_e_requires_review': True})
        return False
    passed = True
    study = {**args.study, 'excluded': result['excluded'],
             'arms': [a for a in args.study['arms'] if a not in result['excluded']]}
    if len(study['arms']) < 2:
        study['terminal_gate'] = 'E'
        passed = False
        write_json(output / 'decision-for-review.json', gates.decision(study, []))
    write_json(output / 'study.json', {**study, 'gate_e_premise_pass': True,
                                       'gate_e_requires_review': bool(result['excluded'])})
    return passed


# ---------------------------------------------------------------- main

PHASES = ('start-test', 'settings', 'transport', 'timed')


def add_arguments(parser):
    parser.add_argument('--output', type=Path, required=True, help='new, empty run directory')
    parser.add_argument('--data-dir', type=Path,
                        help='setup files, prompts, caches and locks (default $TPPB_DATA_DIR or ./tppb-data)')
    parser.add_argument('--model', type=Path,
                        help='local snapshot of the pinned model (default $TPPB_MODEL or the snapshot under $HF_HOME)')
    parser.add_argument('--lock-dir', type=Path, help='per-GPU lock files (default $TPPB_LOCK_DIR or <data-dir>/locks)')
    parser.add_argument('--lock', action='append', default=[], metavar='PATH',
                        help='an extra lock file to hold for the whole run (repeatable); {gpu} expands to 0..7, '
                             'e.g. /shared/locks/gpu-{gpu}.lock to exclude jobs that use that naming')
    parser.add_argument('--prompts', type=Path, help='prompt file (default <data-dir>/prompts.json)')
    parser.add_argument('--probe-receipt', type=Path, help='numactl probe receipt (default <data-dir>/setup/numactl-probe.json)')
    parser.add_argument('--start-receipt', type=Path)
    parser.add_argument('--settings-receipt', type=Path)
    parser.add_argument('--transport-receipt', type=Path)
    parser.add_argument('--deviation-receipt', type=Path)
    parser.add_argument('--port', type=int, default=8150)


def model_ready(path, snapshot):
    ready = json.loads(Path(path).read_text())
    if ready.get('revision') != REVISION or ready.get('repo') != MODEL:
        raise RuntimeError('Pinned model download incomplete')
    if Path(ready.get('snapshot', '')).resolve() != Path(snapshot).resolve():
        raise RuntimeError(f'MODEL_READY.json verified {ready.get("snapshot")}, not {snapshot}')
    return ready


def main(phase, args):
    """One phase. `args` from add_arguments."""
    if phase not in PHASES:
        raise ValueError(phase)
    args.phase = phase
    paths = spec.configure(args.data_dir, args.model, args.lock_dir)
    data = paths['data']
    args.prompts = args.prompts or data / 'prompts.json'
    args.probe_receipt = args.probe_receipt or data / 'setup/numactl-probe.json'
    output = output_path(args.output)
    model_ready(data / 'setup/MODEL_READY.json', paths['model'])
    code_sha256 = launch.code_files()
    args.numactl = launch.numactl_dir(args.probe_receipt)
    args.bank = load_prompts(args.prompts)
    args.deviations = launch.deviations_from_receipt(args.deviation_receipt)
    prompts_sha = digest(args.prompts)
    if args.phase != 'start-test':
        check_prior(args.start_receipt, 'start-test', args.deviations, prompts_sha, code_sha256)
    if args.phase in ('transport', 'timed'):
        check_prior(args.settings_receipt, 'settings', args.deviations, prompts_sha, code_sha256)
    if args.phase == 'timed':
        check_prior(args.transport_receipt, 'transport', args.deviations, prompts_sha, code_sha256)
    args.study = study_from(args) if args.phase != 'start-test' else None
    args.topology = telemetry.topology()
    os.sched_setaffinity(0, args.topology['runner_cpus'])
    args.locks = launch.Locks(output, getattr(args, 'lock', None) or ())
    try:
        write_json(output / 'run-manifest.json', {'code_sha256': code_sha256,
                   'locks': [str(p) for p in args.locks.paths], 'paths': {k: str(v) for k, v in paths.items()},
                   'python': sys.executable, 'topology': {
                   k: v for k, v in args.topology.items() if k != 'lscpu'}, 'numactl': args.numactl,
                   'prompts_sha256': prompts_sha, 'phase': args.phase, 'deviations': args.deviations,
                   'study': args.study, 'start_wall_ns': time.time_ns()})
        passed = True
        if args.phase == 'start-test':
            result = ladder(args, output)
            passed = result['result'] == 'done'
            if not passed:
                study = {'step': None, 'arms': result['arms'], 'infeasible': result['infeasible'],
                         'excluded': [], 'terminal_gate': 'S'}
                write_json(output / 'decision-for-review.json', {**gates.decision(study, []),
                           'ladder_result': result['result'], 'reason': result.get('reason'),
                           'start_test_failures': failures_of(result)})
            result['start_test_failures'] = failures_of(result)
            write_json(output / 'ladder.json', result)
        elif args.phase == 'settings':
            result = settings(args, output)
            passed = settings_study(args, output, result)
        elif args.phase == 'transport':
            results = transport(args, output)
            passed = all(r['pass'] for r in results.values())
            write_json(output / 'study.json', args.study)
            if not passed:
                write_json(output / 'decision-for-review.json',
                           gates.decision({**args.study, 'terminal_gate': 'P'}, []))
        else:
            rounds, terminal, failure = timed(args, output)
            study = {**args.study, 'terminal_gate': terminal}
            if failure:
                study['gate_m_failure'] = failure
            write_json(output / 'accepted-rounds.json', rounds)
            write_json(output / 'decision-for-review.json', gates.decision(study, rounds))
            passed = terminal is None
        evidence = {str(p): digest(p) for p in output.rglob('*')
                    if p.is_file() and p.suffix in ('.json', '.log', '.txt', '.jsonl')}
        write_json(output / 'receipt.json', {'phase': args.phase, 'pass': passed,
                    'prompts_sha256': prompts_sha, 'deviations': args.deviations,
                    'evidence_sha256': evidence, 'finish_wall_ns': time.time_ns()})
        return passed
    except BaseException as exc:
        if isinstance(exc, HostMemoryStop):
            write_json(output / 'STOPPED-host-memory.json',
                       {'note': HOST_MEMORY_NOTE, 'reason': str(exc), 'gate_result': False,
                        'start_test_failure': False, 'ladder_moved': False,
                        'action': 'Phase stopped for review before a server start; see *-host-memory/host-memory.json.'})
        write_json(output / 'INCOMPLETE.json', {'error': f'{type(exc).__name__}: {exc}',
                   'action': 'No result is usable. Inspect retained server before any further launch.'})
        raise
    finally:
        args.locks.close()
