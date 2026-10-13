"""Pure gate, ladder and decision calculations (README, "Gates" and "Decision").

Missing evidence fails closed (EvidenceError or a failing check). Every
log-based gate names the line format it reads, quoted from the pinned source:
vLLM v0.30.0 (commit ced6857a) or NCCL v2.29.7-1.
"""
import math
import re
import statistics

from .spec import (ARMS, GPUS, NUMA_NODES, OUTPUT_TOKENS, PROMPT_TOKENS, ROUNDS, SOCKET_CPUS,
                  STAGE_BOUNDARIES, STOP_TOKEN_IDS, TP_ARMS, WORKER_CPUS, cpu_list)


class EvidenceError(ValueError):
    pass


# ---------------------------------------------------------------- /metrics

def metric_samples(text, name):
    rows = {}
    for line in text.splitlines():
        match = re.fullmatch(re.escape(name) + r'(\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if match:
            key = match[1] or ''
            value = float(match[2])
            if key in rows or not math.isfinite(value):
                raise EvidenceError(f'Invalid duplicate/nonfinite metric: {name}')
            rows[key] = value
    if not rows:
        raise EvidenceError(f'Missing metric {name}')
    return rows


def delta(before, after, name):
    left, right = metric_samples(before, name), metric_samples(after, name)
    if left.keys() != right.keys() or any(right[k] < left[k] for k in left):
        raise EvidenceError(f'Counter reset or label set changed: {name}')
    return sum(right[k]-left[k] for k in left)


def cache_config(metrics):
    """The engine-wide `vllm:cache_config_info` gauge."""
    samples = metric_samples(metrics, 'vllm:cache_config_info')
    if len(samples) != 1:
        raise EvidenceError('Expected one engine-wide cache config')
    labels, value = next(iter(samples.items()))
    if value != 1:
        raise EvidenceError('Cache config info gauge must equal one')
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:[^"\\]|\\[\\"n])*)"(?:,|$)')
    body, pos, result = labels[1:-1], 0, {}
    while pos < len(body):
        match = pattern.match(body, pos)
        if not match or match[1] in result:
            raise EvidenceError('Malformed/duplicate cache config label')
        result[match[1]] = re.sub(r'\\([\\"n])', lambda m: '\n' if m[1] == 'n' else m[1], match[2])
        pos = match.end()
    if result.get('engine') != '0':
        raise EvidenceError('Expected cache configuration for engine 0')
    for name in ('num_gpu_blocks', 'block_size'):
        if not re.fullmatch(r'[1-9][0-9]*', result.get(name, '')):
            raise EvidenceError(f'Missing/invalid cache config {name}')
    return result


def capacity(metrics):
    config = cache_config(metrics)
    return int(config['num_gpu_blocks']) * int(config['block_size'])


# ---------------------------------------------------------------- vLLM log lines

# vllm/v1/worker/gpu_worker.py:546-558 (f-string, logger.info, every worker).
KV_RESERVED = re.compile(r'Initial free memory ([0-9.]+) GiB, reserved ([0-9.]+) GiB memory for KV Cache '
                         r'as specified by kv_cache_memory_bytes config')
# vllm/distributed/device_communicators/cuda_communicator.py:318-324,
# info_once(scope="global"): logged once, by global rank 0 (vllm/logger.py:102-111).
ALLREDUCE = re.compile(r"Using (\[[^\]]*\]) all-reduce backends \(in dispatch order\) for group "
                       r"'([^']*)' out of potential backends: (\[[^\]]*\])")
# vllm/utils/numa_utils.py:354-360, :363-368, :441-446, :497-503.
BIND_CPUS = re.compile(r'Binding worker subprocess \(local_rank=(\d+), gpu_index=(\d+)\) to CPUs (\S+) '
                       r'and NUMA node (\d+)')
BIND_NODE_ONLY = re.compile(r'Binding worker subprocess \(local_rank=(\d+), gpu_index=(\d+)\) to NUMA node (\d+)')
BIND_ENGINE = re.compile(r'Binding EngineCore subprocess \(local_rank=(\d+)\) to NUMA nodes (\S+)')
NUMACTL_REJECTED = re.compile(r'numactl args .* rejected; falling back to ')
# Fatal lines for the start test and gate M. Sources: torch's allocator
# message and exception class ("CUDA out of memory", torch.OutOfMemoryError),
# torch/c10 "CUDA error: ..."; vLLM multiproc_executor.py:315 ("Worker proc %s
# died unexpectedly"), :1055 ("WorkerProc hit an exception."), engine/
# exceptions.py:16 (EngineDeadError message), engine/utils.py:1321 ("Engine
# core initialization failed"); a fatal signal ("Segmentation fault", the
# "Segfault encountered" banner of a crashed worker).
FATAL = re.compile(r'CUDA out of memory|OutOfMemoryError|CUDA error|cudaError|died unexpectedly'
                   r'|WorkerProc hit an exception|EngineCore encountered an issue'
                   r'|Engine core initialization failed|Segmentation fault|Segfault encountered')


def format_gib(value):
    """vllm/utils/mem_utils.py:30-31."""
    return f'{round(value / 1024 ** 3, 2)}'


def kv_reserved(log, kv_bytes):
    """Every start: the pinned pool is what each of the eight workers reserved
    (logger.info in every worker: exactly one line per worker; zero fails)."""
    rows = [{'initial_free_gib': m[1], 'reserved_gib': m[2]} for m in KV_RESERVED.finditer(log)]
    expected = format_gib(kv_bytes)
    return {'pass': len(rows) == len(GPUS) and all(r['reserved_gib'] == expected for r in rows),
            'expected_reserved_gib': expected, 'lines': rows}


def allreduce_lines(log):
    rows = []
    for match in ALLREDUCE.finditer(log):
        enabled = re.findall(r"'([^']*)'", match[1])
        rows.append({'enabled': enabled, 'group': match[2],
                     'potential': re.findall(r"'([^']*)'", match[3])})
    return rows


def allreduce_check(log, arm):
    """Gate P's vLLM clause. Only the first bracket list (enabled) is read."""
    rows = allreduce_lines(log)
    tp = [r for r in rows if re.fullmatch(r'tp:\d+', r['group'])]
    forbidden = [r for r in rows if any(b in ('CUSTOM', 'SYMM_MEM', 'NCCL_SYMM_MEM') for b in r['enabled'])]
    checks = {'no_custom_or_symm_mem_enabled': not forbidden}
    if arm in TP_ARMS:
        checks['tp_group_line_present'] = bool(tp)
        checks['tp_group_enabled_is_pynccl_only'] = bool(tp) and all(r['enabled'] == ['PYNCCL'] for r in tp)
    return {'pass': all(checks.values()), 'checks': checks, 'lines': rows, 'tp_lines': tp,
            'forbidden': forbidden}


def binding(log):
    """Gate N, log part: eight worker lines with the declared CPU list and node,
    and no "numactl args ... rejected" warning."""
    lines = [{'local_rank': int(m[1]), 'gpu_index': int(m[2]), 'cpus': m[3], 'node': int(m[4])}
             for m in BIND_CPUS.finditer(log)]
    node_only = [m[0] for m in BIND_NODE_ONLY.finditer(log)]
    rejected = [m[0] for m in NUMACTL_REJECTED.finditer(log)]
    engine = [{'local_rank': int(m[1]), 'nodes': m[2]} for m in BIND_ENGINE.finditer(log)]
    by_gpu = {}
    for row in lines:
        by_gpu.setdefault(row['gpu_index'], []).append(row)
    correct = (len(lines) == len(GPUS) and set(by_gpu) == set(GPUS)
               and all(len(v) == 1 for v in by_gpu.values())
               and all(by_gpu[g][0]['cpus'] == WORKER_CPUS[NUMA_NODES[g]]
                       and by_gpu[g][0]['node'] == NUMA_NODES[g] for g in GPUS))
    checks = {'eight_declared_worker_lines': correct, 'no_node_only_worker_line': not node_only,
              'no_numactl_rejected_warning': not rejected}
    return {'pass': all(checks.values()), 'checks': checks, 'worker_lines': lines,
            'node_only_lines': node_only, 'rejected_lines': rejected, 'engine_core_lines': engine}


def fatal_lines(log):
    return [line for line in log.splitlines() if FATAL.search(line)]


OOM = re.compile(r'CUDA out of memory|OutOfMemoryError')
ENGINE_DEATH = re.compile(r'died unexpectedly|WorkerProc hit an exception|EngineCore encountered an issue'
                          r'|Engine core initialization failed|Segmentation fault|Segfault encountered|CUDA error'
                          r'|cudaError')
# torch's allocator message names the free memory: "Tried to allocate X GiB. GPU N has a
# total capacity of ... of which ... is free."
OOM_DETAIL = re.compile(r'Tried to allocate .*|of which .* is free.*')


def failure_class(log, server_exited):
    """Start-test failure class: 'oom' (an out-of-memory line), 'engine_death' (a
    fatal line, or the server exited), else 'other' (timeout, failed check, ...)."""
    if OOM.search(log):
        return 'oom'
    if ENGINE_DEATH.search(log) or server_exited:
        return 'engine_death'
    return 'other'


def oom_lines(log):
    return [line for line in log.splitlines() if OOM.search(line) or OOM_DETAIL.search(line)]


# ---------------------------------------------------------------- NCCL lines (gate P)

# debug.cc:371-374 and :385-386: "<host>:<pid>:<tid> [<cudaDev>] NCCL INFO <msg>"
NCCL_INFO = re.compile(r'(\S+):(\d+):(\d+) \[(\d+)\] NCCL INFO (.*)$')
# shm.cc:130 "Channel %02d : %d[%d] -> %d[%d] via SHM/%s/%s" (no connIndex);
# p2p.cc:406-426 "Channel %02d/%01d : ... via P2P/..."; net.cc:317/321/363
# "Channel %02d/%d : %d[%d] -> %d[%d] [send|receive] via NET/...".
NCCL_CHANNEL = re.compile(r'Channel\s+(\d+)(?:/(\d+))?\s*:\s*(\d+)\[(\d+)\]\s*->\s*(\d+)\[(\d+)\]'
                          r'(?:\s*\[\w+\])?\s+via\s+(\S+)')
# transport.cc:113-114
NCCL_P2P_TYPE = re.compile(r'Check P2P Type isAllDirectP2p (\d+) directMode (\d+) isAllCudaP2p (\d+)')
# shm.cc:554/562 (MMAP branch; SHM subsystem), evidence only.
NCCL_MMAP = re.compile(r'MMAP allocated shareable host buffer (\S+) size (\d+)')
VIA_FORBIDDEN = re.compile(r'\bvia\s+(?:P2P|NET)\b')


def nccl_info_lines(log):
    """Every NCCL INFO line with the pid of the process that printed it."""
    rows = []
    for line in log.splitlines():
        match = NCCL_INFO.search(line)
        if match:
            rows.append({'host': match[1], 'pid': int(match[2]), 'tid': int(match[3]),
                         'cuda_dev': int(match[4]), 'message': match[5]})
    return rows


# NCCL_DEBUG_FILE=<dir>/nccl.%h.%p.log: debug.cc:241-283 expands %h and %p and
# opens the file line-buffered (setlinebuf), one file per process, so lines of
# different processes never interleave (unlike a shared, block-buffered stdout).
NCCL_FILE = re.compile(r'nccl\.(.+)\.(\d+)\.log')


def nccl_files(directory):
    """{pid: text} from the per-process NCCL debug files; pid from the file name."""
    from pathlib import Path
    files = {}
    for path in sorted(Path(directory).glob('nccl.*.log')):
        match = NCCL_FILE.fullmatch(path.name)
        if not match:
            raise EvidenceError(f'Unexpected NCCL debug file name {path.name}')
        files[int(match[2])] = path.read_text(errors='replace')
    return files


def transport(files, arm, worker_pids, server_log):
    """Gate P on one arm's transport check.

    files: {pid: text} of the per-process NCCL debug files (nccl_files);
    worker_pids: {gpu index: pid} of the eight workers (ready-cards.json);
    server_log: the vLLM log, for the all-reduce backend line.
    """
    if arm not in ARMS:
        raise EvidenceError('Unknown transport arm')
    workers = {int(g): int(p) for g, p in worker_pids.items()}
    if sorted(workers) != list(GPUS):
        raise EvidenceError('Need the eight worker pids')
    info, forbidden = [], []
    for pid, text in sorted(files.items()):
        for row in nccl_info_lines(text):
            row['file_pid'] = pid
            if row['pid'] != pid:
                raise EvidenceError(f'NCCL line pid {row["pid"]} in the file of pid {pid}')
            info.append(row)
        forbidden += [line for line in text.splitlines() if VIA_FORBIDDEN.search(line)]
    forbidden += [line for line in server_log.splitlines() if VIA_FORBIDDEN.search(line)]
    channels = []
    for row in info:
        message = row['message']
        if re.match(r'Channel\s+\d+(?:/\d+)?\s*:', message) and re.search(r'\bvia\b', message):
            match = NCCL_CHANNEL.search(message)
            if not match:
                raise EvidenceError(f'Unparsed NCCL channel evidence: {message}')
            channels.append({'pid': row['pid'], 'channel': int(match[1]),
                             'connection_index': int(match[2]) if match[2] else None,
                             'source_rank': int(match[3]), 'source_nvml': int(match[4]),
                             'target_rank': int(match[5]), 'target_nvml': int(match[6]),
                             'transport': match[7], 'line': message})
    shm = [c for c in channels if c['transport'].startswith('SHM/')]
    missing_files = sorted(g for g, pid in workers.items() if pid not in files)
    checks = {'nccl_file_from_every_worker': not missing_files,
              'channel_lines_present': bool(channels),
              'every_channel_shm': bool(channels) and len(shm) == len(channels),
              'no_via_p2p_or_net': not forbidden}
    pairs = {(c['source_nvml'], c['target_nvml']) for c in shm}
    boundaries = {f'{a}->{b}': (a, b) in pairs for a, b in STAGE_BOUNDARIES[arm]}
    checks['every_stage_boundary_has_shm_line'] = all(boundaries.values())
    shm_by_pid = {pid: sum(c['pid'] == pid for c in shm) for pid in workers.values()}
    if arm in TP_ARMS:
        checks['shm_line_from_every_tp_rank'] = all(n > 0 for n in shm_by_pid.values())
    p2p_type = []
    for row in info:
        match = NCCL_P2P_TYPE.search(row['message'])
        if match:
            p2p_type.append({'pid': row['pid'], 'tid': row['tid'], 'isAllDirectP2p': int(match[1]),
                             'directMode': int(match[2]), 'isAllCudaP2p': int(match[3])})
    zero_by_pid = {pid: sum(r['pid'] == pid and r['isAllDirectP2p'] == 0 and r['isAllCudaP2p'] == 0
                            for r in p2p_type) for pid in workers.values()}
    checks['check_p2p_type_zero_from_every_worker'] = all(n > 0 for n in zero_by_pid.values())
    vllm = allreduce_check(server_log, arm)
    checks.update({f'allreduce_{k}': v for k, v in vllm['checks'].items()})
    mmap = [{'pid': r['pid'], 'path': m[1], 'size': int(m[2])}
            for r in info for m in [NCCL_MMAP.search(r['message'])] if m]
    passed = all(checks.values())
    return {'pass': passed, 'arm': arm, 'checks': checks, 'stop_for_review': not passed,
            'workers_without_nccl_file': missing_files, 'nccl_file_pids': sorted(files),
            'stage_boundaries': boundaries, 'shm_lines_by_worker_pid': shm_by_pid,
            'zero_check_p2p_type_by_worker_pid': zero_by_pid, 'channels': channels,
            'forbidden_lines': forbidden, 'check_p2p_type': p2p_type,
            'nonzero_check_p2p_type': [r for r in p2p_type if r['isAllDirectP2p'] or r['isAllCudaP2p']],
            'allreduce': vllm, 'mmap_host_buffers_evidence_only': mmap,
            'nccl_info_line_count': len(info)}


# ---------------------------------------------------------------- requests and levels

def request_valid(row):
    usage = row.get('usage') or {}
    return (not row.get('error') and row.get('done') is True and row.get('status') == 200
            and usage.get('prompt_tokens') == PROMPT_TOKENS and usage.get('completion_tokens') == OUTPUT_TOKENS
            and row.get('ttft_s') is not None and math.isfinite(row['ttft_s'])
            and row['ttft_s'] >= 0 and len(row.get('content_ns', [])) > 0)


def level(rows, before, after, startup_metrics, expected, needed):
    """Gates K (capacity and preemption part), T, C and G for one arm at one
    level, plus the round values (medians and output throughput)."""
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise EvidenceError('Missing or duplicate measured request records')
    kept = [r for r in rows if request_valid(r)]
    dropped = len(rows)-len(kept)
    queries = delta(before, after, 'vllm:prefix_cache_queries_total')
    hits = delta(before, after, 'vllm:prefix_cache_hits_total')
    preemptions = delta(before, after, 'vllm:num_preemptions_total')
    if queries <= 0 or hits > queries:
        raise EvidenceError('Missing positive prefix-cache query accounting')
    gap_rows = [r for r in kept if r['content_chunks'] == OUTPUT_TOKENS and r['token_gap_s'] is not None]
    for row in gap_rows:
        if not math.isfinite(row['token_gap_s']) or row['token_gap_s'] < 0:
            raise EvidenceError('Bad token gap')
    tpot = [r['mean_tpot_s'] for r in kept if r['mean_tpot_s'] is not None]
    start, end = min(r['send_ns'] for r in rows), max(r['end_ns'] for r in rows)
    if end <= start:
        raise EvidenceError('Invalid measured window')
    chunks = sum(r.get('content_chunks') == OUTPUT_TOKENS for r in rows)
    cap = capacity(startup_metrics)
    result = {'K_capacity': cap >= needed, 'K_preemption': preemptions == 0,
              'kv_capacity_tokens': cap, 'capacity_needed': needed,
              'T': dropped / expected <= .02, 'C': hits / queries < .01,
              'G': chunks / expected >= .95, 'dropped': dropped,
              'chunk_complete': chunks, 'total': expected,
              'prefix_queries': queries, 'prefix_hits': hits, 'preemptions': preemptions,
              'ttft_s': statistics.median(r['ttft_s'] for r in kept) if kept else None,
              'token_gap_s': statistics.median(r['token_gap_s'] for r in gap_rows) if gap_rows else None,
              'mean_tpot_s': statistics.median(tpot) if tpot else None,
              'output_tokens_per_second': sum(r['usage']['completion_tokens'] for r in kept) / ((end-start)/1e9),
              'window_ns': [start, end]}
    result['K'] = result['K_capacity'] and result['K_preemption']
    result['rerun_level'] = not (result['T'] and result['C'])
    return result


def round_capacity_k(arms_cells):
    """Gate K's equality clause: num_gpu_blocks x block_size identical across the
    arms' timed starts of the round. Applied in place; returns the capacities."""
    caps = {cell['kv_capacity_tokens'] for cells in arms_cells.values() for cell in cells.values()}
    equal = len(caps) == 1
    for cells in arms_cells.values():
        for cell in cells.values():
            cell['K_equal_across_arms'] = equal
            cell['K'] = cell['K_capacity'] and cell['K_preemption'] and equal
    return sorted(caps)


# ---------------------------------------------------------------- gate N

def placement(rows, workers, binding_result):
    """Gate N for one level of one timed start."""
    if not rows or not binding_result:
        raise EvidenceError('Missing process placement evidence')
    per_worker, cpus_ok = {}, {}
    for gpu, pid in workers.items():
        row = rows.get(str(pid)) or {}
        gpu = int(gpu)
        cpus_ok[str(gpu)] = (not row.get('error') and row.get('gpu') == gpu and bool(row.get('Cpus_allowed_list'))
                             and cpu_list(row['Cpus_allowed_list']) == cpu_list(WORKER_CPUS[NUMA_NODES[gpu]]))
        per_worker[str(gpu)] = (cpus_ok[str(gpu)] and row.get('node') == NUMA_NODES[gpu]
                                and row.get('node_fraction_lower_bound') is not None
                                and row['node_fraction_lower_bound'] >= .95)
    complete = len(per_worker) == len(GPUS)
    passed = complete and all(per_worker.values()) and binding_result['pass']
    return {'pass': passed, 'workers': per_worker, 'binding': binding_result['checks'],
            'binding_checks_pass': complete and binding_result['pass'] and all(cpus_ok.values()),
            'worker_cpus_allowed': cpus_ok, 'nccl_shm_pages_by_node_record_only': shm_pages(rows)}


def shm_pages(rows):
    nodes = {}
    for row in rows.values():
        for node, pages in (row.get('nccl_shm_pages') or {}).items():
            nodes[node] = nodes.get(node, 0) + pages
    return nodes


# ---------------------------------------------------------------- gate Q

def _socket(cpus):
    cpus = set(cpus)
    for socket in (0, 1):
        if cpus and cpus <= set(cpu_list(SOCKET_CPUS[socket])):
            return socket
    return None


def _process_delta(first, last):
    total = last['ticks'] + last['child_ticks']
    if first is None:
        return total     # born after the first sample: all its time is inside
    if first['start'] != last['start']:
        return None
    return total - first['ticks'] - first['child_ticks']


def gap_record(window):
    """Intervals of the level window (its two edge intervals included) and those over
    2.5 s. A record, not a gate-Q condition (gate Q has no gap condition)."""
    steps = [(a['monotonic_ns'], b['monotonic_ns']) for a, b in zip(window, window[1:])]
    over = [{'from_monotonic_ns': a, 'to_monotonic_ns': b, 'interval_s': (b - a) / 1e9}
            for a, b in steps if (b - a) / 1e9 > GAP_RECORD_S]
    return {'gate': False, 'threshold_s': GAP_RECORD_S,
            'max_interval_s': max(((b - a) / 1e9 for a, b in steps), default=None),
            'intervals_over_threshold': len(over), 'over_threshold': over}


def level_scale(window, begin, end):
    """When either edge interval of the level's window exceeds 2.5 s, the time outside the
    level would dilute "at most 2% over the level"; the fraction is then also computed
    with the denominator scaled to the level's own duration (every busy and foreign
    tick kept) and gate Q uses the larger of the two. Returns the scale (<= 1) or None."""
    first = (window[1]['monotonic_ns'] - window[0]['monotonic_ns']) / 1e9
    last = (window[-1]['monotonic_ns'] - window[-2]['monotonic_ns']) / 1e9
    if max(first, last) <= GAP_RECORD_S:
        return None
    return (end - begin) / (window[-1]['monotonic_ns'] - window[0]['monotonic_ns'])


SAMPLE_ERROR = 'telemetry sample error'
GAP_RECORD_S = 2.5


def quiet(samples, begin, end):
    """Gate Q. An errored telemetry sample does not abort the study. It fails
    gate Q for this level with the reason 'telemetry sample error' (error texts
    recorded), so gate Q's rerun rule applies. Gate Q's conditions are
    still computed on the remaining samples where they can be; when they cannot,
    the computability problem is recorded beside the reason. Without an errored
    sample nothing changes: missing evidence still raises EvidenceError."""
    errors = [str(s['error']) for s in samples if s.get('error')]
    if not errors:
        return _quiet_samples(samples, begin, end)
    clean = [s for s in samples if not s.get('error')]
    try:
        result = _quiet_samples(clean, begin, end)
    except EvidenceError as exc:
        result = {'not_computable': str(exc)}
    result.update({'pass': False, 'reasons': [SAMPLE_ERROR] + result.get('reasons', []),
                   'telemetry_sample_errors': errors})
    return result


def _quiet_samples(samples, begin, end):
    """Gate Q for one level: no foreign compute process on the eight cards;
    foreign CPU at most 2% of each socket's logical CPUs; each card's clock not
    below 90% of its level median for more than 10% of samples.

    Foreign CPU of socket s = busy ticks of s's CPUs minus the CPU time of the
    runner's process tree (server, client, runner, monitor) spent on s, over
    the samples enclosing the level. A process confined to one socket counts
    there in full; a process whose affinity spans both sockets (the engine core)
    is attributed per thread and per one-second interval to the socket of the
    CPU the thread last ran on (/proc/<pid>/task/<tid>/stat field 39). Ticks
    that cannot be attributed (a thread or process that ended unreaped inside
    the window, reaped children of a spanning process) count as foreign. The
    bound without any spanning-process credit is recorded beside it.
    """
    if len(samples) < 2:
        raise EvidenceError('Missing telemetry samples')
    if samples[0]['monotonic_ns'] > begin or samples[-1]['monotonic_ns'] < end:
        raise EvidenceError('Telemetry does not enclose measured window')
    left = max(i for i, s in enumerate(samples) if s['monotonic_ns'] <= begin)
    right = min(i for i, s in enumerate(samples) if s['monotonic_ns'] >= end)
    window = samples[left:right+1]
    # Gaps in the window (edge intervals included) are recorded, not a gate-Q reason; an
    # edge gap is handled inside the CPU condition (level_scale). An interval over 2.5 s
    # gets no engine-core (spanning) CPU credit.
    steps = [(b['monotonic_ns']-a['monotonic_ns']) / 1e9 for a, b in zip(window, window[1:])]
    uncredited = [i for i, step in enumerate(steps) if step > GAP_RECORD_S]
    gaps = gap_record(window)
    relevant = [s for s in window if begin <= s['monotonic_ns'] <= end]
    if not relevant:
        raise EvidenceError('No samples within measured window')
    foreign_gpu = []
    for s in window:
        if sorted(g['index'] for g in s['gpus']) != list(GPUS):
            raise EvidenceError('Telemetry missing GPU')
        foreign_gpu.extend(p for p in s['compute_apps'] if str(p['pid']) not in s['owned'])
    a, b = window[0], window[-1]
    confined, spanning_pids = {0: 0, 1: 0}, []
    for pid, last in b['owned'].items():
        socket = _socket(cpu_list(last['affinity']))
        if socket is None:
            spanning_pids.append(pid)
            continue
        change = _process_delta(a['owned'].get(pid), last)
        if change is not None and change > 0:
            confined[socket] += change
    spanning = {0: 0, 1: 0}
    for index, (x, y) in enumerate(zip(window, window[1:])):
        if index in uncredited:
            continue        # no spanning credit for an interval over 2.5 s (no-credit rule)
        for pid in spanning_pids:
            first, last = x['owned'].get(pid), y['owned'].get(pid)
            if last is None or 'threads' not in last:
                continue
            if first is not None and first['start'] != last['start']:
                continue
            for tid, thread in last['threads'].items():
                before = (first or {}).get('threads', {}).get(tid)
                if before is not None and before['start'] != thread['start']:
                    before = None
                change = thread['ticks'] - (before['ticks'] if before else 0)
                if change > 0:
                    spanning[_socket([thread['processor']])] += change
    sockets, bounds, unscaled, scaled = {}, {}, {}, {}
    scale = level_scale(window, begin, end)
    for socket in (0, 1):
        cpus = cpu_list(SOCKET_CPUS[socket])
        busy = sum(b['cpu'][str(c)]['busy']-a['cpu'][str(c)]['busy'] for c in cpus)
        total = sum(b['cpu'][str(c)]['total']-a['cpu'][str(c)]['total'] for c in cpus)
        if total <= 0 or busy < 0:
            raise EvidenceError('Bad CPU counters')
        foreign = max(0, busy - confined[socket] - spanning[socket])
        unscaled[str(socket)] = foreign / total
        # An edge interval over 2.5 s -> also the fraction
        # over the level's own duration; gate on the larger.
        scaled[str(socket)] = foreign / (total * scale) if scale else None
        sockets[str(socket)] = max(unscaled[str(socket)], scaled[str(socket)] or 0)
        bounds[str(socket)] = max(0, busy - confined[socket]) / total
    clocks = {}
    for gpu in GPUS:
        values = [next(g['clock_mhz'] for g in s['gpus'] if g['index'] == gpu) for s in relevant]
        med = statistics.median(values)
        if med <= 0:
            raise EvidenceError('Invalid GPU clock')
        clocks[str(gpu)] = sum(v < .9*med for v in values)/len(values)
    reasons = []
    passed = (not foreign_gpu and all(v <= .02 for v in sockets.values())
              and all(v <= .10 for v in clocks.values()))
    return {'pass': passed, 'reasons': reasons, 'foreign_gpu_processes': foreign_gpu, 'foreign_cpu_fraction': sockets,
            'foreign_cpu_fraction_unscaled': unscaled, 'foreign_cpu_fraction_level_scaled': scaled,
            'level_scale': scale,
            'foreign_cpu_fraction_without_spanning_credit': bounds, 'spanning_pids': spanning_pids,
            'low_clock_fraction': clocks, 'telemetry_gaps': gaps}


# ---------------------------------------------------------------- gate E

TOKEN_ID = re.compile(r'token_id:(\d+)')


def _token(text):
    match = TOKEN_ID.fullmatch(str(text))
    if not match:
        raise EvidenceError(f'Token is not a token_id placeholder: {text!r}')
    return int(match[1])


def gate_e_tokens(response):
    """[(token id, logprob, {top id: logprob} or None)] for one response, or
    None if the response carries no logprobs at all (gate E counts that as
    corruption against every other arm)."""
    try:
        choice = response['choices'][0]
        usage = response['usage']
        logprobs = choice.get('logprobs')
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise EvidenceError(f'Gate E response lacks choices/usage: {exc}') from exc
    if len(response['choices']) != 1 or usage.get('completion_tokens') != OUTPUT_TOKENS:
        raise EvidenceError('Gate E response is not one choice of 128 tokens')
    if not logprobs or not logprobs.get('content'):
        return None
    rows = []
    for entry in logprobs['content']:
        if 'token' not in entry:
            raise EvidenceError('Gate E logprob entry lacks its token field')
        top = entry.get('top_logprobs')
        try:
            alternatives = {_token(t['token']): t['logprob'] for t in top} if top else None
        except (KeyError, TypeError) as exc:
            raise EvidenceError(f'Gate E top_logprobs entry malformed: {exc}') from exc
        rows.append((_token(entry['token']), entry.get('logprob'), alternatives))
    if len(rows) != OUTPUT_TOKENS:
        raise EvidenceError('Gate E logprobs do not cover 128 tokens')
    return rows


def compare_outputs(x, y, bound=1.0):
    """Gate E, one prompt, arms X and Y (token rows from gate_e_tokens)."""
    if x is None or y is None:
        return {'status': 'corruption', 'reason': 'response without logprobs'}
    for position, ((a, la, top_x), (b, lb, top_y)) in enumerate(zip(x, y)):
        if a == b:
            continue
        result = {'position': position, 'x_token': a, 'y_token': b}
        if top_x is None or top_y is None or la is None or lb is None:
            return {**result, 'status': 'corruption', 'reason': 'top_logprobs missing at first difference'}
        if b not in top_x or a not in top_y:
            return {**result, 'status': 'corruption', 'reason': 'chosen token not in the other arm\'s top 5'}
        x_margin = la - top_x[b]
        y_margin = lb - top_y[a]
        result.update(x_margin_nat=x_margin, y_margin_nat=y_margin)
        if x_margin < bound and y_margin < bound:
            return {**result, 'status': 'benign'}
        return {**result, 'status': 'corruption', 'reason': f'logprob margin not below {bound} nat'}
    return {'status': 'identical'}


def premise(tokens):
    """Gate E premise check for one arm (not a gate). tokens: {prompt id:
    rows from gate_e_tokens, or None for a response without logprobs}.

    At every position the emitted token's logprob must equal exactly (no
    tolerance) the largest top_logprobs value among tokens that are not stop
    ids (STOP_TOKEN_IDS); ties are allowed. A position whose largest top value
    belongs only to stop ids (strictly above every non-stop value) is counted
    as stop-masked and checked against the largest non-stop value. A position
    whose two or more top_logprobs tokens share that largest non-stop value is
    counted as tied. A position lacking logprob or top_logprobs fails; a
    response without logprobs fails as one record with position None."""
    record = {'positions_checked': 0, 'failures': [], 'stop_masked': 0, 'tied_top': 0}
    for prompt_id, rows in tokens.items():
        if rows is None:
            record['failures'].append({'prompt': str(prompt_id), 'position': None, 'emitted_token': None,
                                       'emitted_logprob': None, 'max_non_stop_logprob': None,
                                       'reason': 'response without logprobs'})
            continue
        for position, (token, logprob, top) in enumerate(rows):
            record['positions_checked'] += 1
            failure = {'prompt': str(prompt_id), 'position': position, 'emitted_token': token,
                       'emitted_logprob': logprob, 'max_non_stop_logprob': None}
            if logprob is None or not top:
                record['failures'].append({**failure, 'reason': 'logprob or top_logprobs missing'})
                continue
            non_stop = [v for t, v in top.items() if t not in STOP_TOKEN_IDS]
            if not non_stop:
                record['failures'].append({**failure, 'reason': 'no non-stop token in top_logprobs'})
                continue
            best = max(non_stop)
            failure['max_non_stop_logprob'] = best
            if any(v > best for t, v in top.items() if t in STOP_TOKEN_IDS):
                record['stop_masked'] += 1
            if sum(v == best for v in non_stop) > 1:
                record['tied_top'] += 1
            if logprob != best:
                record['failures'].append({**failure, 'reason': 'emitted logprob is not the largest non-stop value'})
    return record


def gate_e(responses):
    """responses: {arm: {prompt id: response JSON}} for the arms that reached
    their configuration start. Returns the premise records, the
    per-pair results and the exclusions.

    If the premise fails in any arm: premise_pass False, excluded None (no
    gate E exclusion is applied), corrupted_pairs None and pairs None; the pair
    comparisons are kept under 'pairs_information_only' and are not verdicts."""
    arms = [a for a in ARMS if a in responses]
    prompt_ids = list(responses[arms[0]]) if arms else []
    if arms and (any(list(responses[a]) != prompt_ids for a in arms) or not prompt_ids):
        raise EvidenceError('Gate E arms answered different prompt lists')
    tokens = {a: {p: gate_e_tokens(responses[a][p]) for p in prompt_ids} for a in arms}
    premises = {a: premise(tokens[a]) for a in arms}
    premise_pass = all(not r['failures'] for r in premises.values())
    pairs, corrupted = {}, []
    for i, x in enumerate(arms):
        for y in arms[i+1:]:
            per_prompt = {str(p): compare_outputs(tokens[x][p], tokens[y][p]) for p in prompt_ids}
            bad = [p for p, r in per_prompt.items() if r['status'] == 'corruption']
            pairs[f'{x}|{y}'] = {'corruption': bool(bad), 'corrupted_prompts': bad, 'prompts': per_prompt}
            if bad:
                corrupted.append((x, y))
    head = {'arms': arms, 'stop_token_ids': list(STOP_TOKEN_IDS), 'premise': premises,
            'premise_pass': premise_pass}
    if not premise_pass:
        return {**head, 'excluded': None, 'corrupted_pairs': None, 'pairs': None,
                'pairs_information_only': pairs,
                'note': 'Premise check failed: no gate E exclusion is applied; the pair comparisons '
                        'are recorded for information only and are not verdicts; the failures need review.'}
    return {**head, 'pairs': pairs, 'corrupted_pairs': [list(p) for p in corrupted],
            'excluded': exclusions(arms, corrupted) if len(arms) >= 2 else []}


def exclusions(arms, corrupted):
    """An arm in a corrupted pair with every other non-excluded arm is excluded;
    a corrupted pair that cannot be attributed so excludes both its arms."""
    bad = {frozenset(p) for p in corrupted}
    excluded = {x for x in arms if all(frozenset((x, y)) in bad for y in arms if y != x)}
    for pair in bad:
        if not pair & excluded:
            excluded |= pair
    return [a for a in ARMS if a in excluded]


# ---------------------------------------------------------------- ladder (gate S)

def ladder_step(active, step, results, history):
    """KV-pool ladder. `results`: {arm: passed} at `step` for every
    active arm; `history`: {arm: [failed steps]} of the current pass.

    Returns ('done', step) when every active arm passed; ('next', step + 1);
    ('restart', infeasible arms) after step 3 when some arm failed at every
    step (it is dropped; the rest are re-tested from step 1);
    ('not-measurable', infeasible) when fewer than two arms would remain;
    ('stop', reason) when step 3 failed but no failing arm failed at every step
    (the ladder rule does not decide this case; stop for review).
    """
    if set(results) != set(active):
        raise EvidenceError('Every active arm needs a start-test result at this step')
    for arm, ok in results.items():
        if not ok:
            history.setdefault(arm, []).append(step)
    if all(results.values()):
        return ('done', step)
    if step < 3:
        return ('next', step + 1)
    infeasible = [a for a in active if history.get(a) == [1, 2, 3]]
    if not infeasible:
        return ('stop', 'Ladder undefined: an arm failed at step 3 but not at every step')
    remaining = [a for a in active if a not in infeasible]
    if len(remaining) < 2:
        return ('not-measurable', infeasible)
    return ('restart', infeasible)


# ---------------------------------------------------------------- decision

def pair_verdict(ratios, x, y):
    """Pairwise verdict over five rounds; r < 1 means X is better."""
    if len(ratios) != ROUNDS or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise EvidenceError('Five positive finite paired ratios required')
    if all(r < .9 for r in ratios):
        return f'{x} better'
    if all(r > 1.1 for r in ratios):
        return f'{y} better'
    if all(.9 <= r <= 1.1 for r in ratios):
        return 'no material difference'
    return 'unresolved'


METRICS = ('ttft_s', 'token_gap_s', 'output_tokens_per_second')


def deciding_levels(metric, step_levels):
    """TTFT and token gap at every level; throughput at every level but c = 1
    (step 1: c = 8 and 32; step 2: c = 8 and 16; step 3: c = 8; at c = 1 throughput
    is a function of the other two metrics)."""
    return [c for c in step_levels if metric != 'output_tokens_per_second' or c != 1]


def level_validity(rounds, arms, step_levels):
    """valid[arm][c]: K (all arms of the round), T and C hold in all five rounds."""
    valid = {a: {} for a in arms}
    gap_ok = {}
    for c in step_levels:
        key = str(c)
        k_all = all(all(r['levels'][key][a]['K'] for a in arms) for r in rounds)
        for a in arms:
            valid[a][c] = k_all and all(r['levels'][key][a]['T'] and r['levels'][key][a]['C'] for r in rounds)
        gap_ok[c] = all(all(r['levels'][key][a]['G'] for a in arms) for r in rounds)
    return valid, gap_ok


def pair_cells(rounds, x, y, valid, gap_ok, step_levels):
    verdicts, ratios = {}, {}
    for metric in METRICS:
        for c in deciding_levels(metric, step_levels):
            if not (valid[x][c] and valid[y][c]):
                continue
            if metric == 'token_gap_s' and not gap_ok[c]:
                continue
            values = []
            for r in rounds:
                vx, vy = r['levels'][str(c)][x][metric], r['levels'][str(c)][y][metric]
                if vx is None or vy is None or vx <= 0 or vy <= 0:
                    raise EvidenceError('Missing valid per-round value')
                values.append(vy / vx if metric == 'output_tokens_per_second' else vx / vy)
            key = f'{metric}:c{c}'
            ratios[key] = values
            verdicts[key] = pair_verdict(values, x, y)
    return verdicts, ratios


def table_a(study, verdicts, valid_pair_levels):
    """Rows A0-A6, checked top to bottom; the first that applies is taken."""
    arms = study['arms']
    if study.get('terminal_gate') in ('S', 'P', 'N', 'Q'):
        return 'A0'
    if 'TP8' not in arms or 'PP8' not in arms:
        return 'A1'
    if valid_pair_levels < 2:
        return 'A0'
    outcomes = set(verdicts.values())
    if {'TP8 better', 'PP8 better'} <= outcomes:
        return 'A2'
    if 'unresolved' in outcomes:
        return 'A3'
    if outcomes == {'no material difference'}:
        return 'A4'
    if 'TP8 better' in outcomes:
        return 'A5'
    return 'A6'


def table_a_clause(row, verdicts):
    throughput = [v for k, v in verdicts.items() if k.startswith('output_tokens_per_second:')]
    if row == 'A2':
        return {'throughput_clause_contradicted': 'TP8 better' in throughput}
    if row == 'A5':
        return {'docs_correction': 'TP8 better' in throughput}
    if row == 'A6':
        return {'confirms_docs': 'PP8 better' in throughput}
    return {}


def cell_winners(arms, pair_verdicts, step_levels):
    winners = {}
    for metric in METRICS:
        for c in deciding_levels(metric, step_levels):
            key = f'{metric}:c{c}'
            winner = None
            for x in arms:
                beats = []
                for y in arms:
                    if y == x:
                        continue
                    pair = f'{x}|{y}' if f'{x}|{y}' in pair_verdicts else f'{y}|{x}'
                    beats.append(pair_verdicts.get(pair, {}).get(key) == f'{x} better')
                if beats and all(beats):
                    winner = x
            winners[key] = winner
    return winners


def table_b(study, winners, valid_all_levels):
    if study.get('terminal_gate') in ('S', 'P', 'N', 'Q') or len(study['arms']) < 3 or valid_all_levels < 2:
        return 'B0'
    found = set(w for w in winners.values() if w)
    if 'TP4xPP2' in found and not found & {'TP8', 'PP8'}:
        return 'B1'
    if 'TP4xPP2' in found:
        return 'B2'
    if found:
        return 'B3'
    return 'B4'


# The expectation written before the first run (TP8 against PP8; README "Decision").
EXPECTED = {'ttft_s': 'PP8 better', 'token_gap_s': 'TP8 better', 'output_tokens_per_second': 'PP8 better'}


def expectation_check(row_a, row_b, a_verdicts, pair_verdicts, step_levels):
    """Expectation: A2 and B3; TP8 better on gap, PP8 better on TTFT
    and throughput in every deciding TP8-PP8 cell; token gap ordered
    TP8 < TP4xPP2 < PP8 at every level. A missing expected cell is against it."""
    problems = []
    if row_a != 'A2':
        problems.append(f'table A row {row_a}')
    if row_b != 'B3':
        problems.append(f'table B row {row_b}')
    for metric, wanted in EXPECTED.items():
        for c in deciding_levels(metric, step_levels):
            got = a_verdicts.get(f'{metric}:c{c}')
            if got != wanted:
                problems.append(f'TP8|PP8 {metric}:c{c} is {got}, expected {wanted}')
    for c in step_levels:
        for pair, wanted in (('TP8|TP4xPP2', 'TP8 better'), ('TP4xPP2|PP8', 'TP4xPP2 better')):
            got = pair_verdicts.get(pair, {}).get(f'token_gap_s:c{c}')
            if got != wanted:
                problems.append(f'gap order {pair} c{c} is {got}, expected {wanted}')
    return bool(problems), problems


def decision(study, rounds):
    """Mechanical tables A and B; a calculation to be read, not a claim.

    study: {'step', 'arms' (feasible, non-excluded, ordered as ARMS),
            'infeasible', 'excluded', 'terminal_gate'}.
    """
    from .spec import levels as step_levels_of
    terminal = study.get('terminal_gate')
    base = {'study': study, 'status': 'calculation'}
    if terminal == 'M':
        return {**base, 'table_a': None, 'table_b': None, 'requires_review': True,
                'reason': 'Gate M second failure: arm INFEASIBLE at this step; review the ladder first'}
    arms = [a for a in ARMS if a in study['arms']]
    step_levels = step_levels_of(study['step']) if study.get('step') else ()
    if terminal is None and len(rounds) != ROUNDS:
        raise EvidenceError('Need exactly five accepted rounds')
    pair_verdicts, pair_ratios = {}, {}
    valid_pair_levels = valid_all_levels = 0
    if terminal is None and len(arms) >= 2:
        valid, gap_ok = level_validity(rounds, arms, step_levels)
        for i, x in enumerate(arms):
            for y in arms[i+1:]:
                v, r = pair_cells(rounds, x, y, valid, gap_ok, step_levels)
                pair_verdicts[f'{x}|{y}'], pair_ratios[f'{x}|{y}'] = v, r
        if 'TP8' in arms and 'PP8' in arms:
            valid_pair_levels = sum(valid['TP8'][c] and valid['PP8'][c] for c in step_levels)
        valid_all_levels = sum(all(valid[a][c] for a in arms) for c in step_levels)
    a_verdicts = pair_verdicts.get('TP8|PP8', {})
    row_a = table_a(study, a_verdicts, valid_pair_levels)
    winners = cell_winners(arms, pair_verdicts, step_levels) if terminal is None else {}
    row_b = table_b(study, winners, valid_all_levels)
    against, expectation = expectation_check(row_a, row_b, a_verdicts, pair_verdicts, step_levels)
    requires_review = (row_a in ('A0', 'A1', 'A3', 'A4') or row_b in ('B0', 'B4') or terminal is not None
                    or bool(study.get('excluded')) or against)
    return {**base, 'table_a': {'row': row_a, **table_a_clause(row_a, a_verdicts), 'verdicts': a_verdicts},
            'table_b': {'row': row_b, 'cell_winners': winners},
            'pair_verdicts': pair_verdicts, 'pair_ratios': pair_ratios,
            'valid_levels_tp8_pp8': valid_pair_levels, 'valid_levels_all_arms': valid_all_levels,
            'against_expectation': against, 'expectation_check': expectation, 'requires_review': requires_review}


# ---------------------------------------------------------------- retries (gates N, Q)

class StudyRetries:
    """One N replacement round in the whole study."""
    def __init__(self):
        self.replacement_used = False


class RoundRetries:
    """At most two N/Q reruns of a round; N then may use the study's one replacement."""
    def __init__(self, study=None):
        self.study = study if study is not None else StudyRetries()
        self.n_retries = self.q_retries = 0
        self.replaced = False
        self.terminal_gate = None

    def failed(self, placement_ok, quiet_ok):
        if not quiet_ok:
            if self.q_retries == 2:
                self.terminal_gate = 'Q'
                return 'NOT-MEASURABLE'
            self.q_retries += 1
        if not placement_ok:
            if self.replaced:
                self.terminal_gate = 'N'
                return 'NOT-MEASURABLE'
            if self.n_retries == 2:
                if self.study.replacement_used:
                    self.terminal_gate = 'N'
                    return 'NOT-MEASURABLE'
                self.study.replacement_used = True
                self.replaced = True
                return 'replace_round'
            self.n_retries += 1
        return 'rerun_round'

    def receipt(self):
        return {'n_retries': self.n_retries, 'q_retries': self.q_retries,
                'replaced': self.replaced, 'terminal_gate': self.terminal_gate,
                'study_replacement_used': self.study.replacement_used}
