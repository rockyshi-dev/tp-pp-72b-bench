"""Synthetic evidence in the pinned line formats, for CPU-only tests.

NCCL v2.29.7-1: debug.cc:371-374/385-386 prefix "<host>:<pid>:<tid> [<dev>] NCCL INFO ";
transport.cc:113 "Check P2P Type ..."; shm.cc:130 "Channel %02d : %d[%d] -> %d[%d]
via SHM/%s/%s"; shm.cc:554 "MMAP allocated shareable host buffer ...";
p2p.cc:417 "Channel %02d/%01d : ... via P2P/CUMEM..."; net.cc:317 "... [send] via NET/...".
vLLM v0.30.0: cuda_communicator.py:318-324; numa_utils.py:354-360, 441-446, 497-503;
gpu_worker.py:546-558.
"""
from tppbench.spec import GPUS, NUMA_NODES, SOCKET_CPUS, WORKER_CPUS, cpu_list

WORKERS = {g: 1000 + g for g in GPUS}


def prefix(gpu, tid=None):
    pid = WORKERS[gpu]
    return f'host1:{pid}:{tid or pid} [{gpu}] NCCL INFO '


def channel(gpu, src_rank, dst_rank, dst_gpu, via='SHM/direct/direct', number=0):
    return f'{prefix(gpu)}Channel {number:02d} : {src_rank}[{gpu}] -> {dst_rank}[{dst_gpu}] via {via}'


def vllm_line(gpu, message, where='cuda_communicator.py:318'):
    return f'(Worker_{gpu} pid={WORKERS[gpu]}) INFO 10-05 12:00:00 [{where}] {message}'


def allreduce_line(enabled="['PYNCCL']", group='tp:0',
                   potential="['CUSTOM', 'FLASHINFER', 'SYMM_MEM', 'PYNCCL']"):
    return vllm_line(0, f"Using {enabled} all-reduce backends (in dispatch order) for group '{group}' "
                        f"out of potential backends: {potential}.")


def nccl_log(arm, allreduce=True):
    out = []
    for g in GPUS:
        out += [f'{prefix(g)}Check P2P Type isAllDirectP2p 0 directMode 0 isAllCudaP2p 0',
                f'{prefix(g)}ncclCommInitRank comm 0x7f{g} rank {g} nranks 8 cudaDev {g} nvmlDev {g} '
                f'busId {g}1000 commId 0xabc - Init COMPLETE',
                f'{prefix(g, 5000 + g)}MMAP allocated shareable host buffer /dev/shm/nccl-a{g} size 9637888 '
                f'ptr 0x7f00000{g}000']
        if arm == 'TP8':
            out.append(channel(g, g, (g + 1) % 8, (g + 1) % 8))
        elif arm == 'TP4xPP2':
            base, r = (g // 4) * 4, g % 4
            out.append(channel(g, r, (r + 1) % 4, base + (r + 1) % 4))
            out.append(channel(g, 0, 1, g + 4, number=1) if g < 4 else channel(g, 1, 0, g - 4, number=1))
        else:
            out.append(channel(g, g, (g + 1) % 8, (g + 1) % 8))
    if arm in ('TP8', 'TP4xPP2') and allreduce:
        out.append(allreduce_line())
    out.append(vllm_line(0, "Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'pp:0' "
                            "out of potential backends: ['CUSTOM', 'PYNCCL']."))
    return '\n'.join(out)


def split_log(log):
    """Per-process NCCL debug files {pid: text} and the remaining vLLM log, as
    NCCL_DEBUG_FILE=<dir>/nccl.%h.%p.log produces them."""
    import re
    files, rest = {}, []
    for line in log.splitlines():
        match = re.match(r'host1:(\d+):\d+ \[\d+\] NCCL INFO ', line)
        if match:
            files.setdefault(int(match[1]), []).append(line)
        else:
            rest.append(line)
    return {pid: '\n'.join(lines) + '\n' for pid, lines in files.items()}, '\n'.join(rest)


def binding_log(cpus=None, nodes=None, extra=''):
    lines = []
    for g in GPUS:
        node = NUMA_NODES[g] if nodes is None else nodes[g]
        cpu = WORKER_CPUS[NUMA_NODES[g]] if cpus is None else cpus[g]
        lines.append(f'(EngineCore pid=900) INFO [numa_utils.py:354] Binding worker subprocess '
                     f'(local_rank={g}, gpu_index={g}) to CPUs {cpu} and NUMA node {node}')
    lines.append('(APIServer pid=800) INFO [numa_utils.py:441] Binding EngineCore subprocess (local_rank=0) '
                 'to NUMA nodes 0,1')
    return '\n'.join(lines) + ('\n' + extra if extra else '')


def kv_line(reserved='2.25', free='23.05'):
    return (f'(Worker_0 pid=1000) INFO [gpu_worker.py:558] Initial free memory {free} GiB, reserved {reserved} '
            'GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. '
            'This does not respect the gpu_memory_utilization config.')


def placement_rows(fraction=.99, cpus=None):
    rows = {}
    for g in GPUS:
        rows[str(WORKERS[g])] = {'gpu': g, 'role': 'worker', 'node': NUMA_NODES[g],
                                 'node_fraction_lower_bound': fraction,
                                 'Cpus_allowed_list': WORKER_CPUS[NUMA_NODES[g]] if cpus is None else cpus[g],
                                 'Mems_allowed_list': '0-1', 'nccl_shm_pages': {str(NUMA_NODES[g]): 10}}
    rows['900'] = {'gpu': None, 'role': 'other', 'node': 0, 'node_fraction_lower_bound': .5,
                   'Cpus_allowed_list': '0-103', 'Mems_allowed_list': '0-1', 'nccl_shm_pages': {}}
    return rows


def metrics(q=0, h=0, p=0):
    return '\n'.join(f'vllm:{k}_total{{model_name="m",engine="0"}} {v}' for k, v in
                     [('prefix_cache_queries', q), ('prefix_cache_hits', h), ('num_preemptions', p)])


def cache_metric(blocks=3686, size=16):
    return (f'vllm:cache_config_info{{engine="0",num_gpu_blocks="{blocks}",'
            f'block_size="{size}",gpu_memory_utilization="0.92",enable_prefix_caching="True"}} 1')


def request(i, chunks=128, pt=1536, ct=128):
    from tppbench import client
    row = {'id': i, 'status': 200, 'done': True, 'error': None,
           'usage': {'prompt_tokens': pt, 'completion_tokens': ct},
           'send_ns': i*1000, 'end_ns': i*1000+3_000_000_000,
           'content_ns': [i*1000+1_000_000_000+j*1_000_000 for j in range(chunks)]}
    return client.summarize(row)


ALL = cpu_list(SOCKET_CPUS[0]) + cpu_list(SOCKET_CPUS[1])


def sample(t, busy, owned, apps=(), clock=2500.0):
    """busy: {cpu: busy ticks so far}; every CPU's total advances 100 ticks per second."""
    return {'monotonic_ns': t * 1_000_000_000, 'wall_ns': t, 'owned': owned,
            'cpu': {str(c): {'busy': busy.get(c, 0), 'total': 100 * t} for c in ALL},
            'gpus': [{'index': g, 'clock_mhz': clock} for g in GPUS], 'compute_apps': list(apps)}


def gate_e_response(tokens, top=None, logprob=-0.1):
    """tokens: list of 128 ids; top: {position: {id: logprob}}; default top-5 is the chosen token only."""
    content = []
    for i, t in enumerate(tokens):
        alternatives = (top or {}).get(i, {t: logprob})
        entry = {'token': f'token_id:{t}', 'logprob': alternatives.get(t, logprob),
                 'top_logprobs': [{'token': f'token_id:{k}', 'logprob': v} for k, v in alternatives.items()]}
        content.append(entry)
    return {'choices': [{'index': 0, 'logprobs': {'content': content}}], 'usage': {'completion_tokens': 128}}


def cell(ttft=1., gap=1., tput=1., **flags):
    base = dict(K=True, T=True, C=True, G=True, ttft_s=ttft, token_gap_s=gap, output_tokens_per_second=tput)
    base.update(flags)
    return base


def rounds(values, step_levels=(1, 8, 32), override=None):
    """values: {arm: (ttft, gap, throughput)} or {arm: {c: (ttft, gap, throughput)}}."""
    out = []
    for k in range(5):
        levels = {}
        for c in step_levels:
            levels[str(c)] = {}
            for arm, v in values.items():
                t = v[c] if isinstance(v, dict) else v
                if callable(t):
                    t = t(k)
                levels[str(c)][arm] = cell(*t)
                if override:
                    levels[str(c)][arm].update(override(k, c, arm) or {})
        out.append({'round': k + 1, 'levels': levels})
    return out
