"""Read-only /proc and nvidia-smi evidence for eight cards and two sockets.

Never imports a GPU library. The owned
process set is the runner's whole process tree (server, client processes,
runner and its monitor), and processes whose CPU affinity spans both sockets
(the engine core, bound by vLLM to nodes 0,1) are also sampled per thread so
that gate Q can attribute their CPU time to a socket.
"""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

from .spec import (API_CPUS, CLIENT_CPUS, GPUS, NUMA_NODES, RUNNER_CPUS, SOCKET_CPUS, WORKER_CPUS,
                  cpu_list)


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20)


def format_cpus(cpus):
    cpus = sorted(cpus)
    parts, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(str(cpus[i]) if i == j else f'{cpus[i]}-{cpus[j]}')
        i = j + 1
    return ','.join(parts)


def socket_of(cpus):
    """0 or 1 if every CPU lies in that socket; None if the set spans both."""
    cpus = set(cpus)
    for socket in (0, 1):
        if cpus and cpus <= set(cpu_list(SOCKET_CPUS[socket])):
            return socket
    return None


def topology(lscpu=None, topo=None, p2p=None):
    """Verify the expected node shape before any start; return the declared lists."""
    lscpu = command(['lscpu', '-p=CPU,CORE,SOCKET,NODE']) if lscpu is None else lscpu
    rows = [tuple(map(int, line.split(','))) for line in lscpu.splitlines()
            if line and not line.startswith('#')]
    for socket in (0, 1):
        found = sorted(r[0] for r in rows if r[2] == socket and r[3] == socket)
        if found != cpu_list(SOCKET_CPUS[socket]):
            raise RuntimeError(f'Socket {socket} CPUs differ from the expected layout: {format_cpus(found)}')
    if len(rows) != 104:
        raise RuntimeError('Expected 104 logical CPUs')
    core = {cpu: (s, c) for cpu, c, s, n in rows}
    siblings = {cpu: sorted(x for x, key in core.items() if key == core[cpu] and x != cpu)
                for cpu in cpu_list(RUNNER_CPUS)}
    raw = command(['nvidia-smi', 'topo', '-m']) if topo is None else topo
    for gpu in GPUS:
        node = NUMA_NODES[gpu]
        line = next((x for x in raw.splitlines() if re.match(rf'GPU{gpu}\s', x)), '')
        if not re.search(re.escape(SOCKET_CPUS[node]) + rf'\s+{node}\b', line):
            raise RuntimeError(f'GPU{gpu} is not on NUMA node {node}: {line}')
    return {'api_cpus': cpu_list(API_CPUS), 'client_cpus': cpu_list(CLIENT_CPUS),
            'runner_cpus': cpu_list(RUNNER_CPUS),
            'worker_cpus': {str(g): cpu_list(WORKER_CPUS[NUMA_NODES[g]]) for g in GPUS},
            'socket_cpus': {str(s): cpu_list(SOCKET_CPUS[s]) for s in (0, 1)},
            # CPUs 96-103 are SMT siblings of the client's 44-51;
            # recorded, not avoided.
            'runner_smt_siblings': siblings, 'lscpu': rows, 'gpu_topology': raw,
            'p2p': command(['nvidia-smi', 'topo', '-p2p', 'r']) if p2p is None else p2p}


def _stat(text):
    rest = text[text.rindex(')') + 2:].split()
    return {'state': rest[0], 'ppid': int(rest[1]), 'session': int(rest[3]),
            'ticks': int(rest[11]) + int(rest[12]), 'child_ticks': int(rest[13]) + int(rest[14]),
            'start': int(rest[19]), 'processor': int(rest[36])}


def process_table(proc=Path('/proc')):
    result = {}
    for path in proc.glob('[0-9]*/stat'):
        try:
            result[int(path.parent.name)] = _stat(path.read_text())
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
    return result


def thread_table(pid, proc=Path('/proc')):
    rows = {}
    for path in (proc / str(pid) / 'task').glob('[0-9]*/stat'):
        try:
            row = _stat(path.read_text())
            rows[path.parent.name] = {'ticks': row['ticks'], 'start': row['start'], 'processor': row['processor']}
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            pass
    return rows


def descendants(table, pid):
    found = {pid} if pid in table else set()
    while True:
        more = {p for p, row in table.items() if row['ppid'] in found}
        if more <= found:
            return found
        found |= more


def assert_server_cards(pid, snapshot, table=None):
    """Match owned compute PIDs to physical nvidia-smi indices via UUIDs; one
    worker per card on exactly cards 0-7. Returns {gpu index: worker pid}."""
    owned = descendants(process_table() if table is None else table, pid)
    uuid_to_index = {row['uuid']: row['index'] for row in snapshot['gpus']}
    apps = snapshot['compute_apps']
    if any(row['pid'] not in owned for row in apps):
        raise RuntimeError('Foreign compute process at readiness')
    workers = {}
    for row in apps:
        index = uuid_to_index.get(row['uuid'])
        if index is None:
            raise RuntimeError(f'Unknown GPU UUID {row["uuid"]}')
        workers.setdefault(index, set()).add(row['pid'])
    if set(workers) != set(GPUS) or any(len(p) != 1 for p in workers.values()):
        raise RuntimeError(f'Server compute cards {sorted(workers)} differ from intended {list(GPUS)} '
                           'or a card has more than one compute process')
    if len({next(iter(p)) for p in workers.values()}) != len(GPUS):
        raise RuntimeError('One process holds several cards')
    return {'intended_indices': list(GPUS), 'observed_indices': sorted(workers),
            'workers': {str(g): next(iter(p)) for g, p in sorted(workers.items())},
            'compute_apps': apps, 'uuid_to_index': uuid_to_index, 'owned_pids': sorted(owned)}


def anonymous_placement(raw, node):
    """Lower bound of the anonymous memory on `node`: every page reported on
    another node is charged to anon (COW mappings mix file and anon pages)."""
    total = lower = 0
    for line in raw.splitlines():
        fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
        anon = int(fields.get('anon', '0'))
        if not anon:
            continue
        size = int(fields.get('kernelpagesize_kB', '4'))
        other = sum(int(v) for k, v in fields.items()
                    if re.fullmatch(r'N\d+', k) and k != f'N{node}')
        total += anon * size
        lower += max(0, anon - other) * size
    return {'node': node, 'anonymous_kib': total, 'node_lower_bound_kib': lower,
            'node_fraction_lower_bound': lower / total if total else None}


def nccl_shm_pages(raw):
    """Record-only: pages per NUMA node of NCCL's /dev/shm SHM staging buffers
    (MMAP branch on this driver, shm.cc:554); the mapping keeps its name with a
    ' (deleted)' suffix after unlink (shmutils.cc:110-111)."""
    nodes = {}
    for line in raw.splitlines():
        fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
        if not fields.get('file', '').startswith('/dev/shm/nccl-'):
            continue
        for key, value in fields.items():
            match = re.fullmatch(r'N(\d+)', key)
            if match:
                nodes[match[1]] = nodes.get(match[1], 0) + int(value)
    return nodes


def status_lists(pid, proc=Path('/proc')):
    text = (proc / str(pid) / 'status').read_text()
    found = dict(re.findall(r'^(Cpus_allowed_list|Mems_allowed_list):\s*(\S+)$', text, re.M))
    if set(found) != {'Cpus_allowed_list', 'Mems_allowed_list'}:
        raise ValueError(f'No allowed lists in /proc/{pid}/status')
    return found


def placement(server_pid, workers, directory, proc=Path('/proc')):
    """Gate N evidence after warm-up: per worker its Cpus_allowed_list and the
    share of its anonymous memory on its GPU's node; every other server process
    recorded with its affinity."""
    directory = Path(directory)
    directory.mkdir()
    table = process_table(proc)
    by_pid = {int(pid): int(gpu) for gpu, pid in workers.items()}
    rows = {}
    for p in sorted(descendants(table, server_pid)):
        if table[p]['state'] == 'Z':
            continue
        try:
            raw = (proc / str(p) / 'numa_maps').read_text()
            (directory / f'{p}.numa_maps').write_text(raw)
            gpu = by_pid.get(p)
            node = NUMA_NODES[gpu] if gpu is not None else 0
            rows[str(p)] = {'gpu': gpu, 'role': 'worker' if gpu is not None else 'other',
                            **anonymous_placement(raw, node), 'nccl_shm_pages': nccl_shm_pages(raw),
                            'start': table[p]['start'], **status_lists(p, proc)}
        except (OSError, ValueError) as exc:
            rows[str(p)] = {'gpu': by_pid.get(p), 'role': 'worker' if p in by_pid else 'other',
                            'error': str(exc)}
    return rows


def cpu_snapshot(proc=Path('/proc')):
    rows = {}
    for line in (proc / 'stat').read_text().splitlines():
        fields = line.split()
        if re.fullmatch(r'cpu\d+', fields[0]):
            values = list(map(int, fields[1:9]))  # guest already included in user
            rows[fields[0][3:]] = {'total': sum(values), 'busy': sum(values)-values[3]-values[4]}
    return rows


def gpu_snapshot():
    fields = 'index,uuid,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.total'
    raw = command(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'])
    rows = []
    for parts in csv.reader(raw.splitlines(), skipinitialspace=True):
        idx, uuid, clock, power, temp, util, memory, total = parts
        rows.append({'index': int(idx), 'uuid': uuid, 'clock_mhz': float(clock),
                     'power_w': float(power), 'temperature_c': float(temp),
                     'utilization_pct': float(util), 'memory_mib': float(memory),
                     'memory_total_mib': float(total)})
    if sorted(r['index'] for r in rows) != list(range(8)):
        raise RuntimeError('Cannot observe all eight cards')
    raw = command(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader,nounits'])
    apps = [{'pid': int(p[0]), 'uuid': p[1], 'name': p[2]}
            for p in csv.reader(raw.splitlines(), skipinitialspace=True) if p]
    return {'gpus': rows, 'compute_apps': apps}


# Pids of live touch-and-free helpers (hostmem.run adds and
# discards them). They are children of the runner, but are named here too so
# that a sample counts them as owned even if one were reparented.
HELPER_PIDS = set()


def owned_rows(root, table=None, affinity=os.sched_getaffinity, threads=thread_table):
    """Every process of the runner's tree and every live helper pid, with its CPU
    affinity; processes whose affinity spans both sockets also carry per-thread
    ticks and last CPU."""
    table = process_table() if table is None else table
    rows = {}
    for pid in descendants(table, root) | {p for p in set(HELPER_PIDS) if p in table}:
        try:
            cpus = sorted(affinity(pid))
        except OSError:
            continue
        row = {**{k: table[pid][k] for k in ('start', 'ticks', 'child_ticks', 'state')},
               'affinity': format_cpus(cpus)}
        if socket_of(cpus) is None:
            row['threads'] = threads(pid)
        rows[str(pid)] = row
    return rows


class Monitor:
    def __init__(self, root, directory, cpus):
        self.root = root
        self.cpus = cpus
        self.path = Path(directory) / 'telemetry.jsonl'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        return {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns(),
                'cpu': cpu_snapshot(), 'owned': owned_rows(self.root), **gpu_snapshot()}

    def boundary(self):
        """An enclosing sample; a failure is an errored sample (gate Q fails for the level)."""
        try:
            return self.sample()
        except Exception as exc:
            return {'monotonic_ns': time.monotonic_ns(), 'error': str(exc)}

    def run(self):
        os.sched_setaffinity(0, self.cpus)
        with self.path.open('x') as dest:
            while not self.stop.is_set():
                before = time.monotonic()
                try:
                    row = self.sample()
                except Exception as exc:
                    row = {'monotonic_ns': time.monotonic_ns(), 'error': str(exc)}
                dest.write(json.dumps(row) + '\n')
                dest.flush()
                self.stop.wait(max(0, 1 - (time.monotonic() - before)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=45)
        if self.thread.is_alive():
            raise RuntimeError('Telemetry failed to stop')
