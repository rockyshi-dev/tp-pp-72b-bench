"""The CPU-only `numactl` probe, run before anything else on the machine.

Run, with no GPU process,

  taskset -c 22-25,74-77 numactl --cpunodebind=0,1 --membind=0,1 sh -c \
    "grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status; \
     numactl --physcpubind=0-21,52-73 --membind=0 \
       grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status"

and the same with `26-43,78-95 --membind=1`. Each must exit 0 and the outer level
must print Cpus_allowed_list 0-103 (both nodes) and Mems_allowed_list 0-1.
This is the chain the server uses: API server restricted (22-25,74-77) ->
engine core `--cpunodebind=0,1 --membind=0,1` (vllm/utils/numa_utils.py:372-446)
-> workers `--physcpubind=<list> --membind=<node>` (numa_utils.py:345-369),
each argument set first probed with `numactl <args> true`
(numa_utils.py:476-505).

The inner level must print Cpus_allowed_list exactly the requested CPU list and
Mems_allowed_list equal to the outer level's (0-1) -- `--membind` sets a memory
policy, not the cpuset that Mems_allowed_list shows -- and `numactl --show`, run
at the inner level of the same chain (same outer restriction, same inner
arguments), must print one `membind:` line whose node tokens are exactly the
requested node. That run must exit 0 as well.

Fallback: if the restricted parent cannot widen, the API server is launched
unrestricted and moved with taskset after the workers are up. The probe therefore
runs the same commands without the outer `taskset` when the restricted chain
fails; mode 'fallback' means only the unrestricted chain passed; 'neither' means
the placement cannot be applied on this machine. Everything each command prints
(stdout, stderr, exit code) is recorded.
"""
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from . import spec
from .spec import API_CPUS, ENGINE_CORE_NODES, SOCKET_CPUS, WORKER_CPUS, cpu_list, digest, write_json

GREP = "grep -E 'Cpus_allowed_list|Mems_allowed_list' /proc/self/status"
ALL_CPUS = sorted(set(cpu_list(SOCKET_CPUS[0])) | set(cpu_list(SOCKET_CPUS[1])))   # 0-103
ALL_NODES = [0, 1]


def command(node, restricted=True):
    inner = f'numactl --physcpubind={WORKER_CPUS[node]} --membind={node} {GREP}'
    argv = ['numactl', f'--cpunodebind={ENGINE_CORE_NODES}', f'--membind={ENGINE_CORE_NODES}',
            'sh', '-c', f'{GREP}; {inner}']
    return (['taskset', '-c', API_CPUS] if restricted else []) + argv


def show_command(node, restricted=True):
    """`numactl --show` at the inner level of the same chain."""
    inner = f'numactl --physcpubind={WORKER_CPUS[node]} --membind={node} numactl --show'
    argv = ['numactl', f'--cpunodebind={ENGINE_CORE_NODES}', f'--membind={ENGINE_CORE_NODES}',
            'sh', '-c', inner]
    return (['taskset', '-c', API_CPUS] if restricted else []) + argv


def parse_status(stdout):
    """Two levels, each printing Cpus_allowed_list then Mems_allowed_list
    (their order in /proc/<pid>/status)."""
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(':')
        if key not in ('Cpus_allowed_list', 'Mems_allowed_list') or not value.strip():
            raise ValueError(f'Unexpected probe output line: {line!r}')
        rows.append((key, cpu_list(value.strip())))
    keys = [k for k, _ in rows]
    if keys != ['Cpus_allowed_list', 'Mems_allowed_list'] * 2:
        raise ValueError(f'Expected two levels of Cpus/Mems lines, got {keys}')
    return {'outer': {'cpus': rows[0][1], 'mems': rows[1][1]},
            'inner': {'cpus': rows[2][1], 'mems': rows[3][1]}}


def parse_membind(stdout):
    """The node tokens of the single `membind:` line of `numactl --show`
    (e.g. 'membind: 0 ' -> ['0']); None if there is no such line."""
    lines = [line for line in stdout.splitlines() if line.startswith('membind:')]
    if len(lines) > 1:
        raise ValueError(f'More than one membind line: {lines}')
    return lines[0].partition(':')[2].split() if lines else None


def evaluate(node, returncode, stdout, show_returncode, show_stdout):
    result = {'node': node, 'returncode': returncode, 'stdout': stdout,
              'show_returncode': show_returncode, 'show_stdout': show_stdout,
              'requested_inner_cpus': WORKER_CPUS[node]}
    try:
        levels = parse_status(stdout)
        membind = parse_membind(show_stdout)
    except ValueError as exc:
        return {**result, 'pass': False, 'error': str(exc)}
    # Inner Mems_allowed_list equals the outer level's, and
    # the inner node is read from `numactl --show`'s membind line.
    checks = {'exit_0': returncode == 0,
              'show_exit_0': show_returncode == 0,
              'outer_cpus_0_103': levels['outer']['cpus'] == ALL_CPUS,
              'outer_mems_0_1': levels['outer']['mems'] == ALL_NODES,
              'inner_cpus_requested': levels['inner']['cpus'] == cpu_list(WORKER_CPUS[node]),
              'inner_mems_equal_outer': levels['inner']['mems'] == levels['outer']['mems'],
              'inner_membind_its_node': membind == [str(node)]}
    return {**result, 'levels': levels, 'membind_tokens': membind, 'checks': checks,
            'pass': all(checks.values())}


def run_chain(restricted, env, run=subprocess.run):
    rows = []
    for node in (0, 1):
        argv = command(node, restricted)
        done = run(argv, capture_output=True, text=True, timeout=30, env=env)
        show_argv = show_command(node, restricted)
        show = run(show_argv, capture_output=True, text=True, timeout=30, env=env)
        rows.append({'argv': argv, 'stderr': done.stderr, 'show_argv': show_argv, 'show_stderr': show.stderr,
                     **evaluate(node, done.returncode, done.stdout, show.returncode, show.stdout)})
    return {'restricted': restricted, 'pass': all(r['pass'] for r in rows), 'commands': rows}


def probe(numactl_receipt, run=subprocess.run):
    install = json.loads(Path(numactl_receipt).read_text())
    binary = Path(install['numactl'])
    if digest(binary) != install['numactl_sha256']:
        raise RuntimeError('numactl binary differs from its install receipt')
    path_dir = Path(install['path_dir'])
    if (path_dir / 'numactl').resolve() != binary.resolve():
        raise RuntimeError('PATH entry does not resolve to the recorded binary')
    env = {**os.environ, 'PATH': f'{path_dir}:{os.environ.get("PATH", "/usr/bin:/bin")}',
           'CUDA_VISIBLE_DEVICES': ''}
    chains = [run_chain(True, env, run)]
    if not chains[0]['pass']:
        chains.append(run_chain(False, env, run))
    mode = 'restricted' if chains[0]['pass'] else ('fallback' if chains[-1]['pass'] else 'neither')
    return {'mode': mode, 'pass': mode in ('restricted', 'fallback'), 'chains': chains,
            'numactl': str(binary), 'numactl_sha256': install['numactl_sha256'],
            'path_dir': str(path_dir), 'host': platform.node(), 'kernel': platform.release(),
            'wall_ns': time.time_ns()}


def cli(numactl_receipt=None, output=None):
    numactl_receipt = Path(numactl_receipt or spec.data_dir() / 'setup/numactl-install.json')
    output = Path(output or spec.data_dir() / 'setup/numactl-probe.json')
    if output.exists():
        raise FileExistsError(output)
    result = probe(numactl_receipt)
    write_json(output, result)
    print(json.dumps({'mode': result['mode'], 'pass': result['pass'], 'output': str(output)}))
    return result['pass']
