"""Restrict the API server to its declared CPUs, then exec it.

Equivalent to `taskset -c 22-25,74-77 <argv>`, plus a receipt. No memory
policy is set: the engine core and the workers are placed by vLLM's own
--numa-bind (numa_utils.py), which this process must be able to widen from
(the numactl probe's mode 'restricted').
"""
import argparse
import json
import os
from pathlib import Path
import platform



def cpu_list(text):
    """'0-21,52-73' -> sorted list (same parser as spec.cpu_list; this file runs as a script)."""
    result = set()
    for part in str(text).strip().split(','):
        if '-' in part:
            low, high = map(int, part.split('-', 1))
            if low > high:
                raise ValueError(f'Descending CPU range in {text!r}')
            result.update(range(low, high + 1))
        else:
            result.add(int(part))
    return sorted(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpus', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    argv = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not argv or platform.machine() != 'x86_64':
        raise RuntimeError('Expected x86_64 and an exec command')
    cpus = set(cpu_list(args.cpus))
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError('Affinity not applied')
    args.receipt.write_text(json.dumps({'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)),
                                       'argv': argv}) + '\n')
    os.execvpe(argv[0], argv, os.environ)


if __name__ == '__main__':
    main()
