"""Guarded server lifecycle for eight cards.

Failures retain the owned server for manual handling; the runner never kills a
server that did not answer. Launch requirements (no foreign GPU process, port free,
no inherited engine or transport overrides, every server process gone and GPU memory
released before the next start) are implemented here and in telemetry.py.
"""
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import gates
from . import spec
from . import telemetry
from .spec import (API_CPUS, ARMS, COMMIT, GPUS, LADDER, MODEL, NUMA_NODES, PARALLEL_FLAGS, REVISION, TP_PP,
                   VLLM_VERSION, cpu_list, digest, output_path, worker_cpus, write_json)


class StartFailure(RuntimeError):
    """An observed startup/answer failure, distinct from missing instrumentation."""


class ManualAttention(RuntimeError):
    """A server process is retained; nothing else may start until a person looks."""


HERE = Path(__file__).resolve().parent


def code_files(directory=HERE):
    """sha256 of every executable file of the package; recorded in every run manifest and
    compared by each phase with the manifests of the phases it builds on."""
    return {p.name: digest(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix in ('.py', '.json')}


DENIED_PREFIXES = ('PYTORCH_', 'OMP_', 'VLLM_', 'NCCL_', 'CUDA_', 'TORCH_NCCL_', 'TORCHINDUCTOR_', 'TRITON_')
OWN_CUDA = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')
DEVIATIONS = {'NCCL_CUMEM_HOST_ENABLE': '0', 'NCCL_P2P_DISABLE': '1'}


def launcher_caches():
    root = spec.data_dir()
    return {'TRITON_CACHE_DIR': str(root / 'cache/triton'),
            'TORCHINDUCTOR_CACHE_DIR': str(root / 'cache/inductor')}


def numactl_dir(probe_receipt):
    """The numactl copy outside the Python environment, verified by hash, and the
    probe's placement mode ('restricted' or 'fallback')."""
    probe = json.loads(Path(probe_receipt).read_text())
    if probe.get('pass') is not True or probe.get('mode') not in ('restricted', 'fallback'):
        raise RuntimeError('The numactl probe did not pass; the placement cannot be applied')
    path_dir = Path(probe['path_dir'])
    binary = Path(probe['numactl'])
    if (path_dir / 'numactl').resolve() != binary.resolve() or digest(binary) != probe['numactl_sha256']:
        raise RuntimeError('numactl differs from the probed binary')
    if sorted(p.name for p in path_dir.iterdir()) != ['numactl']:
        raise RuntimeError('The numactl PATH directory must contain only numactl')
    return {'mode': probe['mode'], 'path_dir': str(path_dir), 'numactl': str(binary),
            'numactl_sha256': probe['numactl_sha256']}


SECRET_NAME = re.compile(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|^SSH_', re.I)
REDACTED = '<redacted>'


def redact_environment(env):
    """The environment as recorded in a manifest: the value of every variable whose name
    contains KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL, or starts with SSH_ (any case), is
    replaced by '<redacted>'. The server itself gets the unredacted environment."""
    return {k: (REDACTED if SECRET_NAME.search(k) else v) for k, v in env.items()}


def vllm_bin():
    """The `vllm` entry point of the running Python environment."""
    return Path(sys.executable).parent / 'vllm'


def check_vllm():
    """vLLM 0.30.0, the CUDA 12.9 build, installed from commit ced6857."""
    if importlib.metadata.version('vllm') != VLLM_VERSION:
        raise RuntimeError(f'vLLM {VLLM_VERSION} (the CUDA 12.9 build) is required')
    version = Path(importlib.metadata.distribution('vllm').locate_file('vllm/_version.py')).read_text()
    if COMMIT[:7] not in version:
        raise RuntimeError('Installed vLLM commit not verified')
    if not vllm_bin().is_file():
        raise RuntimeError(f'No vllm entry point next to {sys.executable}')


def environment(deviations, numactl, transport=False, settings=False, nccl_dir=None):
    check_vllm()
    if any(DEVIATIONS.get(k) != v for k, v in deviations.items()):
        raise RuntimeError('Undeclared startup deviation')
    caches = launcher_caches()
    unsafe = [k for k, v in os.environ.items()
              if (k == 'LD_PRELOAD' or k.startswith(DENIED_PREFIXES)) and k not in OWN_CUDA
              and not (k in deviations and deviations[k] == v)
              and not caches.get(k) == v]
    if unsafe:
        raise RuntimeError(f'Unexpected inherited engine/transport overrides: {unsafe}')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, GPUS)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_CACHE_ROOT=str(spec.data_dir() / 'cache/vllm'), **caches)
    # The numactl copy is first on the server's PATH only.
    env['PATH'] = numactl['path_dir'] + os.pathsep + os.environ.get('PATH', '')
    if Path(shutil.which('numactl', path=env['PATH']) or '').resolve() != Path(numactl['numactl']).resolve():
        raise RuntimeError('numactl on the server PATH is not the probed copy')
    env.update(deviations)
    if transport:
        # Transport-record start only. SHM adds the MMAP
        # host-buffer lines (shm.cc:554, 562); the default mask is INIT|BOOTSTRAP|ENV
        # (debug.cc:69), replaced by this list.
        env['NCCL_DEBUG'] = 'INFO'
        env['NCCL_DEBUG_SUBSYS'] = 'INIT,BOOTSTRAP,ENV,SHM'
        # One line-buffered file per process (debug.cc:241-283), so the
        # eight workers' lines never interleave in a shared stdout.
        if nccl_dir is None:
            raise RuntimeError('Transport start needs a per-process NCCL debug directory')
        env['NCCL_DEBUG_FILE'] = str(Path(nccl_dir) / 'nccl.%h.%p.log')
    if settings:
        env['VLLM_LOGGING_LEVEL'] = 'DEBUG'      # configuration start only
    return env


def argv(arm, port, step):
    """Only the parallelism flags differ between arms; the pool
    flag, the ladder's --max-model-len and the NUMA binding are the same in all."""
    if arm not in ARMS or step not in LADDER:
        raise ValueError((arm, step))
    ladder = LADDER[step]
    result = [str(vllm_bin()), 'serve', str(spec.model_dir()), '--served-model-name', MODEL,
              '--host', '127.0.0.1', '--port', str(port), *PARALLEL_FLAGS[arm],
              '--kv-cache-memory-bytes', str(ladder['kv_cache_memory_bytes'])]
    if ladder['max_model_len'] is not None:
        result += ['--max-model-len', str(ladder['max_model_len'])]
    result += ['--numa-bind', '--numa-bind-nodes', *map(str, NUMA_NODES),
               '--numa-bind-cpus', *(worker_cpus(g) for g in GPUS)]
    return result


HTTP_TIMEOUT_S = 15


def http(url, method='GET', timeout=HTTP_TIMEOUT_S):
    with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=timeout) as response:
        return response.read().decode()


def check_port(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))


def preflight(port):
    """Before anything of a start runs: no compute process on
    any card and the port free, the same checks Server.__init__ repeats. Called by
    runner.open_server before the touch-and-free helper."""
    gpu = telemetry.gpu_snapshot()
    if gpu['compute_apps']:
        raise RuntimeError('Foreign compute process present; leave it untouched')
    check_port(port)
    return gpu


def assert_server_identity(process, start, port, model_data, proc=Path('/proc')):
    table = telemetry.process_table(proc)
    if process.poll() is not None or table.get(process.pid, {}).get('start') != start:
        raise RuntimeError('Answering server process exited or changed identity')
    owned = telemetry.descendants(table, process.pid)
    listeners = set()
    for line in (proc / 'net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        address, hexport = fields[1].split(':')
        if int(hexport, 16) == port and fields[3] == '0A' and address in ('0100007F', '00000000'):
            listeners.add(fields[9])
    sockets = set()
    for pid in owned:
        for fd in (proc / str(pid) / 'fd').glob('*'):
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r'socket:\[(\d+)\]', target)
            if match:
                sockets.add(match[1])
    if not listeners or not listeners <= sockets:
        raise RuntimeError('Answering listening socket is not owned by the server tree')
    models = model_data.get('data', [])
    if len(models) != 1 or models[0].get('id') != MODEL or models[0].get('root') != str(spec.model_dir()):
        raise RuntimeError('Answering server model path/name differs from the pinned snapshot')
    return {'pid': process.pid, 'start': start, 'owned_pids': sorted(owned),
            'listening_inodes': sorted(listeners), 'models': model_data}


def lock_paths(output, extra=()):
    """run.lock in the output directory, gpu<N>.lock in the lock directory, and every `--lock`
    pattern: `{gpu}` in a pattern is expanded for each of the eight GPUs, so another job's
    per-GPU lock files can be taken too. Paths are de-duplicated (a path already taken is
    taken once). A pattern without `{gpu}` that names one of the run's own lock files is
    refused: it would lock a file the run already holds."""
    lockroot = spec.lock_dir()
    own = [Path(output) / 'run.lock'] + [lockroot / f'gpu{g}.lock' for g in GPUS]
    paths, seen = [], set()
    for path in own:
        key = Path(os.path.abspath(path))
        seen.add(key)
        paths.append(path)
    own_keys = set(seen)
    for pattern in extra:
        pattern = str(pattern)
        expanded = [pattern.replace('{gpu}', str(g)) for g in GPUS] if '{gpu}' in pattern else [pattern]
        for item in expanded:
            key = Path(os.path.abspath(item))
            if '{gpu}' not in pattern and key in own_keys:
                raise ValueError(f'--lock {pattern} names one of the run\'s own lock files')
            if key not in seen:
                seen.add(key)
                paths.append(key)
    return paths


class Locks:
    def __init__(self, output, extra=()):
        self.files = []
        output = output_path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('Use a new run directory; never overwrite prior evidence')
        self.paths = lock_paths(output, extra)
        try:
            for path in self.paths:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                file = Path(path).open('a+')
                self.files.append(file)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise

    @property
    def fds(self):
        return tuple(f.fileno() for f in self.files)

    def close(self):
        # close, not LOCK_UN: inherited server descriptors keep the flock alive
        # if the runner fails or loses its SSH connection.
        for f in self.files:
            f.close()
        self.files.clear()


def nccl_progress(log):
    marker = log.find('vLLM is using nccl==')
    if marker < 0:
        return False
    later = log[marker:].splitlines()[1:]
    return any(re.search(r'load(?:ing)?.*(?:weight|model|checkpoint)|(?:weight|model).*load|'
                         r'compil|captur.*(?:graph|cuda)|Application startup complete',
                         line, re.I) for line in later)


def startup_signature(log, returncode, timed_out, nccl_observed_seconds=0):
    """The two allowed startup deviations and the failure signatures that permit them."""
    worker_segv = re.search(r'Worker proc .* died unexpectedly \(exit code: -11\)', log)
    fatal_segv = re.search(r'Fatal Python error: Segmentation fault', log)
    if (returncode in (-signal.SIGSEGV, 139) or worker_segv or fatal_segv) and 'cuMemCreate' in log:
        return 'NCCL_CUMEM_HOST_ENABLE'
    if (timed_out and returncode is None and nccl_observed_seconds >= 1800
            and 'vLLM is using nccl==' in log and not nccl_progress(log)):
        return 'NCCL_P2P_DISABLE'
    return None


def deviations_from_receipt(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    result = {}
    for failure in data['failures']:
        manifest_path = Path(failure['manifest'])
        if digest(manifest_path) != failure['manifest_sha256']:
            raise RuntimeError('Startup manifest evidence changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('deviations') != {} or failure.get('deviations') != {}:
            raise RuntimeError('An explicitly default start is required')
        if manifest.get('model') != MODEL or manifest.get('revision') != REVISION:
            raise RuntimeError('Wrong startup model evidence')
        if manifest.get('kind') != 'start-test' or manifest.get('arm') not in ARMS:
            raise RuntimeError('A default start-test artifact is required')
        logpath = Path(failure['log'])
        if digest(logpath) != failure['log_sha256']:
            raise RuntimeError('Startup failure evidence changed')
        key = startup_signature(logpath.read_text(), failure['returncode'], failure['timed_out'],
                                failure.get('nccl_observed_seconds', 0))
        if not key:
            raise RuntimeError('Deviation requires a matching default-start failure')
        result[key] = DEVIATIONS[key]
    return result


class Server:
    def __init__(self, arm, directory, topology, locks, deviations, port, kind, step, numactl):
        self.directory = output_path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.process = None
        self.completed = False
        self.workers = None
        self.url = f'http://127.0.0.1:{port}'
        self.port = port
        self.kind = kind
        self.placement_mode = numactl['mode']
        self.nccl_dir = self.directory / 'nccl' if kind == 'transport' else None
        if self.nccl_dir is not None:
            self.nccl_dir.mkdir()
        env = environment(deviations, numactl, transport=kind == 'transport',
                          settings=kind == 'settings', nccl_dir=self.nccl_dir)
        gpu = telemetry.gpu_snapshot()
        self.before_gpus = gpu
        self.known_processes = {}
        write_json(self.directory / 'before-gpus.json', gpu)
        if gpu['compute_apps']:
            raise RuntimeError('Foreign compute process present; leave it untouched')
        check_port(port)
        for node in (0, 1):
            (self.directory / f'node{node}-memory-before.txt').write_text(
                Path(f'/sys/devices/system/node/node{node}/meminfo').read_text())
        command = argv(arm, port, step)
        if self.placement_mode == 'restricted':
            launched = [sys.executable, str(HERE / 'placement_exec.py'), '--cpus', API_CPUS,
                        '--receipt', str(self.directory / 'placement-exec.json'), '--', *command]
        else:
            launched = command       # fallback: moved with taskset after the workers are up
        manifest = {'arm': arm, 'kind': kind, 'step': step, 'argv': launched, 'server_argv': command,
                    'model': MODEL, 'revision': REVISION, 'vllm_commit': COMMIT, 'deviations': deviations,
                    'tp_pp': TP_PP[arm], 'placement_mode': self.placement_mode, 'numactl': numactl,
                    'versions': {k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers')},
                    'nvidia_dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                                            if d.metadata['Name'].lower().startswith('nvidia-')},
                    'environment': redact_environment(env),
                    'generation_config': json.loads((spec.model_dir() / 'generation_config.json').read_text()),
                    'generation_config_sha256': digest(spec.model_dir() / 'generation_config.json'),
                    'topology': {k: v for k, v in topology.items() if k != 'lscpu'},
                    'start_wall_ns': time.time_ns()}
        if 'NCCL_CUMEM_HOST_ENABLE' in deviations:
            manifest['deviation_notes'] = {'NCCL_CUMEM_HOST_ENABLE': 'no-op below CUDA driver 12.6: '
                                           'cuMem host allocation is off below 12060 (NCCL cudawrap.cc:66-72)'}
        write_json(self.directory / 'manifest.json', manifest)
        self.deviations = dict(deviations)
        self.log = (self.directory / 'server.log').open('x')
        try:
            self.process = subprocess.Popen(launched, stdin=subprocess.DEVNULL, stdout=self.log,
                                            stderr=subprocess.STDOUT, env=env, start_new_session=True,
                                            pass_fds=locks.fds)
            self.identity = telemetry.process_table()[self.process.pid]['start']
            write_json(self.directory / 'server.pid.json', {'pid': self.process.pid, 'start': self.identity})
        except BaseException:
            self.retain('Server constructor failed after possible launch')
            raise

    def log_text(self):
        return (self.directory / 'server.log').read_text(errors='replace')

    def ready(self, timeout=1800):
        deadline = time.monotonic()+timeout
        nccl_seen = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            log = self.log_text()
            if nccl_seen is None and 'vLLM is using nccl==' in log:
                nccl_seen = time.monotonic()
                deadline = max(deadline, nccl_seen + 1800)
            try:
                if http(self.url + '/health') is not None:
                    identity = assert_server_identity(self.process, self.identity, self.port,
                                                      json.loads(http(self.url + '/v1/models')))
                    write_json(self.directory / 'ready-identity.json', identity)
                    snapshot = telemetry.gpu_snapshot()
                    cards = telemetry.assert_server_cards(self.process.pid, snapshot)
                    cards['free_mib_after_start'] = {str(g['index']): g['memory_total_mib'] - g['memory_mib']
                                                     for g in snapshot['gpus']}
                    write_json(self.directory / 'ready-cards.json', cards)
                    self.workers = cards['workers']
                    self.track_processes()
                    self.api_server_placement()
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        code = self.process.poll()
        write_json(self.directory / 'start-failure.json',
                   {'returncode': code, 'timed_out': code is None,
                    'nccl_observed_seconds': time.monotonic()-nccl_seen if nccl_seen is not None else 0,
                    'log': str(self.directory / 'server.log'),
                    'log_sha256': digest(self.directory / 'server.log'),
                    'deviations': self.deviations,
                    'manifest': str(self.directory / 'manifest.json'),
                    'manifest_sha256': digest(self.directory / 'manifest.json')})
        if code is None:
            self.retain('Start timed out with the server alive; manual handling required')
        raise StartFailure('Start failed; server has not been automatically killed')

    def api_server_placement(self, run=subprocess.run, proc=Path('/proc')):
        """The API server holds CPUs 22-25,74-77. Restricted mode
        launched it there; fallback mode moves every thread now that the
        workers are up (`taskset -a -p`; plain `-p` would move one thread)."""
        pid = self.process.pid
        record = {'mode': self.placement_mode, 'pid': pid}
        if self.placement_mode == 'fallback':
            command = ['taskset', '-a', '-p', '-c', API_CPUS, str(pid)]
            done = run(command, capture_output=True, text=True, timeout=20)
            record.update(command=command, returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)
            if done.returncode != 0:
                write_json(self.directory / 'api-server-placement.json', record)
                raise gates.EvidenceError('taskset could not move the API server')
        wanted = set(cpu_list(API_CPUS))
        threads = {t.name: sorted(os.sched_getaffinity(int(t.name))) for t in (proc / str(pid) / 'task').iterdir()}
        record['thread_affinity'] = {k: telemetry.format_cpus(v) for k, v in threads.items()}
        record['pass'] = bool(threads) and all(set(v) == wanted for v in threads.values())
        write_json(self.directory / 'api-server-placement.json', record)
        if not record['pass']:
            raise gates.EvidenceError('API server is not on its declared CPUs')

    def retain(self, reason):
        try:
            write_json(self.directory / 'MANUAL_ATTENTION.json',
                       {'pid': self.process.pid if self.process else None, 'reason': reason,
                        'action': 'Inspect exact pid and elapsed time; user decides stopping'})
        finally:
            if getattr(self, 'log', None):
                self.log.flush()

    def released(self):
        table = self.track_processes()
        alive = {p: start for p, start in self.known_processes.items()
                 if table.get(p, {}).get('start') == start and table[p]['state'] != 'Z'}
        gpu = telemetry.gpu_snapshot()
        baseline = {g['index']: g['memory_mib'] for g in self.before_gpus['gpus']}
        ok = all(g['memory_mib'] <= baseline[g['index']] for g in gpu['gpus'] if g['index'] in baseline)
        return (not alive and not gpu['compute_apps'] and ok), {'alive': alive, 'gpus': gpu}

    def wait_released(self, reason, timeout=60):
        """Every process of the server tree exited and all eight cards' memory back to
        the pre-start values, within 60 s."""
        deadline = time.monotonic() + timeout
        while True:
            self.process.poll()
            done, state = self.released()
            if done:
                write_json(self.directory / 'shutdown.json',
                           {'known_processes': self.known_processes, 'gpus': state['gpus'], 'released': True,
                            'reason': reason})
                self.log.close()
                self.completed = True
                return
            if time.monotonic() >= deadline:
                self.retain(f'Server tree or GPU memory not released within {timeout} s; no next start')
                raise ManualAttention('Shutdown incomplete; no next server is permitted')
            time.sleep(1)

    def close_completed(self, timeout=60):
        """Normal end of an answering start; not a failure/timeout cancellation."""
        self.track_processes()
        if self.process.poll() is not None:
            if self.process.returncode != 0:
                self.retain('Server exited unexpectedly')
                raise RuntimeError('Server exited unexpectedly')
        else:
            if telemetry.process_table()[self.process.pid]['start'] != self.identity:
                raise RuntimeError('PID identity changed')
            os.kill(self.process.pid, signal.SIGTERM)
        self.wait_released('normal shutdown', timeout=timeout)

    def settle_after_failure(self):
        """After a failed start test or a gate M failure: a server that still
        answers is shut down normally; a server that exited must be fully gone
        within 60 s; a live, unanswering server is retained (ManualAttention)."""
        if self.completed:
            return 'closed'
        self.track_processes()
        if self.process.poll() is None:
            try:
                http(self.url + '/health', timeout=5)
            except (OSError, urllib.error.URLError):
                self.retain('Server alive but not answering after a failure')
                raise ManualAttention('Live, unanswering server retained')
            os.kill(self.process.pid, signal.SIGTERM)
            self.wait_released('shutdown after a recorded failure')
            return 'closed'
        self.wait_released('server exited after a failure')
        return 'exited'

    def track_processes(self):
        table = telemetry.process_table()
        owned = {p for p, r in table.items() if r['session'] == self.process.pid}
        if table.get(self.process.pid, {}).get('start') == self.identity:
            owned |= telemetry.descendants(table, self.process.pid)
        for p in owned:
            self.known_processes.setdefault(p, table[p]['start'])
        return table


# Per-start record: compile lines are recorded only. With the pool pinned by
# --kv-cache-memory-bytes the compile state does not move the pool, so no gate
# depends on it.
COMPILE_SCRATCH = r'Compiling a graph for compile range|Compiling model again due to a load failure'
COMPILE_CACHED = (r'Directly load AOT compilation from path|Directly load the compiled graph'
                  r'|reconstructed serializable fn from standalone compile artifacts')


def start_record(log, metrics, step):
    lines = log.splitlines()
    scratch = [x for x in lines if re.search(COMPILE_SCRATCH, x)]
    cached = [x for x in lines if re.search(COMPILE_CACHED, x)]
    return {'step': step, 'kv_capacity_tokens': gates.capacity(metrics),
            'kv_reserved': gates.kv_reserved(log, LADDER[step]['kv_cache_memory_bytes']),
            'torch_compile_lines': [x for x in lines if 'torch.compile took' in x],
            'compile_scratch_lines': scratch, 'compile_cached_lines': cached,
            'compile_mode': 'scratch' if scratch else 'cached' if cached else 'unknown'}


ATTENTION_LINE = r'Using (\S+) attention backend|Using (\S+) backend\.'


def startup_fields(log, metrics, arm):
    """Configuration-start evidence, with source-labelled
    deductions, never defaults."""
    config = gates.cache_config(metrics)
    fields = {}
    for name in ('max_seq_len', 'enable_prefix_caching', 'async_scheduling', 'max_num_batched_tokens',
                 'max_num_seqs', 'max_concurrent_batches'):
        hits = re.findall(r'\b' + name + r'=([^,\s)]+)', log)
        fields[name] = hits[-1] if hits else None
    for name in ('max_num_seqs', 'max_num_batched_tokens'):
        # vllm/engine/arg_utils.py:2990 "Defaulting max_num_seqs to %d for %s usage context."
        matches = re.findall(r'Defaulting ' + name + r' to (\d+) for', log)
        if matches:
            fields[name] = matches[-1]
    # vllm/v1/engine/core.py:215 (DEBUG) "Batch queue is enabled with size %d"
    batches = re.findall(r'Batch queue is enabled with size (\d+)', log)
    if batches:
        fields['max_concurrent_batches'] = batches[-1]
    for name in ('gpu_memory_utilization', 'enable_prefix_caching', 'num_gpu_blocks', 'block_size'):
        fields[name] = config.get(name)
    # vllm/v1/worker/gpu_worker.py:441 info_once("Using V2 Model Runner")
    fields['model_runner'] = 'V2' if 'Using V2 Model Runner' in log else None
    fields['deductions'] = {}
    pp_size = TP_PP[arm][1]
    if fields['max_concurrent_batches'] == str(pp_size + 1):
        # vllm/config/vllm.py:588-599: pp_size + 1 only with async scheduling.
        fields['async_scheduling'] = 'True'
        fields['deductions']['async_scheduling'] = 'Observed batch queue size; vllm/config/vllm.py:588-599'
    if fields['model_runner'] is None and 'gpu_model_runner.py:' in log:
        fields['model_runner'] = 'V1'
        fields['deductions']['model_runner'] = 'V1 module emitted log; gpu_worker.py:469-490'
    fields['cuda_graph_log_lines'] = [x for x in log.splitlines() if 'cudagraph' in x.lower()]
    fields['cache_config_metric_lines'] = [x for x in metrics.splitlines() if x.startswith('vllm:cache_config_info')]
    backend = re.findall(ATTENTION_LINE, log)
    fields['attention_backend'] = (backend[-1][0] or backend[-1][1]) if backend else None
    fields['allreduce'] = gates.allreduce_lines(log)
    fields['binding'] = gates.binding(log)
    # Recorded as unset; environment() refuses every inherited VLLM_* variable.
    fields['VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC'] = 'unset'
    fields['missing_fields'] = [k for k, v in fields.items() if v is None]
    if not fields['cuda_graph_log_lines']:
        fields['missing_fields'].append('cuda_graph_log_lines')
    return fields
