"""Fixed identities of the benchmark: model pin, arms, cards, CPU lists, KV-pool ladder,
workload, rotation. Importing this file never imports vLLM or torch.

Machine paths (data directory, model snapshot, lock directory) are not constants: they come
from command-line flags or environment variables through `configure` (README, "Paths").
"""
import hashlib
import json
import os
from pathlib import Path

# Model
MODEL = 'Qwen/Qwen2.5-72B-Instruct'
REVISION = '495f39366efef23836d0cfae4fbe635880d2be31'
MODEL_FILES = 47           # files of the pinned revision (37 safetensors)
MODEL_SAFETENSORS_BYTES = 145_412_519_312
# vLLM v0.30.0; the +cu129 wheel of this commit
COMMIT = 'ced6857afa0ea7b2e3f0846a62e1394e90f15607'
VLLM_VERSION = '0.30.0+cu129'
# Touch-and-free on both NUMA nodes before every server start.
HOST_MEMORY_NODES = (0, 1)
USABLE_FREE_AFTER_HELPER = 20 * 10**9    # per node, decimal bytes; a precondition, not a gate
HOST_MEMORY_NOTE = 'touch-and-free before every server start (both NUMA nodes); precondition, not a gate'

# ---------------------------------------------------------------- paths (configurable)

_PATHS = {}


def default_model_dir():
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
    return hf_home / 'hub' / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / REVISION


def configure(data_dir=None, model=None, lock_dir=None):
    """Set the machine paths. Each falls back to its environment variable, then to a default:
    data directory TPPB_DATA_DIR (default ./tppb-data), model snapshot TPPB_MODEL (default the
    pinned snapshot under $HF_HOME), lock directory TPPB_LOCK_DIR (default <data>/locks)."""
    data = Path(data_dir or os.environ.get('TPPB_DATA_DIR') or 'tppb-data').resolve()
    _PATHS['data'] = data
    _PATHS['model'] = Path(model or os.environ.get('TPPB_MODEL') or default_model_dir()).resolve()
    _PATHS['locks'] = Path(lock_dir or os.environ.get('TPPB_LOCK_DIR') or data / 'locks').resolve()
    return dict(_PATHS)


def data_dir():
    if 'data' not in _PATHS:
        configure()
    return _PATHS['data']


def model_dir():
    if 'model' not in _PATHS:
        configure()
    return _PATHS['model']


def lock_dir():
    if 'locks' not in _PATHS:
        configure()
    return _PATHS['locks']


# Arms: only the parallelism flags differ between arms.
ARMS = ('TP8', 'TP4xPP2', 'PP8')
PARALLEL_FLAGS = {'TP8': ('--tensor-parallel-size', '8'),
                  'TP4xPP2': ('-tp', '4', '-pp', '2'),
                  'PP8': ('--pipeline-parallel-size', '8')}
TP_PP = {'TP8': (8, 1), 'TP4xPP2': (4, 2), 'PP8': (1, 8)}
TP_ARMS = ('TP8', 'TP4xPP2')
# Gate P minimum matches: adjacent stage boundaries by NVML device index.
STAGE_BOUNDARIES = {'TP8': (), 'TP4xPP2': ((0, 4), (1, 5), (2, 6), (3, 7)),
                    'PP8': tuple((i, i + 1) for i in range(7))}

# Cards and placement (two-socket machine; README "Requirements")
GPUS = tuple(range(8))
NUMA_NODES = (0, 0, 0, 0, 1, 1, 1, 1)
WORKER_CPUS = {0: '0-21,52-73', 1: '26-43,78-95'}      # per NUMA node
API_CPUS = '22-25,74-77'
CLIENT_CPUS = '44-51'
RUNNER_CPUS = '96-103'
SOCKET_CPUS = {0: '0-25,52-77', 1: '26-51,78-103'}
ENGINE_CORE_NODES = '0,1'

# KV pool ladder (bytes per GPU, --max-model-len, levels)
LADDER = {
    1: {'kv_cache_memory_bytes': 2415919104, 'max_model_len': None, 'levels': (1, 8, 32)},
    2: {'kv_cache_memory_bytes': 1342177280, 'max_model_len': 8192, 'levels': (1, 8, 16)},
    3: {'kv_cache_memory_bytes': 805306368, 'max_model_len': 8192, 'levels': (1, 8)},
}

# Workload
PROMPT_TOKENS = 1536
OUTPUT_TOKENS = 128
REQUEST_TOKENS = PROMPT_TOKENS + OUTPUT_TOKENS        # 1,664 (gate K)
# Measured requests per level after a warm-up of 2c: 30, 80, 160 at c = 1, 8, 32;
# c = 16 (top level at ladder step 2): 160.
MEASURED = {1: 30, 8: 80, 16: 160, 32: 160}
# Prompt banks: measured prompts per level.
BANK_MEASURED = {1: 30, 8: 80, 16: 160, 32: 160}
BANK_LEVELS = (1, 8, 16, 32)
BURST_REQUESTS = 64        # start test: one burst of 64 requests at the top level
TRANSPORT_REQUESTS = 8
GATE_E_C1_PROMPTS = 30     # gate E: the 30 measured c = 1 prompts of bank 0
# Gate E premise check: the request's stop ids (eos 151645 and
# generation_config's 151643). min_tokens masks them at every position of a
# 128/128 request while the raw logprobs still carry them.
STOP_TOKEN_IDS = (151645, 151643)
# Gate E requests only (start-test, transport and timed requests keep the model's
# default sampling parameters).
GATE_E_REPETITION_PENALTY = 1.0
ROUNDS = 5
ROTATION = ('TP8', 'TP4xPP2', 'PP8')   # round k: rotated left by k - 1


def cpu_list(text):
    """Parse a Linux / numactl CPU list ('0-21,52-73') into a sorted list."""
    result = set()
    for part in str(text).strip().split(','):
        if not part:
            raise ValueError(f'Empty CPU list element in {text!r}')
        if '-' in part:
            low, high = part.split('-', 1)
            low, high = int(low), int(high)
            if low > high:
                raise ValueError(f'Descending CPU range in {text!r}')
            result.update(range(low, high + 1))
        else:
            result.add(int(part))
    return sorted(result)


def worker_cpus(gpu):
    return WORKER_CPUS[NUMA_NODES[gpu]]


def round_order(number, arms=ARMS):
    """[TP8, TP4xPP2, PP8] rotated left by k - 1; rounds 4 and 5
    repeat the orders of rounds 1 and 2. Arms no longer in the study keep their
    places in the rotation and are skipped."""
    if not 1 <= number <= ROUNDS:
        raise ValueError(number)
    shift = (number - 1) % 3
    order = ROTATION[shift:] + ROTATION[:shift]
    return tuple(a for a in order if a in arms)


def levels(step):
    return LADDER[step]['levels']


def capacity_needed(step):
    return max(levels(step)) * REQUEST_TOKENS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_path(path):
    return Path(path).resolve()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
